"""Scene provenance, supervision, exposure, and interruption recovery."""
import csv
import json
import random
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (
    BookWindows,
    CausalCollator,
    SceneReplayDataset,
    apply_scene_review,
    extract_scene_candidates,
    loss_sums,
    scene_sentence_ends,
    training_exposure,
    training_trainer,
    window_epoch_callback,
    write_scene_sheet,
)


@pytest.fixture
def scene_data(tmp_path):
    path = tmp_path / "train.jsonl"
    ids = [0] + list(range(2, 32))*3 + [0]
    path.write_text(json.dumps({"id": "46", "ids": ids}) + "\n")
    replay = BookWindows(path, 16, random_offsets=True)
    record = {"scene_id": "46-2-13", "book_id": "46", "start_token": 2, "end_token": 13,
              "prefix_tokens": 4, "input_ids": ids[2:13], "original_book_end": False,
              "natural_end": False, "text_sha256": "fixture", "text": "An original passage.", "score": 1.0}
    return replay, record


def test_scene_prefix_is_context_and_real_eos_remains_a_target(scene_data):
    replay, record = scene_data
    mixture = SceneReplayDataset(replay, [{**record, "natural_end": True}], eos_id=0)
    scene = mixture[0]
    assert scene["input_ids"] == record["input_ids"]+[0]
    assert scene["labels"][:4] == [-100]*4 and scene["labels"][-1] == 0
    batch = CausalCollator(1)([scene, replay[0]])
    assert batch["labels"][0, 11] == 0
    assert batch["labels"][0, 12:].tolist() == [-100]*4
    assert batch["attention_mask"][0, :12].tolist() == [1]*12
    totals = loss_sums(torch.randn(2, 16, 32), batch["labels"])
    assert totals[:, 1].tolist() == [8, 15]


def test_automatic_excerpt_cut_does_not_teach_false_eos(scene_data):
    replay, record = scene_data
    mixture = SceneReplayDataset(replay, [record], 0)
    assert mixture[0]["input_ids"] == record["input_ids"]
    assert mixture[0]["labels"].count(0) == 0
    end = len(replay.books[0])-1
    final = {**record, "start_token": end-10, "end_token": end,
             "input_ids": replay.books[0][end-10:end].tolist(), "original_book_end": True}
    assert SceneReplayDataset(replay, [final], 0)[0]["labels"][-1] == 0


@pytest.mark.parametrize("edit", [
    {"book_id": "validation-only"}, {"input_ids": [2, 3, 4]}, {"original_book_end": True},
    {"prefix_tokens": 99}, {"start_token": 3},
])
def test_scene_injection_and_invalid_boundaries_are_rejected(scene_data, edit):
    replay, record = scene_data
    with pytest.raises(ValueError):
        SceneReplayDataset(replay, [{**record, **edit}], 0)


def test_review_only_changes_selection_and_natural_end(scene_data, tmp_path):
    _, record = scene_data
    path = tmp_path/"review.csv"
    write_scene_sheet(path, [record])
    assert not apply_scene_review([record], path)[0]["natural_end"]
    rows = list(csv.DictReader(path.open()))
    rows[0]["natural_end"] = "1"
    def save():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save()
    assert apply_scene_review([record], path)[0]["natural_end"]
    rows[0]["text"] = "A rewritten story from somewhere else."
    save()
    with pytest.raises(ValueError, match="original content"):
        apply_scene_review([record], path)


def test_scene_epoch_covers_replay_and_counts_masked_targets(scene_data):
    replay, record = scene_data
    mixture = SceneReplayDataset(replay, [record], 0)
    assert len(mixture) == 2*len(replay)
    assert [mixture[i] for i in range(1, len(mixture), 2)] == list(replay)
    rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    mixture.set_training_epoch(3)
    expected = {"examples": 0, "scored_targets": 0, "scene_targets": 0, "replay_targets": 0}
    steps, effective_batch = 9, 4
    remaining = steps
    for epoch in range(3):
        mixture.set_training_epoch(epoch)
        per_epoch = (len(mixture)+effective_batch-1)//effective_batch
        order = torch.randperm(len(mixture), generator=torch.Generator().manual_seed(42+epoch)).tolist()
        for i in order[:min(remaining, per_epoch)*effective_batch]:
            row = mixture[i]
            targets = sum(t != -100 for t in row["labels"][1:]) if "labels" in row else len(row["input_ids"])-1
            expected["examples"] += 1
            expected["scored_targets"] += targets
            expected["replay_targets" if i % 2 else "scene_targets"] += targets
        remaining -= min(remaining, per_epoch)
    mixture.set_training_epoch(3)
    actual = training_exposure(mixture, effective_batch, steps, 42)
    assert all(actual[k] == v for k, v in expected.items())
    assert mixture.epoch == 3
    assert random.getstate() == rng and torch.equal(torch.get_rng_state(), torch_rng)


def test_scene_mixture_resume_matches_batches_and_weights(scene_data, tmp_path):
    replay, record = scene_data
    torch.manual_seed(42)
    template = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=1, n_head=2,
                                        resid_pdrop=.05, embd_pdrop=.05, attn_pdrop=.05))
    initial = {k: v.clone() for k, v in template.state_dict().items()}
    def run(path, resume=None):
        model = GPT2LMHeadModel(template.config)
        model.load_state_dict(initial)
        dataset = SceneReplayDataset(replay, [record], 0)
        batches = []
        model.register_forward_pre_hook(lambda m, a, k: batches.append(
            (k["input_ids"].tolist(), k["labels"].tolist())), with_kwargs=True)
        trainer = training_trainer(model=model, args=TrainingArguments(
            output_dir=str(path), max_steps=8, per_device_train_batch_size=2,
            gradient_accumulation_steps=2, save_steps=5, save_total_limit=2,
            learning_rate=5e-5, use_cpu=True, report_to="none", logging_strategy="no",
            disable_tqdm=True, seed=42, data_seed=42), train_dataset=dataset,
            data_collator=CausalCollator(1), callbacks=[window_epoch_callback(dataset)])
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        return model, batches
    full, batches = run(tmp_path/"full")
    resumed, remaining = run(tmp_path/"resumed", tmp_path/"full/checkpoint-5")
    # Epoch has a partial accumulation group; use the actual microbatch suffix.
    assert remaining == batches[-len(remaining):]
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())


def test_extractor_preserves_original_ids_and_paragraphs(tmp_path):
    paragraph = ("The little girl carried a candle to the door. "
                 "Her father was outside in the snow. She opened the door and took his hand. "
                 "The room was warm and the fire was bright. They sat by the fire and shared their bread. "
                 "The little girl smiled when her father told a story. ")
    text = "\n\n".join(paragraph+f"The clock rang {word} times." for word in
                         ["one", "two", "three", "four", "five", "six", "seven", "eight"]*4)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([text]*10, vocab_size=400, special_tokens=["<|endoftext|>", "<|pad|>"])
    bpe.save_model(str(tmp_path))
    tok = GPT2TokenizerFast(vocab_file=str(tmp_path/"vocab.json"), merges_file=str(tmp_path/"merges.txt"),
                           eos_token="<|endoftext|>", bos_token="<|endoftext|>", pad_token="<|pad|>")
    ids = [0]+tok.encode(text)+[0]
    path = tmp_path/"train.jsonl"
    path.write_text(json.dumps({"id": "46", "ids": ids}))
    dataset = BookWindows(path, 512)
    records, stats = extract_scene_candidates(dataset, tok, minimum=128, maximum=300)
    assert records and stats["books"] == 1
    for row in records:
        assert row["input_ids"] == ids[row["start_token"]:row["end_token"]]
        assert tok.decode(row["input_ids"]) == row["text"] and row["text"] in text
        assert not row["natural_end"]
        assert 48 <= row["prefix_tokens"] <= 128
        assert len(row["input_ids"])-row["prefix_tokens"] >= 64
    assert scene_sentence_ends("Mr. Smith came home. He saw Dr. Jones!") == [20, 38]


def test_scene_cli_defaults_do_not_change_capacity(monkeypatch):
    import model
    captured = []
    monkeypatch.setattr(model, "finetune_scenes", captured.append)
    monkeypatch.setattr(model, "compare_capacity", captured.append)
    model.main(["finetune-scenes", "--plan-only"])
    model.main(["compare-capacity"])
    scenes, capacity = captured
    assert (scenes.learning_rate, scenes.max_steps, scenes.warmup_ratio, scenes.benchmark_steps) == (5e-5, 1000, .03, 250)
    assert (capacity.learning_rate, capacity.max_steps, capacity.benchmark_steps) == (3e-4, -1, 2000)


def test_scene_experiment_finalizes_saved_checkpoint_without_extra_training(scene_data, tmp_path, monkeypatch):
    import hashlib

    from transformers import Trainer

    import model

    replay, record = scene_data
    data = tmp_path/"data"
    data.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["The snow falls. The girl opens the door. "*20], vocab_size=300,
                            special_tokens=["<|endoftext|>", "<|pad|>"])
    bpe.save_model(str(data))
    tokenizer = GPT2TokenizerFast(vocab_file=str(data/"vocab.json"), merges_file=str(data/"merges.txt"),
                                 bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>")
    tokenizer.save_pretrained(data/"tokenizer")
    for split in ("train", "validation", "test"):
        (data/f"{split}.jsonl").write_text(json.dumps({"id": "46", "ids": replay.books[0].tolist()}))
    report = {"splits": {"train": {"target_tokens": len(replay.books[0])-1}}}
    (data/"report.json").write_text(json.dumps(report))
    (data/"manifest.json").write_text(json.dumps([{"url": model.sources[0]}]))
    monkeypatch.setattr(model, "prepare", lambda args: report)
    monkeypatch.setattr(model, "extract_scene_candidates", lambda *a: ([record], {"fixture": True}))
    def benchmark(data, tokenizer, signature, *args, **kwargs):
        ids = record["input_ids"][:4]
        return {"dataset_sha256": signature, "sampling_seeds": [42], "generation_tokens": 12,
                "training_eval_windows": 16, "prompts": [{"prompt_id": "p00", "book_id": "46",
                    "input_ids": ids, "prompt": tokenizer.decode(ids)}]}
    monkeypatch.setattr(model, "make_capacity_benchmark", benchmark)
    out = tmp_path/"experiment"
    command = ["finetune-scenes", "--smoke-test", "--data-dir", str(data), "--output-dir", str(out)]
    model.main(command)
    model_file = out/"scenes/best/model.safetensors"
    before = hashlib.sha256(model_file.read_bytes()).hexdigest()
    review = out/"blind_best_full.csv"
    with review.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["notes"] = "Preserve this review on recovery."
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reviewed_bytes = review.read_bytes()
    # Model a Colab disconnect after final optimizer save but before best/ completion.
    (out/"scenes/best/training_summary.json").unlink()
    def forbid_training(*args, **kwargs):
        raise AssertionError("Recovery performed an extra optimizer update")
    monkeypatch.setattr(Trainer, "train", forbid_training)
    model.main(command+["--resume"])
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == before
    assert review.read_bytes() == reviewed_bytes
    summary = json.loads((out/"comparison.json").read_text())
    assert summary["complete"] and not summary["test_evaluated"]
    assert all(row["global_step"] == 3 for row in summary["branches"].values())
