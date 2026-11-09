"""Regression checks for data leakage, token accounting, and export compatibility."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export import validate_tokenizer
from model import (
    BookWindows,
    CausalCollator,
    clean_book,
    contrastive_generate,
    corpus_probe_inputs,
    download_book,
    generate_samples,
    group_books,
    load_continuation,
    loss_sums,
    read_sources,
    repetition_counts,
    select_continuation_result,
    set_continuation_dropout,
    split_groups,
    token_metrics,
    training_trainer,
    window_epoch_callback,
)


def gutenberg(body, language="English"):
    return (f"Title: Winter Stories\nAuthor: A Writer\nLanguage: {language}\n\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK WINTER STORIES ***\n"
            f"{body}\n*** END OF THE PROJECT GUTENBERG EBOOK WINTER STORIES ***\n"
            "FULL PROJECT GUTENBERG LICENSE must not enter training")


def test_sources_are_parsed_without_execution(tmp_path):
    corpus = tmp_path / "corpus.py"
    sentinel = tmp_path / "executed"
    corpus.write_text(f"open({str(sentinel)!r}, 'w').write('unsafe')\nsources = "
                      "['https://www.gutenberg.org/cache/epub/46/pg46.txt']")
    assert len(read_sources(corpus)) == 1
    assert not sentinel.exists()
    corpus.write_text("sources = list()")
    with pytest.raises(ValueError):
        read_sources(corpus)


def test_sources_reject_books_outside_the_original_christmas_list(tmp_path):
    corpus = tmp_path / "corpus.py"
    corpus.write_text("sources = ['https://www.gutenberg.org/cache/epub/84/pg84.txt']")
    with pytest.raises(ValueError, match="outside the approved Christmas corpus"):
        read_sources(corpus)


def test_cleanup_preserves_text_and_verse_but_removes_boilerplate():
    prose = "A child stood by the window and watched the snow fall on the village.\n" * 30
    verse = "The bells ring clear\nThe night draws near\nAnd all is still"
    raw = gutenberg("[Illustration: A tree]\n\n" + prose + "\n" + verse).replace("\n", "\r\n")
    metadata, cleaned = clean_book(raw)
    assert metadata["language"] == "English"
    assert verse in cleaned
    assert "village. A child" in cleaned
    assert "GUTENBERG" not in cleaned
    assert "Illustration" not in cleaned
    assert cleaned.count("A child") == 30


@pytest.mark.parametrize("raw", ["<html>not a book</html>", gutenberg("too short"),
                                   gutenberg("bonjour " * 400, language="French")])
def test_bad_downloads_are_rejected(raw):
    with pytest.raises(ValueError):
        clean_book(raw)


def test_cached_source_preserves_download_bytes(tmp_path):
    path = tmp_path / "raw" / "46.txt"
    path.parent.mkdir()
    raw = "A source with\r\nWindows line endings\r\n"
    path.write_bytes(raw.encode())
    assert download_book("https://www.gutenberg.org/cache/epub/46/pg46.txt", tmp_path, offline=True) == raw


def synthetic_book(book_id, seed, size=1500):
    rng = random.Random(seed)
    text = " ".join(str(rng.randrange(10000000)) for _ in range(size))
    return {"id": str(book_id), "text": text, "words": size}


def test_alternate_editions_removed_and_anthologies_stay_in_same_split():
    one, two, three, four = [synthetic_book(i, i) for i in range(4)]
    duplicate = {**one, "id": "copy", "text": one["text"].replace(" ", "\n")}
    anthology = {"id": "anthology", "text": one["text"] + " " + two["text"], "words": 3000}
    groups, duplicates, overlaps = group_books([one, two, three, four, duplicate, anthology])
    assert duplicates == [{"id": "copy", "duplicate_of": "0"}]
    assert any({x["id"] for x in g} == {"0", "1", "anthology"} for g in groups)
    splits = split_groups(groups, 42)
    assignments = {b["id"]: split for split, books in splits.items() for b in books}
    assert assignments["0"] == assignments["1"] == assignments["anthology"]
    assert all(splits.values())
    assert splits == split_groups(groups, 42)


def test_windows_score_every_target_once_and_never_join_books(tmp_path):
    path = tmp_path / "tokens.jsonl"
    books = [[0, 2, 3, 4, 5, 6, 0], [0, 7, 8, 0]]
    path.write_text("\n".join(json.dumps({"id": i, "ids": ids}) for i, ids in enumerate(books)))
    dataset = BookWindows(path, block_size=4)
    scored = [token for row in dataset for token in row["input_ids"][1:]]
    assert scored == books[0][1:] + books[1][1:]
    assert len(dataset) == 3
    batch = CausalCollator(1)([dataset[0], {"input_ids": [0, 7, 0]}])
    assert batch["labels"][1].tolist() == [-100, 7, 0, -100]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0]


def test_shifted_windows_preserve_books_lengths_edges_and_rng(tmp_path):
    path = tmp_path / "tokens.jsonl"
    books = [[0] + list(range(2, 98)) + [0], [0] + list(range(102, 168)) + [0], [0, 200, 201, 0]]
    path.write_text("\n".join(json.dumps({"id": i, "ids": ids}) for i, ids in enumerate(books)))
    fixed = BookWindows(path, 16)
    shifted = BookWindows(path, 16, random_offsets=True, seed=42)
    rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    first = [row["input_ids"] for row in shifted]
    shifted.set_training_epoch(1)
    second = [row["input_ids"] for row in shifted]
    assert first != second
    assert random.getstate() == rng and torch.equal(torch_rng, torch.get_rng_state())
    assert len(fixed) == len(shifted)
    for epoch in range(4):
        shifted.set_training_epoch(epoch)
        for index, (book, original_start) in enumerate(fixed.windows):
            reference = fixed[index]["input_ids"]
            actual = shifted[index]["input_ids"]
            assert len(actual) == len(reference)
            assert actual.count(0) == reference.count(0)
            assert any(books[book][start:start+len(actual)] == actual for start in range(len(books[book])))
            if original_start == 0 or original_start+16 >= len(books[book]):
                assert actual == reference
    recreated = BookWindows(path, 16, random_offsets=True, seed=42)
    recreated.set_training_epoch(1)
    assert [row["input_ids"] for row in recreated] == second
    fixed.set_training_epoch(10)
    assert [row for row in fixed] == [row for row in BookWindows(path, 16)]


@pytest.mark.parametrize("random_offsets", [False, True])
def test_training_resume_after_epoch_boundary_matches_uninterrupted(tiny_model, tmp_path, random_offsets):
    from accelerate.data_loader import SeedableRandomSampler
    from transformers import TrainingArguments
    template, tokenizer = tiny_model
    path = tmp_path / "tokens.jsonl"
    path.write_text(json.dumps({"id": "46", "ids": [0] + list(range(2, 62))*3 + [0]}))
    initial = {key: value.clone() for key, value in template.state_dict().items()}
    def run(output, resume=None):
        model = GPT2LMHeadModel(template.config)
        model.load_state_dict(initial)
        dataset = BookWindows(path, 16, random_offsets=random_offsets, seed=42)
        batches = []
        def record(module, args, kwargs):
            batches.append((dataset.epoch, kwargs["input_ids"].tolist()))
        model.register_forward_pre_hook(record, with_kwargs=True)
        trainer = training_trainer(model=model, args=TrainingArguments(
            output_dir=str(output), max_steps=12, per_device_train_batch_size=2,
            learning_rate=3e-4, save_steps=9, save_total_limit=2, logging_strategy="no",
            use_cpu=True, report_to="none", disable_tqdm=True, seed=123, data_seed=123),
            train_dataset=dataset, data_collator=CausalCollator(tokenizer.pad_token_id),
            callbacks=[window_epoch_callback(dataset)])
        reference_sampler = SeedableRandomSampler(dataset, data_seed=123)
        for epoch in (0, 1, 2):
            dataset.set_training_epoch(epoch)
            reference_sampler.set_epoch(epoch)
            assert list(trainer._get_train_sampler()) == list(reference_sampler)
        dataset.set_training_epoch(0)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        return model, batches
    full, full_batches = run(tmp_path / "full")
    resumed, resumed_batches = run(tmp_path / "resumed", tmp_path / "full" / "checkpoint-9")
    assert len(full_batches) == 12 and full_batches[7][0] == 1
    assert resumed_batches == full_batches[9:]
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items())


def test_perplexity_is_token_weighted_and_ignores_padding():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor([[0, 2, 3, 4, 0], [0, 7, 0, -100, -100]])
    values = loss_sums(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    class Prediction:
        predictions = values.numpy()
    metrics = token_metrics(Prediction())
    assert metrics["nll"] == pytest.approx(expected.item(), abs=1e-6)
    assert metrics["scored_tokens"] == 6
    assert metrics["perplexity"] == pytest.approx(np.exp(expected.item()), rel=1e-6)


@pytest.fixture
def tiny_model(tmp_path):
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["The snow falls. Christmas Eve was cold. " * 20],
                            vocab_size=300, special_tokens=["<|endoftext|>", "<|pad|>"])
    bpe.save_model(str(tmp_path))
    tokenizer = GPT2TokenizerFast(vocab_file=str(tmp_path/"vocab.json"), merges_file=str(tmp_path/"merges.txt"),
                                 bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                 unk_token="<|endoftext|>", pad_token="<|pad|>")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=64, n_layer=1,
                                     n_embd=32, n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=1))
    return model, tokenizer


def test_samples_preserve_training_rng_and_mode(tiny_model):
    model, tokenizer = tiny_model
    model.train()
    before = torch.get_rng_state().clone()
    a = generate_samples(model, tokenizer, ["The snow"], 4)
    assert torch.equal(before, torch.get_rng_state())
    assert model.training
    assert a == generate_samples(model, tokenizer, ["The snow"], 4)
    assert {row["mode"] for row in a} == {"greedy", "sample"}
    with pytest.raises(ValueError):
        generate_samples(model, tokenizer, ["x " * 200], 4)
    assert model.training


def test_passage_probes_use_exact_tokens_and_never_feed_reference(tiny_model, tmp_path, monkeypatch):
    model, tokenizer = tiny_model
    tokens = [0] + tokenizer.encode("The snow falls. Christmas Eve was cold. " * 12) + [0]
    path = tmp_path / "tokens.jsonl"
    path.write_text(json.dumps({"id": "46", "ids": tokens}))
    dataset = BookWindows(path, model.config.n_positions)
    probes = corpus_probe_inputs(dataset, tokenizer, max_new_tokens=4, seed=42)
    assert probes == corpus_probe_inputs(dataset, tokenizer, max_new_tokens=4, seed=42)
    probe, = probes
    start, length = probe["start_token"], probe["prefix_tokens"]
    assert probe["book_id"] == "46"
    assert probe["input_ids"] == tokens[start:start+length]
    assert probe["reference_ids"] == tokens[start+length:start+length+4]
    # Decode/re-encode can change IDs at byte or BPE boundaries. Corpus probes
    # must pass only the original prefix to generate, never the reference.
    def reject_reencoding(*args, **kwargs):
        raise AssertionError("Corpus probe was re-tokenized")
    calls = []
    original_generate = model.generate
    def capture_generate(inputs, **kwargs):
        calls.append(inputs.tolist()[0])
        return original_generate(inputs, **kwargs)
    monkeypatch.setattr(tokenizer, "encode", reject_reencoding)
    monkeypatch.setattr(model, "generate", capture_generate)
    results = generate_samples(model, tokenizer, probes, max_new_tokens=4)
    assert len(results) == 2
    assert calls == [probe["input_ids"], probe["input_ids"]]


def test_cached_contrastive_matches_independent_full_prefix_search(tiny_model):
    model, tokenizer = tiny_model
    model.eval()
    inputs = torch.tensor([[0, 2, 3, 4, 5]])
    alpha, top_k, steps = .6, 4, 6
    # Full-prefix, one-candidate-at-a-time reference: deliberately no shared
    # cache/branch-selection code from the implementation under test.
    reference = inputs.clone()
    with torch.inference_mode():
        for _ in range(steps):
            output = model(reference, use_cache=False, output_hidden_states=True)
            logits = output.logits[0, -1].float().clone()
            logits[tokenizer.pad_token_id] = -float("inf")
            probabilities = logits.softmax(-1)
            candidates = probabilities.topk(top_k).indices.tolist()
            scores = []
            for candidate in candidates:
                extended = torch.cat((reference, torch.tensor([[candidate]])), dim=1)
                hidden = model(extended, use_cache=False, output_hidden_states=True).hidden_states[-1][0]
                similarity = torch.nn.functional.cosine_similarity(hidden[-1:].float(), hidden[:-1].float(), dim=-1)
                scores.append((1-alpha)*probabilities[candidate].item() - alpha*similarity.max().item())
            selected = candidates[max(range(top_k), key=lambda i: scores[i])]
            reference = torch.cat((reference, torch.tensor([[selected]])), dim=1)
            if selected == tokenizer.eos_token_id:
                break
        cached = contrastive_generate(model, inputs, steps, tokenizer.pad_token_id, tokenizer.eos_token_id,
                                      top_k=top_k, alpha=alpha)
    assert torch.equal(cached, reference)


def test_contrastive_zero_penalty_matches_greedy_and_caps_context(tiny_model):
    model, tokenizer = tiny_model
    model.eval()
    inputs = torch.tensor([[0] + [2, 3, 4]*20])  # Only three positions remain.
    with torch.inference_mode():
        result = contrastive_generate(model, inputs, 10, tokenizer.pad_token_id, tokenizer.eos_token_id, alpha=0)
        greedy = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=3,
                                do_sample=False, suppress_tokens=[tokenizer.pad_token_id])
    assert torch.equal(result, greedy)
    assert result.shape[1] <= model.config.n_positions
    assert tokenizer.pad_token_id not in result[0, inputs.shape[1]:].tolist()


def test_contrastive_stops_at_eos(tiny_model, monkeypatch):
    model, tokenizer = tiny_model
    model.eval()
    forward = model.forward
    def force_eos(*args, **kwargs):
        output = forward(*args, **kwargs)
        output.logits.fill_(-1000)
        output.logits[..., tokenizer.eos_token_id] = 1000
        return output
    monkeypatch.setattr(model, "forward", force_eos)
    with torch.inference_mode():
        output = contrastive_generate(model, torch.tensor([[0, 2, 3]]), 8,
                                      tokenizer.pad_token_id, tokenizer.eos_token_id, alpha=0)
    assert output.tolist() == [[0, 2, 3, tokenizer.eos_token_id]]


def test_decoding_modes_preserve_rng_weights_and_training_mode(tiny_model):
    model, tokenizer = tiny_model
    model.train()
    rng = torch.get_rng_state().clone()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    modes = ("blocked_sample", "contrastive")
    samples = generate_samples(model, tokenizer, ["The snow"], 8, modes=modes, include_token_ids=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training
    assert all(torch.equal(weights[key], value) for key, value in model.state_dict().items())
    assert samples == generate_samples(model, tokenizer, ["The snow"], 8, modes=modes, include_token_ids=True)
    assert repetition_counts(samples[0]["token_ids"])["repeated_ngrams"] == 0
    assert repetition_counts([2]*8) == {"ngrams": 5, "repeated_ngrams": 4}
    assert repetition_counts([2, 3]) == {"ngrams": 0, "repeated_ngrams": 0}


def test_export_rejects_tokenizer_changes(tiny_model):
    from tokenizers import normalizers
    _, tokenizer = tiny_model
    validate_tokenizer(tokenizer)
    tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(ValueError, match="byte-level"):
        validate_tokenizer(tokenizer)


def test_continuation_checks_data_and_tokenizer(tiny_model, tmp_path):
    from tokenizers import normalizers
    model, tokenizer = tiny_model
    source = tmp_path / "best"
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    (source / "training_summary.json").write_text(json.dumps({"dataset_sha256": "original-data"}))
    loaded = load_continuation(source, tokenizer, "original-data")
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="dataset differs"):
        load_continuation(source, tokenizer, "changed-data")
    tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(ValueError, match="tokenizer differs"):
        load_continuation(source, tokenizer, "original-data")


@pytest.mark.parametrize("trained_nll,retain", [(4.1, True), (None, True), (3.8, False)])
def test_continuation_keeps_starting_weights_unless_validation_improves(tiny_model, tmp_path, trained_nll, retain):
    model, _ = tiny_model
    initial = tmp_path / "initial"
    model.save_pretrained(initial)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.5)
    changed = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint, nll, retained = select_continuation_result(model, initial, 3.9, trained_nll, "checkpoint-250")
    assert retained == retain
    assert checkpoint == (str(initial) if retain else "checkpoint-250")
    assert nll == (3.9 if retain else trained_nll)
    expected = original if retain else changed
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in expected.items())


def test_dropout_override_changes_training_and_survives_reload(tiny_model, tmp_path):
    model, _ = tiny_model
    ids = torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8]])
    model.eval()
    with torch.no_grad():
        original_eval = model(ids).logits
    model.train()
    torch.manual_seed(42)
    with torch.no_grad():
        original_training = model(ids).logits
    set_continuation_dropout(model, .05)
    torch.manual_seed(42)
    with torch.no_grad():
        changed_training = model(ids).logits
    assert not torch.allclose(original_training, changed_training)
    model.eval()
    with torch.no_grad():
        assert torch.equal(original_eval, model(ids).logits)
    destination = tmp_path / "changed-dropout"
    model.save_pretrained(destination)
    loaded = GPT2LMHeadModel.from_pretrained(destination)
    assert all(getattr(loaded.config, key) == .05 for key in ("resid_pdrop", "embd_pdrop", "attn_pdrop"))
    assert all(module.p == .05 for module in loaded.modules() if isinstance(module, torch.nn.Dropout))
