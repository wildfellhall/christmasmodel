"""Expansion isolation, target accounting, and interrupted-training recovery."""
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as pipeline
from model import (BookWindows, CausalCollator, WinterReplayDataset, combined_book_windows,
                   copied_training_spans, screen_winter_units, select_winter_text,
                   training_exposure, training_trainer, window_epoch_callback)


@pytest.fixture
def pools(tmp_path):
    replay_file, new_file = tmp_path/'original.jsonl', tmp_path/'new.jsonl'
    replay_file.write_text(json.dumps({'id': 'original', 'ids': [0]+list(range(2, 30))*3+[0]})+'\n')
    # A chapter cut without EOS and a complete story with genuine EOS.
    new_file.write_text('\n'.join(json.dumps(row) for row in [
        {'id': 'chapter', 'ids': [0]+list(range(31, 50))},
        {'id': 'story', 'ids': [0]+list(range(51, 60))+[0]},
    ])+'\n')
    return (BookWindows(replay_file, 16, random_offsets=True),
            BookWindows(new_file, 16, random_offsets=True))


def test_selection_rejects_edition_drift_and_ambiguous_boundaries():
    text = 'CHAPTER I\n\nThe girl opened the door. Her father came inside.'
    body = 'Publisher catalog\n'+text+'\n\nCHAPTER II\n\nLater.'
    spec = {'start': r'^CHAPTER I$', 'end': r'^CHAPTER II$',
            'sha256': hashlib.sha256(text.encode()).hexdigest()}
    assert select_winter_text(body, spec)[0] == text
    with pytest.raises(ValueError, match='changed'):
        select_winter_text(body.replace('father', 'mother'), spec)
    with pytest.raises(ValueError, match='unique'):
        select_winter_text('CHAPTER I\n\n'+body, spec)
    with pytest.raises(ValueError, match='Missing'):
        select_winter_text(body.replace('CHAPTER II', 'AFTERWORD'), spec)


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_heldout_overlap_is_excluded_despite_case_and_punctuation(split):
    words = [f'word{i}' for i in range(80)]
    original = [{'id': 'heldout', 'split': split, 'text': ' '.join(words)}]
    copied = 'NEW OPENING. '+', '.join(w.upper() for w in words[20:60])+'. END.'
    units = [{'id': 'copied', 'text': copied}, {'id': 'new', 'text': 'Other material entirely.'}]
    kept, excluded = screen_winter_units(units, original)
    assert [r['id'] for r in kept] == ['new']
    assert excluded[0]['split'] == split
    assert excluded[0]['reason'] == 'heldout_exact_32_word_overlap'


def test_reprints_are_excluded_against_original_train_and_new_pool():
    text = ' '.join(f'original{i}' for i in range(2200))
    kept, excluded = screen_winter_units([{'id': 'reprint', 'text': text}],
                                        [{'id': 'old', 'split': 'train', 'text': text}])
    assert not kept and excluded[0]['other_id'] == 'old'
    kept, excluded = screen_winter_units([{'id': 'one', 'text': text}, {'id': 'two', 'text': text}], [])
    assert [r['id'] for r in kept] == ['one'] and excluded[0]['other_id'] == 'one'


@pytest.mark.parametrize('natural_end', [False, True])
def test_preparation_preserves_saved_splits_and_only_teaches_real_endings(tmp_path, monkeypatch, natural_end):
    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, ids, **kwargs):
            return bytes(i for i in ids if i).decode()

    tok = Tokenizer()
    before = {}
    for split in ('train', 'validation', 'test'):
        path = tmp_path/f'{split}.jsonl'
        path.write_text(json.dumps({'id': split, 'ids': tok.encode(f'Original {split} unrelated prose.')})+'\n')
        before[split] = path.read_bytes()
    text = 'CHAPTER I\n\n'+('The girl brought a basket into the warm kitchen. '*50).strip()
    raw = ('Title: Winter fixture\nAuthor: Fixture\nLanguage: English\n'
           '*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n'
           +text+'\n\nCHAPTER II\n\nOmitted material.\n'
           '*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n')
    spec = {'start': r'^CHAPTER I$', 'end': r'^CHAPTER II$', 'natural_end': natural_end,
            'sha256': hashlib.sha256(text.encode()).hexdigest()}
    monkeypatch.setattr(pipeline, 'WINTER_SOURCES', [{'id': '1', 'url': 'fixture', 'selections': [spec]}])
    monkeypatch.setattr(pipeline, 'download_book', lambda *args: raw)
    manifest, serialized = pipeline.prepare_winter_material(tmp_path, tmp_path/'cache', tok, 'base')
    ids = json.loads(serialized)['ids']
    assert ids == [0]+tok.encode(text)+([0] if natural_end else [])
    assert manifest['target_tokens'] == len(ids)-1
    assert manifest['base_dataset_sha256'] == 'base'
    assert all((tmp_path/f'{split}.jsonl').read_bytes() == before[split] for split in before)


def test_full_causal_labels_keep_story_eos_but_never_add_chapter_eos(pools):
    replay, new = pools
    mixture = WinterReplayDataset(replay, new)
    for epoch in range(3):
        mixture.set_training_epoch(epoch)
        assert [mixture[i] for i in range(1, len(mixture), 2)] == list(replay)
        for i in range(0, len(mixture), 2):
            row = mixture[i]
            cycle, index = mixture.order[i//2]
            b, _ = mixture.cycles[cycle].windows[index]
            if new.book_ids[b] == 'chapter':
                assert row['input_ids'][-1] != 0
            batch = CausalCollator(1)([row])
            assert batch['labels'][0, 0] == -100
            assert batch['labels'][0, 1:].tolist() == row['input_ids'][1:]


def test_replay_cycles_cover_new_windows_and_exposure_matches_actual_batches(pools):
    mixture = WinterReplayDataset(*pools)
    first_cycle = [index for cycle, index in mixture.order if cycle == 0]
    assert sorted(first_cycle) == list(range(len(pools[1])))
    rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    steps, batch = 9, 4
    per_epoch = (len(mixture)+batch-1)//batch
    expected = dict(examples=0, scored_targets=0, new_targets=0, replay_targets=0)
    remaining = steps
    for epoch in range(3):
        mixture.set_training_epoch(epoch)
        order = torch.randperm(len(mixture), generator=torch.Generator().manual_seed(42+epoch)).tolist()
        for index in order[:min(remaining, per_epoch)*batch]:
            n = len(mixture[index]['input_ids'])-1
            expected['examples'] += 1
            expected['scored_targets'] += n
            expected['new_targets' if index % 2 == 0 else 'replay_targets'] += n
        remaining -= min(remaining, per_epoch)
    mixture.set_training_epoch(5)
    actual = training_exposure(mixture, batch, steps, 42)
    assert all(actual[k] == v for k, v in expected.items())
    assert mixture.epoch == mixture.replay.epoch == 5
    assert random.getstate() == rng and torch.equal(torch.get_rng_state(), torch_rng)


def test_copy_scan_includes_additions_without_crossing_book_boundaries(pools):
    replay, new = pools
    combined = combined_book_windows(replay, new)
    result = copied_training_spans([new.books[0][1:].tolist()], combined, minimum=8)[0]
    assert result['book_id'] == 'chapter' and result['matched_tokens'] == 19
    assert len(combined) == len(replay)+len(new)
    assert combined.original_pool is replay and combined.supplementary_pool is new


def test_winter_resume_replays_exact_batches_and_weights(pools, tmp_path):
    torch.manual_seed(42)
    template = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=1,
                                        n_head=2, resid_pdrop=.05, embd_pdrop=.05, attn_pdrop=.05))
    initial = {k: v.clone() for k, v in template.state_dict().items()}

    def run(path, resume=None):
        model = GPT2LMHeadModel(template.config)
        model.load_state_dict(initial)
        dataset = WinterReplayDataset(*pools)
        batches = []
        model.register_forward_pre_hook(lambda m, a, k: batches.append(
            (k['input_ids'].tolist(), k['labels'].tolist())), with_kwargs=True)
        trainer = training_trainer(model=model, args=TrainingArguments(
            output_dir=str(path), max_steps=9, per_device_train_batch_size=2,
            gradient_accumulation_steps=2, save_steps=5, save_total_limit=2,
            learning_rate=5e-5, use_cpu=True, report_to='none', logging_strategy='no',
            disable_tqdm=True, seed=42, data_seed=42), train_dataset=dataset,
            data_collator=CausalCollator(1), callbacks=[window_epoch_callback(dataset)])
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        return model, batches

    full, batches = run(tmp_path/'full')
    resumed, remaining = run(tmp_path/'resumed', tmp_path/'full/checkpoint-5')
    assert remaining == batches[-len(remaining):]
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
