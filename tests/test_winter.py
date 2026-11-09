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
from model import (BookWindows, ContextBookWindows, CausalCollator, WinterReplayDataset, combined_book_windows,
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


@pytest.mark.parametrize('lookback', [0, 4])
def test_winter_resume_replays_exact_batches_and_weights(pools, tmp_path, lookback):
    torch.manual_seed(42)
    template = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=1,
                                        n_head=2, resid_pdrop=.05, embd_pdrop=.05, attn_pdrop=.05))
    initial = {k: v.clone() for k, v in template.state_dict().items()}

    def run(path, resume=None):
        model = GPT2LMHeadModel(template.config)
        model.load_state_dict(initial)
        contextual = [ContextBookWindows(tmp_path/name, 16, random_offsets=True, lookback=lookback)
                      for name in ('original.jsonl', 'new.jsonl')]
        dataset = WinterReplayDataset(*contextual)
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


@pytest.mark.parametrize('lookback', [0, 4, 8])
def test_context_windows_score_every_fixed_target_once_without_crossing_books(pools, tmp_path, lookback):
    dataset = ContextBookWindows(tmp_path/'new.jsonl', 16, lookback=lookback)
    collected = {i: [] for i in range(len(dataset.books))}
    for i in range(len(dataset)):
        row = dataset[i]
        book, _, _ = dataset.target_window(i)
        targets = [token for token in row['labels'][1:] if token != -100]
        collected[book].extend(targets)
        assert len(row['input_ids']) <= 16
        assert len(targets) == dataset.scored_targets(i)
        # The actual collator/causal shift must agree, including the EOS label.
        batch = CausalCollator(1)([row])
        assert batch['labels'][:, 1:].ne(-100).sum() == len(targets)
    assert all(collected[i] == ids[1:].tolist() for i, ids in enumerate(dataset.books))
    assert collected[0][-1] != 0 and collected[1][-1] == 0
    if lookback == 4:
        assert dataset[1]['input_ids'] == list(range(37, 50))
        assert dataset[1]['labels'] == [-100]*5+list(range(42, 50))


def test_context_mixture_exposure_counts_labels_not_visible_context(pools, tmp_path):
    mixture = WinterReplayDataset(
        ContextBookWindows(tmp_path/'original.jsonl', 16, random_offsets=True, lookback=4),
        ContextBookWindows(tmp_path/'new.jsonl', 16, random_offsets=True, lookback=4))
    expected, visible = 0, 0
    mixture.set_training_epoch(0)
    order = torch.randperm(len(mixture), generator=torch.Generator().manual_seed(42)).tolist()
    for i in order[:8]:
        row = mixture[i]
        expected += int(CausalCollator(1)([row])['labels'][:, 1:].ne(-100).sum())
        visible += len(row['input_ids'])-1
    exposure = training_exposure(mixture, 4, 2, 42)
    assert exposure['scored_targets'] == expected < visible
    assert exposure['new_targets']+exposure['replay_targets'] == expected
    control = mixture.replay
    actual = training_exposure(control, len(control), 1, 42)
    assert actual['scored_targets'] == sum(control.scored_targets(i) for i in range(len(control)))


def test_unrelated_context_diagnostic_changes_only_prefix_and_restores_mode():
    from types import SimpleNamespace

    class PrefixModel(torch.nn.Module):
        device = torch.device('cpu')

        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, inputs, **kwargs):
            self.inputs.append(inputs.clone())
            # A controlled model: evidence is exclusively the first context token.
            logits = torch.nn.functional.one_hot(inputs[:, 0], 8).float()*6
            return SimpleNamespace(logits=logits[:, None, :].expand(-1, inputs.shape[1], -1))

    benchmark = {'context_probe_tokens': 3, 'prompts': [
        {'prompt_id': 'a', 'book_id': 'one', 'input_ids': [2, 3, 4], 'reference_ids': [2, 2, 2]},
        {'prompt_id': 'b', 'book_id': 'two', 'input_ids': [5, 6], 'reference_ids': [5, 5, 5]},
    ]}
    model = PrefixModel()
    rng = torch.get_rng_state().clone()
    report = pipeline.score_context_use(model, benchmark)
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    assert report['scored_tokens_per_condition'] == 6 and report['paired_prompts'] == 2
    assert report['context_gain_nll'] == pytest.approx(6, abs=1e-5)
    assert model.inputs[0].tolist() == [[2, 3, 4, 2, 2, 2], [5, 6, 5, 2, 2, 2]]
    assert model.inputs[1].tolist() == [[5, 6, 5, 5, 5], [2, 3, 5, 5, 5]]
    assert all(p['book_id'] != p['unrelated_book_id'] for p in report['pairs'])
    benchmark['prompts'] = benchmark['prompts'][:1]
    missing = pipeline.score_context_use(model, benchmark)
    assert missing['context_gain_nll'] is None and missing['skipped_prompt_ids'] == ['a']


def test_inline_additions_keep_individual_dates_and_true_endings():
    sources = {r['id']: r for r in pipeline.WINTER_SOURCES}
    new_ids = {'24873', '24874', '24876', '24878', '19766', '17888', '17530',
               '34378', '76681', '56583', '316', '544', '45'}
    assert len(sources) == len(pipeline.WINTER_SOURCES) == 37
    assert sum(len(sources[i]['selections']) for i in new_ids) == 28
    assert all(s['year'] < 1920 for i in new_ids for s in sources[i]['selections'])
    assert {s['year'] for s in sources['24878']['selections']} == {1909, 1910}
    assert sources['17530']['selections'][0]['natural_end']
    assert sources['56583']['selections'][0]['natural_end']
    assert not any(s['natural_end'] for s in sources['45']['selections'])
    assert '24877' not in sources  # The reprinted Anne chapter must not return.


def test_default_source_prefers_previous_winter_run(tmp_path):
    assert pipeline.default_winter_source(tmp_path) == tmp_path/'runs/christmas-scenes/control/best'
    latest = tmp_path/'runs/christmas-winter/expanded/best'
    latest.mkdir(parents=True)
    assert pipeline.default_winter_source(tmp_path) == latest
