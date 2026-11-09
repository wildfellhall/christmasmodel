"""Check objective gradients, context preservation and real checkpoint recovery."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as pipeline


def small_model(positions=256, dropout=0.):
    return GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=positions, n_ctx=positions,
        n_embd=16, n_layer=1, n_head=2, eos_token_id=0, bos_token_id=0, pad_token_id=1,
        resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout))


def objective_config(**overrides):
    return dict(curriculum_fraction=.4, context_weight=.1, context_margin=.2,
                unlikelihood_weight=.1, unlikelihood_every=2, unlikelihood_tokens=32, **overrides)


def test_extension_preserves_old_logits_ties_and_rng_and_trains_new_positions():
    model = small_model(32).eval()
    inputs = torch.arange(2, 25)[None]
    with torch.no_grad():
        expected = model(inputs).logits
    rng = torch.get_rng_state().clone()
    grown, report = pipeline.extend_gpt2_context(model, 64)
    assert torch.equal(rng, torch.get_rng_state()) and report['changed']
    assert grown.config.n_ctx == grown.config.n_positions == 64
    assert grown.lm_head.weight.data_ptr() == grown.transformer.wte.weight.data_ptr()
    with torch.no_grad():
        torch.testing.assert_close(grown(inputs).logits, expected, atol=0, rtol=0)
    grown.train()
    ids = (torch.arange(60) % 60 + 2)[None]
    grown(ids, labels=ids).loss.backward()
    assert grown.transformer.wpe.weight.grad[32:].abs().sum() > 0
    assert pipeline.extend_gpt2_context(grown, 64)[0] is grown
    with pytest.raises(ValueError, match='shrink'):
        pipeline.extend_gpt2_context(grown, 32)


def test_curriculum_anneals_and_preserves_each_pool_target_mass():
    scores = torch.tensor([.5, 1.5, .8, 1.2])
    pools, counts = torch.tensor([0, 0, 1, 1]), torch.tensor([15., 30., 4., 20.])
    weights = pipeline.curriculum_weights(scores, pools, counts, 0., .4)
    assert weights[1] > weights[0] and weights[3] > weights[2]
    for pool in (0, 1):
        mask = pools == pool
        torch.testing.assert_close((counts*weights)[mask].sum(), counts[mask].sum())
    for progress in (.4, .9):
        assert torch.equal(pipeline.curriculum_weights(scores, pools, counts, progress, .4), torch.ones(4))
    assert torch.equal(pipeline.curriculum_weights(scores, pools, counts, 0, 0), torch.ones(4))
    assert .5 <= pipeline.passage_simplicity('') <= 1.5
    assert pipeline.passage_simplicity('She came in. He shut the door.') > pipeline.passage_simplicity(
        ' '.join(['extraordinarily complicated interrelationships']*20))


def paired_batch():
    inputs = torch.stack([torch.arange(224) % 20 + 2, torch.arange(224) % 20 + 25])
    labels = inputs.clone()
    labels[:, :128] = -100
    return inputs, labels, torch.tensor([100, 200])


def test_context_loss_uses_same_suffix_and_distinct_book_prefix_without_dropout():
    model = small_model(dropout=.2).train()
    inputs, labels, keys = paired_batch()
    observed = []
    hook = model.register_forward_pre_hook(lambda m, a, k: observed.append((m.training, a[0].clone())),
                                          with_kwargs=True)
    rng = torch.get_rng_state().clone()
    loss, count = pipeline.context_pair_loss(model, inputs, labels, keys, margin=1.)
    assert count == 2 and model.training and torch.equal(rng, torch.get_rng_state())
    enabled, batch = observed[0]
    assert not enabled and batch.shape == (4, 192)
    for i in range(2):
        assert torch.equal(batch[2*i, :128], inputs[i, :128])
        assert torch.equal(batch[2*i+1, :128], inputs[1-i, :128])
        assert torch.equal(batch[2*i, 128:], batch[2*i+1, 128:])
        assert torch.equal(batch[2*i, 128:], inputs[i, 128:192])
    loss.backward()
    assert model.transformer.h[0].attn.c_attn.weight.grad.abs().sum() > 0
    hook.remove()
    assert pipeline.context_pair_loss(model, inputs, labels, keys*0, .2)[1] == 0
    labels[1] = -100  # No out-of-bounds access if a different-book row has no targets.
    assert pipeline.context_pair_loss(model, inputs, labels, keys, .2)[1] == 0


def test_unlikelihood_targets_only_third_phrase_occurrence_and_has_correct_gradient(monkeypatch):
    phrase = list(range(2, 10))
    assert pipeline.repetition_negative_positions(phrase*2) == []
    assert pipeline.repetition_negative_positions(phrase*3) == [23]
    model = small_model().train()
    calls = []

    def generate(prefix, **kwargs):
        calls.append(kwargs)
        return torch.cat((prefix, prefix.new_tensor([phrase*4])), dim=1)

    monkeypatch.setattr(model, 'generate', generate)
    inputs, _, _ = paired_batch()
    rng = torch.get_rng_state().clone()
    loss, count = pipeline.repetition_unlikelihood(model, inputs, torch.ones_like(inputs), 32)
    assert count == 9 and model.training and torch.equal(rng, torch.get_rng_state())
    assert not calls[0]['do_sample']
    loss.backward()
    assert model.transformer.wte.weight.grad.abs().sum() > 0
    # Finite differences: moving opposite this gradient must lower the penalty.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-.001)
    smaller, _ = pipeline.repetition_unlikelihood(model, inputs, torch.ones_like(inputs), 32)
    assert smaller < loss


def test_ce_gradient_accumulation_matches_a_single_token_weighted_batch():
    model = small_model(32).train()
    rows = [{'input_ids': list(range(2, n)), 'book_key': n, 'simplicity': .5, 'pool': n % 2}
            for n in (12, 18, 21)]
    batch = pipeline.CoherenceCollator(1)(rows)
    standard = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels']).loss
    standard.backward()
    expected = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    cfg = objective_config()
    cfg.update(curriculum_fraction=0, context_weight=0, unlikelihood_weight=0)
    trainer = SimpleNamespace(coherence_config=cfg, coherence_stats=defaultdict(float),
                              state=SimpleNamespace(global_step=0, max_steps=3), _ul_step=-1)
    targets = batch['labels'].ne(-100).sum()
    for row in rows:
        micro = pipeline.CoherenceCollator(1)([row])
        pipeline.coherence_loss(trainer, model, micro, targets).backward()
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, expected[k], atol=2e-7, rtol=1e-5)


def test_unlikelihood_is_applied_once_per_optimizer_update(monkeypatch):
    model = small_model(32).train()
    cfg = objective_config()
    cfg.update(curriculum_fraction=0, context_weight=0)
    trainer = SimpleNamespace(coherence_config=cfg, coherence_stats=defaultdict(float),
                              state=SimpleNamespace(global_step=0, max_steps=3), _ul_step=-1)
    calls = []
    monkeypatch.setattr(pipeline, 'repetition_unlikelihood', lambda *a: (calls.append(1) or model.lm_head.weight.sum()*0+1, 1))
    batch = pipeline.CoherenceCollator(1)([{'input_ids': [2, 3, 4]}])
    losses = [pipeline.coherence_loss(trainer, model, batch, 8) for _ in range(4)]
    assert len(calls) == 1
    torch.testing.assert_close(losses[0]-losses[1], torch.tensor(.1))
    trainer.state.global_step = 1
    pipeline.coherence_loss(trainer, model, batch, 8)
    assert len(calls) == 1
    trainer.state.global_step = 2
    pipeline.coherence_loss(trainer, model, batch, 8)
    assert len(calls) == 2


class IntegerTokenizer:
    eos_token_id, pad_token_id = 0, 1

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


def test_coherence_resume_preserves_batches_gradients_and_all_weights(tmp_path):
    for name, offset in [('original', 2), ('new', 25)]:
        (tmp_path/f'{name}.jsonl').write_text('\n'.join(json.dumps({'id': f'{name}-{i}',
            'ids': [offset+i % 2]*620}) for i in range(3)))
    torch.manual_seed(123)
    initial = small_model(dropout=.05).state_dict()

    def run(path, resume=None):
        model = small_model(dropout=.05)
        model.load_state_dict(initial)
        base = pipeline.WinterReplayDataset(*(pipeline.ContextBookWindows(tmp_path/f'{name}.jsonl',
            256, lookback=128, random_offsets=True) for name in ('original', 'new')))
        dataset = pipeline.CoherenceDataset(base, IntegerTokenizer())
        assert dataset[4]['input_ids'] == base[4]['input_ids']
        assert dataset[4]['labels'] == base[4]['labels']
        assert dataset[4]['pool'] == 1 and dataset[5]['pool'] == 0
        observed = []
        model.register_forward_pre_hook(lambda m, a, k: observed.append(k['input_ids'].tolist())
            if m.training else None, with_kwargs=True)
        trainer = pipeline.training_trainer(model=model, args=TrainingArguments(output_dir=str(path),
            max_steps=5, per_device_train_batch_size=4, gradient_accumulation_steps=2, save_steps=3,
            learning_rate=5e-5, use_cpu=True, report_to='none', logging_strategy='no',
            disable_tqdm=True, seed=42, remove_unused_columns=False), train_dataset=dataset,
            data_collator=pipeline.CoherenceCollator(1), callbacks=[pipeline.window_epoch_callback(dataset)],
            coherence_config=objective_config())
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        exposure = pipeline.training_exposure(dataset, 8, 5, 42)
        assert exposure['scored_targets'] == pipeline.training_exposure(base, 8, 5, 42)['scored_targets']
        return model, observed

    full, batches = run(tmp_path/'full')
    resumed, remaining = run(tmp_path/'resumed', tmp_path/'full/checkpoint-3')
    assert remaining == batches[-len(remaining):]
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key


def test_reranking_preserves_rng_and_mode_and_returns_a_recorded_candidate():
    model = small_model().train()
    model.config.coherence_decoding = {'candidates': 3}
    prompt = {'prompt': 'fixture', 'input_ids': list(range(2, 42))}
    rng = torch.get_rng_state().clone()
    rows = pipeline.generate_samples(model, IntegerTokenizer(), [prompt], 16, modes=('reranked',),
                                      include_token_ids=True)
    assert torch.equal(rng, torch.get_rng_state()) and model.training
    winner = rows[0]['ranking']['selected_candidate']
    assert rows[0]['token_ids'] == rows[0]['ranking']['candidates'][winner]['token_ids']
    assert 1 not in rows[0]['token_ids'] and len(rows[0]['token_ids']) <= 16
    assert rows == pipeline.generate_samples(model, IntegerTokenizer(), [prompt], 16, modes=('reranked',),
                                             include_token_ids=True)


def test_reranker_aligns_positions_and_discourages_loops_and_empty_eos():
    model = small_model().eval()
    records = []
    def observe(m, a, k):
        records.append(k['position_ids'].clone())
    model.register_forward_pre_hook(observe, with_kwargs=True)
    prompt = list(range(2, 42))
    candidates = [[0, 1], [2]*32, list(range(2, 34))]
    with torch.no_grad():
        winner, audit = pipeline.rank_continuations(model, IntegerTokenizer(), prompt, candidates)
    assert winner == candidates[2]
    assert records[1][0, 0] == len(prompt)-16
    assert records[0][0, -1] == records[1][0, -1]
    assert audit['candidates'][1]['repeated_4gram_fraction'] > .9


def test_default_command_enables_five_methods_and_retains_legacy_defaults(monkeypatch):
    captured = []
    monkeypatch.setattr(pipeline, 'finetune_winter', captured.append)
    pipeline.main(['train-coherence'])
    pipeline.main(['finetune-winter'])
    new, old = captured
    assert new.block_size == 1024 and new.winter_lookback == 256
    assert new.batch_size*new.accumulation == 16 and new.learning_rate == 3e-5
    assert new.rerank_candidates == 8 and new.context_loss_weight == new.unlikelihood_weight == .1
    assert new.curriculum_fraction == .4 and new.max_steps == 1000
    assert old.block_size == 512 and old.batch_size*old.accumulation == 32
    with pytest.raises(SystemExit):
        pipeline.main(['train-coherence', '--unlikelihood-every', '0'])


def test_one_run_pipeline_and_final_checkpoint_recovery(tmp_path, monkeypatch):
    """Run actual training/review/save paths; only replace external corpus retrieval."""
    from transformers import TrainerCallback
    data, out = tmp_path/'data', tmp_path/'run'
    data.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(['The snow falls. Christmas Eve was cold. '*20],
        vocab_size=300, special_tokens=['<|endoftext|>', '<|pad|>'])
    bpe.save_model(str(tmp_path))
    tokenizer = GPT2TokenizerFast(vocab_file=str(tmp_path/'vocab.json'), merges_file=str(tmp_path/'merges.txt'),
        bos_token='<|endoftext|>', eos_token='<|endoftext|>', unk_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.save_pretrained(data/'tokenizer')
    for split in ('train', 'validation', 'test'):
        (data/f'{split}.jsonl').write_text('\n'.join(json.dumps({'id': f'{split}-{i}',
            'ids': [0]+[2+i]*1600+[0]}) for i in range(2)))
    report = {'splits': {'train': {'target_tokens': 3202}}}
    (data/'report.json').write_text(json.dumps(report))
    signature = pipeline.dataset_signature(data)
    encoded = '\n'.join(json.dumps({'id': f'winter-{i}', 'ids': [0]+[20+i]*1600+[0]}) for i in range(2))
    manifest = {'sources': [], 'accepted': [], 'excluded': [], 'words': 1000, 'target_tokens': 3202,
        'base_dataset_sha256': signature, 'catalog_sha256': pipeline.json_digest(pipeline.WINTER_SOURCES),
        'train_sha256': pipeline.hashlib.sha256(encoded.encode()).hexdigest()}
    monkeypatch.setattr(pipeline, 'prepare', lambda args: report)
    monkeypatch.setattr(pipeline, 'prepare_winter_material', lambda *a: (manifest, encoded))
    monkeypatch.setattr(pipeline, 'make_capacity_benchmark', lambda *a, **k: {
        'dataset_sha256': signature, 'sampling_seeds': [42], 'generation_tokens': 12,
        'prompts': [{'prompt_id': f'p{i}', 'book_id': f'validation-{i}',
            'input_ids': [2+i]*16, 'prompt': tokenizer.decode([2+i]*16), 'reference_ids': [2+i]*12}
            for i in range(2)]})
    stop = [True]
    trained = []
    original_trainer = pipeline.training_trainer

    class Interrupt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            trained.append(state.global_step)

        def on_save(self, args, state, control, **kwargs):
            if stop[0] and state.global_step == 3:
                raise InterruptedError('Final optimizer checkpoint exists; finalization interrupted')

    def observed_trainer(*a, **kw):
        trainer = original_trainer(*a, **kw)
        trainer.add_callback(Interrupt())
        return trainer

    monkeypatch.setattr(pipeline, 'training_trainer', observed_trainer)
    argv = ['train-coherence', '--data-dir', str(data), '--output-dir', str(out), '--smoke-test',
            '--rerank-candidates', '2', '--unlikelihood-tokens', '32']
    with pytest.raises(InterruptedError):
        pipeline.main(argv)
    assert trained == [1, 2, 3]
    final_weights = GPT2LMHeadModel.from_pretrained(out/'expanded/checkpoint-3').state_dict()
    stop[0] = False
    pipeline.main(argv+['--resume'])
    assert trained == [1, 2, 3]  # No extra update after the saved final optimizer step.
    summary = json.loads((out/'expanded/best/training_summary.json').read_text())
    assert summary['global_step'] == 3 and not summary['test']
    cfg = summary['run']['winter_recipe']['coherence']
    assert cfg['positions'] == 1024 and cfg['context_weight'] == .1
    assert summary['training_exposure']['scored_targets'] > 0
    last = GPT2LMHeadModel.from_pretrained(out/'expanded/last')
    assert last.config.n_positions == 1024
    assert all(torch.equal(v, last.state_dict()[k]) for k, v in final_weights.items())
    review = json.loads((out/'expanded/reviews/step-3/coherence.json').read_text())
    assert review['reranked_samples'] and review['samples']
    assert not review['test_evaluated']
    # Original loss windows remain 512, even though training windows are 1024.
    metrics = pipeline.score_training_subset(last, tokenizer, pipeline.BookWindows(data/'train.jsonl', 512), 16)
    assert review['fixed_training_subset']['nll'] == pytest.approx(metrics['nll'])
    blind = out/'blind_final_full.csv'
    blind.write_text(blind.read_text()+'\nmanual review preserved\n')
    pipeline.main(argv+['--resume'])
    assert blind.read_text().endswith('manual review preserved\n')
    assert trained == [1, 2, 3]
