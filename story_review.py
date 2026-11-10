"""Comparable held-out text metrics, raw generation and explicit model selection."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from story_data import StoryWindows, digest, file_hash, write_json

RATING_FIELDS = ('grammar', 'continuity', 'progression', 'nonrepetition', 'ending')
DECODING_DEFAULTS = {'temperature': .8, 'top_p': .92, 'top_k': 40, 'repetition_penalty': 1.0}


def context_length(model):
    for key in ('max_position_embeddings', 'n_positions'):
        if getattr(model.config, key, None):
            return int(getattr(model.config, key))
    raise ValueError('Model does not declare a supported context limit')


def checkpoint_hash(directory):
    root = Path(directory)
    paths = sorted(set(root.glob('*.safetensors')) | set(root.glob('pytorch_model*.bin')))
    if not paths:
        raise ValueError('No checkpoint weights found')
    paths += [root / 'config.json', root / 'tokenizer.json', root / 'tokenizer_config.json']
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('Checkpoint must contain weights, config and tokenizer')
    return digest({p.name: file_hash(p) for p in paths})


def resolve_selected(path):
    root = Path(path).resolve()
    selection = root if root.is_file() else root / 'selected.json'
    if not selection.is_file():
        raise ValueError(f'No reviewed generation selection at {selection}; use stories select first, '
                         'or explicitly name a checkpoint for a diagnostic export')
    record = json.loads(selection.read_text())
    checkpoint = (selection.parent / record['checkpoint']).resolve()
    if checkpoint_hash(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed after prose review')
    return checkpoint, record


def score_text(model, tokenizer, rows, context, batch_size=4):
    """Every text target once; BPB excludes EOS and divides by original UTF-8 bytes."""
    import torch
    from model import CausalCollator
    from torch.nn import functional as F
    dataset = StoryWindows(rows, tokenizer, context, min(context // 4, 256))
    sums = defaultdict(lambda: {'nll_sum': 0., 'targets': 0, 'text_nll_sum': 0., 'bytes': 0})
    for row in rows:
        sums[row['pool']]['bytes'] += len(row['text'].encode('utf-8'))
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                indices = list(range(start, min(start + batch_size, len(dataset))))
                batch = {k: v.to(model.device) for k, v in CausalCollator(tokenizer.pad_token_id)(
                    [dataset[i] for i in indices]).items()}
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
                labels = batch['labels'][:, 1:]
                losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels,
                                         ignore_index=-100, reduction='none')
                for j, i in enumerate(indices):
                    pool = rows[dataset.windows[i][0]]['pool']
                    sums[pool]['nll_sum'] += float(losses[j].double().sum())
                    sums[pool]['targets'] += int(labels[j].ne(-100).sum())
                    text_mask = labels[j].ne(-100) & labels[j].ne(tokenizer.eos_token_id)
                    sums[pool]['text_nll_sum'] += float(losses[j][text_mask].double().sum())
    finally:
        model.train(was_training)
    sums['all'] = {key: sum(value[key] for value in sums.values())
                   for key in ('nll_sum', 'targets', 'text_nll_sum', 'bytes')}
    return {pool: {**value, 'nll': value['nll_sum'] / max(1, value['targets']),
                   'bits_per_byte': value['text_nll_sum'] / max(1, value['bytes']) / math.log(2),
                   'context': context} for pool, value in sums.items()}


def make_benchmark(rows, count=20, seed=42, custom_prompts=None):
    """Select identical TEXT prefixes across tokenizers; balance pools and source works."""
    from model import prose_sentence_ends
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ends = prose_sentence_ends(row['text'])
        possible = [end for end in ends if 40 <= len(row['text'][:end].split()) <= 100
                    and len(row['text'][end:].split()) >= 40]
        if not possible:
            continue
        end = min(possible, key=lambda e: abs(len(row['text'][:e].split()) - 65))
        grouped[row['pool']][row['group']].append({'unit_id': row['id'], 'group': row['group'],
            'pool': row['pool'], 'prompt': row['text'][:end], 'reference': row['text'][end:end+1600]})
    rng, prompts = random.Random(seed), []
    pools = sorted(grouped)
    orders = {}
    for pool in pools:
        orders[pool] = list(grouped[pool])
        rng.shuffle(orders[pool])
        for candidates in grouped[pool].values():
            rng.shuffle(candidates)
    while len(prompts) < count:
        added = False
        for pool in pools:
            for group in orders[pool][:]:
                if grouped[pool][group]:
                    prompts.append(grouped[pool][group].pop())
                    orders[pool].remove(group)
                    orders[pool].append(group)
                    added = True
                    break
            if len(prompts) >= count:
                break
        if not added:
            break
    for text in custom_prompts or []:
        prompts.append({'unit_id': None, 'group': 'written-probes', 'pool': 'written-probes',
                        'prompt': text, 'reference': None})
    if not prompts:
        raise ValueError('No eligible prose-review prompts')
    for row in prompts:
        row['prompt_id'] = digest(row)[:20]
    return {'schema': 1, 'prompts': prompts, 'seed': seed,
            'note': 'Reference continuations are for review only, never generation input.'}


def generate(model, tokenizer, prompt, tokens=192, seed=42, mode='sample', decoding=None):
    import torch
    params = {**DECODING_DEFAULTS, **(decoding or {})}
    if params['temperature'] <= 0 or not 0 < params['top_p'] <= 1 or params['top_k'] < 0 or params['repetition_penalty'] <= 0:
        raise ValueError('Invalid decoding configuration')
    ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.eos_token_id]
    limit=min(context_length(model),int(getattr(model.config,'story_training_context',context_length(model))))
    room = limit - len(ids)
    if room < 1:
        raise ValueError('Prompt exceeds context; no silent prompt truncation')
    before = model.training
    mps_rng = torch.mps.get_rng_state() if model.device.type == 'mps' else None
    cuda = [model.device.index or 0] if model.device.type == 'cuda' else []
    model.eval()
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda):
            torch.manual_seed(seed)
            inputs = torch.tensor([ids], device=model.device)
            kwargs = {'do_sample': mode == 'sample', 'max_new_tokens': min(tokens, room),
                      'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id,
                      'use_cache': True}
            if tokenizer.pad_token_id != tokenizer.eos_token_id:
                kwargs['suppress_tokens'] = [tokenizer.pad_token_id]
            if mode == 'sample':
                kwargs.update(params)
            elif mode != 'greedy':
                raise ValueError('Use greedy or sample for raw review')
            result = model.generate(inputs, attention_mask=torch.ones_like(inputs), **kwargs)[0, len(ids):].tolist()
    finally:
        model.train(before)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
    continuation=tokenizer.decode(result, skip_special_tokens=True)
    return {'continuation': continuation, 'token_ids': result,
            'generated_words':len(continuation.split()),'requested_new_tokens':tokens,'context_limit':limit,
            'generated_tokens': len(result), 'ended_with_eos': bool(result and result[-1] == tokenizer.eos_token_id),
            'context_limited': room < tokens, 'mode': mode, 'seed': seed}


def review_model(model, tokenizer, rows, benchmark, output, step, context, batch_size=4,
                 tokens=192, seeds=(42, 43), decoding=None, training_rows=None):
    import numpy as np
    from types import SimpleNamespace
    from model import copied_training_spans, repetition_counts
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {str(length): score_text(model, tokenizer, rows, length, batch_size)
               for length in sorted({min(512, context), context})}
    samples = []
    for prompt in benchmark['prompts']:
        for mode in ('greedy', 'sample'):
            for seed in (seeds[:1] if mode == 'greedy' else seeds):
                sample = generate(model, tokenizer, prompt['prompt'], tokens, seed, mode, decoding)
                sample.update(prompt_id=prompt['prompt_id'], pool=prompt['pool'], group=prompt['group'],
                              prompt=prompt['prompt'], reference=prompt['reference'])
                sample['sample_id'] = digest({'prompt': prompt['prompt_id'], 'mode': mode, 'seed': seed})[:20]
                sample.update(repetition_counts(sample['token_ids']))
                samples.append(sample)
    if training_rows:
        bank = SimpleNamespace(book_ids=[r['id'] for r in training_rows],
            books=[np.asarray(tokenizer.encode(r['text'], add_special_tokens=False)) for r in training_rows])
        for sample, match in zip(samples, copied_training_spans([s['token_ids'] for s in samples], bank)):
            sample['training_copy'] = match
    report = {'step': step, 'metrics': metrics, 'benchmark_sha256': digest(benchmark),
              'requested_new_tokens':tokens,
              'decoding': {**DECODING_DEFAULTS, **(decoding or {})}, 'samples': samples,
              'validation_text_sha256': digest(rows), 'test_evaluated': False,
              'coherence_verdict': None}
    write_json(out / 'review.json', report)
    # Never overwrite reviewer work on an interrupted/resumed review.
    if not (out / 'ratings.csv').exists():
        with (out / 'ratings.csv').open('w', newline='') as handle:
            fields = ['sample_id', 'prompt', 'continuation', *RATING_FIELDS, 'notes']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            shuffled = samples.copy()
            random.Random(42).shuffle(shuffled)
            for sample in shuffled:
                writer.writerow({key: sample[key] for key in fields if key in sample})
    return report


def select_checkpoint(run, checkpoint, ratings, reviewer, min_pass_rate=.8, max_nll_regression=.15):
    """Promote only explicitly rated, unchanged raw samples; persist a verifiable selection."""
    run, checkpoint, ratings = Path(run).resolve(), Path(checkpoint).resolve(), Path(ratings).resolve()
    if not 0 <= min_pass_rate <= 1 or max_nll_regression < 0 or not reviewer.strip():
        raise ValueError('Invalid review selection policy')
    if run not in checkpoint.parents:
        raise ValueError('Select a checkpoint from this run')
    provenance = json.loads((checkpoint / 'story_provenance.json').read_text())
    if provenance.get('diagnostic_only'):
        raise ValueError('Diagnostic runs cannot be promoted as generation selections')
    report_path = checkpoint.parent / 'review.json'
    if not report_path.exists():
        report_path = checkpoint / 'review.json'
    report = json.loads(report_path.read_text())
    if checkpoint_hash(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the reviewed weights')
    if digest(report) != provenance['review_sha256']:
        raise ValueError('Review differs from the checkpoint provenance')
    lookup = {s['sample_id']: s for s in report['samples']}
    seen, passed = set(), 0
    per_pool = defaultdict(lambda: [0, 0])
    for row in csv.DictReader(ratings.open(encoding='utf-8-sig',newline='')):
        key = row['sample_id']
        if key not in lookup or key in seen:
            raise ValueError('Unknown or duplicate review sample')
        sample = lookup[key]
        if (row['prompt'], row['continuation']) != (sample['prompt'], sample['continuation']):
            raise ValueError('Rated prose differs from saved raw generation')
        values = [row.get(k) for k in RATING_FIELDS]
        if any(value not in {'0', '1', '2'} for value in values):
            raise ValueError('Rate every dimension 0, 1 or 2 for every sample')
        # Ending is reported separately: a requested continuation need not end a story.
        passed_sample = all(row[k] == '2' for k in RATING_FIELDS[:-1])
        passed += passed_sample
        per_pool[sample['pool']][0] += passed_sample
        per_pool[sample['pool']][1] += 1
        seen.add(key)
    if seen != set(lookup):
        raise ValueError('Rate every raw sample, including failures')
    if passed / len(seen) < min_pass_rate or any(a / b < min_pass_rate for a, b in per_pool.values()):
        raise ValueError('Prose review does not meet the requested overall/per-pool pass rate')
    best = json.loads((run / 'best_nll.json').read_text())
    nll = report['metrics'][str(provenance['context'])]['all']['nll']
    if nll > best['nll'] + max_nll_regression:
        raise ValueError('Selected model exceeds the validation-loss regression allowance')
    record = {'checkpoint': str(checkpoint.relative_to(run)),
              'checkpoint_sha256': checkpoint_hash(checkpoint), 'review_sha256': digest(report),
              'ratings_sha256': file_hash(ratings), 'reviewer': reviewer,
              'samples': len(seen), 'pass_rate': passed / len(seen),
              'per_pool_pass_rates': {p: a/b for p, (a,b) in per_pool.items()},
              'policy': {'min_pass_rate': min_pass_rate, 'max_nll_regression': max_nll_regression},
              'interpretation': 'Reviewer judgment on this frozen validation panel; not a guarantee.'}
    write_json(run / 'selected.json', record)
    return record
