"""Plain causal-LM baselines, matched initialization and reviewed ablations."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from story_data import (StoryWindows, TokenMixture, digest, file_hash, make_tokenizer,
                        prepare_corpus, read_jsonl, verify_corpus, write_json)
from story_review import (DECODING_DEFAULTS, checkpoint_hash, context_length, generate,
                          make_benchmark, resolve_selected, review_model, score_text,
                          select_checkpoint)

DEFAULTS = {'schema': 1, 'initialization': 'scratch', 'base_model': None, 'revision': None,
    'start_from': None,
    'corpus': 'story-data', 'output': 'runs/story-scratch', 'seed': 42,
    'model_seed': 42, 'vocab_size': 8192, 'layers': 6, 'width': 384, 'heads': 6,
    'context': 512, 'lookback': None, 'batch_size': 4, 'accumulation': 4,
    'learning_rate': None, 'weight_decay': .1, 'dropout': .05, 'warmup_ratio': .03,
    'max_steps': 2000, 'token_budget': 32000000, 'eval_steps': 250, 'patience': 6,
    'min_delta': .002, 'precision': 'auto', 'cpu': False, 'gradient_checkpointing': False, 'pool_shares': None,
    'author_balance': 0., 'objective': 'ce', 'preference_file': None,
    'preference_weight': .1, 'unlikelihood_weight': .1, 'unlikelihood_every': 8,
    'sample_tokens': 192, 'sample_seeds': [42, 43], 'benchmark_prompts': 20,
    'written_prompts': [], 'decoding': DECODING_DEFAULTS, 'diagnostic_units': None}


def load_config(path):
    supplied = json.loads(Path(path).read_text()) if not isinstance(path, dict) else path
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    config = {**copy.deepcopy(DEFAULTS), **supplied}
    if not isinstance(config['decoding'],dict) or set(config['decoding'])-set(DECODING_DEFAULTS):
        raise ValueError('Unknown decoding fields')
    config['decoding']={**DECODING_DEFAULTS,**config['decoding']}
    if config['initialization'] not in {'scratch', 'pretrained'} or config['objective'] not in {'ce', 'reviewed_preference', 'repetition'}:
        raise ValueError('Choose scratch/pretrained initialization and one objective')
    if config['initialization'] == 'pretrained' and not config['base_model']:
        raise ValueError('Pretrained initialization requires a base model')
    if config['base_model'] and not Path(config['base_model']).is_dir() and not re.fullmatch(r'[0-9a-f]{40}', config['revision'] or ''):
        raise ValueError('Pin external model and tokenizer to an immutable commit SHA')
    if config['learning_rate'] is None:
        config['learning_rate'] = 3e-4 if config['initialization'] == 'scratch' else 2e-5
    if config['lookback'] is None:
        config['lookback'] = min(config['context'] // 4, 256)
    for key in ('layers','width','heads','vocab_size','context','lookback','batch_size','accumulation',
                'max_steps','token_budget','eval_steps','patience','sample_tokens','benchmark_prompts',
                'unlikelihood_every','seed','model_seed'):
        if type(config[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if not all(0<=config[key]<2**32 for key in ('seed','model_seed')):
        raise ValueError('Seeds must be integers in [0, 2**32)')
    if config['vocab_size']<258:
        raise ValueError('Byte-level BPE needs at least 256 bytes plus special tokens')
    if type(config['gradient_checkpointing']) is not bool:
        raise ValueError('gradient_checkpointing must be a boolean')
    for key in ('layers', 'width', 'heads', 'context', 'batch_size', 'accumulation',
                'learning_rate', 'max_steps', 'token_budget', 'eval_steps', 'patience',
                'sample_tokens', 'benchmark_prompts', 'unlikelihood_every'):
        if not isinstance(config[key], (float, int)) or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if config['width'] % config['heads'] or not 0 <= config['lookback'] <= config['context'] - 2:
        raise ValueError('Invalid head width or lookback')
    if not 0 <= config['dropout'] < 1 or not 0 <= config['warmup_ratio'] < 1:
        raise ValueError('Invalid dropout or warmup')
    if config['objective'] == 'reviewed_preference' and not config['preference_file']:
        raise ValueError('Reviewed preference training requires an annotated file')
    if config['preference_file'] and config['objective'] != 'reviewed_preference':
        raise ValueError('A preference file requires the reviewed_preference objective')
    if not config['sample_seeds']:
        raise ValueError('At least one generation seed is required')
    if (len(set(config['sample_seeds']))!=len(config['sample_seeds']) or
            any(type(seed) is not int or not 0<=seed<2**32 for seed in config['sample_seeds'])):
        raise ValueError('Generation seeds must be distinct integers in [0, 2**32)')
    if config['diagnostic_units'] is not None and config['diagnostic_units'] < 1:
        raise ValueError('diagnostic_units must be positive')
    if not 0<=config['author_balance']<=1:
        raise ValueError('author_balance must be in [0, 1]')
    for key in ('learning_rate','weight_decay','preference_weight','unlikelihood_weight','min_delta'):
        if not isinstance(config[key],(int,float)) or not math.isfinite(config[key]) or config[key]<0:
            raise ValueError(f'{key} must be finite and nonnegative')
    return config


def initialize_model(config, tokenizer, offline=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, set_seed
    set_seed(config['model_seed'])
    if config['base_model']:
        architecture = AutoConfig.from_pretrained(config['base_model'], revision=config['revision'],
                                                   local_files_only=offline)
        if config['initialization'] == 'pretrained':
            model = AutoModelForCausalLM.from_pretrained(config['base_model'], revision=config['revision'],
                                                        local_files_only=offline, torch_dtype=torch.float32)
        else:
            # Loading architecture/tokenizer does not load any pretrained model weights.
            model = AutoModelForCausalLM.from_config(architecture)
    else:
        model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=len(tokenizer),
            n_positions=config['context'], n_ctx=config['context'], n_layer=config['layers'],
            n_embd=config['width'], n_head=config['heads'], n_inner=4*config['width'],
            resid_pdrop=config['dropout'], embd_pdrop=config['dropout'], attn_pdrop=config['dropout']))
    if config['context'] > context_length(model):
        raise ValueError('Training context exceeds the architecture; use an explicitly trained extension')
    # Preserve original vocabulary IDs; only newly appended special tokens are initialized.
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.config.use_cache = False
    return model


def reviewed_pairs(path, rows, tokenizer, context):
    """Require exact source evidence and an explicit contradiction review.

    This validates provenance and annotation structure, not semantic truth. No
    unreviewed automatic corruption or source-vs-arbitrary-rollout labels enter.
    """
    lookup = {r['id']: r for r in rows}
    result, seen = [], set()
    for record in read_jsonl(path):
        if record['id'] in seen or record['unit_id'] not in lookup:
            raise ValueError('Duplicate preference or non-training source')
        seen.add(record['id'])
        if record.get('decision') != 'contradiction' or not record.get('reviewer') or not record.get('reason'):
            raise ValueError('Preference needs an explicit contradiction review; ambiguous examples are excluded')
        if record.get('kind') not in {'entity', 'polarity', 'order', 'location', 'causality'}:
            raise ValueError('Unknown preference kind')
        source = lookup[record['unit_id']]
        if record['source_sha256'] != source['text_sha256']:
            raise ValueError('Preference source changed')
        a, b, c = record['prefix_start'], record['target_start'], record['target_end']
        if not 0 <= a < b < c <= len(source['text']):
            raise ValueError('Invalid preference boundaries')
        prefix, chosen = source['text'][a:b], source['text'][b:c]
        if record.get('evidence') not in prefix or not record.get('evidence', '').strip():
            raise ValueError('Contradiction evidence must be present in the visible prefix')
        rejected = record['rejected']
        if not rejected.strip() or rejected == chosen:
            raise ValueError('Preference needs a distinct rejected continuation')
        if record['kind'] == 'entity':
            entities = record.get('entities', [])
            if len(entities) != 2 or entities[0]['canonical_id'] == entities[1]['canonical_id']:
                raise ValueError('Entity substitutions require two distinct identities, not aliases')
            for entity in entities:
                if not entity.get('name') or entity['name'] not in prefix:
                    raise ValueError('Use full entity names present in the prefix')
            if record.get('agreement_checked') is not True:
                raise ValueError('Entity preference requires grammatical-agreement review')
        # Annotation offsets often put a space at the end of the prefix. A BPE
        # token can join it to the next word: preserve the full source encoding
        # by assigning the shared separator to both continuations instead.
        prefix_without_space=prefix.rstrip()
        separator=prefix[len(prefix_without_space):]+chosen[:len(chosen)-len(chosen.lstrip())]
        prefix=prefix_without_space
        chosen=separator+chosen.lstrip()
        rejected=separator+rejected.lstrip()
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
        rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)
        if (prefix_ids+chosen_ids!=tokenizer.encode(prefix+chosen,add_special_tokens=False) or
                prefix_ids+rejected_ids!=tokenizer.encode(prefix+rejected,add_special_tokens=False)):
            raise ValueError('Preference boundary changes tokenization; keep leading whitespace in the continuation')
        if not prefix_ids or not chosen_ids or max(len(chosen_ids), len(rejected_ids)) + len(prefix_ids) > context:
            raise ValueError('Preference does not fit the context; no silent truncation')
        common = next((i for i, (x,y) in enumerate(zip(chosen_ids, rejected_ids)) if x != y),
                      min(len(chosen_ids), len(rejected_ids)))
        if common >= min(len(chosen_ids), len(rejected_ids)):
            raise ValueError('Preference must differ before the end of either continuation')
        result.append({**record, 'pair_id': record['id'], 'prefix_ids': prefix_ids,
                       'chosen_ids': chosen_ids, 'rejected_ids': rejected_ids, 'difference_start': common})
    if not result:
        raise ValueError('No reviewed preference examples')
    return result


class PreferenceDataset:
    def __init__(self, base, pairs, seed):
        self.base, self.pairs, self.seed = base, pairs, seed
        self.random_offsets = False
        self.set_training_epoch(0)

    def set_training_epoch(self, epoch):
        self.base.set_training_epoch(epoch)
        self.epoch = int(epoch)
        groups = {}
        for pair in self.pairs:
            groups.setdefault(pair['kind'], []).append(pair)
        rng = random.Random(self.seed + self.epoch)
        for group in groups.values():
            rng.shuffle(group)
        # Equal category exposure; rotate independently of original/winter row parity.
        kinds = sorted(groups)
        self.order = [groups[kind][i % len(groups[kind])]
                      for i in range(max(map(len, groups.values()))) for kind in kinds]
        rng.shuffle(self.order)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return {**self.base[index], 'reviewed_pair': self.order[index % len(self.order)]}


def seal_checkpoint(path, progress):
    """Completion marker written after optimizer state and token accounting."""
    path=Path(path)
    write_json(path/'exposure.json',progress)
    files={p.name:p.stat().st_size for p in path.iterdir() if p.is_file() and p.name!='complete.json'}
    write_json(path/'complete.json',{'schema':1,'sizes':files,
        'state_sha256':file_hash(path/'trainer_state.json'),
        'exposure_sha256':file_hash(path/'exposure.json')})


def story_checkpoint(directory, require_scaler=False):
    """A dropped Colab connection must not select a partly written checkpoint."""
    required={'optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','exposure.json','complete.json'}
    if require_scaler:required.add('scaler.pt')
    candidates=[p for p in Path(directory).glob('checkpoint-*') if p.name.split('-')[-1].isdigit()]
    for path in sorted(candidates,key=lambda p:int(p.name.split('-')[-1]),reverse=True):
        if all((path/name).is_file() for name in required) and any(
                (path/name).is_file() for name in ('model.safetensors','pytorch_model.bin')):
            try:
                seal=json.loads((path/'complete.json').read_text())
                if not all((path/name).is_file() and (path/name).stat().st_size==size
                           for name,size in seal['sizes'].items()):continue
                if (file_hash(path/'trainer_state.json')!=seal['state_sha256'] or
                        file_hash(path/'exposure.json')!=seal['exposure_sha256']):continue
                json.loads((path/'trainer_state.json').read_text())
                json.loads((path/'exposure.json').read_text())
            except (OSError,ValueError,KeyError,TypeError):
                continue
            return str(path)
    return None


def train(config, resume=False, offline=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed
    import model as legacy
    cfg = load_config(config)
    out, corpus = Path(cfg['output']), Path(cfg['corpus'])
    manifest = verify_corpus(corpus)
    training_rows, validation_rows = read_jsonl(corpus/'train.jsonl'), read_jsonl(corpus/'validation.jsonl')
    if cfg['diagnostic_units']:
        import hashlib
        def diagnostic_rows(rows):
            bounded=[]
            for row in rows:
                if len(row['text'].split())<80:continue
                text=' '.join(row['text'].split()[:400])
                bounded.append({**row,'text':text,'words':len(text.split()),'complete':False,
                                'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
                if len(bounded)==cfg['diagnostic_units']:break
            if not bounded:raise ValueError('No sufficiently long diagnostic units')
            return bounded
        training_rows = diagnostic_rows(training_rows)
        validation_rows = diagnostic_rows(validation_rows)
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError('Use a new output directory or --resume')
    if resume and not (out/'experiment.json').is_file():
        raise ValueError('No story experiment to resume')
    experiment = {'config': cfg, 'corpus_sha256': digest(manifest),
                  'implementation_sha256':{name:file_hash(Path(__file__).parent/name) for name in
                      ('story_training.py','story_data.py','story_review.py','model.py')},
                  'preference_sha256': file_hash(cfg['preference_file']) if cfg['preference_file'] else None,
                  'test_evaluated': False, 'historical_adaptation_cutoff_inclusive': 1920,
                  'project_holdout_is_unseen_by_this_training_run': True,
                  'external_pretraining_overlap_with_holdout': 'unknown' if cfg['initialization']=='pretrained' else None,
                  'pretrained_knowledge_is_historically_restricted': False if cfg['initialization']=='pretrained' else None}
    parent = None
    if cfg['start_from']:
        parent = Path(cfg['start_from']).resolve()
        if not (parent/'config.json').exists():
            parent, _ = resolve_selected(parent)
        lineage = json.loads((parent/'story_provenance.json').read_text())
        if lineage['corpus_sha256'] != digest(manifest) or lineage['initialization'] != cfg['initialization']:
            raise ValueError('Continuation must preserve the story corpus and initialization lineage')
        if lineage.get('diagnostic_only'):
            raise ValueError('Diagnostic weights cannot initialize a production story run')
        parent_cfg=json.loads((parent/'training_summary.json').read_text())['experiment']['config']
        for key in ('base_model','revision','vocab_size','layers','width','heads','dropout'):
            if parent_cfg[key]!=cfg[key]:
                raise ValueError(f'Continuation must retain parent {key}; architecture/dropout changes need a new initialization')
        if parent == out.resolve() or out.resolve() in parent.parents:
            raise ValueError('Continuation output must not contain its source')
        experiment['parent'] = {'checkpoint': str(parent), 'sha256': checkpoint_hash(parent)}
    if resume:
        prior = json.loads((out/'experiment.json').read_text())
        if prior != experiment:
            raise ValueError('Frozen training config or corpus changed')
    else:
        out.mkdir(parents=True, exist_ok=True)
        # Declare ownership before downloads/initialization can be interrupted.
        write_json(out/'experiment.json', experiment)
    initial_ready=out/'initial/initialized.json'
    if resume and initial_ready.is_file():
        initial=json.loads(initial_ready.read_text())
        if checkpoint_hash(out/'initial/model')!=initial['checkpoint_sha256']:
            raise ValueError('Initial weights changed; do not resume from a damaged snapshot')
        tokenizer = AutoTokenizer.from_pretrained(out/'tokenizer', local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(out/'initial/model', local_files_only=True)
    else:
        if parent:
            tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
            tokenizer.save_pretrained(out/'tokenizer')
            model = AutoModelForCausalLM.from_pretrained(parent, local_files_only=True)
        else:
            tokenizer = make_tokenizer(corpus, out/'tokenizer', cfg['base_model'], cfg['revision'],
                                        cfg['vocab_size'], offline)
            model = initialize_model(cfg, tokenizer, offline)
    if cfg['context'] > context_length(model):
        raise ValueError('Context exceeds the loaded architecture')
    model.config.story_decoding = cfg['decoding']
    model.config.story_training_context = cfg['context']
    if not initial_ready.is_file():
        model.save_pretrained(out/'initial/model',safe_serialization=True)
        tokenizer.save_pretrained(out/'initial/model')
        write_json(initial_ready,{'checkpoint_sha256':checkpoint_hash(out/'initial/model')})
    dataset = StoryWindows(training_rows, tokenizer, cfg['context'], cfg['lookback'])
    mixture = TokenMixture(dataset, cfg['pool_shares'], seed=cfg['seed'], author_balance=cfg['author_balance'])
    horizon = token_horizon(mixture, cfg['batch_size']*cfg['accumulation'],
                            cfg['max_steps'], cfg['token_budget'], cfg['seed'])
    storage_gib=model.num_parameters()*4*(1+math.ceil(horizon['steps']/cfg['eval_steps'])+6)/(1024**3)
    print(f"Planned {horizon['steps']} updates / {horizon['targets']:,} targets; "
          f"approximately {storage_gib:.1f} GiB for retained weights and optimizer states.",flush=True)
    train_set = mixture
    pairs = []
    if cfg['objective'] == 'reviewed_preference':
        pairs = reviewed_pairs(cfg['preference_file'], training_rows, tokenizer, cfg['context'])
        train_set = PreferenceDataset(mixture, pairs, cfg['seed'])
    benchmark = make_benchmark(validation_rows, cfg['benchmark_prompts'], cfg['seed'], cfg['written_prompts'])
    if resume and json.loads((out/'benchmark.json').read_text()) != benchmark:
        raise ValueError('Frozen prose benchmark changed')
    write_json(out/'benchmark.json', benchmark)
    write_json(out/'data_summary.json', {'training_units': len(training_rows), 'validation_units': len(validation_rows),
        'unique_targets': sum(dataset.targets(i) for i in range(len(dataset))),
        'pool_target_shares': mixture.shares, 'epoch_target_exposure': mixture.exposure,
        'schedule': horizon,
        'estimated_weight_and_optimizer_gib':storage_gib,
        'parameters': model.num_parameters(), 'diagnostic_only': bool(cfg['diagnostic_units']),
        'tokenizer_sha256': file_hash(out/'tokenizer/tokenizer.json')})
    precision = legacy.resolve_precision(cfg['precision'], cfg['cpu'])
    checkpoint = story_checkpoint(out/'optimizer', precision=='fp16') if resume else None
    progress = {'tokens': 0, 'pool_tokens': {}, 'preference_ids': {}, 'unlikelihood_negatives': 0,
                'skipped_updates': 0, 'bad_reviews': 0, 'early_stop_nll': None}
    if checkpoint:
        progress.update(json.loads((Path(checkpoint)/'exposure.json').read_text()))
    last_ul = [-1]

    class Collator(legacy.CausalCollator):
        def __call__(self, examples):
            batch = super().__call__(examples)
            batch['reviewed_pairs'] = [row['reviewed_pair'] for row in examples if 'reviewed_pair' in row]
            batch['target_pools'] = [row['pool'] for row in examples]
            return batch

    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if getattr(kwargs['optimizer'], 'step_was_skipped', False):
                progress['skipped_updates'] += 1
            if progress['tokens'] >= cfg['token_budget'] or state.global_step >= cfg['max_steps']:
                control.should_training_stop = control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            # Persist consumed targets before potentially interruptible generation.
            seal_checkpoint(out/'optimizer'/f'checkpoint-{state.global_step}', progress)
            folder = out/'reviews'/f'step-{state.global_step}'
            report = save_review(kwargs['model'], folder, state.global_step)
            nll = report['metrics'][str(cfg['context'])]['all']['nll']
            old = progress['early_stop_nll']
            if old is None or nll < old - cfg['min_delta']:
                progress['early_stop_nll'], progress['bad_reviews'] = nll, 0
            else:
                progress['bad_reviews'] += 1
            progress['reviewed_step']=state.global_step
            seal_checkpoint(out/'optimizer'/f'checkpoint-{state.global_step}', progress)
            if progress['bad_reviews'] >= cfg['patience']:
                control.should_training_stop = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step and (progress.get('reviewed_step')!=state.global_step or
                    not (out/'reviews'/f'step-{state.global_step}'/'review.json').exists()):
                self.on_save(args, state, control, **kwargs)

    def save_review(student, destination, step):
        if step!=0:
            student.save_pretrained(destination/'model', safe_serialization=True)
            tokenizer.save_pretrained(destination/'model')
        report = review_model(student, tokenizer, validation_rows, benchmark, destination, step,
            cfg['context'], cfg['batch_size'], cfg['sample_tokens'], cfg['sample_seeds'],
            cfg['decoding'], training_rows)
        provenance = {'experiment_sha256': digest(experiment), 'corpus_sha256': digest(manifest),
                      'context': cfg['context'], 'step': step, 'review_sha256': digest(report),
                      'checkpoint_sha256': checkpoint_hash(destination/'model'),
                      'diagnostic_only': bool(cfg['diagnostic_units']), 'initialization': cfg['initialization']}
        write_json(destination/'model/story_provenance.json', provenance)
        write_json(destination/'model/training_summary.json', {**provenance, 'experiment': experiment})
        nll = report['metrics'][str(cfg['context'])]['all']['nll']
        best_path = out/'best_nll.json'
        best = json.loads(best_path.read_text()) if best_path.exists() else {'nll': float('inf')}
        if nll < best['nll']:
            write_json(best_path, {'nll': nll, 'step': step, 'checkpoint': str((destination/'model').relative_to(out))})
        print(f'Story review step {step}: NLL {nll:.4f}; '
              f'BPB {report["metrics"][str(cfg["context"])]["all"]["bits_per_byte"]:.4f}; '
              f'raw prose: {destination / "ratings.csv"}', flush=True)
        return report

    trainer = legacy.training_trainer(model=model, processing_class=tokenizer, train_dataset=train_set,
        data_collator=Collator(tokenizer.pad_token_id), callbacks=[legacy.window_epoch_callback(train_set), ProgressCallback()],
        args=TrainingArguments(output_dir=str(out/'optimizer'), max_steps=horizon['steps'],
            per_device_train_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['accumulation'],
            learning_rate=cfg['learning_rate'], weight_decay=cfg['weight_decay'], adam_beta1=.9, adam_beta2=.95,
            max_grad_norm=1., warmup_ratio=cfg['warmup_ratio'], lr_scheduler_type='cosine_with_min_lr',
            lr_scheduler_kwargs={'min_lr_rate': .1}, optim='adamw_torch',
            save_strategy='steps', save_steps=cfg['eval_steps'], save_total_limit=2,
            logging_steps=min(25,cfg['eval_steps']), report_to='none', remove_unused_columns=False,
            gradient_checkpointing=cfg['gradient_checkpointing'],
            gradient_checkpointing_kwargs={'use_reentrant':False} if cfg['gradient_checkpointing'] else None,
            seed=cfg['model_seed'], data_seed=cfg['seed'], use_cpu=cfg['cpu'],
            bf16=precision=='bf16', fp16=precision=='fp16', disable_tqdm=True,
            dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available() and not cfg['cpu']))

    def compute_loss(student, inputs, return_outputs=False, num_items_in_batch=None):
        import torch
        annotated = inputs.pop('reviewed_pairs', [])
        pools = inputs.pop('target_pools', [])
        labels = inputs.pop('labels')
        output = student(**inputs, use_cache=False)
        sums = legacy.loss_sums(output.logits, labels)
        count = sums[:, 1].sum()
        denominator = count if num_items_in_batch is None else torch.as_tensor(num_items_in_batch, device=count.device)
        loss = sums[:, 0].sum() / denominator.clamp_min(1)
        if student.training:
            progress['tokens'] += int(count)
            for pool, amount in zip(pools, labels[:,1:].ne(-100).sum(1).tolist()):
                progress['pool_tokens'][pool]=progress['pool_tokens'].get(pool,0)+amount
            if cfg['objective'] == 'reviewed_preference':
                preference, _, _ = legacy.continuity_preference_loss(student, annotated, tokenizer.pad_token_id)
                loss = loss + cfg['preference_weight']*preference*count/denominator.clamp_min(1)
                for pair in annotated:
                    progress['preference_ids'][pair['id']] = progress['preference_ids'].get(pair['id'], 0)+1
            if (cfg['objective'] == 'repetition' and trainer.state.global_step % cfg['unlikelihood_every'] == 0
                    and last_ul[0] != trainer.state.global_step):
                last_ul[0] = trainer.state.global_step
                # Legacy miner only needs a declared n_positions limit; new architectures
                # use a separate generic implementation below.
                ul, negatives = repetition_loss(student, inputs, tokenizer, cfg['context'])
                loss = loss + cfg['unlikelihood_weight']*ul
                progress['unlikelihood_negatives'] += negatives
        return (loss, output) if return_outputs else loss

    trainer.compute_loss = compute_loss
    trainer.model_accepts_loss_kwargs = True
    if not (out/'initial/review.json').exists():
        save_review(trainer.model, out/'initial', 0)
    if progress['early_stop_nll'] is None:
        initial_report=json.loads((out/'initial/review.json').read_text())
        progress['early_stop_nll']=initial_report['metrics'][str(cfg['context'])]['all']['nll']
    set_seed(cfg['model_seed'])
    if checkpoint:
        saved = json.loads((Path(checkpoint)/'trainer_state.json').read_text())
        complete = (saved['global_step'] >= horizon['steps'] or progress['tokens'] >= cfg['token_budget']
                    or progress['bad_reviews'] >= cfg['patience'])
    else:
        complete = False
    if complete:
        from transformers.trainer_callback import TrainerState
        trainer._load_from_checkpoint(checkpoint)
        trainer.state = TrainerState.load_from_json(str(Path(checkpoint)/'trainer_state.json'))
    else:
        trainer.train(resume_from_checkpoint=checkpoint)
    step = trainer.state.global_step
    final = out/'reviews'/f'step-{step}'
    if not (final/'review.json').exists():
        save_review(trainer.model, final, step)
    write_json(out/'last.json', {'step': step, 'checkpoint': str((final/'model').relative_to(out))})
    # Report actual target exposure in the Trainer's deterministic shuffled order.
    exposure = exposure_for_updates(mixture, cfg['batch_size']*cfg['accumulation'], step, cfg['seed'])
    summary = {'experiment_sha256': digest(experiment), 'global_step': step,
               'exposure': progress, 'pool_target_exposure': progress['pool_tokens'],
               'planned_pool_exposure_for_completed_updates': exposure, 'schedule': horizon,
               'test_evaluated': False,
               'best_nll': json.loads((out/'best_nll.json').read_text()),
               'last': json.loads((out/'last.json').read_text()), 'selected_for_generation': None,
               'diagnostic_only': bool(cfg['diagnostic_units'])}
    write_json(out/'summary.json', summary)
    return summary


def token_horizon(mixture, batch, max_steps, budget, seed):
    """Align the LR schedule with the first optimizer update reaching the budget."""
    import torch
    old, steps, tokens, totals = mixture.epoch, 0, 0, Counter()
    try:
        for epoch in range(math.ceil(max_steps/math.ceil(len(mixture)/batch))):
            mixture.set_training_epoch(epoch)
            order=torch.randperm(len(mixture),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            for start in range(0,len(order),batch):
                for i in order[start:start+batch]:
                    window=mixture.order[i]
                    book=mixture.base.windows[window][0]
                    amount=mixture.base.targets(window)
                    tokens+=amount
                    totals[mixture.base.rows[book]['pool']]+=amount
                steps+=1
                if tokens>=budget or steps>=max_steps:
                    return {'steps':steps,'targets':tokens,'pool_targets':dict(totals),
                            'budget_overshoot':max(0,tokens-budget),
                            'stop_reason':'token_budget' if tokens>=budget else 'max_steps'}
    finally:
        mixture.set_training_epoch(old)


def exposure_for_updates(mixture, batch, steps, seed):
    import torch
    old, remaining, totals = mixture.epoch, steps, Counter()
    per_epoch = math.ceil(len(mixture)/batch)
    try:
        for epoch in range(math.ceil(steps/per_epoch)):
            mixture.set_training_epoch(epoch)
            order = torch.randperm(len(mixture), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            for i in order[:min(remaining, per_epoch)*batch]:
                window = mixture.order[i]
                book = mixture.base.windows[window][0]
                totals[mixture.base.rows[book]['pool']] += mixture.base.targets(window)
            remaining -= min(remaining,per_epoch)
    finally:
        mixture.set_training_epoch(old)
    return dict(totals)


def repetition_loss(model, inputs, tokenizer, context):
    import torch
    from model import repetition_negative_positions
    prefix = inputs['input_ids'][:1, :min(128, int(inputs['attention_mask'][0].sum())-1)]
    before = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(prefix, attention_mask=torch.ones_like(prefix), do_sample=False,
                max_new_tokens=min(64, context-prefix.shape[1]), use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                suppress_tokens=[tokenizer.pad_token_id] if tokenizer.pad_token_id != tokenizer.eos_token_id else [])
        bad = repetition_negative_positions(generated[0,prefix.shape[1]:].tolist())
        if not bad:
            return next(model.parameters()).new_zeros(()), 0
        positions = torch.tensor(bad,device=model.device)+prefix.shape[1]
        logits = model(generated,use_cache=False).logits[0,positions-1].float()
        probabilities = logits.softmax(-1).gather(1,generated[0,positions,None]).squeeze(1)
        return -torch.log1p(-probabilities.clamp(max=1-1e-5)).mean(), len(bad)
    finally:
        model.train(before)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--catalog', default='story_catalog.json')
    prepare.add_argument('--output', default='story-data')
    prepare.add_argument('--cache', default='data')
    prepare.add_argument('--seed',type=int,default=42)
    prepare.add_argument('--offline',action='store_true')
    for name in ('train','plan'):
        p = commands.add_parser(name)
        p.add_argument('--config',required=True)
        p.add_argument('--resume',action='store_true')
        p.add_argument('--offline',action='store_true')
    sweep=commands.add_parser('ablation-configs')
    sweep.add_argument('--config',required=True)
    sweep.add_argument('--output',required=True)
    sweep.add_argument('--preferences')
    comparison=commands.add_parser('compare')
    comparison.add_argument('--reviews',nargs='+',required=True)
    comparison.add_argument('--output',required=True)
    unblind=commands.add_parser('unblind')
    unblind.add_argument('--comparison',required=True)
    unblind.add_argument('--ratings',required=True)
    unblind.add_argument('--output',required=True)
    select = commands.add_parser('select')
    select.add_argument('--run',required=True)
    select.add_argument('--checkpoint',required=True)
    select.add_argument('--ratings',required=True)
    select.add_argument('--reviewer',required=True)
    select.add_argument('--min-pass-rate',type=float,default=.8)
    select.add_argument('--max-nll-regression',type=float,default=.15)
    for name in ('generate','test'):
        p=commands.add_parser(name)
        if name=='generate':
            source=p.add_mutually_exclusive_group(required=True)
            source.add_argument('--run',help='Run directory containing a reviewed selection')
            source.add_argument('--checkpoint',help='Explicit checkpoint for diagnostic sampling, without promotion')
        else:
            p.add_argument('--run',required=True)
        p.add_argument('--output',required=name=='test')
        p.add_argument('--cpu',action='store_true')
        if name=='generate':
            p.add_argument('--prompt',required=True)
            p.add_argument('--tokens',type=int,default=192)
            p.add_argument('--seed',type=int,default=42)
            for key,value in DECODING_DEFAULTS.items():
                p.add_argument('--'+key.replace('_','-'),type=int if key=='top_k' else float,default=value)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        result=prepare_corpus(args.catalog,args.output,args.cache,args.seed,args.offline)
    elif args.command=='train':
        result=train(args.config,args.resume,args.offline)
    elif args.command=='plan':
        cfg=load_config(args.config)
        result={'config':cfg,'corpus':verify_corpus(cfg['corpus']),
                'note':'Plain next-token CE unless an explicit ablation is configured. No training started.'}
    elif args.command=='ablation-configs':
        from story_experiments import ablation_configs
        result=ablation_configs(args.config,args.output,args.preferences)
    elif args.command=='compare':
        from story_experiments import compare_reviews
        result=compare_reviews(args.reviews,args.output)
    elif args.command=='unblind':
        from story_experiments import unblind_ratings
        result=unblind_ratings(args.comparison,args.ratings,args.output)
    elif args.command=='select':
        result=select_checkpoint(args.run,args.checkpoint,args.ratings,args.reviewer,
                                 args.min_pass_rate,args.max_nll_regression)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        diagnostic=args.command=='generate' and args.checkpoint is not None
        if diagnostic:
            checkpoint,selection=Path(args.checkpoint).resolve(),None
        else:
            checkpoint,selection=resolve_selected(args.run)
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True)
        device='cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model.to(device).eval()
        if args.command=='generate':
            result=generate(model,tokenizer,args.prompt,args.tokens,args.seed,
                            decoding={key:getattr(args,key) for key in DECODING_DEFAULTS})
            result.update(checkpoint=str(checkpoint),diagnostic=diagnostic,selection=selection)
            if args.output:write_json(args.output,result)
        else:
            cfg=json.loads((Path(args.run)/'experiment.json').read_text())['config']
            if cfg['diagnostic_units']:
                raise ValueError('A diagnostic run cannot perform final test evaluation')
            manifest=verify_corpus(cfg['corpus'])
            output=Path(args.output)
            if output.exists():
                raise ValueError('Test report exists; choose an explicit new output')
            result={'selection':selection,'corpus_sha256':digest(manifest),'test_evaluated':True,
                    'test':score_text(model,tokenizer,read_jsonl(Path(cfg['corpus'])/'test.jsonl'),
                                      cfg['context'],cfg['batch_size'])}
            write_json(output,result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=='__main__':
    main()
