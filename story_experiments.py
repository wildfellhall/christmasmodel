"""Generate controlled experiment configs and blind raw-prose review panels."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from story_data import digest, file_hash, read_jsonl, write_json
from story_review import RATING_FIELDS


def ablation_configs(config, output, preferences=None):
    from story_training import load_config
    base=load_config(config)
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty ablation-config directory')
    if base['objective']!='ce' or base['preference_file']:
        raise ValueError('Ablate from a plain-CE baseline')
    rows=read_jsonl(Path(base['corpus'])/'train.jsonl')
    pools=sorted({r['pool'] for r in rows})
    variants={'ce':{},'repetition':{'objective':'repetition'},
              'context-1024':{'context':1024,'lookback':256},
              'author-balance':{'author_balance':.25}}
    if preferences:
        variants['reviewed-preference']={'objective':'reviewed_preference',
                                        'preference_file':str(Path(preferences).resolve())}
    if 'winter' in pools and len(pools)>1:
        remaining={p:sum(r['words'] for r in rows if r['pool']==p) for p in pools if p!='winter'}
        for share in (.25,.5):
            shares={p:(1-share)*n/sum(remaining.values()) for p,n in remaining.items()}
            variants[f'winter-{int(100*share)}']={'pool_shares':{'winter':share,**shares}}
    rates=(1e-4,3e-4) if base['initialization']=='scratch' else (1e-5,2e-5,5e-5)
    for rate in rates:
        if rate!=base['learning_rate']:
            variants[f'lr-{rate:g}']={'learning_rate':rate}
    results=[]
    for name,changes in variants.items():
        cfg={**base,**changes,'output':str(Path(base['output']).with_name(Path(base['output']).name+'-'+name))}
        load_config(cfg)
        path=out/(name+'.json')
        write_json(path,cfg)
        results.append({'name':name,'config':str(path),'changed_fields':changes})
    manifest={'baseline':base,'variants':results,'same_corpus':True,'same_target_budget':True,
              'execution':'No runs launched. Change one factor at a time; compare common text BPB and blind prose.',
              'notes':['Max-step caps and early stopping can yield unequal actual targets; inspect summaries.',
                       'A 1024-context continuation requires a parent with at least that context.',
                       'Word proportions only allocate the non-winter remainder; the sampler enforces target-token pool shares.']}
    write_json(out/'manifest.json',manifest)
    return manifest


def compare_reviews(reviews, output):
    """Hide branch identity; compare identical prompts and held-out text only."""
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty comparison directory')
    reports=[]
    for path in reviews:
        path=Path(path).resolve()
        if path.is_dir():path=path/'review.json'
        reports.append((path,json.loads(path.read_text())))
    if len(reports)<2:raise ValueError('At least two reviews are required')
    first=reports[0][1]
    expected={r['sample_id'] for r in first['samples']}
    for _,report in reports[1:]:
        if any(report[key]!=first[key] for key in
               ('benchmark_sha256','validation_text_sha256','decoding','requested_new_tokens')):
            raise ValueError('Reviews must share frozen text, prompts and decoding')
        if {r['sample_id'] for r in report['samples']}!=expected:
            raise ValueError('Reviews must use the same modes and generation seeds')
    out.mkdir(parents=True,exist_ok=True)
    mapping, blind, metrics={},[],[]
    for branch,(path,report) in enumerate(reports):
        tokenizer_sha=file_hash(path.parent/'model/tokenizer.json')
        metrics.append({'branch':branch,'review':str(path),'tokenizer_sha256':tokenizer_sha,
                        'metrics':report['metrics']})
        for sample in report['samples']:
            identity=digest({'review':file_hash(path),'branch':branch,'sample':sample['sample_id']})[:24]
            mapping[identity]={'branch':branch,'review':str(path),'review_sha256':file_hash(path),
                               'sample_id':sample['sample_id'],'prompt':sample['prompt'],
                               'continuation':sample['continuation']}
            blind.append({'sample_id':identity,'prompt':sample['prompt'],'continuation':sample['continuation']})
    random.Random(42).shuffle(blind)
    with (out/'blind_ratings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sample_id','prompt','continuation',*RATING_FIELDS,'notes'])
        writer.writeheader();writer.writerows(blind)
    write_json(out/'unblinding.json',mapping)
    report={'branches':metrics,'panel':str(out/'blind_ratings.csv'),
            'comparison_metric':'Use bits_per_byte on the same text/context across tokenizers. Token NLL is tokenizer-specific.',
            'instructions':'Rate blind_ratings.csv before opening unblinding.json. 0=failed, 1=mixed, 2=clear success.'}
    write_json(out/'comparison.json',report)
    return report


def unblind_ratings(comparison, ratings, output):
    mapping=json.loads((Path(comparison)/'unblinding.json').read_text())
    rows=list(csv.DictReader(Path(ratings).open(encoding='utf-8-sig',newline='')))
    if len(rows)!=len(mapping) or {r['sample_id'] for r in rows}!=set(mapping):
        raise ValueError('Rate every blind sample exactly once')
    by_branch={}
    for row in rows:
        original=mapping[row['sample_id']]
        if file_hash(original['review'])!=original['review_sha256']:
            raise ValueError('Source review changed after blinding')
        if any(row[k]!=original[k] for k in ('prompt','continuation')):
            raise ValueError('Do not edit the raw prose')
        if any(row.get(k) not in {'0','1','2'} for k in RATING_FIELDS):
            raise ValueError('Rate every dimension 0, 1 or 2')
        by_branch.setdefault(original['branch'],[]).append({**row,'sample_id':original['sample_id']})
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    result={}
    for branch,branch_rows in by_branch.items():
        path=out/f'ratings-{branch}.csv'
        if path.exists():raise ValueError('Rating export already exists')
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(branch_rows[0]))
            writer.writeheader();writer.writerows(branch_rows)
        result[branch]=str(path)
    return result
