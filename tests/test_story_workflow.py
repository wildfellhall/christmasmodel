import copy
import csv
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import story_data as data
import story_review as review
import story_training as training


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path/'corpus'
    root.mkdir()
    hashes = {}
    for split in data.SPLITS:
        rows=[]
        for i in range(4):
            text=('Mary carried a red parcel across the snowy garden. John waited beside the kitchen door. '
                  'He opened the door and helped her bring the parcel inside. They put it on the table '
                  'and untied the ribbon together. Inside they found a warm coat for the child. '
                  'Mary smiled and went to find the child.\n\n')*3
            rows.append({'id':f'{split}-{i}','text':text,'text_sha256':data.hashlib.sha256(text.encode()).hexdigest(),
                         'pool':'winter' if i%2 else 'original','group':f'{split}-{i}',
                         'author':f'author-{i}','complete':i%2==0,'words':len(text.split())})
        data.write_jsonl(root/f'{split}.jsonl',rows)
        hashes[split]=data.file_hash(root/f'{split}.jsonl')
    data.write_json(root/'corpus.json',{'schema':1,'files':hashes,'counts':{},'lineage':'new_initializations_only'})
    return root


@pytest.fixture
def tokenizer(corpus,tmp_path):
    return data.make_tokenizer(corpus,tmp_path/'tokenizer',vocab_size=320)


def tiny_config(corpus,tmp_path,**overrides):
    return training.load_config({'corpus':str(corpus),'output':str(tmp_path/'run'),
        'cpu':True,'precision':'fp32','layers':1,'width':16,'heads':2,'vocab_size':320,
        'context':256,'batch_size':2,'accumulation':2,'max_steps':3,'eval_steps':2,
        'sample_tokens':4,'sample_seeds':[42], 'benchmark_prompts':1,'dropout':0.,
        'learning_rate':.001,**overrides})


def test_story_targets_and_endings_are_exact(corpus,tokenizer):
    rows=data.read_jsonl(corpus/'train.jsonl')
    ds=data.StoryWindows(rows,tokenizer,context=64,lookback=16)
    for book,row in enumerate(rows):
        targets=[]
        for i,(index,_,_) in enumerate(ds.windows):
            if index==book:
                batch=ds[i]
                assert len(batch['input_ids'])<=64
                targets.extend(t for t in batch['labels'] if t!=-100)
        expected=tokenizer.encode(row['text'],add_special_tokens=False)
        if row['complete']:expected.append(tokenizer.eos_token_id)
        assert targets==expected
    assert all(ds[i]['labels'][0]==-100 for i in range(len(ds)))


def test_token_mixture_balances_targets_and_resumes(corpus,tokenizer):
    rows=data.read_jsonl(corpus/'train.jsonl')
    rows[1]['text']=rows[1]['text'][:180]
    ds=data.StoryWindows(rows,tokenizer,64,16)
    mix=data.TokenMixture(ds,{'original':.8,'winter':.2},epoch_tokens=20000)
    total=sum(mix.exposure.values())
    assert mix.exposure['winter']/total==pytest.approx(.2,abs=.004)
    mix.set_training_epoch(3); order=mix.order.copy()
    restored=data.TokenMixture(ds,{'original':.8,'winter':.2},epoch_tokens=20000)
    restored.set_training_epoch(3)
    assert order==restored.order
    assert training.exposure_for_updates(mix,8,11,42)==training.exposure_for_updates(restored,8,11,42)


def test_dedup_and_group_split_isolation():
    import random
    rows=[]
    for i in range(12):
        text=' '.join(str(random.Random(i*10000+j).randint(0,999999)) for j in range(500))
        rows.append({'id':str(i),'group':str(i//2),'text':text,'words':500,'pool':'winter' if i<6 else 'original'})
    rows.append({**rows[0],'id':'reprint','group':'another-anthology'})
    splits,duplicates=data.partition_units(rows)
    assert duplicates==[{'id':'reprint','duplicate_of':'0'}]
    assignments={r['id']:s for s,rs in splits.items() for r in rs}
    for i in range(0,12,2):assert assignments[str(i)]==assignments[str(i+1)]
    assert all(splits[s] for s in data.SPLITS)


def test_corpus_edit_is_rejected(corpus):
    data.verify_corpus(corpus)
    with (corpus/'validation.jsonl').open('a') as h:h.write('\n')
    with pytest.raises(ValueError,match='changed'):data.verify_corpus(corpus)


def test_scratch_and_pretrained_preserve_architecture_and_token_ids(corpus,tokenizer,tmp_path):
    base=tmp_path/'base'
    torch.manual_seed(8)
    source=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_layer=1,n_embd=16,n_head=2,n_positions=256))
    source.save_pretrained(base);tokenizer.save_pretrained(base)
    cfg=tiny_config(corpus,tmp_path,base_model=str(base),initialization='pretrained')
    pretrained=training.initialize_model(cfg,tokenizer,offline=True)
    scratch=training.initialize_model({**cfg,'initialization':'scratch'},tokenizer,offline=True)
    assert pretrained.config.n_embd==scratch.config.n_embd
    assert torch.equal(pretrained.transformer.wte.weight,source.transformer.wte.weight)
    assert not torch.equal(scratch.transformer.wte.weight,source.transformer.wte.weight)
    assert pretrained.config.pad_token_id!=pretrained.config.eos_token_id


def test_external_revisions_and_context_defaults_are_explicit():
    with pytest.raises(ValueError,match='immutable'):
        training.load_config({'initialization':'pretrained','base_model':'owner/model','revision':'main'})
    assert training.load_config({'context':1024})['lookback']==256
    assert training.load_config({})['objective']=='ce'
    assert training.load_config({})['decoding']['repetition_penalty']==1.


def test_bpb_uses_same_original_bytes(corpus,tokenizer):
    rows=data.read_jsonl(corpus/'validation.jsonl')[:1]
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_layer=1,n_embd=16,n_head=2,n_positions=256))
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    metrics=review.score_text(model,tokenizer,rows,256)['all']
    text_tokens=len(tokenizer.encode(rows[0]['text']))
    assert metrics['targets']==text_tokens+1
    assert metrics['bytes']==len(rows[0]['text'].encode())
    assert metrics['bits_per_byte']==pytest.approx(text_tokens*math.log2(len(tokenizer))/metrics['bytes'],rel=1e-6)


def test_generation_preserves_rng_and_does_not_suppress_shared_eos(tokenizer):
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_layer=1,n_embd=16,n_head=2,n_positions=256))
    model.train();before=torch.get_rng_state().clone()
    result=review.generate(model,tokenizer,'Mary waited.',tokens=3)
    assert model.training and torch.equal(before,torch.get_rng_state())
    assert result['generated_tokens']<=3


def test_preferences_require_evidence_and_distinct_entities(corpus,tokenizer,tmp_path):
    row=data.read_jsonl(corpus/'train.jsonl')[0]
    prefix=row['text'].index('They put')
    end=row['text'].index('Mary smiled')
    pair={'id':'p1','unit_id':row['id'],'source_sha256':row['text_sha256'],
          'prefix_start':0,'target_start':prefix,'target_end':end,'kind':'entity',
          'evidence':'John waited beside the kitchen door.', 'reviewer':'test reviewer',
          'decision':'contradiction','reason':'Reviewed test fixture',
          'entities':[{'name':'Mary','canonical_id':'mary'},{'name':'John','canonical_id':'mary'}],
          'agreement_checked':True,'rejected':'John left the house with the unopened parcel.'}
    path=tmp_path/'pairs.jsonl';data.write_jsonl(path,[pair])
    with pytest.raises(ValueError,match='aliases'):training.reviewed_pairs(path,[row],tokenizer,512)
    pair['entities'][1]['canonical_id']='john';data.write_jsonl(path,[pair])
    accepted=training.reviewed_pairs(path,[row],tokenizer,512)
    assert len(accepted)==1
    assert accepted[0]['prefix_ids']+accepted[0]['chosen_ids']==tokenizer.encode(row['text'][:end],add_special_tokens=False)
    pair['evidence']='A fact from outside the prompt';data.write_jsonl(path,[pair])
    with pytest.raises(ValueError,match='visible prefix'):training.reviewed_pairs(path,[row],tokenizer,512)


def test_real_training_resume_and_selected_export_contract(corpus,tmp_path,monkeypatch):
    cfg=tiny_config(corpus,tmp_path)
    original=training.review_model
    once=[True]
    def interrupt(*args,**kwargs):
        if args[5]==2 and once[0]:
            once[0]=False
            raise InterruptedError('review interrupted after optimizer save')
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'review_model',interrupt)
    with pytest.raises(InterruptedError):training.train(cfg,offline=True)
    result=training.train(cfg,resume=True,offline=True)
    assert result['global_step']==3 and not result['test_evaluated']
    uninterrupted={**cfg,'output':str(tmp_path/'uninterrupted')}
    training.train(uninterrupted,offline=True)
    a=GPT2LMHeadModel.from_pretrained(Path(cfg['output'])/'reviews/step-3/model')
    b=GPT2LMHeadModel.from_pretrained(Path(uninterrupted['output'])/'reviews/step-3/model')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    initial=GPT2LMHeadModel.from_pretrained(Path(cfg['output'])/'initial/model')
    assert not torch.equal(initial.transformer.wte.weight,a.transformer.wte.weight)
    assert result['exposure']['tokens']==sum(result['pool_target_exposure'].values())
    run=Path(cfg['output']); checkpoint=run/'reviews/step-3/model';ratings=checkpoint.parent/'ratings.csv'
    with pytest.raises(ValueError,match='Rate every'):review.select_checkpoint(run,checkpoint,ratings,'test')
    rows=list(csv.DictReader(ratings.open()));fields=list(rows[0])
    for row in rows:
        row.update({key:'2' for key in review.RATING_FIELDS})
    with ratings.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    selected=review.select_checkpoint(run,checkpoint,ratings,'test fixture only',max_nll_regression=10)
    assert review.resolve_selected(run)[0]==checkpoint.resolve()
    (checkpoint/'config.json').write_text((checkpoint/'config.json').read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):review.resolve_selected(run)
    assert selected['pass_rate']==1.


def test_token_budget_stops_at_optimizer_boundary(corpus,tmp_path):
    cfg=tiny_config(corpus,tmp_path,token_budget=1)
    result=training.train(cfg,offline=True)
    assert result['global_step']==1 and result['last']['step']==1


def test_resume_rebuilds_interrupted_initialization(corpus,tmp_path,monkeypatch):
    cfg=tiny_config(corpus,tmp_path,max_steps=1)
    original=training.initialize_model
    attempts=[]
    def interrupted(*args,**kwargs):
        attempts.append(True)
        if len(attempts)==1:
            raise InterruptedError('disconnect before initial weights were saved')
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'initialize_model',interrupted)
    with pytest.raises(InterruptedError):training.train(cfg,offline=True)
    result=training.train(cfg,resume=True,offline=True)
    assert len(attempts)==2 and result['global_step']==1
    assert (Path(cfg['output'])/'initial/initialized.json').exists()


def test_explicit_diagnostic_generation_does_not_require_promotion(tokenizer,tmp_path):
    checkpoint=tmp_path/'diagnostic'
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_layer=1,n_embd=16,
                                   n_head=2,n_positions=256))
    model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
    output=tmp_path/'sample.json'
    result=training.main(['generate','--checkpoint',str(checkpoint),'--prompt','Mary waited.',
                          '--tokens','2','--cpu','--output',str(output)])
    assert result['diagnostic'] and result['selection'] is None
    assert output.exists() and not (tmp_path/'selected.json').exists()
