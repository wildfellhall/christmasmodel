"""Additional regressions, intended to be executed in the Colab verification cell."""
import ast
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_notebook_cells_are_valid_python():
    notebook=json.loads((ROOT/'Christmasmodel_Colab.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))


def test_matched_initializations_share_architecture_and_tokenizer():
    from story_training import load_config
    scratch=load_config(ROOT/'configs/story-matched-scratch.json')
    pretrained=load_config(ROOT/'configs/story-pretrained.json')
    changed={key for key in scratch if scratch[key]!=pretrained[key]}
    assert changed=={'initialization','output','learning_rate'}
    assert scratch['base_model'] and len(scratch['revision'])==40
    strict=load_config(ROOT/'configs/story-scratch.json')
    assert strict['base_model'] is None and strict['initialization']=='scratch'


def test_source_snapshot_is_never_directly_trainable(tmp_path):
    from story_data import load_units
    with pytest.raises(ValueError,match='reviewed catalog'):
        load_units(ROOT/'story_source_snapshot.json',tmp_path,offline=True)


def test_legacy_names_do_not_replace_partial_multitoken_names():
    from model import continuity_corruptions
    variants=continuity_corruptions('She told Elizabeth Ann and Betsy about the parcel.',
                                    'Elizabeth Ann carried the parcel into the little house.')
    entities=[text for kind,text in variants if kind=='entity']
    assert entities==['Betsy carried the parcel into the little house.']
    variants=continuity_corruptions('She said, “I saw the snow falling.',
        'The little boy had closed the door. His sister carried the lamp upstairs.')
    assert all(kind!='order' for kind,_ in variants)


def test_colab_resume_skips_checkpoint_without_token_accounting(tmp_path):
    from story_training import story_checkpoint, seal_checkpoint
    for step in (1,2):
        folder=tmp_path/f'checkpoint-{step}';folder.mkdir()
        for name in ('optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','model.safetensors'):
            (folder/name).write_text('{}')
    seal_checkpoint(tmp_path/'checkpoint-1',{'tokens':100})
    assert Path(story_checkpoint(tmp_path)).name=='checkpoint-1'
    assert story_checkpoint(tmp_path,require_scaler=True) is None
    seal_checkpoint(tmp_path/'checkpoint-2',{'tokens':200})
    assert Path(story_checkpoint(tmp_path)).name=='checkpoint-2'
    (tmp_path/'checkpoint-2/optimizer.pt').write_text('truncated optimizer file')
    assert Path(story_checkpoint(tmp_path)).name=='checkpoint-1'


def test_budget_schedule_matches_actual_deterministic_target_order():
    from story_data import StoryWindows, TokenMixture
    from story_training import token_horizon, exposure_for_updates
    class Tokenizer:
        eos_token_id=0
        def __call__(self,text,**kwargs):
            return {'input_ids':[ord(c) for c in text],
                    'offset_mapping':[(i,i+1) for i in range(len(text))]}
    rows=[{'id':str(i),'pool':pool,'author':str(i),'text':'a tale in the snow. '*35,'complete':True}
          for i,pool in enumerate(('original','winter','winter'))]
    mixture=TokenMixture(StoryWindows(rows,Tokenizer(),32,8),{'original':.7,'winter':.3},seed=5)
    natural=TokenMixture(mixture.base,seed=5)
    for epoch in (0,1):
        natural.set_training_epoch(epoch)
        assert len(natural.order)==len(natural.base)
        assert set(natural.order)==set(range(len(natural.base)))
    horizon=token_horizon(mixture,4,100,2000,5)
    exposure=exposure_for_updates(mixture,4,horizon['steps'],5)
    assert exposure==horizon['pool_targets']
    assert sum(exposure.values())==horizon['targets']>=2000
    assert horizon['budget_overshoot']<4*23
