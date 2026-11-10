"""Colab-only orchestration. Importing this module does not start a run."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from story_data import digest, file_hash, prepare_corpus, verify_corpus, write_json

SOURCE_FILES=('model.py','export.py','story_data.py','story_training.py','story_review.py',
              'story_experiments.py','build_story_catalog.py','colab_workflow.py',
              'story_source_snapshot.json','story_boundaries.json','historical_additions.json',
              'requirements.txt','requirements-export.txt')


def require_colab(gpu=False):
    if 'google.colab' not in sys.modules and not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Run this workflow in Google Colab; local execution is disabled here')
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Choose Runtime > Change runtime type > GPU before training')


def prepare(workspace, source_root='.'):
    """Build frozen splits once; cache downloads and data in Google Drive."""
    require_colab()
    from build_story_catalog import build
    workspace,source=Path(workspace).resolve(),Path(source_root).resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    inputs={name:file_hash(source/name) for name in
            ('model.py','story_data.py','build_story_catalog.py','story_source_snapshot.json',
             'story_boundaries.json','historical_additions.json')}
    corpus,catalog=workspace/'story-data',workspace/'story_catalog.json'
    receipt=workspace/'preparation.json'
    if (corpus/'corpus.json').exists():
        saved=json.loads(receipt.read_text())
        if saved['input_hashes']!=inputs or saved['catalog_sha256']!=file_hash(catalog):
            raise ValueError('Corpus preparation inputs changed; choose a new versioned workspace')
        return verify_corpus(corpus)
    if corpus.exists():
        raise ValueError('Incomplete corpus directory found; use a new versioned workspace')
    build(workspace/'cache',workspace/'winter-cache',catalog,
          source/'story_boundaries.json',source/'historical_additions.json',offline=False,
          source_catalog=source/'story_source_snapshot.json')
    staging=Path(tempfile.mkdtemp(prefix='preparing-',dir=workspace))
    manifest=prepare_corpus(catalog,staging,workspace/'cache',seed=42,offline=True)
    # Write the receipt before the final rename, so a completed corpus always has it.
    write_json(receipt,{'input_hashes':inputs,'catalog_sha256':file_hash(catalog),
                        'corpus_sha256':digest(manifest)})
    staging.rename(corpus)
    return manifest


def configuration(template, workspace, smoke=False):
    """Freeze explicit run paths; rerunning a cell resumes exactly this config."""
    require_colab()
    from story_training import load_config
    workspace=Path(workspace).resolve()
    cfg=load_config(template)
    name=Path(cfg['output']).name+('-smoke' if smoke else '')
    cfg.update(corpus=str(workspace/'story-data'),output=str(workspace/'runs'/name),cpu=False)
    if smoke:
        cfg.update(max_steps=2,token_budget=16000,eval_steps=1,batch_size=1,accumulation=1,
                   benchmark_prompts=1,written_prompts=[],sample_tokens=8,sample_seeds=[42],
                   diagnostic_units=2,patience=3)
    path=workspace/'configs'/(name+'.json')
    if path.exists() and json.loads(path.read_text())!=cfg:
        raise ValueError('Run config changed; give the experiment a new output name')
    write_json(path,cfg)
    return path


def run(config):
    require_colab(gpu=True)
    import torch
    import transformers
    from story_training import load_config, train
    cfg=load_config(config)
    output=Path(cfg['output'])
    output.mkdir(parents=True,exist_ok=True)
    # The Trainer needs an empty new output. Keep environment/source receipt next
    # to the run until train creates experiment.json, rather than pre-populating it.
    info={'python':platform.python_version(),'torch':torch.__version__,
          'transformers':transformers.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(0),'config':cfg,
          'code':{name:file_hash(Path(__file__).parent/name) for name in SOURCE_FILES}}
    revision=Path(cfg['corpus']).parent/'source_revision.json'
    if revision.exists():info['source_revision']=json.loads(revision.read_text())
    write_json(output.parent/(output.name+'-environment.json'),info)
    try:
        summary=train(config,resume=(output/'experiment.json').exists())
    finally:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output/'environment.json',info)
    return summary
