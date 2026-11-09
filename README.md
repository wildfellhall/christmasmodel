# Christmasmodel

Train a small literary continuation model **from random weights** on the Christmas and winter books in `corpus.py`. `model.py` downloads and cleans the books, learns a tokenizer, trains a `GPT2LMHeadModel(GPT2Config(...))`, prints checkpoint samples, and evaluates the best validation checkpoint on held-out books. `export.py` saves a Hugging Face package and converts the weights to GGUF.

## Corpus findings and model choice

The repository contains **189 Gutenberg URLs**, not an embedded text corpus. All 189 downloaded successfully during the implementation audit. Two alternate editions were removed: 19608 duplicates 10679 (*The Story of the Other Wise Man*), and 17743 duplicates 30907 (*Rosemary*). The remaining 187 books contain **5,119,735 words**. The collection includes overlapping anthologies, short Christmas stories, and longer domestic novels; it is not uniformly Christmas prose.

With the default seed and tokenizer:

| Split | Books | Words | Next-token targets |
| --- | ---: | ---: | ---: |
| Training | 168 | 4,606,880 | 6,675,415 |
| Validation | 9 | 256,540 | 365,092 |
| Test | 10 | 256,315 | 368,648 |

See [corpus_audit.json](corpus_audit.json) for counts, duplicate editions, and detected overlaps, and [corpus_manifest.json](corpus_manifest.json) for titles, source URLs, source hashes, and split assignments. Gutenberg can revise its texts; each preparation writes its own fresh provenance report.

| Setting | Default | Reason |
| --- | --- | --- |
| Parameters | **13,989,888** | Much less capacity than the 46.7M Poetry50M architecture |
| Transformer | 6 layers, width 384, 6 heads | Several attention layers with 64 dimensions per head |
| Context | 512 tokens | A few paragraphs without excessive training cost |
| Vocabulary | 8,192 byte-level BPE tokens | Learned on training books only; about 1.45 tokens per word |
| Embeddings | Tied input/output weights | Reduces parameters; all 256 byte values remain representable |
| Optimizer | AdamW, LR 3e-4, betas (0.9, 0.95) | Gradient clipping at 1.0, weight decay 0.1 |
| Schedule | 5% warmup, cosine to 10% of peak LR | Avoids abruptly large initial updates |
| Regularization | Embedding/residual dropout 0.15, attention dropout 0.1 | Limits overfitting on repeated books |
| Training | Up to 12 epochs; batch 8 × accumulation 4 | About 16K target tokens per optimizer update |
| Selection | Validation token NLL every 250 updates | Stop after 6 checks without ≥0.002 improvement; restore best |

These are justified starting defaults, **not the result of a completed hyperparameter sweep**. There are only 0.48 unique training tokens per parameter; twelve passes do not make twelve distinct corpora. Short, stylistically consistent continuations are the intended use. Fluent output, sustained plot coherence, and novelty must be judged after the full Colab run. This is not a chatbot or an instruction-following model.

## Run in Google Colab

Select **Runtime → Change runtime type → GPU**. A T4 is a suitable starting point; the script selects FP16 on T4 and BF16 on supported GPUs. CPU execution works for verification but is slower. GPU execution has not been tested in the implementation environment.

Run these cells in order:

```python
!git clone https://github.com/wildfellhall/christmasmodel.git
%cd /content/christmasmodel
%pip install -q -r requirements.txt
```

```python
!python model.py train
```

No pretrained weights or pretrained tokenizer are downloaded. You can also paste `model.py` into a notebook cell, with its dependencies installed and `corpus.py` in the working directory, then call `main(["train"])`.

Downloads are sequential, cached, and retried. Failed sources are listed in `data/download_failures.json`; preparation stops if fewer than 90% succeed. Use `--min-success-rate 1` to require every URL, `--offline` to require cached downloads, and `--rebuild` to explicitly regenerate prepared data. Missing or changed preparation settings are not silently reused.

Every 250 **optimizer updates** (after gradient accumulation), samples from three fixed story openings appear in the output and `runs/christmas/samples.jsonl`. Each opening gets a greedy continuation and a sampled continuation. Early checkpoints will be nonsense. Change `--sample-steps` independently of `--eval-steps` if desired.

The final `runs/christmas/best/` contains model weights, tokenizer, model card, final samples, and `training_summary.json`. Checkpoints retain optimizer/scheduler/RNG state for resumption; at most two checkpoint directories are kept. `test_metrics.json` reports test performance after validation-based selection. The selection metric is token-weighted `eval_nll`; the Trainer's separate `eval_loss` averages batches differently when their padding differs. Perplexities from different tokenizers are not directly comparable.

```python
!python model.py generate --prompt "On Christmas Eve, the snow fell softly over the village."
# Re-run test evaluation only when needed; do not tune settings against the test split.
!python model.py evaluate
```

Generation limits the continuation to the available context. It rejects prompts that already fill the context window.

### Preserve work across Colab disconnections

Mount Drive before training and use the same paths and settings when resuming:

```python
from google.colab import drive
from pathlib import Path
import subprocess, sys

drive.mount("/content/drive")
base = Path("/content/drive/MyDrive/christmasmodel")
command = [sys.executable, "model.py", "train",
           "--data-dir", str(base / "data"),
           "--output-dir", str(base / "run")]
subprocess.run(command, check=True)  # First run
# After reconnecting and reinstalling dependencies:
# subprocess.run(command + ["--resume"], check=True)
```

The default `/content` filesystem is temporary. Resume is available from the first saved checkpoint, and verifies that the dataset, tokenizer, and training settings match. Do not use `--rebuild` when resuming. Use a new output directory for another experiment. For less GPU memory, use `--batch-size 4 --accumulation 8` from the start of a new run.

## Export weights and download GGUF

After training, install the exporter dependencies. These preserve Colab's CUDA PyTorch installation.

```python
%pip install -q -r requirements-export.txt
%run export.py --setup --download
```

This creates:

- `exports/christmas/huggingface/`: Safetensors weights, config, tokenizer, model card, and training summary.
- `exports/christmas/christmasmodel-huggingface.zip`: the Hugging Face files as a downloadable archive.
- `exports/christmas/christmasmodel-F16.gguf`: the browser download, roughly 28 MB for the default model.
- `exports/christmas/export_manifest.json`: conversion revision, output hash, and verification status.

Use **`%run`**, rather than `!python`, for `--download`: browser downloads require the Colab notebook kernel. To download the ZIP too:

```python
from google.colab import files
files.download("exports/christmas/christmasmodel-huggingface.zip")
```

For a Drive training run, add `--model-dir /content/drive/MyDrive/christmasmodel/run/best`. Exports refuse to overwrite a nonempty destination; use another `--output-dir` for later exports. Q8 is optional:

```python
%run export.py --setup --outtype q8_0 --output-dir exports/christmas-q8 --download
```

F16 is the default to preserve quality in an already small model. Q8 conversion uses the pinned Python converter and does not require a compiler. The model is a completion model: use a llama.cpp runtime supporting the GPT-2 architecture, with no chat template and a context no greater than 512. GGUF support alone does not imply every frontend supports GPT-2.

The custom vocabulary does not match llama.cpp's stock tokenizer hashes. The exporter validates the tokenizer's unnormalized GPT-2 byte-level pipeline before explicitly selecting the GPT-2 pre-tokenizer; it does not alter vocabulary IDs or merges. It uses the pinned [llama.cpp revision](https://github.com/ggml-org/llama.cpp/tree/7d7f0287530bc8981690f361d8c1627dbae1c99a), including its GPT-2 tensor mapping. To compile a CPU runtime and verify token IDs and greedy first-token predictions, add `--verify` (requires CMake and a C++ compiler):

```python
%run export.py --setup --verify --output-dir exports/christmas-verified --download
```

### Optional Hugging Face upload

Authenticate with a write token via `notebook_login()` or a Colab secret named `HF_TOKEN`, then provide the destination:

```python
from huggingface_hub import notebook_login
notebook_login()
%run export.py --setup --repo-id YOUR_NAME/christmasmodel --output-dir exports/hub --download
```

No upload occurs without `--repo-id`. New repositories are private unless you add `--public`; existing repository visibility is preserved. The upload includes inference weights/tokenizer and GGUF, not optimizer checkpoints or downloaded books. No Hugging Face upload was performed during implementation.

To upload an existing export, or retry an interrupted upload without converting again:

```python
%run export.py --upload-only --repo-id YOUR_NAME/christmasmodel --output-dir exports/christmas
```

## Data preparation and evaluation details

- Parse `sources` using `ast.literal_eval`, without importing/executing `corpus.py`.
- Require English metadata and Gutenberg start/end markers; reject HTML/error pages and very short results. Remove the license wrapper, production credits, illustration blocks, and page markers. Unwrap long prose lines while retaining short verse lines and paragraph breaks. Cleanup is conservative: title pages, tables of contents, and some editorial material can remain.
- Detect near-identical editions using content-sampled 13-word shingles. Group books sharing at least 12 sampled shingles and 5% of the smaller signature; split whole groups approximately 90/5/5 by word count. This keeps detected anthology overlap in one partition. Authors may span splits. Small or heavily edited quotations can escape this heuristic; it is not proof of zero contamination. The audit also found zero cross-split exact paragraphs of at least 80 normalized words.
- Fit BPE only on training books. Begin/end books with EOS. Build windows within each book with a one-token overlap, so every next-token target is scored once and tails are retained. Padding is masked with `-100`; genuine EOS labels remain supervised.
- Shuffle training windows, keep validation/test fixed, and select by token-weighted validation NLL. Print fixed-seed samples without changing training RNG state. Reserve test text until model selection ends.

## Verification

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
python model.py train --smoke-test --output-dir runs/smoke
```

The smoke test uses a tiny CPU model and limited windows; its scores and samples do not measure the full model's quality. Implementation checks cover real-corpus preparation, eleven regression tests, a brief training run with the full 13,989,888-parameter/512-context configuration, and checkpoint resumption with identical final weights. Both F16 (28.7 MB) and Q8 (15.8 MB) exports of that configuration loaded in llama.cpp, matched all five tokenizer probes, and matched three greedy first-token predictions against Hugging Face. A full twelve-epoch training run and output-quality assessment remain for Colab.

The implementation follows the [Transformers GPT-2 interface](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/gpt2) and [Hugging Face Hub upload API](https://huggingface.co/docs/huggingface_hub/guides/upload). [Poetry50M](https://github.com/wildfellhall/Poetry50M) informed the custom-tokenizer approach; its larger architecture and incomplete training script were not copied wholesale.
