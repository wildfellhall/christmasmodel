# Christmasmodel

Train literary continuation models from random weights on Gutenberg fiction. The pasted-cell default in [model.py](model.py) now trains **local continuation preferences directly**, using the original Christmas stories and all approved pre-1920 winter selections. [export.py](export.py) packages Hugging Face weights and converts to GGUF.

The original commands retain the original Christmas-only restriction. Winter, coherence and continuity commands use the approved additions without changing old data, splits or token IDs, or loading outside pretrained weights.

## Direct continuity training (current default)

Paste **all of `model.py` into one Colab GPU cell**, with the existing prepared data and project weights available. The inline source lists and dependencies remain self-contained. Its default is:

```python
NOTEBOOK_ARGS = ["train-continuity", "--output-dir", "runs/christmas-continuity"]
```

This stage trains the model's token probabilities; it adds no external model or inference-only classifier. It prefers `runs/christmas-coherence/expanded/last`, then that run's `best`, followed by the existing winter-v2, winter and scene defaults. Explicit `--continue-from` overrides selection. Context length is inherited, including an already-trained 1,024-position model. Use a new output directory; `--resume` subsequently restores this stage's frozen data, optimizer and settings.

The added supervision is built from actual training books:

1. Select an exact, sentence-bounded **48–160-token prefix and 24–96-token continuation**. Preserve original positive token IDs and exclude chapter transitions. Abbreviations such as “Mrs.” do not split sentences.
2. Construct an alternative by replacing a recurring speaker name with another speaker from the same context, adding/removing auxiliary negation, or swapping two narrative sentences. Names require speaker evidence in the same book. Sentence swaps exclude quoted dialogue; negation additions/removals are balanced to prevent a shortcut that simply suppresses “not.” These are automatic corruption preferences, not human annotations establishing that every alternative is impossible.
3. Compare the original and altered continuation **under the same prefix**. Score at most 32 tokens after their first difference, so a changed name or event is not diluted across an otherwise identical long passage. Paired forwards disable dropout while retaining gradients.
4. Optimize `softplus(2 × (0.2 − gap))`, where `gap = rejected_NLL − original_NLL`, weighted **0.5**. Add **0.5 × original-continuation CE** to anchor positive likelihood. Ordinary source-replay CE remains at weight **1**. At most two preference pairs are processed per microbatch, and all auxiliary terms scale with the accumulation-wide target denominator.

There is also an **online source-versus-model objective**. Every two optimizer updates, the student generates up to 64 tokens from a training prefix, alternating greedy and sampled generation. The actual source continuation is preferred to that rollout using a length-normalized, DPO-style log-ratio objective against a **frozen copy of the starting project model** (beta 2, weight 0.5). This targets the student's own output tendencies, including failures not covered by the three corruption rules. The reference introduces no outside weights. Original CE anchors learning; generated prose is never used as a positive example. Source preference does not mean every other continuation is objectively wrong.

This follows the source-versus-model training idea of [SPIN](https://proceedings.mlr.press/v235/chen24j.html), with an online rollout schedule and length normalization rather than that paper's exact recipe. The reference stays frozen across optimizer resumes. Mining preserves training RNG streams and runs once per scheduled optimizer update, independent of gradient accumulation. A small sample of mined pairs is saved in `online_negatives.jsonl`; logs include the actual pair count and reference-relative preference gap. `--online-preference-weight 0` disables this objective. The reference copy and generated rollouts add memory and runtime.

The loss uses a reference-free preference formulation related to [SimPO](https://arxiv.org/abs/2405.14734), adapted to changed spans and coupled to positive CE. It is not a reproduction of that paper's large-model results. The new stage defaults to **1,500 updates, LR 2e-5, batch 4 × accumulation 4**, inherited dropout, 3% warmup and cosine decay to 2e-6. It keeps the 50/50 original/additional window mixture and generated-loop penalty. The earlier simplicity curriculum and unrelated-book-prefix objective default to zero here so the new supervision has a clear purpose. `--continuity-weight` adjusts its strength; 0 disables the preference and positive-continuation auxiliary terms together.

`material/continuity_train.json` and `continuity_validation.json` freeze every pair and its provenance. Training pairs come only from training books; validation books supply evaluation pairs only. No test-book preference pairs are built. The frozen recipe hashes both banks and rejects changed material on resume. Validation reports preference accuracy and gap **separately for names, polarity and order**, alongside original 512-window NLL and raw generated prose. Preference accuracy is not a narrative-quality score.

Console output now separates **PROMPT** and **GENERATED**, and inference prints the checkpoint path. `expanded/last/` retains the actual final update; `expanded/best/` remains selected by validation NLL. The default export cell now selects `runs/christmas-continuity/expanded/last`. Standard GPT-2 weights remain compatible with Hugging Face and GGUF; reranking is optional inference code, not a requirement for using the learned changes.

Execution tests include actual optimization that reverses a tiny model's initially wrong next-token choice without reranking, frozen-reference gradient checks, and exact interrupted/resumed training with online mining enabled. These test the learning machinery; prose from the full model must still be assessed directly.

**Local execution, 2026-11-09:** two 400-update checks were run on the available 13,989,888-parameter project checkpoint, using 10,250 training pairs from the original/approved supplementary pools, batch 4 × accumulation 2, and six unchanged validation prompts with greedy/sampled 64-token generation. The corruption-only run changed held-out preference accuracy from 29/44 to 30/44. Adding online preferences changed it to 31/44, including entity comparisons from 7/10 to 9/10; order stayed 12/22 and polarity stayed 10/12. Full validation NLL changed from 3.782808 to 3.792165 with the combined objective. **Raw prose still lost narrative meaning and sometimes looped. These runs do not demonstrate the requested coherence level.** They verify actual weight/output changes rather than only execution of tiny test models. Source weights were preserved and no test-set evaluation ran. The full 1,500-update Colab recipe has not been completed locally. The repository test suite has **93 passing tests**.

```bash
python model.py generate --model-dir runs/christmas-continuity/expanded/last --decode-mode sample --prompt "Mary heard a knock at the door. She put down her book and"
```

## Five coherence interventions (previous recipe)

To repeat the previous combined recipe, paste the entire `model.py` and change the arguments below. Keep the project's existing weights and prepared data in the notebook runtime or mounted Drive.

```python
NOTEBOOK_ARGS = ["train-coherence", "--output-dir", "runs/christmas-coherence"]
```

It chooses `runs/christmas-winter-v2/expanded/best` if available, then `runs/christmas-winter/expanded/best`, then `runs/christmas-scenes/control/best`. `--continue-from` and `--data-dir` override these locations. Both 14M and 42M project checkpoints are supported; transformer blocks, vocabulary and dropout are inherited. Source weights remain intact. Start this stage in a new output directory; it cannot resume an old recipe's optimizer.

These changes target separate failures visible in the earlier runs. **They are implemented hypotheses, not demonstrated fixes or optimized hyperparameters.** Low NLL and low repetition have already proved insufficient to establish narrative coherence here.

| Plan | Implementation and parameter decision |
| --- | --- |
| **1. Simplicity curriculum** | Early loss weights favor shorter sentences, fewer long words and fewer capitalized names. This is a surface proxy, not a semantic quality label. Raw scores lie in 0.5–1.5; weights are normalized by scored targets within each corpus pool **per microbatch** and anneal to 1 over the first 40% of updates. All examples remain. Normalization preserves pool weight but can move individual weights outside the raw range. A pool with one example receives no relative reweighting. |
| **2. Context discrimination** | Alongside ordinary next-token loss, rank the same 64-token real continuation under its true 128-token prefix versus another training book's prefix. Hinge margin **0.2 nats/token**, weight **0.1**. Paired forwards disable dropout but keep gradients. Only rows with enough context and another eligible book participate. These negatives can teach topic cues; they do not directly certify character or causal consistency. |
| **3. Generated-repetition unlikelihood** | Every **8 updates**, greedily generate at most **64 tokens** from one training prefix. Penalize the final token of the **third and subsequent occurrences of an eight-token phrase**, weight **0.1**. Original-text cross-entropy remains the main objective; generated text never becomes a positive target. This conservative rule avoids penalizing every repeated word/name. Actual negative counts, including zero, are logged. It targets loops, not all semantic errors. |
| **4. Train with longer context** | Extend GPT-2 to **1,024 positions**, rebuilding causal attention buffers. Copy old positions exactly and initialize new ones by tiling old embeddings; verify old-length logits before training. New positions then receive training. Include **256 context-only preceding tokens**, leaving up to **767 scored targets** per full row. Windows never cross books/ranges. More visible context enables continuity without guaranteeing its use. |
| **5. Generate and rerank** | Sample **8 candidates** at temperature **0.8**, top-p **0.92**, top-k **40**, repetition penalty **1.08**. Rank by mean conditional log-probability, bounded full-versus-last-16-token context gain, and penalties for repeated four-grams, copied prompt eight-grams and very short output. Position IDs stay aligned for context comparisons. This same-model heuristic cannot fix an incoherent candidate pool. All candidate scores and ordinary output are retained. |

Plan 3 uses the sequence-unlikelihood idea from [Welleck et al.](https://arxiv.org/abs/1908.04319), with this project's conservative negative rule. The context objective and reranking weights are project heuristics, not claims of reproducing a published coherence result.

Defaults: **1,000 updates, microbatch 4 × accumulation 4**, fresh AdamW at peak LR **3e-5**, betas **0.9/0.95**, decay **0.1**, clipping **1**, **3% warmup**, cosine decay to **3e-6**. The lower LR limits disruption while adding objectives and position embeddings. Full batches score up to 12,272 positive targets versus the previous 12,256; tails affect actual totals. Extra forwards and generation add compute. BF16/FP16 selection and skipped-update accounting remain. Curriculum and margin use the accumulation-wide target denominator; unlikelihood runs once on its scheduled update.

The 50/50 supplementary/original window mixture includes all approved material listed below. **Validation and dropout-off training diagnostics keep the old 512-token windows**, so longer training context cannot automatically lower comparison loss. No test-set model evaluation occurs. Logs expose `ce_nll`, `context_margin_loss`, `context_pairs`, `unlikelihood_loss`, and `unlikelihood_negative_tokens`. Checkpoint output includes greedy, sampled and reranked prose. Review retains ordinary samples on 20 frozen validation prefixes plus reranked samples on three prefixes, with copying checks against both training pools. These are diagnostics, not automatic coherence scores.

Outputs:

- **`expanded/last/`**: final update, even when validation NLL prefers earlier weights. Default export target; not a declared coherence winner.
- **`expanded/best/`**: lowest-validation-NLL checkpoint, including eligibility of the starting weights.
- **`expanded/reviews/step-N/model/`**: periodic weights; adjacent `coherence.json` and `reranked_review.csv` contain prose and candidate scores.
- `experiment.json`: frozen methods/settings; `comparison.json`: final and NLL-selected paths. Target exposure counts original positive labels, excluding auxiliary negatives and curriculum weights.

Resume with `--resume`, keeping other settings and omitting `--continue-from` if previously supplied. The frozen source snapshot/material are reused. For inference:

```bash
python model.py generate --model-dir runs/christmas-coherence/expanded/last --decode-mode reranked --prompt "Mary heard a knock at the door. She put down her book and"
```

Disable methods with `--curriculum-fraction 0`, `--context-loss-weight 0`, `--unlikelihood-weight 0`, `--coherence-context 512`, or inference `--decode-mode sample`. Changing training settings requires a new output directory. Optional `--with-control` adds an original-corpus-only branch **with the same five methods**; default is just the combined branch.

**Export this earlier recipe:** set `EXPORT_NOTEBOOK_ARGS` to select `runs/christmas-coherence/expanded/last` and an unused export directory. Upload remains opt-in through `--repo-id`. **Reranking lives in `model.py`, not GGUF**; other inference engines need equivalent candidate selection to reproduce it.

**85 tests pass**, covering old-context logit preservation, new-position and auxiliary gradients, accumulation scaling, exact interrupted/resumed training, candidate provenance, and the complete three-update save/recovery path. These small execution tests do not establish full-size prose quality or full Colab GPU execution.

## Winter-fiction continuation v2 (previous recipe)

To repeat this earlier recipe, paste the entire `model.py` into one Colab GPU cell and set `NOTEBOOK_ARGS` as below. The original 189 URLs, all 37 additional Gutenberg sources, exact selection boundaries, bibliographic evidence, dependencies, training and review code are inline. Existing project weights and their prepared `data/` directory are required.

The default source is `runs/christmas-winter/expanded/best` when that directory exists, otherwise `runs/christmas-scenes/control/best`. The chosen path is printed and its provenance/tokenizer verified before training. Explicit `--continue-from` always takes precedence. A different project checkpoint, including the 42M model or a prose-reviewed snapshot, can be supplied. Its tokenizer, context length of 512, architecture and dropout are checked/inherited. This is a new stage in **`runs/christmas-winter-v2`**; an older optimizer run cannot be resumed with this changed corpus and recipe.

```python
NOTEBOOK_ARGS = [
    "finetune-winter",
    "--continue-from", "/content/runs/christmas-winter/expanded/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-winter-v2",
]
```

Use mounted Drive paths for persistence. Downloads are cached in `winter-data/` beside the original data directory; `--winter-cache` can set another location. The original prepared data and tokenizer are not modified. Equivalent CLI: `python model.py finetune-winter --continue-from PATH --data-dir DATA --output-dir RUN`.

### Material used

The audited recipe accepts **60 contiguous selections from all 37 sources: 621,570 words and 899,802 additional next-token targets** with the existing tokenizer. These are additions to the original 168 training books. The newest round contributes **28 selections / 110,509 words / 161,475 targets including genuine EOS endings**; the former 24-source material remains included. Broad all-season collections are not appended wholesale.

| Sources | Selected material |
| --- | --- |
| The Curlytops Snowed In; Six Little Bunkers at Grandpa Ford's; The Bobbsey Twins at Snow Lodge | Complete narratives |
| Bunny Brown and His Sister Sue Giving a Show; The Bobbsey Twins at Home; Dotty Dimple at School | Complete narratives, including their autumn/winter setup and resolutions |
| Jack and Jill | Chapters I–XII: December through February |
| Understood Betsy | Chapters I–VII and XI: winter arrival, home/school life, and final autumn household scene |
| Little Women | Chapters 1–9 and 15–23: winter family life, late-autumn illness and Christmas reunion |
| The Dutch Twins | Chapter VI: skating and St. Nicholas |
| Our Little Swedish Cousin | Chapters I–III: skating, knitting and Yule-tide |
| Our Little Canadian Cousin | Chapters VI–IX: autumn through winter |
| Mari, Our Little Norwegian Cousin | Chapters VII–VIII and X: winter family conversations, lumber-camp visit and holiday frolics |
| Our Little Finnish Cousin | Chapters I–III and V–VII: farm home, school, December and Christmas |
| Heidi, Elisabeth P. Stork translation | Chapters IV and XVIII–XIX: grandmother's hut and winter village life |
| What Katy Did at School | Chapters XI–XII: Christmas boxes and waiting for spring |
| Five Little Peppers Midway | Chapters I–X: baking, the play and midwinter homecoming |
| The Bobbsey Twins in a Great City | Chapters I–V and XXI–XXII: winter opening and family reunion |
| Alcott, Aunt Jo's Scrap-Bag I | Tessa's Surprises; Tilly's Christmas |
| Alcott, Aunt Jo's Scrap-Bag III | Kate's Choice |
| Alcott, Aunt Jo's Scrap-Bag V | What a Shovel Did |
| Alcott, Aunt Jo's Scrap-Bag VI | How It All Happened; An Old-Fashioned Thanksgiving |
| Alcott, Kitty's Class Day and Other Stories | A Country Christmas |
| Alcott, Lulu's Library I | A Christmas Dream, and How It Came True |
| Montgomery short stories, Gutenberg 24873 and 24874 | A Christmas Inspiration; A Christmas Mistake; Aunt Cyrilla's Christmas Basket; The Josephs' Christmas; The Osbornes' Christmas |
| Montgomery short stories, Gutenberg 24876 | Bertie's New Year; Clorinda's Gifts; Ida's New Year Cake; The Christmas Surprise at Enderly Road; The Falsoms' Christmas Dinner; The Light on the Big Dipper; The Unforgotten One |
| Montgomery short stories, Gutenberg 24878 | Christmas at Red Butte (1909); Uncle Richard's New Year's Dinner (1910). Later stories in the same collection are excluded. |
| Freeman, Young Lucretia and Other Stories | Young Lucretia; Where the Christmas-Tree Grew; Ann Mary: Her Two Thanksgivings |
| Freeman, Comfort Pease and her Gold Ring | Complete narrative |
| Maida's Little Shop | XII–XIV: The First Snow, The Fair, Christmas Happenings; stop before the sequel advertisement |
| Hans Brinker | I, III–VII, IX: skating, family care and Saint Nicholas; omit the intervening travelogue chapters |
| Liliecrona's Home, Anna Barwell English translation | I–II: Christmas storm, shelter and the household spinning room; English edition 1914 |
| Patty in the City | XXIII: the complete Christmas chapter, through the last conversation and before publisher advertisements |
| The Golden Road | III–IV: Christmas and New Year resolutions |
| Anne's House of Dreams | 15–17: Christmas at Four Winds, New Year's Eve at the Light, A Four Winds Winter |
| Anne of Green Gables | XVIII–XIX and XXIV–XXV: winter rescue/visit and Christmas concert/gifts. The chapter reprinted in Gutenberg 24877 is not added again. |

Dates and edition caveats are recorded per source in `WINTER_SOURCES`. The Finnish book and Curlytops transcription use later printings of pre-1920 works; publisher catalogs are excluded. Selected prose retains original wording, spellings, dialogue and embedded verse/folklore. Chapter selection is editorial, not a claim that every paragraph is a simple self-contained scene.

All 37 downloads must succeed. Each selection must match its reviewed content hash and unambiguous start/end headings; changed editions stop preparation instead of silently substituting text. The supplementary manifest records raw/selected hashes and boundaries. Complete selections sharing an exact normalized **32-word passage** with any original validation/test book are excluded. Substantial reprints against original training books and earlier accepted selections are also screened. The curated catalog yielded no exclusions on the cached editions. Shorter overlaps and paraphrases can remain. Test text is read for contamination screening only; the test model score is never computed by this command.

### Training decisions

| Setting | Default and reason |
| --- | --- |
| Initialization | Continue the user's existing model, with a fresh optimizer; no outside pretrained weights or tokenizer |
| Mixture | 50% additional / 50% original **windows** per mixture epoch, approximately 50% of scored targets. Every original replay window appears once; new windows cycle without replacement, with new deterministic offsets each cycle. Longer selections contribute proportionally more windows. |
| Context and supervision | **128 extra preceding tokens** from the same document, inside the existing 512 positions. A full window scores 383 targets; added context and padding have no loss. Both pools use this policy, avoiding the earlier scene pilot's unequal supervision. Shifted target windows retain fixed target counts; fixed offsets cover each target once, while shifted offsets sample interior coverage. No cross-document windows or invented EOS at chapter cuts. |
| Budget | **1,000 updates; batch 8 × accumulation 4.** Audited default: 6,031,321 supplementary + 6,115,032 original targets; 49.66% supplementary and about 6.70 passes over that pool. 32,000 examples cover part of one 1,095-update mixture epoch. Added context costs compute without adding targets; this is not a matched-target comparison with v1. |
| Optimizer | AdamW, peak LR 5e-5, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1, gradient clip 1.0. Inherit dropout. Conservative continuation settings, not measured optima. |
| Schedule/precision | 3% warmup, cosine decay to 5e-6. BF16 on supported CUDA GPUs, otherwise scaled CUDA FP16; CPU/MPS FP32. Skipped optimizer updates are counted. |
| Validation | Original full validation split, every 250 updates and at the final step. The unchanged source remains eligible for lowest-NLL selection. All new selections belong to training; there is no new held-out supplementary split. |
| Prose review | Save exportable review weights every 250 updates. Same frozen 20 validation prefixes × seeds 42/43, up to 192 tokens; also show their first 96 tokens. Console samples include greedy and sampled text every 250 updates. |
| Diagnostics | Original dropout-off training/validation loss windows stay fixed. Add a **paired context-use diagnostic**: score up to 64 reference tokens with the true prefix and an unrelated equal-length prefix from another validation book. `context_gain_nll = unrelated - true`; positive is evidence of context sensitivity, including topic/word cues, not a coherence verdict. Copy checks cover both training pools for all comparators. |
| Optional control | Add `--with-control` to run original-corpus continuation from identical starting weights with the **same look-back** first. Same update budget/settings; compute and target counts are not exactly matched. Default compares the expanded branch to unchanged baseline weights. |

The supplementary corpus supplies approximately **13.5% more distinct targets** than the original training pool. Longer visible context is an experimental training change, not a measured coherence improvement. It uses the context-only loss masking described in [Hugging Face's sliding-window discussion](https://huggingface.co/docs/transformers/perplexity); that documentation concerns evaluation and does not establish this training setting as optimal. Original validation scoring is deliberately unchanged, so a lower reported NLL cannot come merely from changing its windowing.

`--winter-lookback 0` disables the new context policy for an ablation in a separate output directory. Keep the source weights, new catalog, benchmark and other options identical; report target exposure because equal updates do not equal scored tokens. The default 128 reserves roughly one quarter of context for preceding prose without expanding positional embeddings or architecture. LR, inherited dropout, batch and 1,000-update budget are retained to avoid simultaneously making unsupported optimizer/capacity changes. Selection by NLL and selection by prose quality remain separate.

### Inspect, resume and export

Add `--plan-only` to prepare and freeze everything without training. Add `--resume` to execute that plan or recover after a disconnect; **remove `--continue-from` and its path** when resuming. Keep the other settings, including `--with-control` if used. The saved initial weights and material remove the need to redownload or keep the external source-model directory. The original prepared data remain required. Edited material/settings are rejected on resume.

Outputs include:

- `material/supplementary_sources.json` and `material/train.jsonl`: exact additions, inclusion/exclusion results and frozen token IDs.
- `experiment.json`, `benchmark.json`, `planned_exposure.json`: source provenance, rationale, frozen prompts and budget.
- `expanded/initial/`: unchanged baseline prose, losses and context-use diagnostic.
- `expanded/best/`: model/tokenizer selected by original validation NLL, possibly the unchanged starting weights.
- `expanded/reviews/step-N/model/`: retained weights at each review, independently selectable for prose quality.
- `comparison.json`, `blind_best_full.csv`, `blind_final_full.csv`, corresponding short panels, and `review_key.json`: losses, initial/selected context-use summaries, and blind baseline/treatment comparisons. Human ratings are preserved on completed resume. No automatic coherence winner is declared. Full paired diagnostic details are in each `coherence.json`.

`export.py` accepts the best or any review model and carries `supplementary_sources.json` into the HF package. For this older recipe, change `EXPORT_NOTEBOOK_ARGS` to select `runs/christmas-winter-v2/expanded/best` and an unused export directory. The cell uses explicit arguments, so Jupyter's internal `-f .../kernel.json` is never parsed. It does not upload unless you add `--repo-id`.

With a repository checkout, the explicit `%run` command remains:

```python
%run export.py --model-dir /content/runs/christmas-winter-v2/expanded/best --output-dir /content/exports/winter --setup --outtype f16 --verify --download
```

Add `--repo-id YOUR_ACCOUNT/YOUR_MODEL` only when you want to upload to Hugging Face. Training does not upload. Test evaluation remains a separate explicit command after model selection.

For an execution check, use `--smoke-test --output-dir runs/winter-smoke`: all sources are prepared, but a tiny randomly initialized model trains for three CPU updates on capped windows. This checks plumbing, not the 14M model's quality.

Verification: **74 tests pass**, including context-only masking, genuine EOS and tail coverage, exact target accounting, exact resumed batches/weights with look-back enabled, contamination checks, paired context diagnostics, and pasted Colab argument handling. All 60 real selections matched their inline hashes and passed screening. Both control and expanded branches completed a three-update CPU smoke run. Recovery after the final optimizer save reproduced identical selected weights and blind sheets without another update. These checks validate execution; a full-size quality-training run is still needed to assess coherence.

## Option 1: scene-focused fine-tuning

The completed capacity experiment did not demonstrate sustained coherence. Its 42M branch overfit substantially. This bounded pilot changes the training examples and supervision, while retaining the original Christmas corpus, tokenizer, architecture, and source dropout. It is an experiment, not a promised coherence fix.

**To repeat the earlier scene experiment**, paste all of `model.py` into a Colab GPU cell and change `NOTEBOOK_ARGS` to the following. All original source URLs and required code remain inline; the cell installs pinned dependencies:

```python
NOTEBOOK_ARGS = ["finetune-scenes", "--output-dir", "runs/christmas-scenes"]
```

This expects the source model at `runs/christmas-capacity/14m/best` and its original prepared corpus at `data/`, relative to the notebook's working directory. For the reported run, that source is the best 14M model selected at step 7,750. **The reports ZIP does not contain weights or prepared data.** Keep or restore both from the original training session; this command will not rebuild the corpus or download a pretrained replacement.

For explicit Colab paths, edit the arguments before running the same complete cell:

```python
NOTEBOOK_ARGS = [
    "finetune-scenes",
    "--continue-from", "/content/runs/christmas-capacity/14m/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-scenes",
]
```

Use mounted Drive paths for persistence. The command verifies the source's dataset fingerprint and tokenizer, then saves an unchanged starting snapshot. A different prepared corpus, even with similar text, is rejected.

| Decision | Default and rationale |
| --- | --- |
| Source | Existing 6-layer, width-384, 6-head GPT-2; 512-token context. Reuses the learned vocabulary and language patterns. |
| Excerpts | Exact contiguous **192–448-token** training-book slices, starting and ending at paragraph boundaries; at least four detected sentences. No rewriting, outside text, or validation/test passages enter the training pool. |
| Selection | Prefer shorter sentences, fewer rare words, and fewer capitalized-name proxies; reject obvious headings, verse, and unanchored openings. Keep at most 16 nonoverlapping excerpts per book and remove exact duplicate passages. These are surface heuristics, not semantic certification. |
| Prefix supervision | Choose an original sentence boundary at **48–128 tokens**, preferably near 96, leaving at least 64 continuation tokens. Prefix tokens remain visible to attention but their labels are `-100`; the original continuation is supervised normally. |
| Mixture | **50% scene examples / 50% original windows per mixture epoch.** Every original replay window occurs once per epoch. This is an example ratio, not a token-loss ratio; masked prefixes and shorter excerpts reduce the scene share of scored targets. Both counts are reported. |
| Endings | Automatic paragraph cuts receive **no invented EOS target**. Original book endings retain EOS. A reviewed CSV can explicitly mark a natural scene ending for EOS supervision. |
| Optimizer | Fresh AdamW, peak **5e-5**, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, gradient clipping `1.0`. The peak is one sixth of the original training peak to limit disruption during specialization. Dropout is inherited. |
| Budget | **1,000 updates per branch**, batch 8 × accumulation 4. 3% warmup, cosine decay to 5e-6. Both branches run the fixed budget; these are starting settings, not tuned optima. |
| Control | Ordinary continuation from exactly the same starting weights, learning rate, and update budget. The scene branch tests the complete data/supervision intervention. Compute and scored-target counts are **not matched**. |
| Checkpoints | Validation, console samples, full prose review, and retained review weights every **250 updates**, plus the final partial interval. Standard optimizer checkpoints keep the best and recent checkpoints for resume. |
| Selection | `best/` is chosen by full-validation token NLL, with the unchanged source also eligible. Every periodic review retains weights so a better-writing checkpoint can be selected independently. No automatic coherence winner. |

The local original-corpus extraction produced 2,255 candidates across 163 training books, containing 556,905 excerpt tokens. These are candidate counts, not a claim that every passage is a complete simple scene. Actual counts are printed for your prepared data. A real pilot requires at least 128 selected passages to avoid silently running on a tiny repeated subset.

### Inspect or curate the excerpts

`--plan-only` prepares `scene_candidates.csv`, `scene_selection.json`, the frozen benchmark, and an unchanged source snapshot without training. Add `--resume` to execute that saved plan unchanged.

For manual curation, copy `scene_candidates.csv`, set `include` to `0` for passages to omit, and mark `natural_end=1` only where the excerpt provides a reasonable stopping point. Leave source text, token boundaries, IDs, and hashes unchanged. Start a **new** output directory with `--scene-review` pointing to the edited CSV. Only selected training-book slices can enter the dataset; edits to source content are rejected. The original script runs automatically without a reviewed CSV and labels that selection honestly as heuristic candidates.

### Resume, compare prose, and export

After a disconnect, rerun the full cell with the same settings, **replace `--continue-from` and its path with `--resume`**, and retain the same data/output directories. The saved starting snapshot makes the original source directory unnecessary for resumption. Dataset, initial weights, settings, selection annotations, and benchmark are checked against the frozen plan. Completed branches are reused. A completed final optimizer checkpoint can be finalized without another update.

The output directory contains:

- `comparison.json`: baseline and selected validation scores, exact target exposure, skipped optimizer steps, and paths. `coherence_winner` remains `null` for review.
- `control/initial/`: unchanged baseline prose, validation NLL, and fixed training-subset NLL.
- `control/best/` and `scenes/best/`: model/tokenizer packages selected by validation loss; these may retain the original model if no new checkpoint improves NLL.
- `scenes/reviews/step-250/model/`, etc.: retained, exportable weights even if validation NLL worsened.
- `blind_best_full.csv` and `blind_final_full.csv`: shuffled, opaque-label comparisons of the unchanged baseline, continuation control, and scene branch on the same 20 validation prefixes and two sampling seeds.
- `blind_best_short.csv` and `blind_final_short.csv`: the first 96 tokens of those same 192-token outputs. These are correlated views, not additional independent samples.
- `review_key.json`: reveals branch/checkpoint identities after prose review. Existing blind-review ratings are preserved on resume.

Rate grammar, character/scene continuity, connected events, and repetition. Exact training-copy spans of at least 32 generated tokens are reported; zero detected matches does not establish novelty. The full validation-book split stays unchanged, and the test set is not evaluated. Console samples include greedy and sampled generation; full reviews use the existing temperature 0.8, top-p 0.92, top-k 40, repetition penalty 1.08, without phrase blocking.

For a tiny **execution check**, use `--smoke-test --output-dir runs/scenes-smoke`. This creates a tiny random model on the existing prepared data and runs three CPU updates per branch. It does not load or assess the quality of the 14M source.

The existing exporter accepts either `scenes/best` or any retained review's `model/` directory. For example, from a repository checkout:

```python
%pip install -q -r requirements-export.txt
%run export.py --model-dir runs/christmas-scenes/scenes/best --output-dir exports/scenes --setup --outtype f16 --verify --download
```

Use the existing `--repo-id` option when you explicitly want to upload an export to Hugging Face. Training itself neither uploads weights nor evaluates the test split.

Implementation verification: 52 tests pass, including source-slice validation, prefix/padding/EOS target accounting, deterministic mixture exposure, exact resumed batches/weights, and recovery after the final optimizer save without another update. A tiny end-to-end run completed both branches on the prepared corpus; its F16 GGUF passed five tokenizer probes and three greedy first-token checks. These checks establish execution and export compatibility, not coherence or full Colab GPU-run quality.

## Earlier fresh capacity comparison

**Select `compare-capacity` explicitly to repeat the earlier fresh 14M versus 42M experiment.** Both models start from random weights and train on the same Christmas books. The uploaded checkpoint is not loaded. The experiment tests capacity under a shared recipe; it does not claim that 42M will be sufficient or that either model's optimal hyperparameters have been found.

Copy all of [model.py](model.py) into one Colab GPU cell. Its source list, installation bootstrap, cleaning, tokenizer training, training loop, and evaluation code are inline. Change the arguments near the top to:

~~~python
NOTEBOOK_ARGS = ["compare-capacity", "--output-dir", "runs/christmas-capacity"]
~~~

The equivalent repository command is **python model.py compare-capacity**. It reuses prepared data/tokenizer files when present. On a fresh runtime it downloads the original URLs, removes detected duplicate editions, groups overlapping works across splits, and learns BPE on training books only. All 189 downloads must succeed. This command rejects corpus subsets, rebuilding existing data, continuation weights, and architecture overrides. Gutenberg can revise its texts; retain prepared data to reproduce the earlier corpus byte for byte.

For persistence, mount Drive first and change the two paths in the pasted cell:

~~~python
from google.colab import drive
drive.mount("/content/drive")
~~~

~~~python
NOTEBOOK_ARGS = [
    "compare-capacity",
    "--data-dir", "/content/drive/MyDrive/christmasmodel/data",
    "--output-dir", "/content/drive/MyDrive/christmasmodel/capacity-40",
    # Add "--resume" when restarting this same experiment.
]
~~~

Use **--plan-only** to freeze the settings and prompts and print the budget without training; add **--resume** to execute that saved plan afterward. Use **--smoke-test --output-dir runs/capacity-smoke** for two distinct tiny CPU architectures, three updates each. Smoke tests validate execution, not prose quality.

| Decision | Configuration | Reason and limitation |
| --- | --- | --- |
| Capacity | 6 layers, width 384, 6 heads: **13,989,888** parameters; 12 layers, width 512, 8 heads: **42,286,080** | Tests a substantial difference from initialization. Both retain 64 dimensions per head and 4× MLP expansion. The larger model costs more per update; compute is not matched. |
| Architecture | Standard GPT-2, gelu_new, tied embeddings, GPT-2 initialization | Preserves HF/GGUF export compatibility. The library scales residual output-projection initialization with depth. No inserted identity layers or external weights. |
| Text/tokenizer | Existing 168/9/10 book splits; 8,192-token byte BPE | Keeps language evidence and evaluation difficulty fixed. No reweighting, rewritten stories, or changed training boundaries in this comparison. |
| Context | 512; shifted training windows; fixed validation windows | Reuses tested windowing. Windows never cross books; padding is ignored and genuine EOS targets remain supervised. Shifts preserve row lengths and target counts. |
| Batch | 8 sequences × 4 accumulation = 32 | Same effective batch and shuffled examples for both sizes. Before starting, 4 × 8 is an alternative if memory is tight. Effective batch must remain 32; do not change it on resume. |
| Dropout | **0.05 for embeddings, residuals, and attention** | Uses the previously more promising low-dropout regime. Attention dropout is explicitly set, avoiding the historical 0.1 default. |
| AdamW | LR 3e-4; betas (0.9, 0.95); epsilon 1e-8; decay 0.1; clip norm 1.0 | Shared starting recipe with responsive second-moment estimates and bounded gradients. Trainer excludes bias/norm parameters from weight decay. These are experimental choices, not measured optima. |
| Schedule | 2% warmup; cosine decay to 3e-5 | On this corpus, 329 warmup updates span about 0.8 epoch. This avoids spending the original 5% warmup's roughly two full epochs warming up a 40-epoch run. No optimizer/schedule restarts between epochs. |
| Budget | **40 epochs: 16,440 steps and 267,016,600 supervised targets per branch** on the audited corpus | Early stopping is disabled to preserve equal exposure. Short tails and partial final batches are counted exactly. Repetition is not new language evidence. Set a shorter --max-steps budget for both before starting if desired. |
| Precision | BF16 on supported CUDA GPUs; otherwise CUDA FP16 with scaling; CPU/MPS FP32 | Retains FP32 parameters/optimizer state. FP16 overflow skips are reported separately from attempted steps. Explicit --precision fp32 is available; precision is frozen on resume. |
| Seeds | Data/order/benchmark 42; model/dropout 42 | Example order is independent of architecture and generation RNG. For an independent initialization replicate, change only --model-seed and use a new output directory; the prepared split/tokenizer and prompts stay fixed. |
| Validation/checkpoints | Every 250 updates, every review milestone, and the final update | Retains lowest validation token NLL and latest resumable state. The final partial interval is evaluated. Test books remain reserved. |
| Training fit diagnostic | Fixed, evenly spread subset of up to 1,024 training windows at each prose review | Scores with dropout disabled and fixed offsets, without changing training RNG or the data iterator. Read this beside full validation NLL; dropout-on minibatch loss is not directly comparable. |
| Quick samples | Three fixed openings, greedy and sampled, every 500 updates | Makes gross failures visible. Sampling preserves training RNG and model mode. |
| Full prose panel | **20 frozen validation prefixes × seeds 42/43 × up to 192 generated tokens**, every 2,000 updates and at the end | Paragraph starts and sentence endings avoid prompts cut inside words. Prefixes span 64–160 tokens, leaving room for generation. Book-balanced selection occurs before either model trains; references never enter model input. |
| Panel decoding | Temperature 0.8, top-p 0.92, top-k 40, repetition penalty 1.08 | Same established settings across models/checkpoints; no phrase blocking that mechanically forces repetition down. Length and EOS endings are reported. |

Actual values and rationale are saved in **experiment.json**. Forty epochs is a fixed budget, not a prediction that coherence will appear. One initialization seed cannot establish a robust architecture advantage.

### Outputs and the coherence decision

- **experiment.json / benchmark.json:** immutable settings, data fingerprints, exact budget, and prompts.
- **14m/ / 42m/:** independent runs. Each keeps up to two full optimizer checkpoints and a best/ model selected by validation NLL.
- **reviews/step-N/** within each branch: sampled text, an editable review.csv, and a weights-only model/ snapshot. Earlier snapshots remain available if their prose is preferable. The default creates nine per branch; allow roughly 4 GB for model/checkpoint artifacts plus data.
- **comparison.json:** losses, selected steps, exact exposure, applied/skipped updates, and paths. No automatic coherence winner.
- **blind_review.csv:** mixed lowest-NLL checkpoint outputs with opaque IDs. Selection steps can differ between models; the summary records them.
- **blind_final_review.csv:** outputs from both models at the same final training step, for a comparison at equal exposure. **review_key.json** reveals the model/checkpoint for both panels afterward.

Rate grammar, character/scene continuity, event progression, and nonrepetition separately: **0 = clear failure, 1 = mixed/uncertain, 2 = sustained success**. Read failures and early endings as well as fluent examples. A valid continuation need not match the original reference. Also inspect periodic reviews: lowest validation NLL may not select the best prose.

Each continuation is checked for the longest exact training-token span of at least 32 tokens. Only generated tokens are searched, excluding prompts. Hash candidates are verified against actual IDs. Zero means no match under this rule, not proof of originality. Copying and repetition are diagnostics, not coherence scores.

### Resume and export

Rerun with **--resume** and the same paths/settings. Completed branches are skipped. An interrupted branch restores optimizer, scheduler, RNG, offsets, and shuffled order from its latest complete checkpoint. Partially written checkpoint directories are skipped; FP16 recovery also requires the saved gradient scaler. Before the first checkpoint, it restarts from its seed. After the final checkpoint, it finalizes without an extra update. Precision, settings, dataset fingerprint, and prompts must match; bitwise replay across different hardware/library versions is not promised.

Do not extend the budget on resume, since that would change the LR schedule. Recovery may regenerate a checkpoint's review CSV, so keep completed human ratings in a separate copy.

After reviewing prose, export either best/ or a retained reviews/step-N/model using the existing exporter:

~~~python
%run export.py --model-dir /content/runs/christmas-capacity/42m/best --output-dir /content/exports/christmas-reviewed --setup --verify --download
~~~

Use the existing **evaluate --model-dir PATH --data-dir DATA** command for test evaluation only after selecting a model. The paired experiment never evaluates the test set.

## Corpus findings and original model choice

The repository contains **189 Gutenberg URLs**, not an embedded text corpus. All 189 downloaded successfully during the implementation audit. Two alternate editions were removed: 19608 duplicates 10679 (*The Story of the Other Wise Man*), and 17743 duplicates 30907 (*Rosemary*). The remaining 187 books contain **5,119,735 words**. The collection includes overlapping anthologies, short Christmas stories, and longer domestic novels; it is not uniformly Christmas prose.

With the default seed and tokenizer:

| Split | Books | Words | Next-token targets |
| --- | ---: | ---: | ---: |
| Training | 168 | 4,606,880 | 6,675,415 |
| Validation | 9 | 256,540 | 365,092 |
| Test | 10 | 256,315 | 368,648 |

See [corpus_audit.json](corpus_audit.json) for counts, duplicate editions, and detected overlaps, and [corpus_manifest.json](corpus_manifest.json) for titles, source URLs, source hashes, and split assignments. Gutenberg can revise its texts; each preparation writes its own fresh provenance report.

The following defaults describe the original single-model train command, not the new pasted-cell comparison.

| Setting | Legacy train default | Reason |
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

These were starting heuristics, **not the result of a completed hyperparameter sweep**. Subsequent runs have not demonstrated sustained coherent prose. There are only 0.48 unique training tokens per parameter; twelve passes do not make twelve distinct corpora. Short, stylistically consistent continuations are the intended use. Fluent output, sustained plot coherence, and novelty must be judged after the full Colab run. This is not a chatbot or an instruction-following model.

## Original single-model workflow in Google Colab

Select **Runtime → Change runtime type → GPU**. A T4 is a suitable starting point; the script selects FP16 on T4 and BF16 on supported GPUs. CPU execution works for verification but is slower. CUDA execution has not been tested in the implementation environment; CPU regression tests and an actual 42M forward/backward update on the local Apple accelerator have passed.

**One continuous cell:** open [model.py](model.py), copy the entire file into one code cell in a fresh Colab GPU runtime, set `NOTEBOOK_ARGS = ["train"]`, and run it. The cell installs its dependencies, uses its embedded list of all 189 URLs, and starts fresh training. It does not need a repository clone, `corpus.py`, or `requirements.txt`.

Edit `NOTEBOOK_ARGS` near the top before running if needed:

```python
NOTEBOOK_ARGS = ["train"]                # Original single-model training
# NOTEBOOK_ARGS = ["train", "--resume"]  # Resume using the same data/output paths
# NOTEBOOK_ARGS = ["train", "--smoke-test", "--output-dir", "runs/smoke"]
```

Alternatively, use the repository files and run these cells in order:

```python
!git clone https://github.com/wildfellhall/christmasmodel.git
%cd /content/christmasmodel
%pip install -q -r requirements.txt
```

```python
!python model.py train
```

No pretrained weights or pretrained tokenizer are downloaded. Both notebook and command-line runs use the embedded URLs by default. To select a subset of those books explicitly, pass `--corpus path/to/corpus.py`; URLs outside the embedded list are rejected. Existing prepared data and checkpoints from the original source list remain compatible.

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

### Continue after a completed run

The first reported full run reached 12 epochs with validation NLL still falling: 3.9746 at step 3500 to 3.9171 at step 4750. Its test perplexity was 56.88, while the shared sample still contained broken clauses and inconsistent descriptions. This supports testing additional training; it does not establish that further loss improvement will produce coherent stories. Training loss is measured with dropout enabled, so its numerical gap from validation loss is not a precise overfitting estimate.

To continue that model, paste the **updated** `model.py` into a cell and replace `NOTEBOOK_ARGS` near the top with:

```python
NOTEBOOK_ARGS = [
    "train",
    "--continue-from", "/content/runs/christmas/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-stage2",
    "--epochs", "6",
    "--learning-rate", "1e-4",
    "--warmup-ratio", "0.03",
    "--patience", "4",
]
```

Use the actual saved paths if training under another working directory or on Drive. Both the source model and its original prepared `data/` must still exist. `--continue-from` also accepts `/content/runs/christmas/checkpoint-4750`. Initial runs still train from random weights; continuation loads only your existing project model. Nothing is downloaded from a pretrained model repository.

The continuation inherits the model's architecture, context, dropout, and tokenizer. It verifies the dataset fingerprint, starts a fresh optimizer and schedule, warms up to `1e-4`, and decays to `1e-5`. Six epochs is an upper bound; four evaluations without meaningful improvement can stop it earlier. Keep other settings unchanged for this experiment. Step numbers restart at zero for the new stage.

The original run remains untouched. The new output directory saves an `initial/` snapshot, `initial_samples.json`, and `continuation.json`. It evaluates the starting model on the same validation data and only selects new weights if they beat that baseline. Compare the fixed-prompt initial samples with the new `best/samples.json` to assess grammar and scene consistency as well as loss. This selection uses validation, not test results.

If the new stage is interrupted, use the same options but **replace** `--continue-from` and its path with `--resume`. This restores the new stage's optimizer/scheduler/RNG rather than starting another stage. `--resume` preserves an existing training budget; changing its epoch or LR settings is not a continuation mechanism.

### Compare dropout after stage 2

The reported continuation improved validation NLL to 3.8586 and test perplexity to 54.00, but greedy generation looped and conservative sampling still lost narrative meaning. This does not establish that dropout is the cause. The following bounded experiment tests that hypothesis while keeping the original corpus, tokenizer, architecture, and weights.

Paste the updated `model.py` into a fresh cell in the existing session, with:

```python
NOTEBOOK_ARGS = [
    "compare-dropout",
    "--continue-from", "/content/runs/christmas-stage2/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-dropout-comparison",
    "--max-steps", "500",         # 500 optimizer updates per branch
    "--learning-rate", "5e-5",
    "--warmup-ratio", "0.03",
    "--eval-steps", "100",
    "--sample-steps", "100",
    "--continuation-dropout", "0.05",
]
```

Both branches start independently from the same stage-2 weights and use the same shuffled data, seed, learning-rate schedule, and update budget. `control/` inherits the existing dropout (normally 0.15 embedding/residual and 0.1 attention); `lower-dropout/` uses 0.05 throughout. The override changes the live dropout modules as well as the serialized configuration. Architecture flags do not resize an existing model. Early-stopping patience is raised above the comparison budget so both branches receive the same number of updates.

This uses **1,000 updates in total**, roughly 2.4 passes through the audited training data at the default batch size. It does not evaluate the test set. `comparison.json` contains both validation results and fixed-seed greedy/sampled passages, and each branch retains the starting weights if no new checkpoint improves validation. Examine grammar, repetition, and consistent references across both sets of passages; a small NLL gain alone is insufficient to select a prose-generation recipe.

The reported 500-update comparison reached validation NLL **3.853164** for inherited dropout and **3.835750** for dropout 0.05, from the same 3.858588 starting point. That is about 1.7% lower validation perplexity for the lower-dropout branch versus control. Both branches still looped under greedy decoding and produced inconsistent sampled scenes. This single paired run supports a modest likelihood improvement; it does not demonstrate a solution to coherence or establish which limitation is responsible.

Use a new output directory for a new paired experiment. To resume an interrupted individual branch, use `train --resume` with that branch's output directory, the same numerical training arguments, `--skip-test`, and the saved patience from its `run_config.json`; omit `--continue-from` and `--continuation-dropout`. After resuming, its results are in that branch's `best/` directory (the parent comparison report is not automatically regenerated). For a single new continuation, `--continuation-dropout` is optional; omitting it preserves the source dropout. `--skip-test` is also available for other development runs.

### Diagnose the existing model before further training

Logged training losses include dropout; validation losses do not. Measure both splits with dropout disabled before using their gap to guide another experiment. The diagnostic uses your saved model and original prepared data, performs no training, and does not evaluate the test set.

Paste the updated `model.py` into a cell with:

```python
NOTEBOOK_ARGS = [
    "diagnose",
    "--model-dir", "/content/runs/christmas-dropout-comparison/lower-dropout/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-diagnostics",
]
```

Use the actual saved paths and a new, empty output directory. No corpus rebuild or additional downloads are needed. The command checks the dataset fingerprint, tokenizer, and source URLs before evaluation. It measures token-weighted NLL/perplexity on up to 1,024 evenly spaced windows per split; with the audited 512-token setup, that covers the whole validation split and a subset of training. Increase `--diagnostic-windows` to include more training windows if needed.

It also prints three deterministic passage probes from each split: a prefix taken directly from a book, its actual next passage, and greedy/sampled continuations. Generation receives only the original prefix token IDs; the reference passage is shown for comparison and is never supplied to generation. Prefixes can begin or end inside words because probes preserve token boundaries. These examples help assess whether familiar training passages behave differently from held-out passages; they are not an automatic coherence score.

The final `DIAGNOSTIC SUMMARY` and `diagnostics.json` record both losses, their gap, and whether input/output weights remain tied. A large gap can motivate investigating generalization; a small gap does not by itself prove insufficient capacity or optimization. Whole-book style differences also affect the gap. Review it alongside the passage probes before choosing another training run. The existing checkpoint and prepared dataset are unchanged.

### Compare decoding on the existing checkpoint

The reported dropout-disabled diagnostic measured training NLL **3.499251** on 518,000 targets and validation NLL **3.835750** on all 365,092 validation targets. Weights remained correctly tied. The shared training-book probe lost its soldier/ruined-home scene and entered an "old man" loop under greedy decoding; sampled output also lost scene consistency. The validation probe had similar failures. These examples show that the problem occurs on familiar text too. They do not establish whether model capacity, optimization, or the training objective is chiefly responsible.

The next bounded comparison changes decoding while keeping weights fixed. [Contrastive search](https://arxiv.org/abs/2210.14140) ranks plausible next tokens using their probabilities and a penalty for similarity to earlier hidden representations. Its published results concern other models; improvement on this tiny Christmas model remains unverified. A second comparison simply blocks repeated four-token phrases during sampling. Neither method establishes coherence by reducing repetition alone.

Paste the updated `model.py` into a Colab cell with:

```python
NOTEBOOK_ARGS = [
    "compare-decoding",
    "--model-dir", "/content/runs/christmas-dropout-comparison/lower-dropout/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-decoding",
    "--sample-tokens", "64",
]
```

This generates from the three existing fixed openings and the same three train/three validation book prefixes. Every method starts with the same exact prefix tokens and a 64-token generation cap. The methods are greedy, the original sampling settings, those same sampling settings with `no_repeat_ngram_size=4`, and contrastive search with `k=4, alpha=0.6`. Both sampling methods use seeds 42 and 43 by default; the two other methods are deterministic. That produces 54 continuations, streamed to the notebook and `samples.jsonl`, with the complete report in `comparison.json`. It performs no training or test evaluation and downloads no model or custom generation code. Original weights, tokenizer, data, and training defaults are preserved.

Read all passages for grammatical continuity, consistent characters/locations, and a plausible connection to the opening, especially the first two or three sentences. A valid continuation need not reproduce the reference. The summary reports duplicate token four-grams, output length, and EOS endings; it deliberately selects no automatic winner. Phrase blocking mechanically reduces the repetition metric and can still produce nonsense. Shorter outputs can also have fewer repetitions. If both alternatives remain incoherent across the passages, this comparison does not justify more decoding tweaks or another long run with the same training recipe.

To generate from a single opening with one of these methods:

```python
NOTEBOOK_ARGS = [
    "generate",
    "--model-dir", "/content/runs/christmas-dropout-comparison/lower-dropout/best",
    "--decode-mode", "contrastive",  # Or blocked_sample, sample, greedy, both
    "--prompt", "There was a knock at the door. Mary put down her book",
    "--sample-tokens", "64",
]
```

Contrastive search is implemented directly in this file using the [published ranking rule](https://huggingface.co/blog/introducing-csearch). It evaluates four candidate tokens per step and costs more than greedy/sampling. The implementation matches a full-prefix reference calculation in CPU tests, caps output at the model's context limit, suppresses padding, and stops on EOS. GGUF weights remain exportable, but this Python decoding procedure is not encoded in GGUF; an external runtime may not provide the same search method.

### Train with varied window positions: a controlled experiment

The reported decoding comparison found duplicate token four-gram rates of **46.99%** for greedy, **0.18%** for sampling, and **0%** for blocked sampling and contrastive search. Every output reached 64 tokens, so shorter outputs did not account for these differences. The shared sampled passages still lost characters and meaning; contrastive output also lost the situation and repeated "her aunt." This comparison has not demonstrated a coherence fix. Low repetition is insufficient as a quality target.

The next experiment changes training-data presentation. The original training windows are fixed: a token sees the same available preceding context and the same position within its window on every pass. With `--random-window-offsets`, each book gets a deterministic new offset each epoch. Interior windows move by up to 511 tokens with the default context size; they remain contiguous, in order, and entirely inside that same book. First and last windows, EOS occurrences, row lengths, and the number of training examples are preserved. This provides different contextual views of the existing Christmas text without adding or rewriting stories. It is an unvalidated hypothesis for this model, not a demonstrated cause or cure for incoherence.

Run the paired comparison from your existing lower-dropout checkpoint:

```python
NOTEBOOK_ARGS = [
    "compare-windowing",
    "--continue-from", "/content/runs/christmas-dropout-comparison/lower-dropout/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-window-comparison",
    "--max-steps", "1000",       # Per branch: 2,000 updates total
    "--learning-rate", "1e-4",
    "--warmup-ratio", "0.03",
    "--eval-steps", "200",
    "--sample-steps", "200",
    "--sample-tokens", "64",
]
```

Both branches start independently from exactly the same model, inherit its architecture/tokenizer/dropout, and get the same optimizer, LR schedule, shuffled example indices, target counts, and 1,000-update budget. `fixed/` uses the original chunks; `shifted/` varies interior positions. At the default batch size this is about 2.4 passes per branch. Both use a fresh optimizer, 30 warmup updates, and decay from 1e-4 to 1e-5. These are experimental settings, not established optimal values. The control distinguishes gains from additional optimization at this LR from gains associated with the changed windows.

Validation remains the original fixed whole-book split, and neither branch evaluates the test set. No prepared data are rebuilt. The source checkpoint remains untouched; each branch retains its starting model if no checkpoint beats its initial validation NLL. `comparison.json` records starting/best NLL, selected paths, fixed-opening samples, and continuations from the same three train/three validation passages. The latter are also saved as `best/passage_samples.json`. Check sentence structure, character identity, locations, and the connection to the opening across both branches. A small loss improvement alone does not establish that this experiment met the coherence target.

Shifted windows can overlap or omit some interior targets on an individual pass. They provide sampled coverage rather than the original exactly-once target coverage; their example lengths and total target counts still match the control. The underlying corpus and split assignments remain unchanged. Offset choices and shuffling depend only on the saved seed and training epoch, so a resumed run reproduces them. Regression checks cover resuming partway through a later epoch, with matching batches and final weights for both fixed and shifted modes. The sampler also avoids an epoch-reset issue in the pinned library's skipped-batch loader.

For one independent training stage, add `--random-window-offsets` to `train --continue-from ...`. Existing runs keep fixed windows by default. To resume an interrupted comparison branch, use `train --resume` with that branch's output directory, the same numerical settings, `--skip-test`, and the saved patience from `run_config.json`. Include `--random-window-offsets` for the shifted branch only, and omit `--continue-from`. After an individual resume, branch results are updated but the parent `comparison.json` is not regenerated automatically.

### Test additional depth while retaining the learned model

The reported 1,000-update windowing comparison improved validation NLL to **3.818915** with fixed windows and **3.799722** with shifted windows, from the same 3.835750 starting point. Shifted windows reduced perplexity by about 1.90% versus control. The shared passages still lost narrative meaning and looped under greedy decoding. The 14M model has therefore not met the intended coherence target, despite the likelihood improvements. These results do not establish that capacity is the cause; the following experiment tests one way of increasing it.

The depth comparison expands the existing six-layer model to twelve layers while preserving its initial predictions with dropout disabled:

| Branch | Layers | Parameters | Initialization |
| --- | ---: | ---: | --- |
| `control/` | 6 | 13,989,888 | Your shifted-window checkpoint |
| `expanded/` | 12 | 24,636,672 | The same learned weights plus six initially identity blocks |

The original blocks retain their order at positions 0, 2, 4, 6, 8, and 10. Each inserted block has random attention/MLP input weights and zero output projections, so its residual connections initially pass the input through unchanged. All layers then remain trainable. The tokenizer, embeddings, width, heads, context length, dropout, corpus, and splits are retained. This applies the residual-layer expansion principle in [Gesmundo and Maile, Section 3.6](https://arxiv.org/pdf/2308.06103); preservation of initial behavior does not guarantee better behavior after training.

Run from the actual shifted checkpoint:

```python
NOTEBOOK_ARGS = [
    "compare-depth",
    "--continue-from", "/content/runs/christmas-window-comparison/shifted/best",
    "--data-dir", "/content/data",
    "--output-dir", "/content/runs/christmas-depth-comparison",
    "--grow-layers", "12",
    "--random-window-offsets",
    "--max-steps", "1000",       # Per branch: 2,000 updates total
    "--learning-rate", "1e-4",
    "--warmup-ratio", "0.03",
    "--eval-steps", "200",
    "--sample-steps", "200",
    "--sample-tokens", "64",
]
```

Both branches use the same source checkpoint, shifted windows, shuffled example indices, training targets, optimizer settings, LR schedule, and update budget. The larger model costs more compute per update; this is an equal-token/update comparison, **not an equal-compute comparison**. It is also a test of this particular depth expansion and training budget, not a general verdict on model size. No external pretrained weights, new stories, or teacher-generated data are used.

Before any expanded-model training, the script checks initial logits against the source on the three fixed openings and refuses to proceed if they differ beyond numerical tolerance. Both branches also evaluate their initial validation NLL, which should be approximately **3.799722** for the reported source checkpoint. The output records depth, parameter counts, layer mapping, the logit check, validation results, and the same train/validation passage probes. The selected weights remain ordinary `GPT2LMHeadModel` checkpoints; no special inference architecture is required.

The final output prints `DEPTH COMPARISON` followed by `MATCHED PASSAGES`, with one train and one validation prefix and both branches' continuations side by side. Include both sections when sharing the result; the full set of probes remains in `comparison.json`.

The source checkpoint remains intact. Each branch retains its initial weights if no new checkpoint improves validation, and neither evaluates the test set. Inspect the actual checkpoint and final passages for grammatical and narrative continuity. A likelihood improvement accompanied by the same incoherence does not justify calling the experiment successful. If the expanded branch retains its initial model, its extra layers have not earned their cost in this run.

For a single expanded continuation, use `train --continue-from ... --grow-layers 12`. For resumption, replace `--continue-from` with `--resume`, **omit `--grow-layers`**, and retain the stage's numerical arguments, `--random-window-offsets`, and `--skip-test`. The saved expanded architecture and optimizer are restored without inserting more layers. Older runs without growth metadata remain compatible. As with the windowing comparison, resuming an individual branch does not regenerate the parent comparison report.

To export the expanded candidate, pass its path to the existing exporter, for example `--model-dir /content/runs/christmas-depth-comparison/expanded/best`, with a new export output directory. Identity insertion uses standard GPT-2 blocks, so Hugging Face and GGUF export retain the ordinary model format.

Implementation verification includes 28 passing tests, unchanged initial logits/samples/validation NLL, gradients and learning in the new blocks, and exact resumption of a short expanded-model run. The expanded 24.6M configuration was also exported as F16 (50.1 MB) and Q8 (27.2 MB); both loaded in llama.cpp and passed five tokenizer probes and three greedy first-token checks. These local checks used a briefly trained test checkpoint and do not establish prose quality on the user's trained model.

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

- Use the `sources` list embedded in `model.py`. For an explicit `--corpus` override, parse `sources` using `ast.literal_eval`, without importing/executing that file.
- Require English metadata and Gutenberg start/end markers; reject HTML/error pages and very short results. Remove the license wrapper, production credits, illustration blocks, and page markers. Unwrap long prose lines while retaining short verse lines and paragraph breaks. Cleanup is conservative: title pages, tables of contents, and some editorial material can remain.
- Detect near-identical editions using content-sampled 13-word shingles. Group books sharing at least 12 sampled shingles and 5% of the smaller signature; split whole groups approximately 90/5/5 by word count. This keeps detected anthology overlap in one partition. Authors may span splits. Small or heavily edited quotations can escape this heuristic; it is not proof of zero contamination. The audit also found zero cross-split exact paragraphs of at least 80 normalized words.
- Fit BPE only on training books. Begin/end books with EOS. Default training windows and all evaluation windows use a one-token overlap, so every next-token target is scored once and tails are retained. Optional shifted training windows sample different contextual views, with the same row lengths and preserved book edges. Padding is masked with `-100`; genuine EOS labels remain supervised.
- Shuffle training windows, keep validation/test fixed, and select by token-weighted validation NLL. Print fixed-seed samples without changing training RNG state. Reserve test text until model selection ends.

## Verification

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
python model.py train --smoke-test --output-dir runs/smoke
```

The smoke test uses a tiny CPU model and limited windows; its scores and samples do not measure the full model's quality. Implementation checks cover real-corpus preparation, regression tests, brief training and continuation runs with the full 13,989,888-parameter/512-context configuration, and checkpoint resumption with identical final weights. Continuation checks cover source-weight identity, data/tokenizer compatibility, retaining the starting weights when validation does not improve, and dropout behavior after reload. A short paired dropout run confirmed identical initial weights and validation scores, equal update counts, and skipped test evaluation. Diagnostic checks cover rejecting outside sources, preserving exact passage tokens, withholding reference continuations from generation, and an evaluation-only run on real-corpus windows. Decoding checks cover cached versus full-prefix contrastive ranking, the zero-penalty greedy limit, context limits, EOS stopping, and preserving weights/RNG/training mode. A separate gradient check on unequal-length real-corpus windows matched full-batch and accumulated gradients within 3e-8 in float32 with dropout disabled. Both F16 (28.7 MB) and Q8 (15.8 MB) exports of that configuration loaded in llama.cpp, matched all five tokenizer probes, and matched three greedy first-token predictions against Hugging Face. The full-run quality results above were supplied by the user; no improved prose quality is claimed for the dropout experiment, diagnostic, or decoding comparison.

The implementation follows the [Transformers GPT-2 interface](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/gpt2) and [Hugging Face Hub upload API](https://huggingface.co/docs/huggingface_hub/guides/upload). [Poetry50M](https://github.com/wildfellhall/Poetry50M) informed the custom-tokenizer approach; its larger architecture and incomplete training script were not copied wholesale.
