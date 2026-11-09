"""Colab-ready GPT-2 trained from random weights on corpus.py's Gutenberg books.

Colab GPU runtime, separate cells:
    !git clone https://github.com/wildfellhall/christmasmodel.git
    %cd /content/christmasmodel
    %pip install -q -r requirements.txt
    !python model.py train
    !python model.py generate --prompt "On Christmas Eve, the snow"

Or paste this file after installing requirements and fetching corpus.py, then
call main(["train"]). No pretrained model or tokenizer is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
EOT, PAD = "<|endoftext|>", "<|pad|>"
PROMPTS = [
    "On Christmas Eve, the snow fell softly over the village. At the window,",
    '\"Come in, my dear,\" said the old woman, \"and warm yourself by the fire.',
    "The little girl had saved one candle for her father. When the clock struck",
]
CLEANING_VERSION = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_sources(path):
    """Read the literal list without executing Python from the corpus file."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "sources" for t in node.targets
        ):
            sources = ast.literal_eval(node.value)
            if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
                raise ValueError("corpus.py must define sources as a literal list of URLs")
            for url in sources:
                if not re.fullmatch(r"https://www\.gutenberg\.org/cache/epub/(\d+)/pg\1\.txt", url):
                    raise ValueError(f"Unexpected corpus URL: {url}")
            return list(dict.fromkeys(sources))
    raise ValueError("No literal sources list found")


def clean_book(raw):
    """Conservative Gutenberg cleanup; keep prose, dialogue, and verse layout."""
    text = unicodedata.normalize("NFC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    if "<html" in text[:1000].lower() or "<!doctype html" in text[:1000].lower():
        raise ValueError("Received HTML instead of a book")
    start = re.search(r"^\s*\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*[^\n]*\n", text, re.M | re.I)
    end = re.search(r"^\s*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.M | re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Missing or invalid Gutenberg boundary markers; refusing unstripped text")
    header, body = text[:start.start()], text[start.end():end.start()]
    metadata = {}
    for field in ("Title", "Author", "Language"):
        match = re.search(rf"^{field}:\s*(.+(?:\n[ \t]+[^\n]+)*)", header, re.M)
        metadata[field.lower()] = re.sub(r"\s+", " ", match.group(1)).strip() if match else "Unknown"
    if metadata["language"].lower() != "english":
        raise ValueError(f"Unexpected language: {metadata['language']}")
    body = body.replace("\ufeff", "").replace("\u00ad", "").replace("\xa0", " ")
    body = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", body)
    body = re.sub(r"(?im)^\s*(?:Produced by|Transcribed by|This eBook was produced by)[^\n]*(?:\n[^\n]+)*\n\s*\n", "\n\n", body)
    body = re.sub(r"(?is)\[(?:Illustration|Transcriber[’']?s? note)[^\]]*\]", "", body)
    body = re.sub(r"(?im)^\s*\[?(?:Page|Pg\.)\s+\d+\]?\s*$", "", body)
    paragraphs = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        # Prose is usually wrapped near 70 columns. Retain short-line verse.
        prose = len(lines) > 1 and sum(len(x.strip()) >= 55 for x in lines[:-1]) / (len(lines)-1) >= .65
        prose = prose and not all(x.startswith("    ") for x in lines)
        paragraphs.append(" ".join(x.strip() for x in lines) if prose else "\n".join(lines))
    body = re.sub(r"[ \t]+", " ", "\n\n".join(paragraphs)).strip()
    if len(body.split()) < 300:
        raise ValueError("Book has fewer than 300 words after cleaning")
    if re.search(r"END OF THE PROJECT GUTENBERG|FULL PROJECT GUTENBERG LICENSE", body, re.I):
        raise ValueError("Boilerplate survived cleanup")
    return metadata, body


def download_book(url, cache, offline=False):
    import requests
    book_id = url.split("/")[-2]
    path = Path(cache) / "raw" / f"{book_id}.txt"
    if path.exists():
        # Preserve CRLF so source hashes agree between downloads and cache hits.
        with path.open(encoding="utf-8", newline="") as handle:
            return handle.read()
    if offline:
        raise FileNotFoundError(f"No cached download: {path}")
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60), headers={"User-Agent": "christmasmodel-corpus/1.0"})
            response.raise_for_status()
            raw = response.content.decode("utf-8-sig", errors="strict")
            clean_book(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(raw, encoding="utf-8")
            tmp.replace(path)
            time.sleep(.25)  # Sequential, cached, modest requests to Gutenberg.
            return raw
        except (requests.RequestException, UnicodeError, ValueError) as exc:
            error = exc
            if isinstance(exc, requests.HTTPError) and exc.response.status_code == 404:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: {error}")


def fingerprints(text):
    """Content-sampled 13-word shingles, independent of line/paragraph wrapping."""
    words = re.findall(r"\w+", text.casefold())
    result = set()
    for i in range(len(words) - 12):
        value = int.from_bytes(hashlib.blake2b(" ".join(words[i:i+13]).encode(), digest_size=8).digest(), "little")
        if value % 32 == 0:
            result.add(value)
    return result


def group_books(books):
    """Remove near-identical editions; group overlapping anthologies for splitting."""
    kept, signatures, duplicates = [], [], []
    for book in books:
        sig = fingerprints(book["text"])
        duplicate = next((i for i, old in enumerate(signatures)
                          if min(len(sig), len(old)) and len(sig & old) / max(len(sig), len(old)) >= .85), None)
        if duplicate is not None:
            duplicates.append({"id": book["id"], "duplicate_of": kept[duplicate]["id"]})
        else:
            kept.append(book)
            signatures.append(sig)
    parents = list(range(len(kept)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    overlaps = []
    for i in range(len(kept)):
        for j in range(i):
            shared = len(signatures[i] & signatures[j])
            containment = shared / max(1, min(len(signatures[i]), len(signatures[j])))
            if shared >= 12 and containment >= .05:
                parents[find(i)] = find(j)
                overlaps.append({"ids": [kept[j]["id"], kept[i]["id"]], "containment": round(containment, 4)})
    groups = defaultdict(list)
    for i, book in enumerate(kept):
        groups[find(i)].append(book)
    return list(groups.values()), duplicates, overlaps


def split_groups(groups, seed):
    """Choose whole overlap groups near 90/5/5 by word count, not random windows."""
    if len(groups) < 3:
        raise ValueError("Need at least three independent book groups for train/validation/test")
    sizes = [sum(b["words"] for b in g) for g in groups]
    total = sum(sizes)
    rng, best = random.Random(seed), None
    for _ in range(1000):
        order = list(range(len(groups)))
        rng.shuffle(order)
        assigned = {"train": [], "validation": [], "test": []}
        remaining = order[:]
        for split in ("validation", "test"):
            count = 0
            while remaining and (not assigned[split] or count < total * .05):
                if len(remaining) <= (1 if split == "test" else 2) and assigned[split]:
                    break
                i = remaining.pop()
                assigned[split].append(i)
                count += sizes[i]
        assigned["train"] = remaining
        if not remaining:
            continue
        score = sum(abs(sum(sizes[i] for i in assigned[s])/total - .05) for s in ("validation", "test"))
        if best is None or score < best[0]:
            best = (score, assigned)
    if best is None:
        raise ValueError("Unable to form nonempty grouped splits")
    return {s: [b for i in ids for b in groups[i]] for s, ids in best[1].items()}


def prepare(args):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    if (data / "report.json").exists() and not args.rebuild:
        report = json.loads((data / "report.json").read_text())
        source_hash = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
        if (report["source_sha256"], report["seed"], report["requested_vocab_size"], report["cleaning_version"]) != (
            source_hash, args.seed, args.vocab_size, CLEANING_VERSION
        ):
            raise ValueError("Preparation settings/corpus changed; use a new --data-dir or --rebuild")
        if report["downloaded"] / report["sources"] < args.min_success_rate:
            raise ValueError("Cached corpus misses the requested success rate; retry with --rebuild")
        required = ["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "tokenizer/tokenizer.json"]
        if any(not (data / name).is_file() for name in required):
            raise ValueError("Prepared corpus is incomplete; restore it or use --rebuild")
        print(f"Using prepared corpus: {data / 'report.json'}", flush=True)
        return report
    sources = read_sources(args.corpus)
    books, failures = [], []
    for i, url in enumerate(sources):
        try:
            raw = download_book(url, data, args.offline)
            metadata, text = clean_book(raw)
            books.append(dict(id=url.split("/")[-2], url=url, text=text, words=len(text.split()),
                              sha256=hashlib.sha256(raw.encode()).hexdigest(), **metadata))
            print(f"[{i+1}/{len(sources)}] {metadata['title'][:70]}: {len(text.split()):,} words", flush=True)
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"SKIPPED: {exc}", flush=True)
    write_json(data / "download_failures.json", failures)
    if len(books) < 3 or len(books) / len(sources) < args.min_success_rate:
        raise RuntimeError("Too few valid books; inspect download_failures.json and retry. No training performed.")
    print("Detecting duplicate editions and overlapping stories...", flush=True)
    groups, duplicates, overlaps = group_books(books)
    splits = split_groups(groups, args.seed)
    tokenizer_dir = data / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    bpe = ByteLevelBPETokenizer(add_prefix_space=False)
    bpe.train_from_iterator((b["text"] for b in splits["train"]), vocab_size=args.vocab_size,
                            min_frequency=3, special_tokens=[EOT, PAD], show_progress=True)
    bpe.save_model(str(tokenizer_dir))
    tokenizer = GPT2TokenizerFast(vocab_file=str(tokenizer_dir / "vocab.json"),
                                 merges_file=str(tokenizer_dir / "merges.txt"),
                                 bos_token=EOT, eos_token=EOT, unk_token=EOT, pad_token=PAD,
                                 add_prefix_space=False, add_bos_token=False, model_max_length=1_000_000)
    tokenizer.save_pretrained(tokenizer_dir)
    stats, manifest = {}, []
    for split, subset in splits.items():
        token_count = 0
        with (data / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for book in subset:
                ids = tokenizer.encode(book["text"], add_special_tokens=False)
                token_count += len(ids) + 1
                handle.write(json.dumps({"id": book["id"], "ids": [tokenizer.eos_token_id] + ids + [tokenizer.eos_token_id]}) + "\n")
                clean_path = data / "clean" / f"{book['id']}.txt"
                clean_path.parent.mkdir(exist_ok=True)
                clean_path.write_text(book["text"], encoding="utf-8")
                manifest.append({k: v for k, v in book.items() if k != "text"} | {"split": split, "tokens": len(ids)})
        stats[split] = {"books": len(subset), "words": sum(b["words"] for b in subset), "target_tokens": token_count}
    report = {"cleaning_version": CLEANING_VERSION, "source_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
              "seed": args.seed, "requested_vocab_size": args.vocab_size, "vocab_size": len(tokenizer),
              "sources": len(sources), "downloaded": len(books), "failures": failures,
              "duplicates": duplicates, "overlaps": overlaps, "independent_groups": len(groups), "splits": stats}
    write_json(data / "manifest.json", manifest)
    write_json(data / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


class BookWindows:
    """One-token overlap preserves every next-token target, including book tails.

    Windows never cross books. Real EOS labels stay supervised; only padding is
    -100. No random-window split, and no synthetic next-book transitions.
    """
    def __init__(self, path, block_size, limit=None):
        import numpy as np
        self.books, self.windows = [], []
        self.block_size = block_size
        for line in Path(path).read_text().splitlines():
            book = json.loads(line)
            ids = np.asarray(book["ids"], dtype=np.int32)
            index = len(self.books)
            self.books.append(ids)
            self.windows.extend((index, start) for start in range(0, len(ids)-1, block_size-1))
        if limit:
            # Spread smoke/evaluation subsets across the corpus, not just book 1.
            indices = np.linspace(0, len(self.windows)-1, min(limit, len(self.windows)), dtype=int)
            self.windows = [self.windows[i] for i in indices]
        if not self.windows:
            raise ValueError(f"Empty token dataset: {path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        book, start = self.windows[index]
        return {"input_ids": self.books[book][start:start+self.block_size].tolist()}


class CausalCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch
        width = max(len(x["input_ids"]) for x in examples)
        ids = torch.full((len(examples), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, example in enumerate(examples):
            n = len(example["input_ids"])
            ids[i, :n] = torch.tensor(example["input_ids"])
            labels[i, :n] = ids[i, :n]
            mask[i, :n] = 1
        # Position zero is context, never a next-token target. Mask it explicitly
        # so Trainer's accumulation denominator counts the shifted targets too.
        labels[:, 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def loss_sums(logits, labels):
    """Reduce logits before storing evaluation outputs; retain token-weighted NLL."""
    import torch
    from torch.nn import functional as F
    if isinstance(logits, tuple):
        logits = logits[0]
    targets = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                             reduction="none", ignore_index=-100)
    return torch.stack((losses.sum(dim=1), (targets != -100).sum(dim=1)), dim=1)


def token_metrics(prediction):
    values = prediction.predictions
    nll = float(values[:, 0].sum(dtype="float64") / values[:, 1].sum(dtype="float64"))
    return {"nll": nll, "perplexity": math.exp(min(nll, 50)), "scored_tokens": int(values[:, 1].sum())}


def generate_samples(model, tokenizer, prompts, max_new_tokens=96, seed=1234):
    import torch
    device = next(model.parameters()).device
    was_training = model.training
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    # Sampling must not change the random stream used by training dropout.
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    results = []
    model.eval()
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for prompt in prompts:
                ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.eos_token_id]
                room = model.config.n_positions - len(ids)
                if room <= 0:
                    raise ValueError("Prompt fills the context window; shorten it")
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                for mode in ("greedy", "sample"):
                    kwargs = dict(max_new_tokens=min(max_new_tokens, room),
                                  do_sample=mode == "sample", use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                  suppress_tokens=[tokenizer.pad_token_id])
                    if mode == "sample":
                        kwargs.update(temperature=.8, top_p=.92, top_k=40, repetition_penalty=1.08)
                    output = model.generate(inputs, attention_mask=torch.ones_like(inputs), **kwargs)
                    results.append({"prompt": prompt, "mode": mode,
                                    "continuation": tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)})
    finally:
        model.train(was_training)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    return results


def dataset_signature(data):
    digest = hashlib.sha256()
    for name in ("report.json", "train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer/tokenizer.json"):
        with (Path(data) / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def training(args):
    import torch
    from transformers import (
        AutoTokenizer,
        EarlyStoppingCallback,
        GPT2Config,
        GPT2LMHeadModel,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint
    if args.resume and args.rebuild:
        raise ValueError("Do not rebuild a dataset while resuming an existing run")
    report = prepare(args)
    if args.smoke_test:
        args.layers, args.width, args.heads, args.block_size = 2, 64, 2, 64
        args.batch_size, args.accumulation, args.max_steps = 2, 1, 3
        args.eval_steps, args.sample_steps, args.sample_tokens = 1, 1, 12
        args.train_limit, args.eval_limit, args.cpu = 16, 8, True
    set_seed(args.seed)
    data, out = Path(args.data_dir), Path(args.output_dir)
    resume = None
    if args.resume:
        resume = get_last_checkpoint(str(out)) if args.resume == "latest" else str(Path(args.resume).resolve())
        if not resume:
            raise ValueError("No checkpoint found to resume")
        if Path(resume).resolve().parent != out.resolve():
            raise ValueError("Resume checkpoint must belong to this --output-dir")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a new --output-dir")
    tokenizer = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    tokenizer.model_max_length = args.block_size
    signature = dataset_signature(data)
    settings = {"dataset_sha256": signature, "layers": args.layers, "width": args.width,
                "heads": args.heads, "block_size": args.block_size, "dropout": args.dropout,
                "batch_size": args.batch_size, "accumulation": args.accumulation,
                "learning_rate": args.learning_rate, "epochs": args.epochs, "max_steps": args.max_steps,
                "seed": args.seed, "train_limit": args.train_limit, "eval_limit": args.eval_limit,
                "eval_steps": args.eval_steps, "patience": args.patience, "smoke_test": args.smoke_test}
    if resume:
        prior = json.loads((out / "run_config.json").read_text())
        if prior != settings:
            raise ValueError("Run settings or dataset differ from saved run_config.json; use a new run")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", settings)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=args.block_size, n_ctx=args.block_size,
                        n_embd=args.width, n_layer=args.layers, n_head=args.heads, n_inner=4*args.width,
                        activation_function="gelu_new", resid_pdrop=args.dropout,
                        embd_pdrop=args.dropout, attn_pdrop=.1, initializer_range=.02,
                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True, use_cache=False)
    # This constructor initializes ALL weights randomly. Never load pretrained GPT-2.
    model = GPT2LMHeadModel(config)
    parameters = model.num_parameters()
    train_set = BookWindows(data / "train.jsonl", args.block_size, args.train_limit)
    val_set = BookWindows(data / "validation.jsonl", args.block_size, args.eval_limit)
    effective_batch = args.batch_size * args.accumulation
    print(f"Random GPT-2: {parameters:,} parameters; {len(train_set):,} training windows; "
          f"effective batch {effective_batch} (~{effective_batch*(args.block_size-1):,} targets/update).", flush=True)
    print(f"Unique training target tokens/parameter: {report['splits']['train']['target_tokens']/parameters:.2f}. "
          "Repeated epochs do not add new language evidence.", flush=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    class SampleCallback(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step % args.sample_steps == 0 and state.is_world_process_zero:
                samples = generate_samples(kwargs["model"], tokenizer, PROMPTS, args.sample_tokens)
                with (out / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    for sample in samples:
                        handle.write(json.dumps({"step": state.global_step, **sample}, ensure_ascii=False) + "\n")
                        print(f"\n[step {state.global_step} | {sample['mode']}] "
                              f"{sample['prompt']}{sample['continuation']}\n", flush=True)

    training_args = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        weight_decay=.1, adam_beta1=.9, adam_beta2=.95, adam_epsilon=1e-8,
        max_grad_norm=1., warmup_ratio=.05, lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": .1}, optim="adamw_torch",
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        logging_steps=min(25, args.eval_steps), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="nll", greater_is_better=False,
        fp16=use_cuda and not bf16, bf16=bf16, use_cpu=args.cpu,
        dataloader_num_workers=0, dataloader_pin_memory=use_cuda,
        seed=args.seed, data_seed=args.seed, report_to="none", save_safetensors=True,
        restore_callback_states_from_checkpoint=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=CausalCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                      preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics,
                      callbacks=[SampleCallback(), EarlyStoppingCallback(early_stopping_patience=args.patience,
                                                                         early_stopping_threshold=.002)])
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train_set)/effective_batch)*args.epochs
    if planned_steps < args.eval_steps:
        raise ValueError("Training would finish before its first evaluation; lower --eval-steps")
    if resume and json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"] >= planned_steps:
        raise ValueError("This run already reached its step budget; use generate/evaluate or start a new run")
    trainer.train(resume_from_checkpoint=resume)
    # Trainer has restored the best VALIDATION checkpoint, not necessarily the last.
    model.config.use_cache = True
    best = out / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(best)
    trainer.save_state()
    trainer.remove_callback(EarlyStoppingCallback)
    test_set = BookWindows(data / "test.jsonl", args.block_size, args.eval_limit)
    metrics = trainer.evaluate(test_set, metric_key_prefix="test")
    summary = {"parameters": parameters, "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_validation_nll": trainer.state.best_metric, "global_step": trainer.state.global_step,
               "dataset_sha256": signature, "smoke_test": args.smoke_test,
               "limited_data": bool(args.train_limit or args.eval_limit), "test": metrics,
               "corpus": report, "run": settings}
    write_json(best / "training_summary.json", summary)
    write_json(out / "test_metrics.json", metrics)
    final_samples = generate_samples(model, tokenizer, PROMPTS, args.sample_tokens)
    write_json(best / "samples.json", final_samples)
    (best / "README.md").write_text(model_card(summary), encoding="utf-8")
    print(f"Best model and tokenizer: {best}\nHeld-out test: {json.dumps(metrics, indent=2)}", flush=True)


def model_card(summary):
    return f"""---
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
- gpt2
- from-scratch
- christmas
---
# Christmasmodel

A {summary['parameters']:,}-parameter GPT-2 language model initialized randomly
and trained on the Gutenberg works listed in
[christmasmodel](https://github.com/wildfellhall/christmasmodel/blob/main/corpus.py).
Its byte-level BPE vocabulary was learned on training books only.

This is a literary continuation model, not an instruction/chat model. Use a
story opening as the prompt. Coherence, novelty, and factual accuracy are not
guaranteed. The limited historical corpus can produce memorized passages,
repetition, dated language, and historical biases. Repeated epochs are not new data.

Best validation NLL: {summary['best_validation_nll']}
Test token perplexity: {summary['test'].get('test_perplexity')}
Smoke test: {summary['smoke_test']}; limited data: {summary['limited_data']}.
Scores use this model's tokenizer and are not directly comparable to other vocabularies.
Overlap-group splits reduce, but cannot guarantee the absence of, shared passages.
Full corpus counts, settings, and evaluation are in training_summary.json.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
path = "./runs/christmas/best"  # or the uploaded Hugging Face repository ID
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
inputs = tokenizer("On Christmas Eve, the snow", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=96, do_sample=True,
                        temperature=0.8, top_p=0.92, top_k=40,
                        repetition_penalty=1.08,
                        pad_token_id=tokenizer.pad_token_id,
                        suppress_tokens=[tokenizer.pad_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
"""


def inference(args):
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
    path = Path(args.model_dir or Path(args.output_dir) / "best")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(path, local_files_only=True)
    if args.command == "generate":
        device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else
                                        "mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        for sample in generate_samples(model, tokenizer, [args.prompt], args.sample_tokens, args.seed):
            print(f"\n[{sample['mode']}] {sample['prompt']}{sample['continuation']}")
        return
    summary = json.loads((path / "training_summary.json").read_text())
    if summary["dataset_sha256"] != dataset_signature(args.data_dir):
        raise ValueError("Evaluation data/tokenizer differ from this model's training run")
    dataset = BookWindows(Path(args.data_dir) / "test.jsonl", model.config.n_positions, args.eval_limit)
    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=str(path.parent / "evaluation"),
                                             per_device_eval_batch_size=args.batch_size,
                                             use_cpu=args.cpu, report_to="none"),
                      data_collator=CausalCollator(tokenizer.pad_token_id),
                      preprocess_logits_for_metrics=loss_sums, compute_metrics=token_metrics)
    print(json.dumps(trainer.evaluate(dataset, metric_key_prefix="test"), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "generate"])
    parser.add_argument("--corpus", default=str(ROOT / "corpus.py"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Require cached Gutenberg downloads")
    parser.add_argument("--min-success-rate", type=float, default=.9)
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "christmas"))
    parser.add_argument("--model-dir", help="Saved model for evaluate/generate; defaults to output-dir/best")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=float, default=12)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional optimizer-step cap")
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=96)
    parser.add_argument("--patience", type=int, default=6, help="Validation checks without a meaningful improvement")
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume latest or a checkpoint directory")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny CPU model, 3 updates, limited data; not quality training")
    parser.add_argument("--train-limit", type=int, help="Diagnostic window cap; omitted for full training")
    parser.add_argument("--eval-limit", type=int, help="Diagnostic window cap; omitted for full evaluation")
    parser.add_argument("--prompt", default=PROMPTS[0])
    args = parser.parse_args(argv)
    if args.heads <= 0 or args.width % args.heads or args.block_size < 8 or args.vocab_size < 258:
        parser.error("width must divide by heads; block-size >= 8; vocab-size >= 258")
    if not 0 < args.min_success_rate <= 1 or not 0 <= args.dropout < 1:
        parser.error("min-success-rate must be in (0,1]; dropout in [0,1)")
    if any(x <= 0 for x in [args.layers, args.heads, args.width, args.batch_size, args.accumulation,
                            args.eval_steps, args.sample_steps, args.sample_tokens, args.patience,
                            args.epochs, args.learning_rate]):
        parser.error("Model sizes, batch sizes, intervals, epochs, and learning rate must be positive")
    if any(x is not None and x <= 0 for x in [args.train_limit, args.eval_limit]):
        parser.error("Diagnostic limits must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 (use epochs) or a positive step budget")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        training(args)
    else:
        inference(args)


if __name__ == "__main__" and "__file__" in globals():
    # Works with python and %run; a pasted notebook cell calls main(["train"]).
    main()
