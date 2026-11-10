"""Versioned, source-only narrative material and token-budgeted sampling.

This workflow is independent of legacy data/checkpoints. Text never comes from
the distillation files. A catalog names exact ranges of hash-checked books.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = 1
SPLITS = ('train', 'validation', 'test')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def clean_unit(text):
    """Remove explicit apparatus only. Keep dialogue, spelling and paragraph breaks."""
    text = re.sub(r'\[(?:Pg\.?|Page)\s*\d+\]', '', text, flags=re.I)
    text = re.sub(r'\[(?:Illustration|Frontispiece|Footnote)[^\]]*\]', '', text, flags=re.I | re.S)
    text = re.sub(r'(?m)^\s*\[[A-Z0-9]+\]:[^\n]*(?:\n(?!\s*\n)[^\n]*)*', '', text)
    text = re.split(r'(?im)^[ \t]*(?:Transcriber[’\']?s? Notes?\b|End of (?:the )?Project Gutenberg\b).*$', text)[0]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'(?m)^[ \t]*(?:\*[ \t]*){3,}$', '', text)
    text = re.sub(r'\n(?:[ \t]*\n){2,}', '\n\n', text)
    return text.strip()


def load_units(catalog_path, cache, offline=False):
    """Verify source and selection hashes BEFORE cleaning; no guessed replacements."""
    import model as legacy
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog.get('source_snapshot_only'):
        raise ValueError('Build the reviewed catalog first with build_story_catalog.py')
    if catalog.get('schema') != SCHEMA:
        raise ValueError('Unknown story catalog schema')
    sources, units, seen = {}, [], set()
    for source in catalog['sources']:
        key = str(source['id'])
        if key in sources:
            raise ValueError('Duplicate source ID')
        # Existing approved material includes 1920. New additions need dated evidence.
        if source.get('new_addition') and (not isinstance(source.get('year'), int)
                or source['year'] > 1920 or not source.get('date_evidence')):
            raise ValueError('New additions require publication evidence through 1920')
        if not re.fullmatch(r'https://www\.gutenberg\.org/cache/epub/(\d+)/pg\1\.txt', source['url']):
            raise ValueError('Catalog must reference a Gutenberg text URL')
        raw = legacy.download_book(source['url'], cache, offline)
        if hashlib.sha256(raw.encode()).hexdigest() != source['raw_sha256']:
            raise ValueError(f'Source edition changed: {key}')
        _, body = legacy.clean_book(raw)
        if hashlib.sha256(body.encode()).hexdigest() != source['body_sha256']:
            raise ValueError(f'Source cleanup changed: {key}')
        sources[key] = (source, body)
    for spec in catalog['units']:
        if spec['id'] in seen:
            raise ValueError('Duplicate narrative unit ID')
        seen.add(spec['id'])
        source, body = sources[str(spec['source_id'])]
        start, end = spec['start'], spec['end']
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(body)):
            raise ValueError(f'Invalid boundaries: {spec["id"]}')
        selected = body[start:end]
        if hashlib.sha256(selected.encode()).hexdigest() != spec['sha256']:
            raise ValueError(f'Changed narrative boundaries: {spec["id"]}')
        if spec.get('status') != 'accepted':
            continue
        if spec.get('complete') and spec.get('kind') != 'story':
            raise ValueError('Only a reviewed story ending may receive EOS')
        text = clean_unit(selected)
        if len(text.split()) < 40 or re.search(r'Project Gutenberg|E-text prepared|Transcriber[’\']?s? Notes?', text, re.I):
            raise ValueError(f'Invalid narrative text: {spec["id"]}')
        units.append({**spec, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'author': spec.get('author', source.get('author', 'Unknown')),
                      'source_url': source['url'], 'words': len(text.split()),
                      'group': str(spec.get('group', source['id']))})
    if not units:
        raise ValueError('Catalog contains no accepted narrative units')
    return catalog, units


def partition_units(units, seed=42):
    """Deduplicate story reprints and group overlapping works before any split.

    Source volumes stay together, including all ranges from a supplementary
    book. An overlap joins groups even when one duplicate is subsequently removed.
    """
    from model import fingerprints
    parents = list(range(len(units)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        parents[find(a)] = find(b)

    source_groups, signatures, normalized = {}, [], []
    for i, row in enumerate(units):
        if row['group'] in source_groups:
            join(i, source_groups[row['group']])
        source_groups[row['group']] = i
        normalized.append(' '.join(re.findall(r'\w+', row['text'].casefold())))
        signatures.append(fingerprints(row['text']))
    duplicates, excluded = [], set()
    for i in range(len(units)):
        for j in range(i):
            shared = len(signatures[i] & signatures[j])
            containment = shared / max(1, min(len(signatures[i]), len(signatures[j])))
            exact = normalized[i] == normalized[j]
            if exact or (shared >= 12 and containment >= .05):
                join(i, j)
            similarity = shared / max(1, max(len(signatures[i]), len(signatures[j])))
            if i not in excluded and j not in excluded and (exact or (shared >= 12 and similarity >= .85)):
                excluded.add(i)
                duplicates.append({'id': units[i]['id'], 'duplicate_of': units[j]['id']})
                # Continue joining overlaps with other works, including an omitted
                # duplicate, so a reprint cannot bridge train and validation.
    groups = defaultdict(list)
    for i, row in enumerate(units):
        if i not in excluded:
            groups[find(i)].append(row)
    grouped = list(groups.values())
    if len(grouped) < 3:
        raise ValueError('At least three independent work groups are required')
    # Search whole-group assignments, balancing each pool as well as total words.
    totals = Counter()
    for group in grouped:
        for row in group:
            totals[row['pool']] += row['words']
    rng, best = random.Random(seed), None
    for _ in range(1000):
        order = list(range(len(grouped)))
        rng.shuffle(order)
        assigned = {'train': [], 'validation': [], 'test': []}
        sizes = {key: Counter() for key in SPLITS}
        remaining_pool = Counter(p for group in grouped for p in {r['pool'] for r in group})
        for position, index in enumerate(order):
            amount = Counter()
            for row in grouped[index]:
                amount[row['pool']] += row['words']
            choices = []
            required = Counter()
            for pool in amount:
                missing = [s for s in SPLITS if not sizes[s][pool]]
                if remaining_pool[pool] <= len(missing):
                    required.update(missing)
            empty = [s for s in SPLITS if not assigned[s]]
            if len(order)-position <= len(empty):
                required.update(empty)
            for split, fraction in [('validation', .05), ('test', .05), ('train', .9)]:
                delta = sum(((sizes[split][p] + amount[p]) / total - fraction)**2
                            - (sizes[split][p] / total - fraction)**2 for p, total in totals.items())
                choices.append((-required[split], delta, split))
            split = min(choices)[2]
            assigned[split].extend(grouped[index])
            sizes[split].update(amount)
            remaining_pool.subtract(amount.keys())
        if any(not assigned[s] for s in SPLITS):
            continue
        score = sum((sizes[s][p] / total - f)**2 for s, f in
                    [('train', .9), ('validation', .05), ('test', .05)] for p, total in totals.items())
        # Prefer coverage of every pool where at least three source groups exist.
        for pool in totals:
            eligible = sum(any(r['pool'] == pool for r in g) for g in grouped)
            if eligible >= 3:
                score += sum(not sizes[s][pool] for s in SPLITS)
        if best is None or score < best[0]:
            best = score, assigned
    if best is None:
        raise ValueError('Unable to make nonempty work-level splits')
    return best[1], duplicates


def prepare_corpus(catalog_path, output, cache, seed=42, offline=False):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new corpus directory; prepared splits are immutable')
    catalog, units = load_units(catalog_path, cache, offline)
    splits, duplicates = partition_units(units, seed)
    out.mkdir(parents=True, exist_ok=True)
    hashes, counts = {}, {}
    for split, rows in splits.items():
        write_jsonl(out / f'{split}.jsonl', rows)
        hashes[split] = file_hash(out / f'{split}.jsonl')
        counts[split] = {pool: {'units': sum(r['pool'] == pool for r in rows),
                               'words': sum(r['words'] for r in rows if r['pool'] == pool)}
                         for pool in sorted({r['pool'] for r in rows})}
    manifest = {'schema': SCHEMA, 'catalog_sha256': digest(catalog), 'seed': seed,
                'files': hashes, 'counts': counts, 'duplicates': duplicates,
                'historical_cutoff_inclusive': 1920,
                'lineage': 'new_initializations_only',
                'legacy_holdout_warning': 'New splits are unseen only for new scratch/pretrained branches. '
                'Existing project checkpoints trained on some supplementary holdouts.',
                'test_evaluated': False}
    write_json(out / 'corpus.json', manifest)
    return manifest


def verify_corpus(directory):
    root = Path(directory)
    manifest = json.loads((root / 'corpus.json').read_text())
    for split, expected in manifest['files'].items():
        if file_hash(root / f'{split}.jsonl') != expected:
            raise ValueError(f'Frozen {split} text changed')
    return manifest


def make_tokenizer(corpus, output, model_id=None, revision=None, vocab_size=8192, offline=False):
    from transformers import AutoTokenizer, GPT2TokenizerFast
    if model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=offline)
        if tokenizer.eos_token_id is None:
            raise ValueError('Base tokenizer must define EOS')
        # Never suppress EOS by accidentally using it as PAD during generation.
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({'pad_token': '<|christmas_pad|>'})
    else:
        from tokenizers import ByteLevelBPETokenizer
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator((r['text'] for r in read_jsonl(Path(corpus) / 'train.jsonl')),
                                vocab_size=vocab_size, min_frequency=3,
                                special_tokens=['<|endoftext|>', '<|pad|>'])
        Path(output).mkdir(parents=True, exist_ok=True)
        bpe.save_model(str(output))
        tokenizer = GPT2TokenizerFast(vocab_file=str(Path(output) / 'vocab.json'),
            merges_file=str(Path(output) / 'merges.txt'), eos_token='<|endoftext|>',
            bos_token='<|endoftext|>', unk_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 1_000_000
    tokenizer.save_pretrained(output)
    return tokenizer


class StoryWindows:
    """Paragraph-aligned target spans with preceding context, never crossing a unit."""
    def __init__(self, rows, tokenizer, context=512, lookback=128, stride=None):
        if not 0 <= lookback <= context - 2:
            raise ValueError('Lookback must leave causal targets')
        self.context, self.lookback = context, lookback
        self.rows, self.books, self.windows = rows, [], []
        self.book_ids = [r['id'] for r in rows]
        self.random_offsets, self.epoch = False, 0
        capacity = context - lookback - 1
        for index, row in enumerate(rows):
            encoded = tokenizer(row['text'], add_special_tokens=False, return_offsets_mapping=True)
            tokens = encoded['input_ids']
            # EOS conditions the opening; a final EOS is reserved for complete stories.
            ids = [tokenizer.eos_token_id] + tokens
            if row.get('complete'):
                ids.append(tokenizer.eos_token_id)
            self.books.append(ids)
            boundaries = {1 + j for j, (start, _) in enumerate(encoded['offset_mapping'])
                          if start > 0 and '\n\n' in row['text'][max(0, start-3):start]}
            start = 1
            while start < len(ids):
                stop = min(len(ids), start + (stride or capacity))
                possible = [b for b in boundaries if start + capacity // 2 <= b <= stop]
                if possible and stop < len(ids):
                    stop = max(possible)
                self.windows.append((index, start, stop))
                start = stop
        if not self.windows:
            raise ValueError('Empty narrative dataset')

    def __len__(self):
        return len(self.windows)

    def set_training_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        book, start, end = self.windows[index]
        begin = max(0, start - 1 - self.lookback)
        ids = self.books[book][begin:end]
        labels = [-100] * (start - begin) + ids[start - begin:]
        return {'input_ids': ids, 'labels': labels}

    def targets(self, index):
        _, start, end = self.windows[index]
        return end - start


class TokenMixture:
    """Sample pools by target-token share, and works by a configurable balance.

    An immutable epoch schedule makes target exposure inspectable and exact on
    resume. No microbatch parity determines preference-example selection.
    """
    def __init__(self, dataset, shares=None, epoch_tokens=None, seed=42, author_balance=0.):
        self.base, self.seed = dataset, seed
        if not 0 <= author_balance <= 1:
            raise ValueError('author_balance must be in [0, 1]')
        self.random_offsets, self.author_balance = False, author_balance
        self.natural_epoch = shares is None and epoch_tokens is None and author_balance == 0
        self.pools = defaultdict(list)
        for i, (book, _, _) in enumerate(dataset.windows):
            self.pools[dataset.rows[book]['pool']].append(i)
        totals = {p: sum(dataset.targets(i) for i in indices) for p, indices in self.pools.items()}
        self.epoch_tokens = epoch_tokens or sum(totals.values())
        self.shares = shares if shares is not None else {p: n / sum(totals.values()) for p, n in totals.items()}
        if set(self.shares) != set(self.pools) or any(v <= 0 for v in self.shares.values()):
            raise ValueError('Specify a positive share for every training pool')
        mass = sum(self.shares.values())
        self.shares = {p: v / mass for p, v in self.shares.items()}
        # Fixed row count gives Trainer stable epoch/update and resumption semantics.
        mean = 1 / sum(self.shares[p] / (totals[p] / len(self.pools[p])) for p in self.pools)
        self.size = len(dataset) if self.natural_epoch else max(1, math.ceil(self.epoch_tokens / mean))
        self.set_training_epoch(0)

    def set_training_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch)
        if self.natural_epoch:
            # With no requested reweighting, visit every target window once.
            self.order=list(range(len(self.base)))
            rng.shuffle(self.order)
            self.exposure={p:sum(self.base.targets(i) for i in indices) for p,indices in self.pools.items()}
            return
        orders, positions = {}, Counter()
        for pool, indices in self.pools.items():
            orders[pool] = indices.copy()
            rng.shuffle(orders[pool])
        spent = Counter()
        self.order = []
        authors = {p: defaultdict(list) for p in self.pools}
        for pool, indices in self.pools.items():
            for i in indices:
                book = self.base.windows[i][0]
                authors[pool][self.base.rows[book].get('author', 'Unknown')].append(i)
        for _ in range(self.size):
            pool = min(self.shares, key=lambda p: spent[p] / self.shares[p])
            if rng.random() < self.author_balance:
                author = rng.choice(sorted(authors[pool]))
                i = rng.choice(authors[pool][author])
            else:
                if positions[pool] == len(orders[pool]):
                    rng.shuffle(orders[pool])
                    positions[pool] = 0
                i = orders[pool][positions[pool]]
                positions[pool] += 1
            self.order.append(i)
            spent[pool] += self.base.targets(i)
        self.exposure = dict(spent)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        window=self.order[index]
        book=self.base.windows[window][0]
        return {**self.base[window], 'pool':self.base.rows[book]['pool']}
