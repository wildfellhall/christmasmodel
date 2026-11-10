"""Freeze source ranges for the story workflow from the existing cached books.

Unresolved anthologies are listed for boundary review, never concatenated into
training documents. Chapter cuts are passages without EOS, not invented endings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import model
from story_data import clean_unit, write_json

# These volumes contain multiple works. Their explicit boundary overrides are
# required; a filename or keyword heuristic is not used to certify story ends.
COLLECTIONS = set('35397 28307 28198 20425 43763 17770 28308 48747 71698 58409 '
    '54803 79846 34571 28098 47333 69416 70291 16052 31103 2731 1467 47897 '
    '10723 20018 34665 71907 65830 27741 43190 70198 72621 26015 34465 14606'.split())


def paragraphs(body):
    return list(re.finditer(r'\S(?:[^\n]|\n(?!\s*\n))*', body))


def prose_start(body, after=0, minimum_words=3):
    for block in paragraphs(body[after:]):
        text = block.group()
        if len(text.split()) >= minimum_words and re.search(r'[.!?][\"”\']?(?:\s|$)', text) and not text.isupper():
            return after+block.start()
    raise ValueError('No narrative paragraph found')


def apparatus_end(body, after=None):
    matches=list(re.finditer(r'(?im)^[ \t]*(?:THE END[.!]?|FINIS[.!]?|FOOTNOTES?[:.]?|'
        r'Transcriber[’\']?s? Notes?\b[^\n]*|ADVERTISEMENTS|End of (?:the )?Project Gutenberg\b[^\n]*)[ \t]*$',body))
    matches=[m for m in matches if m.start()>(len(body)*.5 if after is None else after)]
    return matches[0].start() if matches else len(body)


def document_ranges(body):
    """Conservative prose ranges, split at chapter/section headings without EOS."""
    end=apparatus_end(body)
    contents=list(re.finditer(r'(?im)^\s*(?:TABLE OF )?CONTENTS[.:]?\s*$',body[:min(end,15000)]))
    gate=contents[-1].end() if contents else 0
    chapter=list(re.finditer(r'(?m)^\s*(?:(?:CHAPTER|STAVE|BOOK|PART)\s+(?:[IVXLCDM]+|\d+|ONE)\b[^\n]*|[IVXLCDM]+[.]?)\s*$',body))
    # A contents list may repeat headings. First prose AFTER the contents skips it.
    opening=list(re.finditer(r'(?m)^[ \t]*(?:(?:CHAPTER|STAVE)\s+(?:I|1|ONE)[.:]?(?:[ \t]+[^\n]*)?|I[.]?)[ \t]*$',body[gate:end]))
    # Contents entries have trailing page numbers and are not prose headings.
    opening=[m for m in opening if not re.search(r'\s\d+\s*$',m.group())]
    if opening:
        gate+=opening[0].end()
    start=prose_start(body[:end],gate)
    if not contents and re.search(r'(?im)^\s*(?:PREFACE|INTRODUCTION)\s*$',body[:start]):
        following=[m for m in chapter if start<m.start()<min(end,30000)]
        if following:start=prose_start(body[:end],following[0].end())
    cuts=[(start,start)]
    for match in chapter:
        if start<match.start()<end:
            try:following=prose_start(body[:end],match.end())
            except ValueError:continue
            cuts.append((match.start(),following))
    ranges=[]
    for i,(_,begin) in enumerate(cuts):
        stop=cuts[i+1][0] if i+1<len(cuts) else end
        if stop>begin and len(body[begin:stop].split())>=40:
            ranges.append((begin,stop,False))
    return ranges


def children_christmas_ranges(body):
    """The 35 numbered stories in the 1913 Dickinson/Skinner anthology."""
    headings=list(re.finditer(r'(?m)^([IVXL]+)\n\n([\"A-Z][^\n]+)\n\n',body))
    if len(headings)!=35:
        raise ValueError('Expected 35 reviewed anthology story headings')
    result=[]
    for i,heading in enumerate(headings):
        end=headings[i+1].start() if i+1<len(headings) else apparatus_end(body,heading.end())
        start=prose_start(body[:end],heading.end(),minimum_words=3)
        # Dickens extracts and the historical nonfiction selection are not whole stories.
        complete=i+1 not in {1,20,26,27}
        result.append((start,end,complete,heading[2]))
    return result


def build(data_dir, winter_cache, output, overrides=None, additions=None, offline=True,
          source_catalog='story_source_snapshot.json'):
    data_dir,winter_cache=Path(data_dir),Path(winter_cache)
    sources,units,pending={},[],[]
    overrides=json.loads(Path(overrides).read_text()) if overrides else {}
    snapshot=json.loads(Path(source_catalog).read_text())
    frozen_sources={str(s['id']):s for s in snapshot['sources']}
    original_ids={str(key) for key in snapshot.get('original_source_ids',[])}
    original_ids.update(str(u['source_id']) for u in snapshot.get('units',[]) if u['pool']=='original')
    original_ids.update(str(u['source_id']) for u in snapshot.get('pending_anthologies',[]))
    # Fresh Colab runtimes have neither a legacy manifest nor a pre-existing cache.
    (data_dir/'raw').mkdir(parents=True,exist_ok=True)

    def source_record(book_id,cache,pool,extra=None):
        (Path(cache)/'raw').mkdir(parents=True,exist_ok=True)
        url=f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
        raw=model.download_book(url,cache,offline)
        meta,body=model.clean_book(raw)
        source={'id':book_id,'url':url,**meta,'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                'body_sha256':hashlib.sha256(body.encode()).hexdigest(),**(extra or {})}
        sources[book_id]=source
        if book_id in frozen_sources and source['raw_sha256']!=frozen_sources[book_id]['raw_sha256']:
            if source['body_sha256']!=frozen_sources[book_id]['body_sha256']:
                raise ValueError(f'Source edition changed: {book_id}; review before updating its frozen hash')
            source['boilerplate_only_update_from_raw_sha256']=frozen_sources[book_id]['raw_sha256']
        # Consolidate the immutable downloads for one offline preparation cache.
        target=data_dir/'raw'/f'{book_id}.txt'
        if not target.exists():target.write_text(raw)
        return source,body

    def add(book_id,body,ranges,pool,method):
        for i,item in enumerate(ranges):
            start,end,complete,*title=item
            while end>start and body[end-1].isspace():end-=1
            while start<end and body[start].isspace():start+=1
            if len(clean_unit(body[start:end]).split())<40:continue
            units.append({'id':f'{pool}-{book_id}-{i+1}','source_id':book_id,'group':book_id,
                'pool':pool,'start':start,'end':end,'sha256':hashlib.sha256(body[start:end].encode()).hexdigest(),
                'kind':'story' if complete else 'passage','complete':complete,'status':'accepted',
                'title':title[0] if title else sources[book_id]['title'], 'boundary_method':method})

    for book_id in sorted(original_ids):
        source,body=source_record(book_id,data_dir,'original')
        if book_id in overrides:
            entry=overrides[book_id]
            if entry['body_sha256']!=source['body_sha256']:raise ValueError('Boundary override edition changed')
            ranges=[(r['start'],r['end'],r.get('complete',False),r['title']) for r in entry['ranges']]
            add(book_id,body,ranges,'original','reviewed_exact_ranges')
        elif book_id=='28308':
            add(book_id,body,children_christmas_ranges(body),'original','reviewed_numbered_anthology_35')
        elif book_id in COLLECTIONS:
            pending.append({'source_id':book_id,'title':source['title'],'body_sha256':source['body_sha256'],
                            'reason':'Needs explicit constituent-story boundaries; omitted from training'})
        else:
            try:ranges=document_ranges(body)
            except ValueError as exc:raise ValueError(f'{book_id}: {exc}') from exc
            add(book_id,body,ranges,'original','conservative_chapter_passages_no_eos')
    for row in model.WINTER_SOURCES:
        book_id=str(row['id'])
        source,body=source_record(book_id,winter_cache,'winter')
        ranges=[]
        for spec in row['selections']:
            _,start,end=model.select_winter_text(body,spec)
            start=prose_start(body[:end],start)
            ranges.append((start,end,spec['natural_end'],spec.get('title',source['title'])))
        add(book_id,body,ranges,'winter','existing_reviewed_winter_selection')
    for row in json.loads(Path(additions).read_text()) if additions else []:
        book_id=str(row['id'])
        source,body=source_record(book_id,data_dir,'historical',{'new_addition':True,
            'year':row['year'],'date_evidence':row['date_evidence']})
        if row.get('anthology'):
            ranges=[]
            for spec in row['selections']:
                start=body.index(spec['start'])
                end=body.index(spec['end'],start+len(spec['start'])) if spec.get('end') else apparatus_end(body)
                start=prose_start(body[:end],start+len(spec['start']))
                ranges.append((start,end,True,spec['title']))
        elif row.get('opening'):
            matches=list(re.finditer(re.escape(row['opening']),body))
            if len(matches)!=1:
                raise ValueError(f'Historical opening is not unique: {book_id}')
            begin=matches[0].start()
            # These reviewed novels have no publisher advertisements. "The End"
            # can be a chapter title (Curdie), so it must not truncate the plot.
            end=len(body)
            if row.get('ending_before'):
                end=min(end,body.index(row['ending_before'],begin))
            ranges=[(begin,end,True)]
        else:
            raise ValueError('New novels need an explicit reviewed opening; anthologies need story boundaries')
        add(book_id,body,ranges,'historical','reviewed_historical_addition')
    used={u['source_id'] for u in units}
    catalog={'schema':1,'cutoff_inclusive':1920,'sources':[sources[k] for k in sorted(used)],
             'units':units,'pending_anthologies':pending,
             'note':'Original material is retained through 1920. Provisional chapter passages have no EOS; '
                    'unresolved multi-story volumes are excluded until exact boundaries are supplied.'}
    write_json(output,catalog)
    print(json.dumps({'sources':len(used),'units':len(units),'complete_stories':sum(u['complete'] for u in units),
                      'pending_anthologies':len(pending),'catalog':str(output)},indent=2))
    return catalog


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--winter-cache',default='winter-data')
    parser.add_argument('--output',default='story_catalog.json')
    parser.add_argument('--source-catalog',default='story_source_snapshot.json')
    parser.add_argument('--overrides',default='story_boundaries.json')
    parser.add_argument('--additions',default='historical_additions.json')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    build(args.data_dir,args.winter_cache,args.output,args.overrides,args.additions,
          not args.download,args.source_catalog)
