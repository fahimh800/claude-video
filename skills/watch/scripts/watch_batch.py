#!/usr/bin/env python3
"""Watch several videos one at a time, spaced out to avoid rate limits.

Each video gets its own report file (the normal watch.py report). An index of
all reports is printed to stdout at the end. Channel or playlist URLs can be
expanded to their newest videos with --expand.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from download import is_url  # noqa: E402
from runtime import configure_stdio, diagnostic, run_text  # noqa: E402

DEFAULT_GAP = 30.0


def expand(url: str, limit: int) -> list[str]:
    """List up to `limit` video URLs from a channel/playlist without downloading anything."""
    result = run_text(['yt-dlp', '--flat-playlist', '--print', 'url', '-I', f'1:{limit}', '--', url])
    if result.returncode != 0:
        raise SystemExit(f'Could not list videos for {url}: {diagnostic(result.stderr, 400)}')
    return [line.strip() for line in result.stdout.splitlines() if is_url(line.strip())]


def read_list(path: str) -> list[str]:
    lines = Path(path).expanduser().read_text(encoding='utf-8-sig').splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')]


def report_field(text: str, name: str) -> str:
    match = re.search(rf'^- \*\*{re.escape(name)}:\*\* (.+)$', text, re.M)
    return match.group(1).strip() if match else ''


def main() -> int:
    ap = argparse.ArgumentParser(prog='watch_batch', description=__doc__.splitlines()[0])
    ap.add_argument('sources', nargs='*', help='Video URLs or local paths')
    ap.add_argument('--list', dest='list_file', help='Text file with one URL or path per line (# comments allowed)')
    ap.add_argument('--expand', type=int, default=0, metavar='N',
                    help='Treat channel/playlist URLs as lists and take their first N videos')
    ap.add_argument('--question', default=None, help='Question passed to every run (used by Gemini)')
    ap.add_argument('--gap', type=float, default=DEFAULT_GAP, help=f'Seconds between videos (default {DEFAULT_GAP:g})')
    ap.add_argument('--out-dir', required=True, help='Folder for reports and per-run work directories')
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]
    if args.gap < 0 or args.expand < 0:
        ap.error('--gap and --expand must not be negative')

    sources = list(args.sources) + (read_list(args.list_file) if args.list_file else [])
    if args.expand:
        expanded = []
        for source in sources:
            expanded += expand(source, args.expand) if is_url(source) else [source]
        sources = expanded
    sources = list(dict.fromkeys(sources))
    if not sources:
        ap.error('no sources given')

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, source in enumerate(sources, 1):
        if number > 1 and args.gap:
            print(f'[watch-batch] waiting {args.gap:g}s before the next video…', file=sys.stderr)
            time.sleep(args.gap)
        print(f'[watch-batch] {number}/{len(sources)}: {source}', file=sys.stderr)
        report = out / f'report-{number:02d}.md'
        cmd = [sys.executable, str(SCRIPT_DIR / 'watch.py'), source, '--out-dir', str(out), *passthrough]
        if args.question:
            cmd += ['--question', args.question]
        with report.open('w', encoding='utf-8') as stream:
            code = subprocess.run(cmd, stdout=stream, encoding='utf-8', errors='replace').returncode
        text = report.read_text(encoding='utf-8', errors='replace')
        rows.append((number, source, report_field(text, 'Title'), report_field(text, 'Engine'), code, report))

    print('# watch batch: index')
    print()
    print('Read each report file below; each one lists its own frames and transcript.')
    print()
    print('| # | Source | Title | Engine | Status | Report |')
    print('|---|---|---|---|---|---|')
    for number, source, title, engine, code, report in rows:
        status = 'ok' if code == 0 else f'failed (exit {code})'
        print(f'| {number} | {source} | {title or "—"} | {engine or "local"} | {status} | `{report}` |')
    return 0 if any(row[4] == 0 for row in rows) else 1


if __name__ == '__main__':
    configure_stdio()
    raise SystemExit(main())
