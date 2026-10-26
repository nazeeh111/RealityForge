"""Reacquire the small licensed evaluation subset. Requires FFmpeg and network access.

Run from the repository root. No API token or paid service is needed.
Existing files are skipped, never overwritten. Different FFmpeg versions can
produce different encoded bytes; the checked-in manifest hashes identify the
exact delivered files, not a promise of cross-version bitwise transcoding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Separate destination; original samples stay intact')
    parser.add_argument('--include-noncommercial-iris', action='store_true', help='Include IRIS CC BY-NC 4.0 recordings')
    args = parser.parse_args()
    if not shutil.which('ffmpeg'):
        parser.error('FFmpeg is required and was not found on PATH')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'samples/provenance.json').read_text())
    for record in manifest['records']:
        if record['dataset'] == 'IRIS' and not args.include_noncommercial_iris:
            continue
        destination = args.output / record['path']
        if destination.exists():
            print(f'Skip existing {destination}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning',
            '-ss', str(record['source_start_seconds']), '-i', record['source_url'],
            '-t', str(record['requested_duration_seconds']), '-vf', 'scale=960:-2',
            '-r', str(record['output_fps']), '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '20', str(destination),
        ]
        subprocess.run(command, check=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(f'{destination}: sha256 {digest}; {record["license"]}')
    (args.output / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
