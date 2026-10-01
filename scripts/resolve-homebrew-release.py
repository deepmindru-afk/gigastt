#!/usr/bin/env python3
"""Validate data from one successful Release run before proposing formula pins."""
import argparse
import json
from pathlib import Path
import re
import sys


def validate_tag(tag):
    number = r'(?:0|[1-9][0-9]*)'
    if not isinstance(tag, str) or not re.fullmatch(rf'v{number}\.{number}\.{number}(?:-[0-9A-Za-z.-]+)?', tag):
        raise ValueError('expected a release version tag')
    if '-' in tag:
        for part in tag.split('-', 1)[1].split('.'):
            if not part or (part.isdigit() and len(part) > 1 and part[0] == '0'):
                raise ValueError('invalid prerelease identifier')
    return tag


def resolve(metadata, event, repository):
    run = event.get('workflow_run', {})
    if (run.get('conclusion') != 'success' or run.get('name') != 'Release'
            or run.get('path') != '.github/workflows/release.yml'
            or run.get('event') not in ('push', 'workflow_dispatch')
            or run.get('head_repository', {}).get('full_name') != repository):
        raise ValueError('expected a successful Release run from this repository')
    if (type(metadata.get('schema')) is not int or metadata['schema'] != 1
            or metadata.get('repository') != repository):
        raise ValueError('invalid release metadata schema or repository')
    for key, event_key in [('run_id', 'id'), ('run_attempt', 'run_attempt')]:
        value = metadata.get(key)
        if type(value) is not int or value < 1 or value != run.get(event_key):
            raise ValueError('release metadata does not belong to this run and attempt')
    commit = metadata.get('commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('invalid immutable release commit')
    return validate_tag(metadata.get('tag'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tag', help='explicit manual Homebrew selection')
    source.add_argument('--metadata', type=Path)
    parser.add_argument('--event', type=Path)
    parser.add_argument('--repository')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.tag is not None:
        tag = validate_tag(args.tag)
    else:
        if args.event is None or not args.repository:
            raise ValueError('event and repository required for release metadata')
        # Data only: no imports, evaluation or subprocesses from the artifact.
        if args.metadata.stat().st_size > 4096:
            raise ValueError('release metadata is too large')
        metadata = json.loads(args.metadata.read_text())
        event = json.loads(args.event.read_text())
        if not isinstance(metadata, dict) or not isinstance(event, dict):
            raise ValueError('expected JSON objects')
        tag = resolve(metadata, event, args.repository)
    with args.output.open('a') as output:
        output.write(f'tag={tag}\n')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f'Homebrew release selection failed: {error}')
