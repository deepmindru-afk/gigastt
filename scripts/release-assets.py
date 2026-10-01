#!/usr/bin/env python3
"""List required release assets, adding signatures only when signing is enabled."""
import argparse
from pathlib import Path
import sys


def assets(directory, signed):
    files = []
    for pattern in ('*.tar.gz', '*.sha256', '*.deb', 'SHA256SUMS.txt', '*.cdx.json'):
        matches = sorted(path for path in directory.glob(pattern) if path.is_file())
        if not matches:
            raise ValueError(f'missing required release artifact: {pattern}')
        files.extend(matches)
    if signed:
        signatures = [Path(str(path) + '.minisig') for path in files
                      if not path.name.endswith('.sha256')]
        for signature in signatures:
            if not signature.is_file():
                raise ValueError(f'missing required signature: {signature.name}')
        files.extend(signatures)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--signed', action='store_true')
    mode.add_argument('--unsigned', action='store_true')
    args = parser.parse_args()
    print('\n'.join(str(path) for path in assets(args.directory, args.signed)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f'Release asset validation failed: {error}')
