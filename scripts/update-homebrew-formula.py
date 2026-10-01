#!/usr/bin/env python3
"""Pin a formula only after validating all release checksum entries."""
import argparse
from pathlib import Path
import re

TARGETS = ("aarch64-apple-darwin", "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")
TAG = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?")


def rewrite(formula, tag, sums):
    if not TAG.fullmatch(tag):
        raise ValueError("Expected a release version tag")
    version = tag[1:]
    checksums = {}
    for target in TARGETS:
        asset = f"gigastt-{version}-{target}.tar.gz"
        matches = [line.split() for line in sums.splitlines() if line.split() and line.split()[-1].lstrip("*") == asset]
        if len(matches) != 1 or len(matches[0]) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", matches[0][0]):
            raise ValueError(f"Expected exactly one valid SHA256 for {asset}")
        checksums[target] = matches[0][0].lower()

    updated, count = re.subn(r'^  version "[^"]+"$', f'  version "{version}"', formula, flags=re.M)
    if count != 1:
        raise ValueError("Expected exactly one formula version")
    for target in TARGETS:
        pattern = (r'(url "https://github\.com/ekhodzitsky/gigastt/releases/download/)'
                   r'[^/]+/gigastt-[^/"\s]+-' + re.escape(target) +
                   r'\.tar\.gz"(\s+)sha256 "[0-9a-fA-F]{64}"')
        updated, count = re.subn(pattern, lambda m: f'{m[1]}{tag}/gigastt-{version}-{target}.tar.gz"{m[2]}sha256 "{checksums[target]}"', updated)
        if count != 1:
            raise ValueError(f"Expected exactly one formula pin for {target}")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("checksums", type=Path)
    parser.add_argument("--formula", type=Path, default=Path("Formula/gigastt.rb"))
    args = parser.parse_args()
    updated = rewrite(args.formula.read_text(), args.tag, args.checksums.read_text())
    args.formula.write_text(updated)


if __name__ == "__main__":
    main()
