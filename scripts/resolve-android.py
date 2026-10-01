#!/usr/bin/env python3
"""Select Android source using the same tag/version rules as desktop releases."""
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

release = runpy.run_path(str(Path(__file__).with_name('resolve-release.py')))


def main():
    tag = os.environ.get('RELEASE_TAG', '')
    if tag:
        source = release['resolve'](tag)
    else:
        if os.environ.get('ANDROID_PUBLISH') == 'true':
            raise ValueError('Maven publication requires an explicit release tag')
        commit = os.environ['GITHUB_SHA']
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('untagged builds require the dispatch commit SHA')
        commit = release['git']('rev-parse', '--verify', commit + '^{commit}')
        source = {'tag': '', 'commit': commit, 'version': release['workspace_version'](commit)}
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        for key, value in source.items():
            output.write(f'{key}={value}\n')
    print(f"Android source: {source['commit']} (version {source['version']})")


if __name__ == '__main__':
    try:
        main()
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f'Android source validation failed: {error}')
