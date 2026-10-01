#!/usr/bin/env python3
"""Check the Android runtime pin against resolved Rust API features and AAR."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ABIS = ('arm64-v8a', 'armeabi-v7a', 'x86_64')


def check(metadata, pin, aar=None):
    ids = [p['id'] for p in metadata['packages'] if p['name'] == 'ort-sys']
    if len(ids) != 1:
        raise ValueError('expected exactly one ort-sys dependency')
    nodes = [n for n in metadata['resolve']['nodes'] if n['id'] == ids[0]]
    if len(nodes) != 1:
        raise ValueError('missing resolved ort-sys features')
    # API 17 is the crate baseline; each api-N feature includes earlier APIs.
    requested = max([17] + [int(f[4:]) for f in nodes[0]['features'] if re.fullmatch(r'api-\d+', f)])
    version = re.fullmatch(r'1\.(\d+)\.\d+', pin['version'])
    if not version or not re.fullmatch('[0-9a-f]{64}', pin['sha256']):
        raise ValueError('invalid Android runtime version/checksum pin')
    provided = int(version[1])
    if requested > provided:
        raise ValueError(f'Rust requires API {requested}, Android runtime provides API {provided}')
    if aar is not None:
        with Path(aar).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != pin['sha256']:
            raise ValueError('Android runtime archive checksum mismatch')
        with zipfile.ZipFile(aar) as archive:
            header = archive.read('headers/onnxruntime_c_api.h').decode()
            api = re.search(r'^#define ORT_API_VERSION (\d+)\s*$', header, re.MULTILINE)
            if not api or int(api[1]) != provided:
                raise ValueError('Android runtime header API disagrees with pin')
            for abi in ABIS:
                name = f'jni/{abi}/libonnxruntime.so'
                if name not in archive.namelist():
                    raise ValueError(f'Android runtime archive missing {abi}')
                with archive.open(name) as library:
                    if library.read(4) != b'\x7fELF':
                        raise ValueError(f'Android runtime library for {abi} is not ELF')
    print(f'Android runtime {pin["version"]} supports resolved Rust API {requested}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--pin', type=Path, default=Path('packaging/android/onnxruntime.json'))
    parser.add_argument('--aar', type=Path)
    args = parser.parse_args()
    check(json.loads(args.metadata.read_text()), json.loads(args.pin.read_text()), args.aar)


if __name__ == '__main__':
    try:
        main()
    except (KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
        sys.exit(f'Android runtime validation failed: {error}')
