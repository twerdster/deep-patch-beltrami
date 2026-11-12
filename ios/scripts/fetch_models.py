#!/usr/bin/env python3
"""Fetch verified portable Core ML models and compile them for the iOS app."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(record, destination):
    destination = Path(destination)
    size = record.get('size', record.get('bytes'))
    if destination.is_file() and destination.stat().st_size == size and sha256(destination) == record['sha256']:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    try:
        request = urllib.request.Request(record['url'], headers={'User-Agent': 'deep-patch-beltrami-ios/1.1'})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        if temporary.stat().st_size != size or sha256(temporary) != record['sha256']:
            raise ValueError(f"Asset size/SHA256 mismatch: {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract(archive_path, destination):
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError(f'Unexpected archive path: {member.name}')
            target = destination.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError(f'Unexpected archive entry: {member.name}')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigmas', type=int, nargs='+', default=[5,10,15,20,25,30,40,50],
                        help='Noise levels to install (default: all eight)')
    parser.add_argument('--output', type=Path, default=root/'ios/DeepBeltramiApp/ModelAssets')
    parser.add_argument('--cache-dir', type=Path, default=Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'deep-patch-beltrami/coreml/v1.1.0')
    parser.add_argument('--archive-dir', type=Path, help='Use local verified release archives instead of downloading')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    manifest_path = root/'ios/models.json'
    if not manifest_path.is_file():
        parser.error('Missing ios/models.json release manifest')
    manifest = json.loads(manifest_path.read_text())
    selected = [(name, record) for name, record in manifest['assets'].items() if record['sigma'] in args.sigmas]
    missing = set(args.sigmas)-{record['sigma'] for _, record in selected}
    if missing:
        parser.error(f'No released model for sigma {sorted(missing)}')
    args.output.mkdir(parents=True, exist_ok=True)
    staging = root/'.build-ios/coreml-staging'
    staging.mkdir(parents=True, exist_ok=True)
    installed = []
    for name, record in selected:
        if args.archive_dir:
            archive_path = args.archive_dir/name
            if not archive_path.is_file() or sha256(archive_path) != record['sha256'] or archive_path.stat().st_size != record.get('size', record.get('bytes')):
                raise ValueError(f'Local release archive does not match manifest: {name}')
        else:
            print(f"Fetching sigma {record['sigma']}: {name}", flush=True)
            archive_path = fetch(record, args.cache_dir/name)
        if args.download_only:
            continue
        with tempfile.TemporaryDirectory(prefix='compile-', dir=staging) as temporary:
            temporary = Path(temporary)
            extract(archive_path, temporary)
            packages = sorted(temporary.glob('models/*.mlpackage'))
            if len(packages) != 3:
                raise ValueError(f'Expected three model packages in {name}, found {len(packages)}')
            compiled = temporary/'compiled'
            compiled.mkdir()
            for package in packages:
                subprocess.run(['xcrun','coremlcompiler','compile',str(package),str(compiled)], check=True)
                output = compiled/(package.stem+'.mlmodelc')
                if not output.is_dir():
                    raise ValueError(f'Core ML compiler did not produce {output.name}')
                target = args.output/output.name
                # Complete the copy on the destination volume before replacing
                # an installed model; retain it if copying or replacement fails.
                with tempfile.TemporaryDirectory(prefix='.install-', dir=args.output) as installation:
                    installation = Path(installation)
                    staged = installation/output.name
                    backup = installation/'previous'
                    shutil.copytree(output, staged)
                    if target.exists():
                        target.rename(backup)
                    try:
                        staged.rename(target)
                    except OSError:
                        if backup.exists():
                            backup.rename(target)
                        raise
                installed.append(target.name)
                print(f'Installed {target.name}', flush=True)
            license_file = temporary/'LICENSE.txt'
            if license_file.is_file():
                shutil.copyfile(license_file, args.output/'LICENSE.txt')
    print(json.dumps({'verified_sigmas':args.sigmas,'compiled_models':installed,'output':str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
