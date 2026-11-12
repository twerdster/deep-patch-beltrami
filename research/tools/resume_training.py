"""Validate trusted checkpoint provenance before invoking an unchanged trainer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from beltrami_research.artifacts import IMPLEMENTATIONS, validate_resume


def preflight(implementation, arguments):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("resume", "config", "manifest", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    args, _ = parser.parse_known_args(arguments)
    return validate_resume(args.resume, args.config, implementation, args.manifest,
                           args.validation_manifest or args.manifest, args.model_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("trainer_arguments", nargs=argparse.REMAINDER,
                        help="Place -- then the trainer's exact arguments, including --resume")
    args = parser.parse_args()
    forwarded = args.trainer_arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    print(json.dumps({"resume_preflight": preflight(args.implementation, forwarded)}), flush=True)
    return subprocess.call([sys.executable, "-m", args.implementation + ".train", *forwarded])


if __name__ == "__main__":
    sys.exit(main())
