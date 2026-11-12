"""Strip optimizer, RNG state and local metadata from a trusted training checkpoint."""
import argparse
import json
from pathlib import Path

from beltrami_research.artifacts import IMPLEMENTATIONS, export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-checkpoint", type=Path, required=True,
                        help="Only a checkpoint created locally by this trainer; PyTorch pickle is trusted")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = export(args.trusted_checkpoint, args.config, args.implementation, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
