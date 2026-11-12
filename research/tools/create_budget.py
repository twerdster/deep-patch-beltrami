"""Create an exclusive, empty ledger for a user-chosen wall-time allowance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def create(path, hours):
    if not 0 < hours <= 12:
        raise ValueError("Choose a positive aggregate allowance of at most 12 hours")
    value = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "authorized_gpu_seconds": hours * 3600,
             "consumed_gpu_seconds": 0, "status": "idle", "entries": []}
    # Exclusive creation preserves the accounting of previous jobs and failures.
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("BUDGET.json"))
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    create(args.output, args.hours)


if __name__ == "__main__":
    main()
