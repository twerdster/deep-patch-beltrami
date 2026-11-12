"""Run one process group within a durable shared compute allowance."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget-file', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.hours <= 12:
        p.error('Provide a command and positive allowance no greater than12 hours')
    lock = args.budget_file.with_suffix('.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another compute lease owns the budget lock')
    ledger = json.loads(args.budget_file.read_text())
    if any(e['state'] == 'running' for e in ledger['entries']):
        raise SystemExit('Unresolved running lease; inspect its process/service and evidence before recovery')
    if any(e['run_id'] == args.run_id for e in ledger['entries']):
        raise SystemExit('Lease identifier already exists; resumes need a new explicit lease')
    spent = sum(e['charged_seconds'] for e in ledger['entries'])
    allowance = args.hours * 3600
    # Keep a wall-clock margin for scheduler and context teardown latency.
    if spent + allowance > ledger['authorized_gpu_seconds'] - 60:
        raise SystemExit('Requested lease exceeds the remaining aggregate allowance')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(run_id=args.run_id, allowance_seconds=allowance, started_utc=utc(),
                 state='running', supervisor_pid=os.getpid(), command=command,
                 log=str(args.log), charged_seconds=0)
    ledger['entries'].append(entry)
    ledger['status'] = 'running'
    write(args.budget_file, ledger)
    began = time.monotonic()
    stop_requested = False
    def stop(_signal, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    grace = min(20.0, allowance / 5)
    term_sent = None
    returncode = 1
    process = None
    try:
        with args.log.open('a', buffering=1) as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            entry['process_pid'] = process.pid
            last_write = 0
            while process.poll() is None:
                elapsed = time.monotonic() - began
                if term_sent is None and (stop_requested or elapsed >= allowance - grace):
                    os.killpg(process.pid, signal.SIGTERM)
                    term_sent = time.monotonic()
                    entry['termination_reason'] = 'signal' if stop_requested else 'lease_budget'
                if term_sent is not None and time.monotonic() - term_sent >= grace:
                    os.killpg(process.pid, signal.SIGKILL)
                    entry['forced_kill'] = True
                    break
                if elapsed - last_write >= 5:
                    entry['charged_seconds'] = elapsed
                    entry['last_checked_utc'] = utc()
                    ledger['consumed_gpu_seconds'] = spent + elapsed
                    write(args.budget_file, ledger)
                    last_write = elapsed
                time.sleep(.5)
            returncode = process.wait()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        entry.update(charged_seconds=time.monotonic() - began, ended_utc=utc(),
                     returncode=returncode, state='complete' if returncode == 0 else 'failed')
        ledger['consumed_gpu_seconds'] = spent + entry['charged_seconds']
        ledger['status'] = 'idle'
        write(args.budget_file, ledger)
    print(json.dumps(entry), flush=True)
    return returncode


if __name__ == '__main__':
    sys.exit(main())
