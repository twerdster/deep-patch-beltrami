# Sequential prepared continuations

`run_prepared_promotions.py` executes existing `promotion.json` commands only.
It never prepares or resumes a run implicitly. First prepare every
entry in the reviewed `promotion-plan.json` using `promote_run.py`, while no
compute lease is active. Numerical namespaces remain frozen.

The plan contains only reviewed basenames and explicit new allowances. For
example, one prepared continuation can be recorded as:

```json
{
  "schema": 1,
  "state": "prepared_not_launched",
  "decision": "promotion-decision.txt",
  "runs": [
    {
      "parent": "pilot-control-001",
      "output": "continued-control-001",
      "config": "pilot-finetune-frozen.json",
      "hours": 0.3
    }
  ]
}
```

Prepare that exact new output with `promote_run.py --hours 0.3` **without**
`--launch` first. The decision file must record the validation-only reason;
add each matched candidate/control continuation and its allowance to the
reviewed plan before validating the whole queue. Reuse the same budget ledger,
model directory, GPU lock and Python interpreter used during preparation.

On Linux, from `research/` and using the same virtual-environment interpreter
used for preparation:

```bash
python tools/run_prepared_promotions.py \
  --plan promotion-plan.json --status promotion-status.json \
  --model-dir "$MODEL_DIR" --gpu-lock "$PWD/gpu.lock" --validate-only
```

The validation mode checks the entire ordered plan, exact saved command
schema, parent lease/status and frozen checkpoint copies, decision/config/data/
numerical-source hashes, finite checkpoint state, and the sum of all new lease
allowances. It requires the shared budget lock to be idle and preserves at
least 1,800 seconds for final evaluation plus the global 60-second guard.
It does not touch CUDA, access BSD68, write queue status, or launch training.

After review and live GPU/service/budget checks, remove `--validate-only` and
run inside a bounded systemd user service with `KillMode=control-group`.
Choose the service deadline to cover the reviewed plan's outer allowances and
orderly orchestration/cleanup without increasing those allowances. Every child
independently uses the existing `lease_runner`.
Verify the live queue, training process, fresh status/log/checkpoints and active
recurring progress/health monitor before leaving it unattended.

`promotion-status.json` records queue PID, plan/artifact pins, current run and
lease PID, timestamps and verified terminal results. It updates every two
seconds while a lease runs. A completed result includes terminal training
status, successful lease, and last/best hashes. Clean inner budget stops are
expected; lease-forced/signal stops, failed status, stale or nonfinite artifacts,
and missing update progress fail the queue. No later entry starts after a
failure or signal. SIGTERM/SIGINT is forwarded to the current lease process
group, whose supervisor terminates its training group; systemd remains the
whole-cgroup hard-stop backstop.

An existing status file, output execution evidence, used lease identifier,
live budget lock or unresolved ledger entry refuses execution. Never delete
these to retry. Preserve evidence, inspect actual process/service identity,
diagnose recovery and create a newly reviewed plan with new output and lease
identifiers if continuation is justified. Parent runs remain untouched.

CPU verification:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_promotion_queue.py
```

The synthetic fixtures cover prepared-command mutation, decision/source/config/
data/checkpoint guards, no-relaunch behavior, aggregate reserve, terminal
status/checkpoint freshness and finite optimizer/model state. No GPU is used.
