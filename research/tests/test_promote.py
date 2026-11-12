"""Continuation preparation must fail before touching parent or output data."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "promote_run", Path(__file__).resolve().parents[1] / "tools/promote_run.py")
promote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote)


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(promote, "ROOT", tmp_path)
    source = tmp_path / "src/beltrami_operator"
    source.mkdir(parents=True)
    (source / "model.py").write_text("# Frozen numerical source.\n")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train400", "validation32"):
        (data / (name + ".json")).write_text(json.dumps({"split": name}) + "\n")
    config = tmp_path / "config.json"
    contents = {"steps": 60000, "learning_rate": 1e-5}
    config.write_text(json.dumps(contents) + "\n")
    parent = tmp_path / "runs/pilot-control-001"
    parent.mkdir(parents=True)
    (parent / "status.json").write_text(json.dumps({
        "state": "stopped", "reason": "budget", "step": 1000}))
    (parent / "metadata.json").write_text(json.dumps({"run": parent.name}))
    checkpoint = {
        "source_sha256": promote.numerical_source_hash(),
        "config_sha256": promote.digest(config), "config": contents,
        "manifest_sha256": promote.digest(data / "train400.json"),
        "validation_manifest_sha256": promote.digest(data / "validation32.json"),
        "model": {"weight": torch.tensor([1.25])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.1])}}},
        "step": 1000, "gpu_seconds": 1224.0, "best_psnr": 28.4,
    }
    torch.save(checkpoint, parent / "last.pt")
    best = copy.deepcopy(checkpoint)
    best.update(step=500, gpu_seconds=600.0)
    torch.save(best, parent / "best.pt")
    ledger = {"authorized_gpu_seconds": 43200, "entries": [{
        "run_id": parent.name, "state": "complete", "returncode": 0,
        "charged_seconds": 1230.0}]}
    (tmp_path / "BUDGET.json").write_text(json.dumps(ledger))
    decision = tmp_path / "decision.txt"
    decision.write_text("Promote using validation32 only; matched control retained.\n")
    return {
        "parent": parent, "output": tmp_path / "runs/promotion-control-001",
        "config": config, "hours": 1.5, "decision": decision,
        "model_dir": tmp_path / "original-models", "gpu_lock": tmp_path / "gpu.lock",
    }


def change_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def change_checkpoint(path, update):
    value = torch.load(path, map_location="cpu", weights_only=False)
    update(value)
    torch.save(value, path)


def test_prepare_preserves_parent_and_best_and_extends_only_inner_time(prepared_inputs):
    args = prepared_inputs
    original_hashes = {p.name: promote.digest(p) for p in args["parent"].iterdir()}
    plan = promote.prepare(**args)
    assert {p.name: promote.digest(p) for p in args["parent"].iterdir()} == original_hashes
    assert promote.digest(args["output"] / "parent-last.pt") == original_hashes["last.pt"]
    assert promote.digest(args["output"] / "best.pt") == original_hashes["best.pt"]
    assert plan["new_lease_hours"] == 1.5
    assert plan["cumulative_inner_hours"] == pytest.approx(1224 / 3600 + 1.5 - .02)
    command = plan["command"]
    assert command[command.index("--hours") + 1] == "1.5"
    assert command[command.index("--resume") + 1] == str(args["output"] / "parent-last.pt")
    assert command[command.index("--config") + 1] == str(args["config"])
    assert json.loads((args["output"] / "promotion.json").read_text()) == plan


@pytest.mark.parametrize("artifact,field", [
    ("last.pt", "source_sha256"), ("best.pt", "source_sha256"),
    ("last.pt", "config_sha256"), ("best.pt", "manifest_sha256"),
    ("last.pt", "validation_manifest_sha256"),
])
def test_checkpoint_hash_guards_fail_before_creating_output(prepared_inputs, artifact, field):
    args = prepared_inputs
    change_checkpoint(args["parent"] / artifact, lambda value: value.update({field: "wrong"}))
    with pytest.raises(ValueError, match="differs"):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("relative_path", [
    "src/beltrami_operator/model.py", "config.json", "data/train400.json", "data/validation32.json",
])
def test_current_source_config_and_manifest_changes_are_rejected(prepared_inputs, relative_path):
    args = prepared_inputs
    path = promote.ROOT / relative_path
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("change,match", [
    (lambda v: v["config"].update(steps=120000), "config contents"),
    (lambda v: v.update(step=999), "terminal status steps"),
    (lambda v: v.update(gpu_seconds=float("nan")), "elapsed time"),
    (lambda v: v.update(best_psnr=float("nan")), "best score"),
    (lambda v: v["model"]["weight"].fill_(float("inf")), "not finite"),
    (lambda v: v["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")), "not finite"),
])
def test_invalid_last_state_is_rejected(prepared_inputs, change, match):
    args = prepared_inputs
    change_checkpoint(args["parent"] / "last.pt", change)
    with pytest.raises(ValueError, match=match):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("change,match", [
    (lambda v: v.update(step=1001), "later than"),
    (lambda v: v.update(gpu_seconds=1300), "later than"),
    (lambda v: v.update(best_psnr=28.3), "disagree"),
])
def test_stale_or_inconsistent_best_is_rejected(prepared_inputs, change, match):
    args = prepared_inputs
    change_checkpoint(args["parent"] / "best.pt", change)
    with pytest.raises(ValueError, match=match):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("change,match", [
    (lambda v: v["entries"][0].update(state="running"), "unresolved lease"),
    (lambda v: v["entries"][0].update(returncode=1), "successfully terminated"),
    (lambda v: v["entries"].clear(), "successfully terminated"),
    (lambda v: v["entries"][0].update(charged_seconds=36000), "evaluation reserve"),
    (lambda v: v["entries"][0].update(charged_seconds=-1), "invalid charged time"),
    (lambda v: v.update(authorized_gpu_seconds=43201), "must not exceed"),
    (lambda v: v["entries"].append({"run_id": "promotion-control-001", "state": "complete", "returncode": 0, "charged_seconds": 1}), "unique new lease"),
])
def test_ledger_guards(prepared_inputs, change, match):
    args = prepared_inputs
    change_json(promote.ROOT / "BUDGET.json", change)
    with pytest.raises(ValueError, match=match):
        promote.prepare(**args)
    assert not args["output"].exists()


def test_exact_final_reserve_boundary_is_allowed(prepared_inputs):
    args = prepared_inputs
    charge = 43200 - 60 - 1800 - args["hours"] * 3600
    change_json(promote.ROOT / "BUDGET.json", lambda v: v["entries"][0].update(charged_seconds=charge))
    promote.prepare(**args)


def test_signal_stopped_parent_and_existing_output_are_rejected(prepared_inputs):
    args = prepared_inputs
    change_json(args["parent"] / "status.json", lambda v: v.update(reason="signal"))
    with pytest.raises(ValueError, match="separate diagnosis"):
        promote.prepare(**args)
    assert not args["output"].exists()
    change_json(args["parent"] / "status.json", lambda v: v.update(reason="budget"))
    args["output"].mkdir()
    with pytest.raises(FileExistsError):
        promote.prepare(**args)


def test_copy_race_is_detected(prepared_inputs, monkeypatch):
    args = prepared_inputs
    copy_file = promote.shutil.copy2

    def changing_copy(source, destination):
        result = copy_file(source, destination)
        if Path(source).name == "last.pt":
            with Path(source).open("ab") as output:
                output.write(b"changed after validation")
        return result

    monkeypatch.setattr(promote.shutil, "copy2", changing_copy)
    with pytest.raises(IOError, match="changed while"):
        promote.prepare(**args)
    assert not (args["output"] / "promotion.json").exists()


def precision_inputs(args, precision="float64"):
    """Synthetic source/metadata/checkpoints; no implementation is imported."""
    source = promote.ROOT / "src/beltrami_precision"
    source.mkdir()
    (source / "model.py").write_text("# Separate high-precision numerical source.\n")
    change_json(args["config"], lambda value: value.update(model={"metric_precision": precision}))
    contents = json.loads(args["config"].read_text())
    change_json(args["parent"] / "metadata.json", lambda value: value.update(
        implementation="beltrami_precision", config=contents,
        numerics={"metric_precision": precision}))
    for name in ("last.pt", "best.pt"):
        change_checkpoint(args["parent"] / name, lambda value: value.update(
            source_sha256=promote.numerical_source_hash("beltrami_precision"),
            config_sha256=promote.digest(args["config"]), config=contents))
    return args


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_namespace_default_and_explicit_dispatch(prepared_inputs, explicit):
    args = prepared_inputs
    if explicit:
        change_json(args["parent"] / "metadata.json", lambda value: value.update(
            implementation="beltrami_operator"))
    plan = promote.prepare(**args)
    assert plan["implementation"] == "beltrami_operator"
    assert plan["metric_precision"] == "float32"
    assert plan["source_sha256"] == promote.numerical_source_hash()
    assert plan["command"][plan["command"].index("-m") + 1] == "beltrami_operator.train"


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_precision_namespace_dispatch_and_own_source_hash(prepared_inputs, precision):
    args = precision_inputs(prepared_inputs, precision)
    plan = promote.prepare(**args)
    assert plan["implementation"] == "beltrami_precision"
    assert plan["metric_precision"] == precision
    assert plan["source_sha256"] == promote.numerical_source_hash("beltrami_precision")
    assert plan["source_sha256"] != promote.numerical_source_hash("beltrami_operator")
    assert plan["command"][plan["command"].index("-m") + 1] == "beltrami_precision.train"
    assert promote.digest(args["output"] / "parent-last.pt") == promote.digest(args["parent"] / "last.pt")
    assert promote.digest(args["output"] / "best.pt") == promote.digest(args["parent"] / "best.pt")


@pytest.mark.parametrize("implementation", [
    "os", "beltrami_precision.train", "../beltrami_operator", "", None, 32, ["beltrami_precision"],
])
def test_unknown_implementation_rejected_before_checkpoint_loading(prepared_inputs, monkeypatch, implementation):
    args = prepared_inputs
    change_json(args["parent"] / "metadata.json", lambda value: value.update(
        implementation=implementation))

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("Unknown implementations must be rejected before checkpoint loading")

    monkeypatch.setattr(promote.torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="Unknown operator implementation"):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("implementation", ["os", "beltrami_precision.train", "../beltrami_operator"])
def test_source_hash_implementation_allowlist(prepared_inputs, implementation):
    with pytest.raises(ValueError, match="Unknown operator implementation"):
        promote.numerical_source_hash(implementation)


@pytest.mark.parametrize("parent_implementation", ["beltrami_operator", "beltrami_precision"])
@pytest.mark.parametrize("artifact", ["last.pt", "best.pt"])
def test_cross_namespace_checkpoint_source_is_rejected(prepared_inputs, parent_implementation, artifact):
    args = precision_inputs(prepared_inputs)
    other = "beltrami_operator"
    if parent_implementation == "beltrami_operator":
        contents = {"steps": 60000, "learning_rate": 1e-5}
        args["config"].write_text(json.dumps(contents) + "\n")
        change_json(args["parent"] / "metadata.json", lambda value: value.update(
            implementation="beltrami_operator", config=contents))
        for name in ("last.pt", "best.pt"):
            change_checkpoint(args["parent"] / name, lambda value: value.update(
                source_sha256=promote.numerical_source_hash("beltrami_operator"),
                config_sha256=promote.digest(args["config"]), config=contents))
        other = "beltrami_precision"
    change_checkpoint(args["parent"] / artifact, lambda value: value.update(
        source_sha256=promote.numerical_source_hash(other)))
    with pytest.raises(ValueError, match="numerical source differs"):
        promote.prepare(**args)
    assert not args["output"].exists()


def test_precision_source_is_isolated_from_legacy_namespace(prepared_inputs):
    args = precision_inputs(prepared_inputs)
    (promote.ROOT / "src/beltrami_operator/model.py").write_text("# Unrelated legacy namespace change.\n")
    plan = promote.prepare(**args)
    assert plan["source_sha256"] == promote.numerical_source_hash("beltrami_precision")


def test_changed_precision_source_is_rejected(prepared_inputs):
    args = precision_inputs(prepared_inputs)
    (promote.ROOT / "src/beltrami_precision/model.py").write_text("# Changed precision namespace.\n")
    with pytest.raises(ValueError, match="numerical source differs"):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("precision", [None, "float16", "float128", 64])
def test_precision_namespace_requires_explicit_supported_config(prepared_inputs, precision):
    args = precision_inputs(prepared_inputs, precision)
    with pytest.raises(ValueError, match="explicit float64 or diagnostic float32"):
        promote.prepare(**args)
    assert not args["output"].exists()


def test_precision_namespace_rejects_missing_precision_config(prepared_inputs):
    args = precision_inputs(prepared_inputs)
    change_json(args["config"], lambda value: value.update(model={}))
    with pytest.raises(ValueError, match="explicit float64 or diagnostic float32"):
        promote.prepare(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("explicit_legacy", [False, True])
def test_precision_config_cannot_be_dispatched_as_legacy(prepared_inputs, explicit_legacy):
    args = precision_inputs(prepared_inputs)
    if explicit_legacy:
        change_json(args["parent"] / "metadata.json", lambda value: value.update(
            implementation="beltrami_operator"))
    else:
        change_json(args["parent"] / "metadata.json", lambda value: value.pop("implementation"))
    with pytest.raises(ValueError, match="Legacy implementation cannot declare"):
        promote.prepare(**args)
    assert not args["output"].exists()


def test_parent_precision_metadata_must_match_config(prepared_inputs):
    args = precision_inputs(prepared_inputs)
    change_json(args["parent"] / "metadata.json", lambda value: value["numerics"].update(
        metric_precision="float32"))
    with pytest.raises(ValueError, match="precision metadata and model config differ"):
        promote.prepare(**args)
    assert not args["output"].exists()
