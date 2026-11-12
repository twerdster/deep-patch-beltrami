"""CPU-only promotion-report lineage regressions; checkpoint bytes are never loaded."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "summarize_sweep_lineage_subject", Path(__file__).with_name("summarize_sweep.py"))
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PromotionLineageTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "runs").mkdir()
        (self.root / "configs").mkdir()
        source = self.root / "src/beltrami_operator"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text('"""Synthetic frozen operator source."""\n')
        self.config = {
            "sigma": 25, "offset": 25, "seed": 20261112,
            "validation_seed": 742913, "steps": 60000, "batch_size": 2,
            "patch_size": 160, "learning_rate": 1e-5, "weight_decay": 0,
            "gradient_clip": 1, "deterministic": True, "evaluate_every": 50,
            "checkpoint_every": 50, "log_every": 20,
            "model": {"stages": 5, "width": 80, "metric": "frozen",
                      "reaction": False, "zero_dc": False, "step_scale": 1000.0},
        }
        self.config_file = self.root / "configs/pilot-finetune-frozen.json"
        put(self.config_file, self.config)
        self.parent = self.root / "runs/pilot-finetune-frozen-001"
        self.child = self.root / "runs/promote-finetune-frozen-001"
        self.parent.mkdir()
        self.child.mkdir()
        recorded = str(summary.RECORDED_PROJECT_ROOT)
        self.metadata = {
            "implementation": "beltrami_operator", "config": self.config,
            "config_sha256": digest(self.config_file),
            "environment": {"source_sha256": summary.local_source_hash(self.root)},
            "initialization": {"source_sha256": "a" * 64, "function_preserving": True},
            "parameters": 3195600, "trainable_parameters": 3195600,
            "manifest_sha256": summary.TRAIN_SHA,
            "validation_manifest_sha256": summary.VALIDATION_SHA,
            "train_images": 400, "validation_images": 32,
            "paths": {"config": recorded + "/configs/pilot-finetune-frozen.json",
                      "manifest": recorded + "/data/train400.json",
                      "validation_manifest": recorded + "/data/validation32.json"},
        }
        self.parent_status = {"state": "stopped", "reason": "budget", "step": 100,
                              "gpu_seconds": 10.0, "utc": "2026-11-12T10:00:00Z"}
        put(self.parent / "metadata.json", self.metadata)
        put(self.parent / "status.json", self.parent_status)
        put(self.child / "parent-metadata.json", self.metadata)
        put(self.child / "parent-status.json", self.parent_status)
        child_metadata = copy.deepcopy(self.metadata)
        child_metadata["paths"]["resume"] = recorded + "/runs/" + self.child.name + "/parent-last.pt"
        put(self.child / "metadata.json", child_metadata)
        put(self.child / "status.json", {
            "state": "training", "step": 200, "gpu_seconds": 25.0,
            "utc": "2026-11-12T11:00:00Z"})
        # These are inert identity fixtures, not serialized model objects.
        for path in (self.parent / "last.pt", self.child / "parent-last.pt"):
            path.write_bytes(b"synthetic immutable parent LAST bytes\n")
        (self.parent / "best.pt").write_bytes(b"synthetic immutable parent BEST bytes\n")
        (self.child / "best.pt").write_bytes((self.parent / "best.pt").read_bytes())
        self.plan = {
            "parent": recorded + "/runs/" + self.parent.name, "parent_step": 100,
            "parent_last_sha256": digest(self.child / "parent-last.pt"),
            "parent_best_sha256": digest(self.parent / "best.pt"),
            "implementation": "beltrami_operator", "metric_precision": "float32",
            "source_sha256": self.metadata["environment"]["source_sha256"],
            "config_sha256": self.metadata["config_sha256"],
            "new_lease_hours": 1.5, "cumulative_inner_hours": 10 / 3600 + 1.5 - .02,
        }
        put(self.child / "promotion.json", self.plan)
        self.make_validations(self.parent, ((0, 0.0, 1), (50, .04, 5), (100, .02, 10)))
        self.make_validations(self.child, ((100, .025, 15), (150, .03, 20), (200, .035, 25)))
        self.ledger = {"authorized_gpu_seconds": 43200, "consumed_gpu_seconds": 29,
                       "entries": [
                           {"run_id": self.parent.name, "charged_seconds": 12,
                            "allowance_seconds": 20, "state": "complete", "returncode": 0},
                           {"run_id": self.child.name, "charged_seconds": 17,
                            "allowance_seconds": 5400, "state": "running"}]}
        put(self.root / "BUDGET.json", self.ledger)

    def make_validations(self, directory, observations):
        events = []
        for step, gain, elapsed in observations:
            images = [{"image": f"image{i:02d}.png", "pixels_sha256": f"{i:064x}",
                       "psnr": summary.INITIAL_PSNR + gain + (i - 15.5) * .01}
                      for i in range(32)]
            mean = sum(item["psnr"] for item in images) / 32
            put(directory / f"validation-{step:07d}.json", {
                "mean_psnr": mean, "images": images, "crop": 0,
                "clipping": "prediction only, float [0,255]"})
            events.append({"event": "validation", "step": step, "gpu_seconds": elapsed,
                           "mean_psnr": mean})
        events.append({"event": "progress", "step": observations[-1][0],
                       "gpu_seconds": observations[-1][2], "loss": .001,
                       "gradient_norm": .01, "images_seen": observations[-1][0] * 2})
        (directory / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events))

    def load(self):
        return summary.load_run(self.child, self.root, self.ledger)

    def assert_untrusted(self):
        run = self.load()
        self.assertIsNone(run["initial"], "untrusted parent cannot supply step0")
        self.assertNotEqual(run.get("lineage_run_ids"), [self.parent.name, self.child.name])
        self.assertTrue(run["issues"], "rejected lineage must retain diagnostic evidence")

    def test_inherits_parent_best_and_global_curve_without_replaying_exposure(self):
        run = self.load()
        self.assertEqual(run["initial"]["step"], 0)
        self.assertEqual(run["best"]["step"], 50)
        self.assertEqual([item["step"] for item in run["validations"]], [0, 50, 100, 150, 200])
        boundary = next(item for item in run["validations"] if item["step"] == 100)
        self.assertAlmostEqual(boundary["mean_psnr"], summary.INITIAL_PSNR + .025)
        self.assertEqual(Path(boundary["path"]).parent, self.child.resolve())
        self.assertEqual(run["updates"], 200)
        self.assertEqual(run["images_seen"], 400)
        self.assertEqual(run["pixels_seen"], 200 * 2 * 160 * 160)
        self.assertEqual(run["continuation_updates"], 100)
        self.assertEqual(run["lineage_run_ids"], [self.parent.name, self.child.name])
        self.assertEqual(run["lease_seconds_snapshot"], 29)
        self.assertEqual(run["continuation_lease_seconds_snapshot"], 17)
        self.assertEqual(run["inner_cumulative_seconds"], 25)

    def test_report_promote_glob_inherits_evidence_but_has_only_promoted_rows(self):
        report = summary.report(self.root, self.root / "report", repeats=100,
                                plot=False, pattern="promote-*")
        self.assertEqual(len(report["runs"]), 1)
        run = report["runs"][0]
        self.assertEqual(run["run_id"], self.child.name)
        self.assertEqual(run["initial"]["step"], 0)
        self.assertEqual(run["best"]["step"], 50)
        self.assertEqual(run["delta_initial"]["candidate_step"], 50)
        self.assertAlmostEqual(run["delta_initial"]["mean_delta_db"], .04)
        self.assertEqual(report["pending_expected_pilots"], [])
        self.assertEqual(run["control_run_id"], self.child.name)

    def test_saved_parent_metadata_must_agree_with_actual_parent(self):
        changed = copy.deepcopy(self.metadata)
        changed["config"]["validation_seed"] += 1
        put(self.child / "parent-metadata.json", changed)
        self.assert_untrusted()

    def test_parent_status_cannot_silently_move_past_resume_step(self):
        changed = dict(self.parent_status, step=101)
        put(self.child / "parent-status.json", changed)
        self.assert_untrusted()

    def test_promotion_guards_cannot_be_inferred_from_run_name(self):
        for field, value in (("source_sha256", "b" * 64),
                             ("config_sha256", "c" * 64),
                             ("implementation", "beltrami_precision"),
                             ("metric_precision", "float64"), ("parent_step", 99)):
            with self.subTest(field=field):
                plan = dict(self.plan, **{field: value})
                put(self.child / "promotion.json", plan)
                self.assert_untrusted()
        put(self.child / "promotion.json", self.plan)

    def test_external_traversal_and_self_parent_are_rejected(self):
        recorded = str(summary.RECORDED_PROJECT_ROOT)
        for parent in ("/tmp/" + self.parent.name,
                       recorded + "/runs/../runs/" + self.parent.name,
                       recorded + "/runs/" + self.child.name):
            with self.subTest(parent=parent):
                put(self.child / "promotion.json", dict(self.plan, parent=parent))
                self.assert_untrusted()

    def test_symlink_parent_cannot_escape_snapshot(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / self.parent.name
            external.mkdir()
            link = self.root / "runs/linked-parent"
            link.symlink_to(external, target_is_directory=True)
            recorded = str(summary.RECORDED_PROJECT_ROOT)
            put(self.child / "promotion.json", dict(self.plan, parent=recorded + "/runs/linked-parent"))
            self.assert_untrusted()

    def test_individual_retained_metadata_symlink_cannot_escape_snapshot(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "parent-metadata.json"
            put(external, self.metadata)
            retained = self.child / "parent-metadata.json"
            retained.unlink()
            retained.symlink_to(external)
            self.assert_untrusted()

    def test_cyclic_parent_chain_never_supplies_initial_or_compute_claims(self):
        recorded = str(summary.RECORDED_PROJECT_ROOT)
        parent_plan = dict(self.plan, parent=recorded + "/runs/" + self.child.name,
                           parent_step=200)
        put(self.parent / "promotion.json", parent_plan)
        put(self.parent / "parent-metadata.json", json.loads(
            (self.child / "metadata.json").read_text()))
        put(self.parent / "parent-status.json", json.loads(
            (self.child / "status.json").read_text()))
        self.assert_untrusted()

    def test_absent_promotion_hash_is_not_a_wildcard(self):
        plan = dict(self.plan)
        del plan["source_sha256"]
        put(self.child / "promotion.json", plan)
        self.assert_untrusted()

    def test_missing_parent_does_not_mislabel_continuation_lease_as_total(self):
        put(self.child / "promotion.json", dict(
            self.plan, parent=str(summary.RECORDED_PROJECT_ROOT) + "/runs/missing-parent"))
        run = self.load()
        self.assertIsNone(run["initial"])
        self.assertIsNone(run["lease_seconds_snapshot"], "cumulative ancestry cost is unknown")
        self.assertEqual(run["continuation_lease_seconds_snapshot"], 17)

    def test_missing_parent_lease_keeps_cumulative_cost_unknown(self):
        self.ledger["entries"] = self.ledger["entries"][1:]
        run = self.load()
        self.assertIsNone(run["lease_seconds_snapshot"])
        self.assertEqual(run["continuation_lease_seconds_snapshot"], 17)

    def test_invalid_or_missing_ancestor_charge_never_becomes_zero(self):
        original = copy.deepcopy(self.ledger)
        for charge in (None, -1, True, "12", float("nan"), float("inf")):
            with self.subTest(charge=charge):
                self.ledger = copy.deepcopy(original)
                if charge is None:
                    del self.ledger["entries"][0]["charged_seconds"]
                else:
                    self.ledger["entries"][0]["charged_seconds"] = charge
                run = self.load()
                self.assertIsNone(run["lease_seconds_snapshot"])
                self.assertEqual(run["continuation_lease_seconds_snapshot"], 17)
                self.assertTrue(run["issues"])

    def test_manifest_paths_map_only_inside_snapshot_and_exact_recorded_root(self):
        target = (self.root / "data/validation32.json").resolve()
        recorded = str(summary.RECORDED_PROJECT_ROOT)
        for declared in ("data/validation32.json", recorded + "/data/validation32.json"):
            with self.subTest(declared=declared):
                self.assertEqual(summary.local_manifest_path(self.root, declared), target)
        custom_root = "/archives/research-snapshot"
        self.assertEqual(summary.local_manifest_path(
            self.root, custom_root + "/data/validation32.json", custom_root), target)
        for declared in ("../data/validation32.json", "data/../data/validation32.json",
                         recorded + "/runs/../data/validation32.json",
                         recorded + "-sibling/data/validation32.json",
                         "/tmp/validation32.json", "", "data/invalid\x00name.json"):
            with self.subTest(rejected=declared):
                with self.assertRaises(ValueError):
                    summary.local_manifest_path(self.root, declared)
        with self.assertRaises(ValueError):
            summary.local_manifest_path(self.root, "data/validation32.json", "relative/root")

    def test_manifest_symlink_escape_is_rejected(self):
        data = self.root / "data"
        data.mkdir()
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "validation32.json"
            put(external, {"images": []})
            (data / "validation32.json").symlink_to(external)
            with self.assertRaises(ValueError):
                summary.local_manifest_path(self.root, "data/validation32.json")

    def test_manifest_file_hash_checks_exact_bytes_and_records_mismatch_evidence(self):
        data = self.root / "data"
        data.mkdir()
        manifest = data / "validation32.json"
        put(manifest, {"images": []})
        metadata = copy.deepcopy(self.metadata)
        metadata["validation_manifest_sha256"] = digest(manifest)
        put(self.parent / "metadata.json", metadata)
        matching = summary.load_run(self.parent, self.root, self.ledger)
        self.assertEqual(matching["checks"]["local_manifests"]["validation"], "matches recorded SHA")
        original_sha = digest(manifest)
        manifest.write_text(manifest.read_text() + "\n")
        mismatching = summary.load_run(self.parent, self.root, self.ledger)
        self.assertEqual(mismatching["checks"]["local_manifests"]["validation"], "MISMATCH")
        evidence = mismatching["checks"]["manifest_path_evidence"]["validation"]
        self.assertEqual(evidence["recorded_sha256"], original_sha)
        self.assertEqual(evidence["local_sha256"], digest(manifest))
        self.assertNotEqual(evidence["local_sha256"], evidence["recorded_sha256"])

    def test_resource_trend_uses_new_unique_steps_and_floor_quarters(self):
        observations = [{"step": 100, "mean_psnr": 1000.0}]
        observations += [{"step": 100 + index, "mean_psnr": index * .001}
                         for index in range(1, 8)]
        observations.insert(1, {"step": 101, "mean_psnr": -1000.0})
        trend = summary.resource_trend(observations, 100)
        self.assertEqual(trend["unique_new_validation_count"], 7)
        self.assertEqual(trend["quarter_count"], 1, "floor differs from rounded quarter at n=7")
        self.assertEqual(trend["evaluated_global_steps"], list(range(101, 108)))
        self.assertAlmostEqual(trend["last_minus_first_db"], .006)
        self.assertFalse(trend["no_further_extension"])
        observations.append({"step": 108, "mean_psnr": .008})
        trend = summary.resource_trend(observations, 100)
        self.assertEqual(trend["quarter_count"], 2)
        self.assertAlmostEqual(trend["first_quarter_median_psnr"], .0015)
        self.assertAlmostEqual(trend["last_quarter_median_psnr"], .0075)

    def test_resource_trend_minimum_count_and_strict_threshold(self):
        for count in (0, 1, 2, 3):
            with self.subTest(count=count):
                trend = summary.resource_trend([
                    {"step": 101 + index, "mean_psnr": 0.0}
                    for index in range(count)], 100)
                self.assertIsNone(trend["no_further_extension"])
        values = [{"step": 101 + index, "mean_psnr": 0.0} for index in range(4)]
        self.assertTrue(summary.resource_trend(values, 100)["no_further_extension"])
        values[-1]["mean_psnr"] = .002
        self.assertFalse(summary.resource_trend(values, 100)["no_further_extension"],
                         "the declared comparison is strictly less than .002dB")


if __name__ == "__main__":
    unittest.main()
