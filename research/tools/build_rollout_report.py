#!/usr/bin/env python3
"""Build path-free long-rollout results from completed, frozen author evidence.

This maintainer-only CPU tool validates a hash-pinned private evaluator and raw
checkpoints. Public reproduction uses evaluate_rollout_released.py instead.
Create a draft without --budget, replay it under a new lease, then rebuild into
a new directory with the completed idle ledger. Stable replay bytes must match.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from urllib.parse import urlsplit

import torch
from beltrami_research import artifacts, rollout_artifacts
from deep_patch_beltrami.assets import metadata

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

replay = module("rollout_report_replay", ROOT / "tools/evaluate_rollout_released.py")
legacy = module("rollout_report_existing", ROOT / "tools/build_public_report.py")
read, require, same = legacy.read, legacy.require, legacy.same
protocol = replay.protocol
UNCERTAINTY = ("95% paired-image percentile bootstrap intervals condition on the frozen checkpoints, "
               "one fixed noise realization and one training seed. They exclude training-seed, "
               "noise-draw, model-selection and prior benchmark-exposure uncertainty.")
PRIVATE_PROTOCOL = {"axes": "N,C,W,H", "input": "archivedunclippedfloat32+25", "crop": 0,
                    "output": "rejectnonfinite,subtract25,clipfloat[0,255]", "mse_dtype": "float64",
                    "aggregation": "meanper-imagePSNRdB", "TTA": False}


def verify_results(selection_path, results_path, results_sha, status_path, private_evaluator_path):
    replay.digest(results_sha)
    require(protocol.file_hash(results_path) == results_sha, "Results file SHA256 differs")
    selection = read(selection_path)
    require(protocol.file_hash(private_evaluator_path) == selection["evaluator_sha256"], "Private evaluator SHA256 differs")
    private = module("rollout_private_frozen_verifier", private_evaluator_path)
    checked, selection_sha = private.load_selection(selection_path)
    require(checked == selection, "Selection changed during verification")
    raw, status = read(results_path), read(status_path)
    require(raw.get("schema") == private.SCHEMA and raw.get("complete") is True
            and not raw.get("failure") and raw.get("dataset") == "bsd68" and raw.get("image_count") == 68,
            "Private evaluation incomplete or wrong dataset")
    require(raw["selection"] == selection and raw["selection_sha256"] == selection_sha,
            "Frozen selection identity differs")
    require(selection["protocol"] == PRIVATE_PROTOCOL and selection["comparisons"] == replay.COMPARISONS,
            "Fixed numerical protocol/comparisons differ")
    require([x["name"] for x in selection["models"]] == list(replay.MODEL_NAMES)
            and [x["role"] for x in selection["models"]] == ["control", "candidate", "candidate", "benchmark"],
            "Fixed cohort/order/roles differ")
    require("reused" in selection["evaluation_scope"] and "not untouched" in selection["evaluation_scope"],
            "Private benchmark scope must disclose prior exposure")
    for key, minimum in (("timing_warmup", 2), ("timing_repeats", 3), ("bootstrap_samples", 1000), ("bootstrap_seed", 0)):
        require(type(selection[key]) is int and selection[key] >= minimum, "Invalid timing/bootstrap setting")
    require(status.get("state") == "complete" and status.get("images") == 68 and status.get("models") == 4,
            "Final evaluation terminal status incomplete")
    require(datetime.fromisoformat(selection["created_utc"]) <= datetime.fromisoformat(raw["started_utc"])
            <= datetime.fromisoformat(raw["ended_utc"]), "Freeze/evaluation chronology differs")
    manifest = metadata("bsd68.json")
    provenance = {"loader": "deep_patch_beltrami.reproduce.PublicBSD68",
        "public_bsd68_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "noise_asset_sha256": metadata("release-assets.json")["assets"]["bsd68_noise"]["sha256"]}
    require(raw["dataset_provenance"] == provenance, "Archived public input provenance differs")
    images = {x["name"]: x for x in manifest["images"]}
    cases = {x["name"]: x for x in manifest["cases"] if x["sigma"] == 25}
    require([x["name"] for x in raw["models"]] == list(replay.MODEL_NAMES), "Result model set/order differs")
    for entry, result in zip(selection["models"], raw["models"]):
        require(result.get("complete") is True and result["role"] == entry["role"]
                and result["parameters"] == entry["parameters"], "Model completion/identity differs")
        require([x["name"] for x in result["images"]] == sorted(images), "Image set/order differs")
        for image in result["images"]:
            name = image["name"]
            require(image["shape_native"] == images[name]["shape"]
                    and image["label_native_sha256"] == images[name]["label_native_sha256"]
                    and image["noisy_native_sha256"] == cases[name]["noisy_native_sha256"], "Input/label hash or shape differs")
            require(isinstance(image["mse"], (int, float)) and math.isfinite(image["mse"]) and image["mse"] > 0,
                    "Invalid finite MSE")
            require(image.get("perfect_prediction") is False and type(image["clipped_pixels"]) is int
                    and 0 <= image["clipped_pixels"] <= math.prod(image["shape_native"]), "Invalid clipping evidence")
            same(image["psnr"], 10 * math.log10(255 ** 2 / image["mse"]), "PSNR/MSE differs")
            times = image["seconds_forward"]
            require(len(times) == selection["timing_repeats"]
                    and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in times), "Invalid timing repeats")
            same(image["median_seconds_forward"], statistics.median(times), "Image timing median differs")
        same(result["mean_psnr"], statistics.fmean(x["psnr"] for x in result["images"]), "Mean PSNR differs")
        same(result["median_seconds_forward"], statistics.median(x["median_seconds_forward"] for x in result["images"]),
             "Model timing median differs")
        same(result["sum_seconds_forward"], sum(sum(x["seconds_forward"]) for x in result["images"]), "Timing total differs")
    require([{k: x[k] for k in ("candidate", "reference")} for x in raw["comparisons"]] == replay.COMPARISONS,
            "Comparison set/order differs")
    by_name = {x["name"]: x for x in raw["models"]}
    for pair in raw["comparisons"]:
        expected = protocol.paired_summary(by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"],
                                          selection["bootstrap_seed"], selection["bootstrap_samples"])
        same({key: pair[key] for key in expected}, expected, "Paired statistics differ")
    return selection, raw, selection_sha, provenance


def release_url(value, filename):
    url = urlsplit(value)
    require(url.scheme == "https" and url.hostname == "github.com" and not url.username
            and not url.password and not url.query and not url.fragment and "/releases/download/" in url.path
            and Path(url.path).name == filename, "Invalid GitHub release URL")


def verify_exports(spec_path, selection):
    spec = read(spec_path)
    require(spec.get("schema") == 1 and [x["model"] for x in spec["assets"]] == list(replay.MODEL_NAMES[:3]),
            "Supply allthree rollout exports in fixed order")
    assets = []
    filenames = set()
    for entry, item in zip(selection["models"][:3], spec["assets"]):
        filename = item["filename"]
        require(legacy.TOKEN.fullmatch(filename) and filename.endswith(".pt") and filename not in filenames,
                "Invalid/duplicate safe filename")
        filenames.add(filename)
        release_url(item["url"], filename)
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path(spec_path).resolve().parent / path
        sha = protocol.file_hash(path)
        model, value = rollout_artifacts.load_safe(path, expected_sha256=sha)
        meta = {key: item for key, item in value.items() if key != "state_dict"}
        require(value["model_config"] == entry["model_config"] and value["source_pins"] == selection["source_pins"]
                and value["training_step"] == entry["training_step"] and value["full_depth_updates"] == entry["full_depth_updates"]
                and value["phase_updates"] == entry["phase_updates"] and value["parameters"] == entry["parameters"],
                "Safe rollout architecture/source/selection chronology differs")
        provenance = value["provenance"]
        require(provenance["training_checkpoint_sha256"] == entry["checkpoint_sha256"]
                and provenance["training_config_sha256"] == entry["config_sha256"]
                and provenance["warmstart_sha256"] == replay.REACTION_SHA256, "Safe export provenance differs")
        require(protocol.file_hash(entry["checkpoint"]) == entry["checkpoint_sha256"], "Raw selected checkpoint changed")
        # Only a trusted, explicitly frozen maintainer checkpoint is unpickled.
        checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        require(rollout_artifacts.tensor_hash(checkpoint["model"]) == value["tensor_sha256"], "Safe/raw tensor bytes differ")
        require(provenance["training_manifest_sha256"] == checkpoint["manifest_sha256"]
                and provenance["validation_manifest_sha256"] == checkpoint["validation_manifest_sha256"],
                "Safe/raw data provenance differs")
        selected_validation = Path(entry["checkpoint"]).parent / f"validation-{entry['training_step']:07d}.json"
        require(protocol.file_hash(selected_validation) == provenance["selection_validation_sha256"],
                "Safe selected validation pin differs")
        require(protocol.file_hash(path) == sha, "Safe export changed during verification")
        assets.append({"model": entry["name"], "role": entry["role"], "filename": filename, "url": item["url"],
                       "bytes": path.stat().st_size, "sha256": sha, "metadata": meta})
        del model, value, checkpoint
    entry = selection["models"][3]
    path = Path(entry["path"])
    require(protocol.file_hash(path) == entry["sha256"] == replay.REACTION_SHA256, "Released reaction pin differs")
    model, value = artifacts.load(path)
    filename = "dpb-reaction-sigma25-step52000.pt"
    assets.append({"model": "released-reaction", "role": "benchmark", "filename": filename,
        "url": "https://github.com/twerdster/deep-patch-beltrami/releases/download/v1.2.0/" + filename,
        "bytes": path.stat().st_size, "sha256": replay.REACTION_SHA256,
        "metadata": {key: item for key, item in value.items() if key != "state_dict"}})
    require(len({x["metadata"]["provenance"]["training_manifest_sha256"] for x in assets}) == 1
            and len({x["metadata"]["provenance"]["validation_manifest_sha256"] for x in assets}) == 1,
            "Cohort data/selection manifest identities differ")
    return assets


def markdown(public):
    lines = ["# Deep Beltrami long-rollout comparison", "", replay.SCOPE, "",
        "| Model | Steps | Selected updates | Full-depth updates | PSNR (dB) | Median forward (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in public["models"]:
        lines.append(f"| {row['name']} | {row['model_config']['stages']} | {row['training_step']} | "
                     f"{row.get('full_depth_updates', 'previous study')} | {row['mean_psnr']:.6f} | "
                     f"{row['median_seconds_forward'] * 1000:.3f} |")
    lines += ["", "| Declared pair | Mean ΔdB | Conditional 95% interval | Improved / tied / worse |",
              "| --- | ---: | --- | --- |"]
    for pair in public["comparisons"]:
        low, high = pair["bootstrap_mean_95_percentile_ci_db"]
        lines.append(f"| {pair['candidate']} vs {pair['reference']} | {pair['mean_delta_db']:+.6f} | "
                     f"[{low:+.6f}, {high:+.6f}] | {pair['improved_images']} / {pair['tied_images']} / {pair['worse_images']} |")
    lines += ["", UNCERTAINTY, "", "Depth, positive trainable metrics, identity refinements, depth curriculum, optimizer schedule "
        "and intermediate loss form one combined recipe. This is not an isolated causal depth comparison. "
        "The 180px crops lack full nominal context at32/64steps; positive metrics do not prove convergence. "
        "Nominal stages do not establish useful denoising stages.", "", "Equal charged caps do not imply equal updates, "
        "full-depth exposure or checkpoint-selection opportunities. The recorded exposure and eligible validation counts accompany each model. "
        "The shared reaction warmstart and previously observed benchmark constrain interpretation.", "",
        "Runtime is synchronized whole-image forward work on the recorded shared GPU; transfers, clipping and scoring are excluded. "
        "This is not a modern state-of-the-art, stability guarantee or mobile-deployment claim. Prior releases and iOS models remain unchanged.", ""]
    accounting = public["compute_accounting"]
    if accounting["available"]:
        lines += [f"Aggregate charged process wall time: {accounting['consumed_process_wall_hours']:.6f} / "
                  f"{accounting['authorized_process_wall_hours']:.6f} hours, including startup, failures and evaluation; not GPU busy time.", ""]
    return "\n".join(lines)


def build(selection_path, results_path, results_sha, status_path, exports_path, output, private_evaluator_path, budget_path=None):
    selection, raw, selection_sha, provenance = verify_results(selection_path, results_path, results_sha, status_path, private_evaluator_path)
    assets = verify_exports(exports_path, selection)
    public = {"schema": replay.SCHEMA, "purpose": "bsd68", "complete": True, "image_count": 68,
        "evaluation_scope": replay.SCOPE, "sigma": 25, "offset": 25, "protocol": legacy.PROTOCOL,
        "selection_sha256": selection_sha, "results_sha256": results_sha,
        "private_evaluator_sha256": selection["evaluator_sha256"], "protocol_helper_sha256": selection["protocol_helper_sha256"],
        "publication_tool_sha256": protocol.file_hash(__file__), "source_pins": selection["source_pins"],
        "selection_created_utc": selection["created_utc"], "evaluation_started_utc": raw["started_utc"],
        "evaluation_ended_utc": raw["ended_utc"], "dataset_provenance": provenance,
        "environment": legacy.public_environment(raw["environment"]),
        **{key: selection[key] for key in ("timing_warmup", "timing_repeats", "bootstrap_seed", "bootstrap_samples")},
        "uncertainty_scope": UNCERTAINTY, "compute_accounting": legacy.compute_accounting(budget_path), "models": [], "comparisons": []}
    if budget_path is not None:
        ledger = read(budget_path)
        require(ledger.get("status") == "idle" and ledger["authorized_gpu_seconds"] == 43200,
                "Final accounting requires the existing idle12hledger")
    image_rows, timing_rows, pair_rows = [], [], []
    for entry, result, asset in zip(selection["models"], raw["models"], assets):
        model = {"name": entry["name"], "role": entry["role"], "parameters": entry["parameters"],
                 "training_step": entry["training_step"], "implementation": asset["metadata"]["implementation"],
                 "model_config": asset["metadata"]["model_config"], "asset_sha256": asset["sha256"]}
        if entry["kind"] == "rollout-checkpoint":
            config = read(entry["config"])
            require(type(config["seed"]) is int and config["microbatch"] > 0 and config["accumulation"] > 0,
                    "Invalid training exposure settings")
            for key in ("full_depth_updates", "phase_updates", "eligible_validation_records", "terminal_step", "terminal_phase_updates",
                        "charged_seconds", "training_gpu_seconds_at_best", "checkpoint_sha256", "config_sha256"):
                model[key] = entry[key]
            batch = config["microbatch"] * config["accumulation"]
            model.update(training_seed=config["seed"], training_patch_size=config["patch_size"], effective_batch=batch,
                         training_images_seen=entry["training_step"] * batch,
                         full_depth_images_seen=entry["full_depth_updates"] * batch)
        model.update({key: result[key] for key in ("mean_psnr", "median_seconds_forward", "sum_seconds_forward")})
        model["images"] = [{key: image[key] for key in ("name", "shape_native", "noisy_native_sha256", "label_native_sha256",
            "mse", "psnr", "clipped_pixels", "seconds_forward", "median_seconds_forward")} for image in result["images"]]
        public["models"].append(model)
        for image in model["images"]:
            image_rows.append({"model": entry["name"], "role": entry["role"], **{key: image[key] for key in
                ("name", "noisy_native_sha256", "label_native_sha256", "mse", "psnr", "clipped_pixels", "median_seconds_forward")}})
            timing_rows.extend({"model": entry["name"], "role": entry["role"], "image": image["name"],
                "width": image["shape_native"][2], "height": image["shape_native"][3], "repeat": i + 1, "seconds_forward": seconds}
                for i, seconds in enumerate(image["seconds_forward"]))
    require(len({x["training_seed"] for x in public["models"] if "training_seed" in x}) == 1, "Shared single-seed statement differs")
    roles = {x["name"]: x["role"] for x in public["models"]}
    by_name = {x["name"]: x for x in public["models"]}
    for pair in raw["comparisons"]:
        public["comparisons"].append({"candidate": pair["candidate"], "reference": pair["reference"],
            "candidate_role": roles[pair["candidate"]], "reference_role": roles[pair["reference"]],
            **protocol.paired_summary(by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"],
                selection["bootstrap_seed"], selection["bootstrap_samples"])})
        pair_rows.extend({"candidate": pair["candidate"], "reference": pair["reference"], "image": left["name"],
            "noisy_native_sha256": left["noisy_native_sha256"], "label_native_sha256": left["label_native_sha256"],
            "candidate_psnr": left["psnr"], "reference_psnr": right["psnr"], "delta_db": left["psnr"] - right["psnr"]}
            for left, right in zip(by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"]))
    require(protocol.file_hash(results_path) == results_sha and protocol.file_hash(selection_path) == selection_sha,
            "Evidence changed during publication verification")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save("results.json", public)
    save("reference-results.json", {k: v for k, v in public.items() if k != "compute_accounting"})
    manifest = {key: public[key] for key in ("schema", "purpose", "evaluation_scope", "sigma", "offset", "selection_sha256",
        "results_sha256", "private_evaluator_sha256", "protocol_helper_sha256", "source_pins", "selection_created_utc",
        "timing_warmup", "timing_repeats", "bootstrap_seed", "bootstrap_samples")}
    manifest.update(assets=assets, comparisons=replay.COMPARISONS,
        lease_exclusions=[*selection["budget_at_freeze"]["existing_run_ids"], raw["lease_id"]],
        expected_results={"filename": "reference-results.json", "sha256": protocol.file_hash(output / "reference-results.json")})
    replay.validate_cohort(manifest)
    save("models.json", manifest)
    for name, rows in (("per-image.csv", image_rows), ("runtime.csv", timing_rows), ("paired-per-image.csv", pair_rows)):
        legacy.write_csv(output / name, list(rows[0]), rows)
    (output / "REPORT.md").write_text(markdown(public))
    save("files.json", {"schema": 1, "files": [{"name": path.name, "sha256": protocol.file_hash(path)} for path in sorted(output.iterdir())]})
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "results", "exports", "private-evaluator", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--budget", type=Path)
    args = parser.parse_args()
    result = build(args.selection, args.results, args.results_sha256, args.status or args.results.with_name("status.json"),
                   args.exports, args.output, args.private_evaluator, args.budget)
    print(json.dumps({"complete": True, "models": len(result["models"]), "images": result["image_count"], "benchmark_reused": True}))


if __name__ == "__main__":
    main()
