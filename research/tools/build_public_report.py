"""Verify completed frozen BSD68 evidence and emit a path-free publication bundle."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_evaluation_protocol", ROOT / "tools/evaluate_finalists.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
EXPORT_KEYS = {"format", "implementation", "source_sha256", "model_config", "sigma",
               "grayscale_offset", "state_dict", "tensor_sha256", "parameters", "training_step", "provenance"}
PROVENANCE_KEYS = {"training_checkpoint_sha256", "training_config_sha256", "training_manifest_sha256",
                   "validation_manifest_sha256", "original_checkpoint_sha256", "license", "citation"}
UNCERTAINTY = ("95% paired-image percentile bootstrap intervals condition on the frozen checkpoints, "
               "one fixed noise realization and one training seed. They exclude training-seed, "
               "noise-draw and model-selection uncertainty.")
PROTOCOL = {"input_axes": "N,C,W,H", "input_scale": "0-255 float32, unclipped",
            "output": "subtract fixed offset; reject nonfinite; clip float [0,255]",
            "mse_dtype": "float64", "peak": 255, "crop": 0,
            "augmentation": "none", "aggregation": "mean per-image PSNR in dB"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def same(actual, expected, description):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), description)
        for key, value in expected.items():
            same(actual[key], value, description)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), description)
        for a, b in zip(actual, expected):
            same(a, b, description)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), description)
    else:
        require(actual == expected, description)


def public_environment(environment):
    require(all(environment.get(key) is not None for key in ("python", "torch", "numpy")), "Missing runtime software versions")
    result = {}
    for key in ("python", "torch", "numpy", "cuda"):
        value = environment.get(key)
        if value is not None:
            match = re.match(r"\d+\.\d+(?:\.\d+)?(?:[A-Za-z0-9.+-]*)", str(value))
            require(match is not None, f"Invalid public {key} version")
            result[key] = match.group()
    for key in ("device", "device_name"):
        value = environment[key]
        require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ()+_.:-]{0,99}", value),
                f"Invalid public {key} label")
        result[key] = value
    for key in ("cpu_threads", "max_cuda_gib", "deterministic_cudnn", "tf32"):
        if key in environment:
            require(isinstance(environment[key], (int, float, bool)) and math.isfinite(environment[key]),
                    f"Invalid runtime setting: {key}")
            result[key] = environment[key]
    return result


def verify_results(selection_path, results_path, expected_results_sha, status_path):
    require(bool(HEX.fullmatch(expected_results_sha)), "Expected results SHA256 is required")
    require(evaluation.file_hash(results_path) == expected_results_sha, "Results file hash mismatch")
    selection, selection_sha = evaluation.read_frozen_selection(selection_path)
    report, status = read(results_path), read(status_path)
    require(selection["purpose"] == "bsd68" and report.get("dataset") == "bsd68", "Require final BSD68 purpose")
    require(report.get("schema") == 1 and report.get("complete") is True and not report.get("failure"),
            "Final evaluation is incomplete or failed")
    require(report.get("selection_sha256") == selection_sha and report.get("selection") == selection,
            "Results selection identity mismatch")
    require(selection["protocol"] == PROTOCOL, "Frozen scoring protocol differs")
    require(all(type(selection[key]) is int and selection[key] >= minimum for key, minimum in
                (("timing_warmup", 2), ("timing_repeats", 3), ("bootstrap_samples", 1000), ("bootstrap_seed", 0))),
            "Invalid predeclared timing/bootstrap settings")
    require(status.get("state") == "complete" and status.get("images") == 68
            and status.get("models") == len(selection["models"]), "Terminal evaluation status is incomplete")
    require(datetime.fromisoformat(selection["created_utc"]) <= datetime.fromisoformat(report["started_utc"])
            <= datetime.fromisoformat(report["ended_utc"]), "Selection/evaluation timestamp order differs")
    require(selection["evaluator_sha256"] == evaluation.file_hash(evaluation.__file__), "Evaluator hash mismatch")
    evaluation.verify_implementation_sources(selection)
    evaluation.verify_artifacts(selection)
    from deep_patch_beltrami.assets import metadata
    manifest = metadata("bsd68.json")
    provenance = {"loader": "deep_patch_beltrami.reproduce.PublicBSD68",
                  "public_bsd68_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  "noise_asset_sha256": metadata("release-assets.json")["assets"]["bsd68_noise"]["sha256"]}
    require(report.get("dataset_provenance") == provenance, "Public input provenance differs")
    require(report.get("image_count") == 68, "Require all 68 images")
    expected_images = {item["name"]: item for item in manifest["images"]}
    cases = {item["name"]: item for item in manifest["cases"] if item["sigma"] == selection["sigma"]}
    entries = selection["models"]
    names = [item["name"] for item in entries]
    require(len(names) == len(set(names)) and all(TOKEN.fullmatch(name) for name in names), "Invalid model names")
    require([item["role"] for item in entries].count("baseline") == 1
            and {"control", "candidate"} <= {item["role"] for item in entries}
            and all(item["role"] in ("baseline", "control", "candidate") for item in entries), "Missing explicit model roles")
    baseline = next(item for item in entries if item["role"] == "baseline")
    original_names = [f"PB_{selection['sigma']}_80_5__iter_40000.caffemodel"] + [
        f"stage_{index:03d}_deploy.prototxt" for index in range(1, 6)]
    pinned = {item["name"]: item["sha256"] for item in metadata("models.json")["files"]}
    require(baseline["kind"] == "historical" and len(baseline["files"]) == 6
            and {Path(item["path"]).name: item["sha256"] for item in baseline["files"]}
            == {name: pinned[name] for name in original_names}, "Historical checkpoint file pins differ")
    require([item["name"] for item in report["models"]] == names, "Result model order/set differs")
    for entry, model in zip(entries, report["models"]):
        require(model.get("complete") is True and model["role"] == entry["role"]
                and model["parameters"] == entry["parameters"], "Model status/role/parameter mismatch")
        require([item["name"] for item in model["images"]] == sorted(expected_images), "Image order/set differs")
        for image in model["images"]:
            name = image["name"]
            require(image["noisy_native_sha256"] == cases[name]["noisy_native_sha256"]
                    and image["label_native_sha256"] == expected_images[name]["label_native_sha256"]
                    and image["shape_native"] == expected_images[name]["shape"], "Per-image input hash/shape mismatch")
            require(isinstance(image["mse"], (int, float)) and math.isfinite(image["mse"]) and image["mse"] > 0,
                    "Invalid finite image MSE")
            require(image.get("perfect_prediction") is False and type(image["clipped_pixels"]) is int
                    and 0 <= image["clipped_pixels"] <= math.prod(image["shape_native"]), "Invalid clipping/finite evidence")
            same(image["psnr"], 10 * math.log10(255 ** 2 / image["mse"]), "Image PSNR/MSE mismatch")
            times = image["seconds_forward"]
            require(len(times) == selection["timing_repeats"] and all(math.isfinite(t) and t > 0 for t in times),
                    "Invalid timing repeats")
            same(image["median_seconds_forward"], statistics.median(times), "Image timing summary mismatch")
        same(model["mean_psnr"], statistics.fmean(item["psnr"] for item in model["images"]), "Model mean mismatch")
        same(model["median_seconds_forward"], statistics.median(item["median_seconds_forward"] for item in model["images"]),
             "Model timing summary mismatch")
        same(model["sum_seconds_forward"], sum(sum(item["seconds_forward"]) for item in model["images"]),
             "Model timing total mismatch")
    require([{key: pair[key] for key in ("candidate", "reference")} for pair in report["comparisons"]]
            == selection["comparisons"] and bool(selection["comparisons"]), "Comparison pairs/order differ from frozen selection")
    by_name = {item["name"]: item for item in report["models"]}
    for pair in report["comparisons"]:
        expected = evaluation.paired_summary(by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"],
                                             selection["bootstrap_seed"], selection["bootstrap_samples"])
        same({key: pair[key] for key in expected}, expected, "Paired comparison/bootstrap mismatch")
    return selection, report, selection_sha, provenance


def verify_exports(spec_path, selection):
    from beltrami_research import artifacts
    spec = read(spec_path)
    require(spec.get("schema") == 1, "Unsupported export specification")
    entries = {item["name"]: item for item in selection["models"] if item["kind"] == "checkpoint"}
    require([item["model"] for item in spec["assets"]] == list(entries),
            "Supply every frozen checkpoint export in selection order; no implicit asset/model selection")
    filenames, verified = set(), []
    for asset in spec["assets"]:
        entry = entries[asset["model"]]
        require(TOKEN.fullmatch(asset["filename"]) and asset["filename"].endswith(".pt")
                and asset["filename"] not in filenames, "Invalid/duplicate release filename")
        filenames.add(asset["filename"])
        url = urlsplit(asset["url"])
        require(url.scheme == "https" and url.hostname == "github.com" and not url.username
                and not url.password and not url.query and not url.fragment
                and "/releases/download/" in url.path and Path(url.path).name == asset["filename"], "Invalid release URL")
        path = Path(asset["path"])
        path = path if path.is_absolute() else Path(spec_path).resolve().parent / path
        digest = evaluation.file_hash(path)
        model, value = artifacts.load(path, "cpu")
        require(set(value) == EXPORT_KEYS and set(value["provenance"]) == PROVENANCE_KEYS, "Export contains extra metadata")
        require(value["implementation"] == entry.get("implementation", "beltrami_operator")
                and value["source_sha256"] == entry["training_source_sha256"], "Export source identity differs")
        _, Config = artifacts.classes(value["implementation"])
        same(value["model_config"], asdict(Config(**entry["model_config"])), "Export architecture differs")
        for key, item in value["model_config"].items():
            if isinstance(item, str):
                require(item in ("frozen", "positive", "unconstrained", "float32", "float64"), "Invalid public model setting")
            else:
                require(isinstance(item, (int, float, bool)) and math.isfinite(item), "Invalid public model setting")
        require(value["sigma"] == selection["sigma"] and value["grayscale_offset"] == selection["offset"]
                and value["parameters"] == entry["parameters"] == sum(p.numel() for p in model.parameters())
                and value["training_step"] == entry["training_step"],
                "Export sigma/offset/parameter/step differs")
        provenance = value["provenance"]
        for key, expected in {"training_checkpoint_sha256": entry["checkpoint_sha256"],
                              "training_config_sha256": entry["config_sha256"],
                              "training_manifest_sha256": selection["training_manifest_sha256"],
                              "validation_manifest_sha256": selection["validation_manifest_sha256"],
                              "original_checkpoint_sha256": entry["initialization"]["source_sha256"]}.items():
            require(provenance[key] == expected, "Export checkpoint/data provenance differs")
        # Frozen training checkpoints are trusted maintainer artifacts. Public inference never reads them.
        checkpoint = evaluation.load_checkpoint(entry["checkpoint"])
        config = read(entry["config"])
        evaluation.validate_checkpoint_identity(checkpoint, config, entry["config_sha256"], selection["sigma"],
            selection["offset"], value["source_sha256"], entry["initialization"]["source_sha256"],
            selection["training_manifest_sha256"], selection["validation_manifest_sha256"])
        require(artifacts.tensor_hash(checkpoint["model"]) == value["tensor_sha256"], "Exported tensor bytes differ from frozen checkpoint")
        require(provenance["license"] == "MIT" and provenance["citation"] ==
                "Wetzler, Topics in Computational Geometric Vision, 2017, Chapter 4 Section 4.5", "Export license/citation differs")
        require(evaluation.file_hash(path) == digest, "Export changed during verification")
        verified.append({"model": entry["name"], "role": entry["role"], "filename": asset["filename"],
                         "url": asset["url"], "bytes": path.stat().st_size, "sha256": digest,
                         "tensor_sha256": value["tensor_sha256"], "implementation": value["implementation"],
                         "source_sha256": value["source_sha256"], "model_config": value["model_config"],
                         "sigma": value["sigma"], "grayscale_offset": value["grayscale_offset"],
                         "parameters": value["parameters"], "training_step": value["training_step"], "provenance": provenance})
        del model, value, checkpoint
    return verified


def markdown(report):
    lines = [f"## Frozen BSD68 comparison (σ = {report['sigma']})", "",
             "| Model | Role | Parameters | New updates | Mean PSNR (dB) | Median forward (ms) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for model in report["models"]:
        updates = str(model["training_step"]) if model["kind"] == "checkpoint" else "historical 40,000"
        lines.append(f"| {model['name']} | {model['role']} | {model['parameters']:,} | {updates} | "
                     f"{model['mean_psnr']:.6f} | {model['median_seconds_forward'] * 1000:.3f} |")
    lines += ["", "| Declared comparison | Mean ΔPSNR (dB) | Conditional 95% interval | Improved / tied / worse |",
              "| --- | ---: | --- | --- |"]
    for pair in report["comparisons"]:
        lower, upper = pair["bootstrap_mean_95_percentile_ci_db"]
        lines.append(f"| {pair['candidate']} ({pair['candidate_role']}) vs {pair['reference']} ({pair['reference_role']}) | "
                     f"{pair['mean_delta_db']:+.6f} | [{lower:+.6f}, {upper:+.6f}] | "
                     f"{pair['improved_images']} / {pair['tied_images']} / {pair['worse_images']} |")
    lines += ["", UNCERTAINTY, "", "Runtime measures synchronized whole-image forward passes; transfers, clipping and scoring are excluded. "
              "It does not establish phone latency. Original reproduction/iOS checkpoints remain unchanged.", "",
              f"Measured on {report['environment']['device_name']} with PyTorch {report['environment']['torch']}; "
              f"{report['timing_warmup']} warmups and {report['timing_repeats']} timed repeats per image. "
              "New updates exclude the preserved 40,000-step pretraining; checkpoint time is not total job accounting.", "",
              "All models and comparison pairs follow the frozen selection; no post-test model ranking or selection is performed.", ""]
    accounting = report["compute_accounting"]
    if accounting["available"]:
        lines += [f"Aggregate recorded process wall time: {accounting['consumed_process_wall_hours']:.6f} hours "
                  f"within {accounting['authorized_process_wall_hours']:.6f} authorized hours, including startup, "
                  "evaluation and failures. This is not GPU busy time.", ""]
    return "\n".join(lines)


def write_csv(path, fields, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def compute_accounting(path):
    if path is None:
        return {"available": False, "note": "Checkpoint inner seconds do not establish aggregate compute consumption."}
    ledger = read(path)
    charges = [entry["charged_seconds"] for entry in ledger["entries"]]
    require(all(entry["state"] in ("complete", "failed") for entry in ledger["entries"])
            and all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in charges),
            "Compute ledger is active, unresolved or contains invalid charges")
    total, allowance = sum(charges), ledger["authorized_gpu_seconds"]
    require(isinstance(allowance, (int, float)) and math.isfinite(allowance) and 0 < allowance
            and total <= allowance, "Compute exceeds recorded aggregate allowance")
    same(ledger["consumed_gpu_seconds"], total, "Compute ledger total differs from lease charges")
    return {"available": True, "ledger_sha256": evaluation.file_hash(path),
            "authorized_process_wall_hours": allowance / 3600, "consumed_process_wall_hours": total / 3600,
            "leases": len(charges), "failed_leases": sum(entry["state"] == "failed" for entry in ledger["entries"]),
            "scope": "Conservative process wall time including startup, evaluation and failures; not GPU busy time."}


def validate_release_pairs(baseline, assets, comparisons):
    spec = importlib.util.spec_from_file_location("released_comparison_contract", ROOT / "tools/evaluate_released.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_comparisons({"baseline": baseline, "assets": assets, "comparisons": comparisons})


def build(selection_path, results_path, results_sha, status_path, exports_path, output, budget_path=None):
    selection, raw, selection_sha, dataset = verify_results(selection_path, results_path, results_sha, status_path)
    assets = verify_exports(exports_path, selection)
    baseline_entry = next(item for item in selection["models"] if item["role"] == "baseline")
    validate_release_pairs(baseline_entry, assets, selection["comparisons"])
    public = {"schema": 1, "purpose": "bsd68", "complete": True, "image_count": 68,
              "sigma": selection["sigma"], "offset": selection["offset"], "protocol": selection["protocol"],
              "selection_sha256": selection_sha, "results_sha256": results_sha,
              "evaluator_sha256": selection["evaluator_sha256"], "publication_tool_sha256": evaluation.file_hash(__file__),
              "training_manifest_sha256": selection["training_manifest_sha256"],
              "validation_manifest_sha256": selection["validation_manifest_sha256"],
              "selection_created_utc": selection["created_utc"], "evaluation_started_utc": raw["started_utc"],
              "evaluation_ended_utc": raw["ended_utc"], "dataset_provenance": dataset,
              "environment": public_environment(raw["environment"]), "timing_warmup": selection["timing_warmup"],
              "timing_repeats": selection["timing_repeats"], "uncertainty_scope": UNCERTAINTY,
              "compute_accounting": compute_accounting(budget_path),
              "models": [], "comparisons": []}
    image_rows, runtime_rows, paired_rows = [], [], []
    by_name = {item["name"]: item for item in raw["models"]}
    roles = {entry["name"]: entry["role"] for entry in selection["models"]}
    for entry, result in zip(selection["models"], raw["models"]):
        model = {key: entry[key] for key in ("name", "role", "kind", "parameters", "training_step")}
        if entry["kind"] == "checkpoint":
            config = read(entry["config"])
            require(type(config["seed"]) is int, "Training seed must be an integer")
            require(all(type(config[key]) is int and config[key] > 0 for key in ("batch_size", "patch_size"))
                    and math.isfinite(entry["training_gpu_seconds"]) and entry["training_gpu_seconds"] >= 0,
                    "Invalid training exposure/time")
            model.update(implementation=entry.get("implementation", "beltrami_operator"), model_config=entry["model_config"],
                         training_seed=config["seed"], checkpoint_inner_seconds=entry["training_gpu_seconds"],
                         training_batch_size=config["batch_size"], training_patch_size=config["patch_size"],
                         training_images_seen=entry["training_step"] * config["batch_size"],
                         training_pixels_seen=entry["training_step"] * config["batch_size"] * config["patch_size"] ** 2,
                         checkpoint_sha256=entry["checkpoint_sha256"], config_sha256=entry["config_sha256"],
                         source_sha256=entry["training_source_sha256"],
                         original_checkpoint_sha256=entry["initialization"]["source_sha256"])
        else:
            model["original_files"] = [{"name": Path(item["path"]).name, "sha256": item["sha256"]} for item in entry["files"]]
        model.update({key: result[key] for key in ("mean_psnr", "median_seconds_forward", "sum_seconds_forward")})
        model["images"] = [{key: image[key] for key in ("name", "shape_native", "noisy_native_sha256",
                            "label_native_sha256", "mse", "psnr", "seconds_forward", "median_seconds_forward", "clipped_pixels")}
                           for image in result["images"]]
        public["models"].append(model)
        for image in result["images"]:
            image_rows.append({"model": entry["name"], "role": entry["role"], **{key: image[key] for key in
                              ("name", "noisy_native_sha256", "label_native_sha256", "mse", "psnr", "clipped_pixels", "median_seconds_forward")}})
            for repeat, seconds in enumerate(image["seconds_forward"], 1):
                runtime_rows.append({"model": entry["name"], "role": entry["role"], "image": image["name"],
                                     "width": image["shape_native"][2], "height": image["shape_native"][3],
                                     "repeat": repeat, "seconds_forward": seconds})
    seeds = {model["training_seed"] for model in public["models"] if model["kind"] == "checkpoint"}
    require(len(seeds) == 1, "This report's uncertainty statement requires one shared training seed")
    for pair in raw["comparisons"]:
        expected = evaluation.paired_summary(by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"],
                                             selection["bootstrap_seed"], selection["bootstrap_samples"])
        public["comparisons"].append({"candidate": pair["candidate"], "candidate_role": roles[pair["candidate"]],
                                      "reference": pair["reference"], "reference_role": roles[pair["reference"]], **expected})
        for candidate, reference in zip(by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"]):
            paired_rows.append({"candidate": pair["candidate"], "candidate_role": roles[pair["candidate"]],
                                "reference": pair["reference"], "reference_role": roles[pair["reference"]],
                                "image": candidate["name"], "noisy_native_sha256": candidate["noisy_native_sha256"],
                                "label_native_sha256": candidate["label_native_sha256"], "candidate_psnr": candidate["psnr"],
                                "reference_psnr": reference["psnr"], "delta_db": candidate["psnr"] - reference["psnr"]})
    require(evaluation.file_hash(results_path) == results_sha
            and evaluation.file_hash(selection_path) == selection_sha, "Results/selection changed during publication verification")
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save("results.json", public)
    # Public replay must keep the same manifest when its final compute charge is added.
    save("reference-results.json", {key: value for key, value in public.items() if key != "compute_accounting"})
    baseline = next(item for item in public["models"] if item["role"] == "baseline")
    save("models.json", {"schema": 1, "purpose": "bsd68", "sigma": selection["sigma"], "offset": selection["offset"],
                          "selection_sha256": selection_sha, "results_sha256": results_sha,
                          "evaluator_sha256": selection["evaluator_sha256"], "license": "MIT",
                          "baseline": {**{key: baseline[key] for key in ("name", "role", "kind", "parameters", "training_step")},
                                       "files": baseline["original_files"]},
                          "assets": assets, "comparisons": selection["comparisons"],
                          **{key: selection[key] for key in ("timing_warmup", "timing_repeats", "bootstrap_seed", "bootstrap_samples")},
                          "expected_results": {"filename": "reference-results.json",
                                               "sha256": evaluation.file_hash(output / "reference-results.json")}})
    write_csv(output / "per-image.csv", list(image_rows[0]), image_rows)
    write_csv(output / "paired-per-image.csv", list(paired_rows[0]), paired_rows)
    write_csv(output / "runtime.csv", list(runtime_rows[0]), runtime_rows)
    table = markdown(public)
    (output / "README-snippet.md").write_text(table)
    (output / "REPORT.md").write_text("# Learned Beltrami operator comparison\n\n" + table +
        "\nThe selection/checkpoint/evaluator/input hashes are retained in `results.json`. "
        "Per-image scores, paired changes and timing repeats accompany this report. "
        "The code and author-trained parameters are MIT licensed; dataset photographs retain separate terms. "
        "Cite Aaron Wetzler, *Topics in Computational Geometric Vision*, Technion PhD thesis, 2017, Chapter 4 §4.5.\n")
    save("files.json", {"schema": 1, "files": [{"name": path.name, "sha256": evaluation.file_hash(path)}
                                                for path in sorted(output.iterdir())]})
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--status", type=Path, help="Default: status.json beside results")
    parser.add_argument("--exports", required=True, type=Path)
    parser.add_argument("--budget", type=Path, help="Optional completed aggregate ledger; only totals/hashes are published")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.selection, args.results, args.results_sha256,
                   args.status or args.results.with_name("status.json"), args.exports, args.output, args.budget)
    print(json.dumps({"complete": True, "models": len(report["models"]), "images": report["image_count"]}))


if __name__ == "__main__":
    main()
