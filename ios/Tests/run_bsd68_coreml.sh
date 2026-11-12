#!/usr/bin/env bash
# Run with an existing Python environment; no packages are installed here.
set -euo pipefail

tests_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$tests_dir/../.." && pwd)"
python_bin="${PYTHON:-python3}"
models_dir="$repo_dir/ios/DeepBeltramiApp/ModelAssets"
build_dir="$repo_dir/.build-ios/bsd68-native"
work_dir=""
limit=""
sigmas="5,10,15,20,25,30,40,50"
cache_dir=""
asset_dir=""
clean_dir=""

usage() {
    cat <<'HELP'
Usage: bash ios/Tests/run_bsd68_coreml.sh --work-dir PATH [options]

Requires macOS 15+, Swift command-line tools, and an existing Python environment
with this project installed. Fetch the Core ML models first with fetch_models.py.
Default: all 544 original cases, FP32, CPU+GPU, eight published mean checks.
Prefer SSD storage: about 380 MB prepared arrays, plus 324 MB if noise is uncached.

  --work-dir PATH   Required; fixtures, cache, logs and results live here.
  --build-dir PATH  Native filesystem build directory; default .build-ios/bsd68-native.
  --python PATH     Existing Python interpreter (default: PYTHON or python3).
  --models PATH     ModelAssets directory; defaults to the app's ModelAssets.
  --limit N         Images per sigma for a smoke run; default all 68.
  --sigmas LIST     Comma-separated sigmas; default all eight.
  --asset-dir PATH  Existing pinned release assets; avoids noise redownload.
  --cache-dir PATH  Existing reproduction cache; default WORK_DIR/cache.
  --clean-dir PATH  Existing pinned BSD68 PNGs.
HELP
}

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --work-dir|--build-dir|--python|--models|--limit|--sigmas|--asset-dir|--cache-dir|--clean-dir)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --work-dir) work_dir="$2" ;; --python) python_bin="$2" ;;
                --build-dir) build_dir="$2" ;;
                --models) models_dir="$2" ;; --limit) limit="$2" ;;
                --sigmas) sigmas="$2" ;; --asset-dir) asset_dir="$2" ;;
                --cache-dir) cache_dir="$2" ;; --clean-dir) clean_dir="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$work_dir" ]] || { usage >&2; exit 2; }
[[ "$(uname -s)" == Darwin ]] || { echo "This runner requires macOS 15 or later." >&2; exit 2; }
os_version="$(sw_vers -productVersion)"
(( ${os_version%%.*} >= 15 )) || { echo "This runner requires macOS 15 or later." >&2; exit 2; }
command -v swiftc >/dev/null || { echo "Install the Swift command-line tools first." >&2; exit 2; }
"$python_bin" -c 'import deep_patch_beltrami, h5py, numpy' || {
    echo "Choose --python from an environment with this project and its dependencies installed." >&2
    exit 2
}
[[ -d "$models_dir" ]] || { echo "ModelAssets is missing; run fetch_models.py first." >&2; exit 2; }
mkdir -p "$work_dir" "$build_dir"

prepare=("$tests_dir/prepare_bsd68_coreml.py" --work-dir "$work_dir" --sigmas "$sigmas")
[[ -z "$limit" ]] || prepare+=(--limit "$limit")
[[ -z "$asset_dir" ]] || prepare+=(--asset-dir "$asset_dir")
[[ -z "$cache_dir" ]] || prepare+=(--cache-dir "$cache_dir")
[[ -z "$clean_dir" ]] || prepare+=(--clean-dir "$clean_dir")
"$python_bin" "${prepare[@]}" | tee "$work_dir/preparation.log"

architecture="$(uname -m)"
binary="$build_dir/run-bsd68-coreml"
if ! swiftc -O -parse-as-library -target "$architecture-apple-macosx15.0" \
    "$repo_dir/ios/DeepBeltramiCoreML/DeepBeltramiCoreML.swift" \
    "$tests_dir/RunBSD68CoreML.swift" -o "$binary" > "$work_dir/swift-build.log" 2>&1; then
    cat "$work_dir/swift-build.log" >&2
    exit 1
fi
run=(--models "$models_dir" --fixtures "$work_dir/fixtures" \
     --output "$work_dir/bsd68-coreml.json" --sigmas "$sigmas")
[[ -z "$limit" ]] || run+=(--limit "$limit")
"$binary" "${run[@]}" | tee "$work_dir/bsd68-coreml.log"
