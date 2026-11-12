#!/bin/bash
# Build the physical-device app with the installed iPhoneOS SDK.
set -euo pipefail

TASK_PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_APP_NAME="DeepBeltramiApp"
TASK_MODELS="$TASK_PROJECT_ROOT/ios/$TASK_APP_NAME/ModelAssets"
TASK_COMPILE_ONLY=false

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: TEAM_ID=YOUR_TEAM_ID ios/scripts/build_ios.sh [--compile-only]"
    echo "TEAM_ID signs for a physical device; without it, build unsigned for compilation checks."
    echo "--compile-only skips model preparation checks for an unsigned source/CI build."
    echo "Optional BUNDLE_ID overrides the application identifier for signing."
    echo "Prepare models first: python3 ios/scripts/fetch_models.py"
    exit 0
fi
if [[ "${1:-}" == "--compile-only" ]]; then
    TASK_COMPILE_ONLY=true
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Unexpected arguments. Use --help for build instructions." >&2
    exit 2
fi
if $TASK_COMPILE_ONLY && [[ -n "${TEAM_ID:-}" ]]; then
    echo "--compile-only is unsigned only. Omit TEAM_ID, or prepare the models and run a normal signed build." >&2
    exit 2
fi

if ! $TASK_COMPILE_ONLY && [[ ! -d "$TASK_MODELS/DeepBeltrami_sigma25_fp32.mlmodelc" ]]; then
    echo "Missing compiled sigma-25 FP32 model. Run python3 ios/scripts/fetch_models.py first." >&2
    exit 1
fi
if $TASK_COMPILE_ONLY; then
    mkdir -p "$TASK_MODELS" "$TASK_PROJECT_ROOT/ios/$TASK_APP_NAME/Fixtures"
fi

TASK_BUILD_ARGS=(
    -project "$TASK_PROJECT_ROOT/ios/$TASK_APP_NAME.xcodeproj"
    -target "$TASK_APP_NAME"
    -configuration Release
    -sdk iphoneos
    -arch arm64
    "SYMROOT=$TASK_PROJECT_ROOT/.build-ios/Build/Products"
    "OBJROOT=$TASK_PROJECT_ROOT/.build-ios/Build/Intermediates.noindex"
)
if [[ -n "${TEAM_ID:-}" ]]; then
    TASK_BUILD_ARGS+=("DEVELOPMENT_TEAM=$TEAM_ID" "CODE_SIGN_STYLE=Automatic" -allowProvisioningUpdates)
else
    TASK_BUILD_ARGS+=("CODE_SIGNING_ALLOWED=NO" "CODE_SIGNING_REQUIRED=NO" "CODE_SIGN_IDENTITY=")
fi

if [[ -n "${BUNDLE_ID:-}" ]]; then
    TASK_BUILD_ARGS+=("PRODUCT_BUNDLE_IDENTIFIER=$BUNDLE_ID")
fi

xcodebuild "${TASK_BUILD_ARGS[@]}" build
echo "Built $TASK_PROJECT_ROOT/.build-ios/Build/Products/Release-iphoneos/$TASK_APP_NAME.app"
if [[ -z "${TEAM_ID:-}" ]]; then
    echo "Unsigned compilation check. Set TEAM_ID to sign for a physical device."
fi
if $TASK_COMPILE_ONLY; then
    echo "Model preparation was not checked. A fresh-checkout bundle cannot denoise until models are fetched and the app is rebuilt."
fi
