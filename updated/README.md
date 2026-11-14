# Deep Patch-Beltrami updates

## Updated and ported for modern hardware

A new PyTorch executor runs the original five-stage Caffe deployment graphs
and learned tensors, with commands that reconstruct and check the archived
BSD68 inputs without a Caffe or MATLAB runtime. A separate SwiftUI app runs
converted Core ML models, including qualified mixed-precision and tiling paths,
with native Mac and iPhone checks. Later learned-operator studies add separately
labeled trained models; they do not replace the historical checkpoints or the
models used by the iPhone app.

### Code and setup

- [Python graph executor](../src/deep_patch_beltrami/inference.py),
  [quick start](../README.md#quick-start) and
  [reproduction guide](../docs/reproduction.md).
- [iOS installation and precision modes](../ios/README.md) and
  [Core ML engine](../ios/DeepBeltramiCoreML/).
- [Research package](../research/README.md),
  [operator-study results](../research/docs/RESULTS.md) and
  [32/64-stage study](../research/docs/LONG_ROLLOUTS.md): separate models,
  reproduction commands and protocol limitations.

### Validation and limits

The [Python tests](../tests/),
[saved BSD68 result](../docs/results/bsd68.json),
[native validation](../ios/VALIDATION.md) and
[iPhone record](../ios/benchmarks/iphone12.json) establish different scopes.
The full native Mac FP32 check is distinct from phone mixed-precision examples;
matching displayed PSNR does not imply identical output pixels across backends.
[Reproducibility notes](../docs/REPRODUCIBILITY.md) preserve the original
40,000-step checkpoints, intensity offsets and historical training discrepancies.
The research studies use a reused benchmark and do not establish modern
state-of-the-art performance.

[Project reference, citation and licensing](../README.md)
