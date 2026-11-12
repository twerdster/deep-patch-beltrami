# Training and validation provenance

The [Berkeley BSDS page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
describes the dataset's research/education terms and requested citation. The
repository's MIT license does not relicense photographs. Acquire data under
the applicable terms; images are downloaded separately, never bundled here.

## Frozen experiment inputs

| Manifest | Images | SHA256 of manifest bytes |
| --- | ---: | --- |
| `data/train400.json` | 400 | `f8a22e7177107990eff632eef8385a17e40398903bf2d6be0bc5a8cae448c8c0` |
| `data/validation32.json` | 32 | `27736aa4658ffbd9f072c8990b9d370ae8734431b70818edf5a04fbe4d59780e` |

Each record pins the PNG file, decoded uint8 grayscale pixels and physical H,W
shape. The trainer verifies file and pixel SHA256. Preserve the manifest bytes
when resuming these runs. Its per-record `split` is authoritative; the retained
`split_policy` description comes from a generic manifest generator and is not
the source-identity selection rule for this experiment.

All 400 recovered training crops were matched to 400 unique BSDS500 source
photographs: all 200 official training and all 200 official test images.
BSD68 maps to 68 unique official validation photographs. The remaining 32
official validation photographs form the independent selection set. These
three source-identity pools are disjoint. The evidence in
`manifests/source-identities.json` includes image identifiers, crop coordinates,
measured match quality and hashes, without photographs or machine paths.

Matching used downsampled candidate retrieval followed by full-resolution crop
normalized cross correlation. Minimum final training-crop NCC was
0.9999821186; maximum pixel MSE against canonical source crops was 0.0035494
on the 0–255 scale. Small historical grayscale rounding differences remain.
The 68 benchmark sources have maximum full-image grayscale MSE 0.00125 against
the separately pinned benchmark PNGs. This supports source-identity exclusion,
not byte identity or absence of near-duplicate photographic scenes. Historical
development exposure cannot be completely reconstructed.

## Exact training continuation

Exact continuation requires the author-recovered 180×180 grayscale PNG crops
matching `data/train400.json`, supplied independently under the applicable
dataset terms. The crop mapping does not provide a byte-exact reconstruction
recipe because historical pixel rounding is not completely recovered. A fresh
BSDS500 crop extraction is a different training-input protocol and needs a new
manifest. No automatic source download is represented as exact recreation of
the archived training inputs or their optimizer trajectory.

The original eight released 40,000-step models remain fully usable without any
training images; see the [main reproduction command](../../README.md).

## Validation reconstruction

`manifests/validation32-sources.json` pins the 32 upstream JPEG files in
[BIDS/BSDS500 revision a04b7c6](https://github.com/BIDS/BSDS500/tree/a04b7c6c3a9f0ace74bf205c72a43d32e1c72722).
The preparation command downloads just these sources, verifies SHA256, uses
Pillow RGB decoding followed by OpenCV BT.601 uint8 grayscale conversion, then
Pillow 13 PNG encoding. Both pixel and file digests must match the frozen
manifest. PNG compressor/decoder versions can change encoded bytes; matching
pixels alone does not bypass the frozen file check. A version mismatch fails
explicitly rather than rewriting the experiment's manifest.

Training noise is fresh at every sampled patch. Selection noise uses seed
742913 mixed with the decoded image hash for each image; a shared secondary
draw may be declared in advance for robustness checks. BSD68 retains the
released original noise residuals and its own independently pinned image data.

For alternative images, make separate explicit train/validation manifests.
Check source-level independence yourself: an image/crop hash overlap check
cannot detect every shared source photograph. Report the changed data and
exposure instead of calling that exact repetition of the frozen experiment.
