# Attribution and external materials

The [MIT license](LICENSE) covers this repository's Deep Patch-Beltrami code and documentation. It does not relicense benchmark photographs, external datasets, papers or third-party software.

## Deep Patch-Beltrami research and artifacts

The method is described in Aaron Wetzler's 2017 PhD thesis, *Topics in Computational Geometric Vision*, Chapter 4, especially Section 4.5 and Tables 4.5–4.6. Cite the thesis when using the learned Deep Patch-Beltrami model. The research was carried out with collaborators under Ron Kimmel's supervision; the released model preserves the author's trained parameters rather than retraining them.

The companion [release](https://github.com/twerdster/deep-patch-beltrami/releases/tag/v1.0.0) provides eight author-trained checkpoints and five deployment graphs. Its model archive includes the full MIT license. The files are inference-only Caffe-format protobufs: training metadata was removed, while every learned tensor's shape and float32 payload bytes were verified against the original checkpoint. Original and released file hashes are retained in the model manifest.

This repository does not redistribute the Caffe implementation, generated Caffe protobuf code, C++ source or binaries. Its Python reader implements the small file-format subset required to read the released tensors; PyTorch executes the deployment graphs.

The second release asset contains author-generated Gaussian residuals. These noise-only arrays reconstruct the historical noisy inputs when combined with the matching separately downloaded clean images. Neither clean images nor noisy photographs are included in the asset. See [the reproduction guide](docs/reproduction.md) and [protocol limits](docs/REPRODUCIBILITY.md).

## BSD68 images

The BSD68 test set derives from the Berkeley Segmentation Dataset. Its photographs are downloaded separately and checked against pinned file and decoded-array hashes. They are not included in the source repository or companion release assets and are not covered by the code's MIT license.

The [Berkeley dataset page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) describes its non-commercial research and educational download terms and requests citation of David Martin, Charless Fowlkes, Doron Tal and Jitendra Malik, *A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics*, ICCV 2001.

The exact PNG versions used by the public fetcher are obtained from the BSD68 directory in [SCUNet commit 52e440a80a655b01e0b41e9dd9bfe599bc11625e](https://github.com/cszn/SCUNet/tree/52e440a80a655b01e0b41e9dd9bfe599bc11625e/testsets/bsd68). This is an image provenance link, not inclusion of SCUNet's source code or pretrained models, nor a new grant of rights to the photographs. The upstream code license does not replace the underlying dataset terms.

Portrait images from the historical archive are not redistributed. Portrait experiments require separately supplied data under the applicable terms. No thesis PDF, private manuscript source, correspondence, backup inventory or personal filesystem metadata is included.

## Runtime dependencies

NumPy, Pillow, h5py and PyTorch are installed as dependencies rather than copied into the source tree. Their original licenses and notices remain in the installed distributions. Installing this package does not relicense those projects or their optional components.
