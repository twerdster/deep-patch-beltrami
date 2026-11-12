"""Synthetic forward-graph checks; no downloads, checkpoints or CUDA required."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deep_patch_beltrami.inference import CaffeGraph, DeepBeltrami


def _graph(tmp_path, text, weights):
    path = tmp_path / "synthetic.prototxt"
    path.write_text(text)
    return CaffeGraph(path, weights, device="cpu")


def _correlate(image, kernel, bias, stride=(1, 1), padding=(0, 0), groups=1):
    """Independent small NCHW correlation reference with explicit scalar sums."""
    image = np.pad(image, ((0, 0), (0, 0), (padding[0], padding[0]),
                           (padding[1], padding[1])), mode="constant")
    batch, _, height, width = image.shape
    outputs, inputs_per_group, kh, kw = kernel.shape
    result = np.empty((batch, outputs, (height-kh)//stride[0]+1,
                        (width-kw)//stride[1]+1), dtype=np.float32)
    for n in range(batch):
        for output in range(outputs):
            group = output // (outputs//groups)
            for row in range(result.shape[2]):
                for col in range(result.shape[3]):
                    value = np.float32(bias[output])
                    for channel in range(inputs_per_group):
                        source_channel = group*inputs_per_group + channel
                        for i in range(kh):
                            for j in range(kw):
                                value += (image[n, source_channel, row*stride[0]+i,
                                                col*stride[1]+j]
                                          * kernel[output, channel, i, j])
                    result[n, output, row, col] = value
    return result


@pytest.mark.parametrize("method", ["REPEAT", "CONSTANT"])
def test_spatial_graph_math_and_fanout(tmp_path, method):
    """Asymmetric borders, correlation axes, branch lifetime and nonlinear math."""
    graph = _graph(tmp_path, f'''
      layer {{ name: "border" type: "Padding" bottom: "data" top: "padded"
        padding_param {{ pad_l: 1 pad_r: 0 pad_t: 0 pad_b: 1
                          pad_method: {method} pad_value: 7 }} }}
      layer {{ name: "filter" type: "Convolution" bottom: "padded" top: "features"
        convolution_param {{ stride_h: 1 stride_w: 2 pad_h: 0 pad_w: 1 }} }}
      layer {{ name: "square" type: "Power" bottom: "features" top: "squared"
        power_param {{ scale: 0.5 shift: -1 power: 2 }} }}
      layer {{ name: "root" type: "Power" bottom: "features" top: "rooted"
        power_param {{ power: 0.5 }} }}
      layer {{ name: "product" type: "Eltwise" bottom: "squared" bottom: "rooted"
        top: "product" eltwise_param {{ operation: PROD }} }}
      layer {{ name: "combine" type: "Eltwise" bottom: "product" bottom: "features"
        top: "result" eltwise_param {{ operation: SUM coeff: 0.25 coeff: -0.75 }} }}
    ''', {"filter": [np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                      np.array([.5], dtype=np.float32)]})
    array = np.arange(1, 7, dtype=np.float32).reshape(1, 1, 2, 3) / 5
    tensor = torch.from_numpy(array.copy()).requires_grad_(True)
    actual = graph(tensor, debug=True)
    if method == "REPEAT":
        padded = np.pad(array, ((0, 0), (0, 0), (0, 1), (1, 0)), mode="edge")
    else:
        padded = np.pad(array, ((0, 0), (0, 0), (0, 1), (1, 0)),
                        mode="constant", constant_values=7)
    features = _correlate(padded, np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                          np.array([.5], dtype=np.float32), stride=(1, 2), padding=(0, 1))
    expected = .25*(.5*features-1)**2*np.sqrt(features) - .75*features
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-5, rtol=2e-6)
    np.testing.assert_array_equal(tensor.detach().numpy(), array)
    assert actual.shape == (1, 1, 2, 3) and not actual.requires_grad


def test_grouped_convolution_keeps_feature_groups_separate(tmp_path):
    kernel = np.array([[[[1, -2], [3, 4]]], [[[5, 6], [-7, 8]]]], dtype=np.float32)
    bias = np.array([.5, -1], dtype=np.float32)
    graph = _graph(tmp_path, '''
      layer { name: "grouped" type: "Convolution" bottom: "data" top: "result"
        convolution_param { group: 2 stride: 1 pad: 0 } }
    ''', {"grouped": [kernel, bias]})
    array = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4) / 7
    expected = _correlate(array, kernel, bias, groups=2)
    np.testing.assert_allclose(graph(torch.from_numpy(array)).numpy(), expected,
                               atol=1e-5, rtol=2e-6)


def test_model_channels_offsets_and_final_clipping(tmp_path):
    """A known one-layer model checks the wrapper without a trained checkpoint."""
    graph = _graph(tmp_path, '''
      layer { name: "affine" type: "Convolution" bottom: "data" top: "result"
        convolution_param { kernel_size: 1 } }
    ''', {"affine": [np.array([[[[2]]]], dtype=np.float32), np.array([3], dtype=np.float32)]})
    model = DeepBeltrami.__new__(DeepBeltrami)
    model.device, model.graphs = "cpu", [graph]
    image = np.array([-20, -1, 0, 1, 7, 300]*6, dtype=np.float32).reshape(2, 3, 2, 3)
    image[1, 2] = np.array([[9, 8, 7], [6, 5, 4]], dtype=np.float32)
    original = image.copy()
    actual = model(image, offset=5.5, debug=True)
    expected = np.clip(2*image + 8.5, 0, 255)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)
    assert actual.shape == image.shape and actual.dtype == np.float32


def test_debug_identifies_nonfinite_layer(tmp_path):
    graph = _graph(tmp_path, '''
      layer { name: "invalid_root" type: "Power" bottom: "data" top: "result"
        power_param { power: 0.5 } }
    ''', {})
    with pytest.raises(FloatingPointError, match="invalid_root"):
        graph(torch.tensor([[[[-1., 4.]]]], dtype=torch.float32), debug=True)
