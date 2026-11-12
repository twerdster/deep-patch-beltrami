import hashlib
import numpy as np
import pytest
import torch
from beltrami_operator.data import PatchSampler, validation_pairs
from beltrami_operator.train import evaluate


def record(image):
    return {"path": "fixture.png", "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


def test_fresh_noise_and_exact_sampler_resume():
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    sampler = PatchSampler([image], 16, 281)
    first, _ = sampler.batch(2, 25, 25)
    state = sampler.rng.bit_generator.state
    second, second_clean = sampler.batch(2, 25, 25)
    assert not torch.equal(first, second)
    sampler.rng.bit_generator.state = state
    resumed, resumed_clean = sampler.batch(2, 25, 25)
    assert torch.equal(resumed, second)
    assert torch.equal(resumed_clean, second_clean)


def test_validation_noise_is_stable_and_axes_are_native():
    image = np.arange(11 * 13, dtype=np.uint8).reshape(11, 13)
    first = list(validation_pairs([image], [record(image)], 25, 17, 25))[0]
    second = list(validation_pairs([image], [record(image)], 25, 17, 25))[0]
    assert first[0].shape == (1, 1, 13, 11)
    assert torch.equal(first[0], second[0])
    np.testing.assert_array_equal(first[1][0, 0], image.T)


def test_validation_rejects_infinity_before_clipping():
    class Infinite(torch.nn.Module):
        def forward(self, value):
            return torch.full_like(value, float("inf"))
    image = np.ones((11, 13), dtype=np.uint8)
    with pytest.raises(FloatingPointError, match="Non-finite validation"):
        evaluate(Infinite(), [image], [record(image)],
                 {"sigma": 25, "validation_seed": 17, "offset": 25}, torch.device("cpu"))
