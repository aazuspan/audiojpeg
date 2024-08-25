from pathlib import Path
import tempfile

from hypothesis import given, strategies as st
import pytest
import numpy as np
from PIL import Image
from scipy.io import wavfile

from audiojpeg.encoding import encode_wav_to_image, decode_image_to_audio


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="module")
def wav_file(tmp_dir):
    """
    Create a temporary stereo WAV file with parameterized sample rate and duration.
    """
    fp = tmp_dir / "test.wav"

    n_samples = 10_000
    signal = np.random.choice([-10_000, 10_000], size=(2, n_samples)).astype(np.int16)

    wavfile.write(fp, rate=44_100, data=signal)
    return fp


@given(
    st.integers(min_value=87, max_value=1024),
    st.sampled_from(["C", "F"]),
)
def test_wav_to_image_roundtrip(tmp_dir, wav_file, width, order):
    img_array = encode_wav_to_image(wav_file, width=width, order=order)

    img_path = tmp_dir / "test.png"
    Image.fromarray(img_array).save(img_path)
    audio = decode_image_to_audio(img_path)

    _, original = wavfile.read(wav_file)
    assert np.allclose(audio, original)
