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


def make_wav_file(tmp_dir, channels: int = 2, duration: int = 1, rate: int = 44_100):
    """
    Create a temporary stereo WAV file with parameterized sample rate and duration.
    """
    fp = tmp_dir / "test.wav"

    n_samples = int(duration * rate)
    signal = np.random.choice([-10_000, 10_000], size=(n_samples, channels)).astype(
        np.int16
    )

    wavfile.write(fp, rate=rate, data=signal)
    return fp


@given(
    st.integers(min_value=95, max_value=1024),
    st.sampled_from(["C", "F"]),
    st.sampled_from([1, 2, 6]),
    st.floats(min_value=0.1, max_value=2),
    st.integers(min_value=8_000, max_value=48_000),
)
def test_wav_to_image_roundtrip(tmp_dir, width, order, channels, duration, rate):
    wav_file = make_wav_file(tmp_dir, channels=channels, duration=duration, rate=rate)
    img_array = encode_wav_to_image(wav_file, width=width, order=order)

    img_path = tmp_dir / "test.png"
    Image.fromarray(img_array).save(img_path)
    audio = decode_image_to_audio(img_path)

    _, original = wavfile.read(wav_file)
    assert np.allclose(audio, original)
