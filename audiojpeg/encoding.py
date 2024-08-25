from typing import Literal

import numpy as np
from PIL import Image
from scipy.io import wavfile

from audiojpeg.metadata import Metadata


def encode_wav_to_image(
    fp: str, width: int = 512, dynamic_range: int = 100, order: Literal["C", "F"] = "C"
) -> np.ndarray:
    """
    Encode a stereo WAV file into an RGB image.
    """
    sample_rate, samples = wavfile.read(fp)

    # Convert mono to stereo
    if samples.ndim == 1:
        samples = np.tile(samples, (2, 1))

    # Flatten the stereo channels into a single array
    samples = samples.ravel()

    # Clip the dynamic range and rescale to byte
    amplitude_min = np.percentile(samples, 100 - dynamic_range).astype(np.int16)
    amplitude_max = np.percentile(samples, dynamic_range).astype(np.int16)
    rescaled_array = (
        (samples - amplitude_min) / (amplitude_max - amplitude_min) * 255
    ).astype(np.uint8)

    # Calculate the minimum image height
    n_channels = 3
    n_samples = len(samples)
    height = np.ceil(n_samples / (width * n_channels)).astype(int)

    # Pad with zeros to allow reshaping to a rectangular image array
    pad_samples = int(width * height * n_channels - n_samples)
    padded_array = np.pad(rescaled_array, (0, pad_samples))

    # Reshape from raveled samples to RGB image
    reshaped_array = padded_array.reshape(height, width, n_channels, order=order)
    meta = Metadata(
        sample_rate=sample_rate,
        amplitude_max=amplitude_max,
        amplitude_range=amplitude_max - amplitude_min,
        pad_samples=pad_samples,
        order=order,
    )

    # Add metadata header row to the top
    reshaped_array = np.insert(
        reshaped_array,
        0,
        values=np.tile(meta.to_header(width=width), (3, 1)).T,
        axis=0,
    )

    return reshaped_array.astype(np.uint8)


def decode_image_to_audio(fp: str) -> np.ndarray:
    """
    Decode an image into a stereo array of samples.
    """
    image_with_header = np.array(Image.open(fp))

    image = image_with_header[1:, :, :]

    # Extract the metadata header and average values across bands for error correction
    header = image_with_header[0, :, :].mean(axis=-1).astype(np.uint8)
    meta = Metadata.from_header(header)

    # Rescale to the original amplitude range
    amplitude_min = meta.amplitude_max - meta.amplitude_range
    image = image / 255 * (meta.amplitude_max - amplitude_min) + amplitude_min
    samples = image.ravel(order=meta.order)

    # Remove padding
    if meta.pad_samples > 0:
        samples = samples[: -meta.pad_samples]

    # Reshape to stereo samples
    return samples.reshape((2, -1))
