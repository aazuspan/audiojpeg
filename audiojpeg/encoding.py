from typing import Literal

import numpy as np
from PIL import Image
from scipy.io import wavfile

from audiojpeg.metadata import Metadata


def encode_wav_to_image(
    fp: str, width: int = 512, dynamic_range: int = 100, order: Literal["C", "F"] = "C"
) -> np.ndarray:
    """
    Encode a WAV file into an RGB image.
    """
    sample_rate, samples = wavfile.read(fp)

    # Add a channel dimension if the samples are mono
    if samples.ndim == 1:
        samples = np.expand_dims(samples, axis=1)

    samples_per_channel, channels = samples.shape
    n_samples = channels * samples_per_channel

    # Flatten audio channels
    samples = samples.ravel()

    # Clip the dynamic range and rescale to byte
    amplitude_min = int(np.percentile(samples, 100 - dynamic_range))
    amplitude_max = int(np.percentile(samples, dynamic_range))
    rescaled_array = (
        (samples - amplitude_min) / (amplitude_max - amplitude_min) * 255
    ).astype(np.uint8)

    # Calculate the minimum image height
    n_bands = 3
    height = np.ceil(n_samples / (width * n_bands)).astype(int)

    # Pad with zeros to allow reshaping to a rectangular image array
    pad_samples = int(width * height * n_bands - n_samples)
    padded_array = np.pad(rescaled_array, (0, pad_samples))

    # Reshape from raveled samples to RGB image
    reshaped_array = padded_array.reshape(height, width, n_bands, order=order)
    meta = Metadata(
        sample_rate=sample_rate,
        channels=channels,
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
    Decode an image into an array of samples.
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

    # Reshape to multi-channel samples
    if meta.channels > 1:
        samples = samples.reshape((-1, meta.channels))

    return samples
