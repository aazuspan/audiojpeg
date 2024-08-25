from __future__ import annotations

import numpy as np


class Metadata:
    PARAMETER_WIDTHS = {
        "sampling_rate": 32,
        "amplitude_range": 16,
        "amplitude_max": 16,
        "pad_samples": 16,
    }

    def __init__(self, *, sampling_rate: int, amplitude_max: int, amplitude_range: int, pad_samples: int):
        self.sampling_rate = sampling_rate
        """The sampling rate of the audio signal, in hertz."""
        
        self.amplitude_max = amplitude_max
        """The maximum amplitude of the audio signal, prior to scaling."""

        # We store range instead of the min to avoid the complexity of encoding negative
        # numbers.        
        self.amplitude_range = amplitude_range
        """The range between the min and max amplitude, prior to scaling."""
        
        self.pad_samples = pad_samples
        """The number of samples padded onto the audio signal."""

        self._parameters = {
            name: (getattr(self, name), width) for name, width in self.PARAMETER_WIDTHS.items()
        }


    def _encode(self):
        encoded = []
        start = 0
        for value, width in self._parameters.values():
            encoded.append((value << start))
            start += width
        return sum(encoded)
    
    @classmethod
    def _decode(cls, encoded: int):
        decoded = {}
        start = 0
        for name, width in cls.PARAMETER_WIDTHS.items():
            mask = (1 << width) - 1
            value = (encoded >> start) & mask
            decoded[name] = value
            start += width

        return decoded
        

    def to_header(self, width: int=128) -> np.ndarray:
        """Encode the metadata into an image header array."""
        encoded_bits = np.array([int(bit) for bit in f"{self._encode():0{width}b}"]) * 255

        if len(encoded_bits) > width:
            raise ValueError(f"Width must be >= {len(encoded_bits)}")
        
        return encoded_bits


    @classmethod
    def from_header(cls, header: np.ndarray) -> Metadata:
        """Decode metadata from an image header array."""
        # Threshold the header bits to 0 or 1 based on their original 0 or 255 value.
        corrected_header = (header > 127).astype(int)
        encoded = int("".join(corrected_header.astype(str)), 2)
        decoded = cls._decode(encoded)

        return cls(**decoded)
