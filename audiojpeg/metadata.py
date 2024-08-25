from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Type
import numpy as np


T = TypeVar("T")


class Parameter(ABC, Generic[T]):
    width: int

    def __new__(cls, value: T, width: int):
        obj = super().__new__(cls, value)
        obj.width = width
        return obj

    @abstractmethod
    def encode(self) -> int:
        """Encode the parameter value to an integer."""
        ...

    @staticmethod
    @abstractmethod
    def decode(value: int) -> T:
        """Decode the parameter value from an integer"""
        ...


class IntParameter(Parameter[int], int):
    def encode(self) -> int:
        return self

    @staticmethod
    def decode(value: int) -> int:
        return value


class StrParameter(Parameter[str], str):
    def encode(self) -> int:
        return ord(self)

    @staticmethod
    def decode(value: int) -> str:
        return chr(value)


@dataclass
class ParameterSpec:
    width: int
    type: Type[Parameter]


@dataclass
class Metadata:
    PARAMETER_SPECS = {
        "sample_rate": ParameterSpec(32, IntParameter),
        "amplitude_range": ParameterSpec(16, IntParameter),
        "amplitude_max": ParameterSpec(16, IntParameter),
        "pad_samples": ParameterSpec(16, IntParameter),
        "order": ParameterSpec(7, StrParameter),
    }

    sample_rate: int
    """The sampling rate of the audio signal, in hertz."""

    amplitude_max: int
    """The maximum amplitude of the audio signal, prior to scaling."""

    amplitude_range: int
    """The range between the min and max amplitude, prior to scaling."""

    pad_samples: int
    """The number of samples padded onto the audio signal."""

    order: str
    """The order used when reshaping arrays with Numpy."""

    def _encode(self):
        encoded = []
        start = 0

        for param, spec in self.PARAMETER_SPECS.items():
            raw_value = getattr(self, param)
            encoded_value = spec.type(raw_value, spec.width).encode()
            encoded.append((encoded_value << start))
            start += spec.width

        return sum(encoded)

    @classmethod
    def _decode(cls, encoded: int):
        decoded = {}
        start = 0
        for param, spec in cls.PARAMETER_SPECS.items():
            mask = (1 << spec.width) - 1
            raw_value = (encoded >> start) & mask
            decoded_value = spec.type.decode(raw_value)
            decoded[param] = decoded_value
            start += spec.width

        return decoded

    def to_header(self, width: int = 128) -> np.ndarray:
        """Encode the metadata into an image header array."""
        encoded_bits = (
            np.array([int(bit) for bit in f"{self._encode():0{width}b}"]) * 255
        )

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
