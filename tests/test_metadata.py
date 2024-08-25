from audiojpeg.metadata import Metadata

from hypothesis import given, strategies as st


@given(
    st.sampled_from([128, 256, 512]),
    st.integers(min_value=0, max_value=2 ** 32 - 1),
    st.integers(min_value=0, max_value=2 ** 16 - 1),
    st.integers(min_value=0, max_value=2 ** 16 - 1),
    st.integers(min_value=0, max_value=2 ** 16 - 1),
)
def test_header_roundtrip(width, sample_rate, amplitude_max, amplitude_range, pad_samples):
    meta = Metadata(sample_rate=sample_rate, amplitude_max=amplitude_max, amplitude_range=amplitude_range, pad_samples=pad_samples)
    assert Metadata.from_header(meta.to_header(width=width)) == meta