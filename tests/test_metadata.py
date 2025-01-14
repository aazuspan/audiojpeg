from hypothesis import given, strategies as st

from audiojpeg.metadata import Metadata


@given(
    st.sampled_from([128, 256, 512]),
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=2**8 - 1),
    st.integers(min_value=0, max_value=2**16 - 1),
    st.integers(min_value=0, max_value=2**16 - 1),
    st.integers(min_value=0, max_value=2**16 - 1),
    st.sampled_from(["C", "F"]),
)
def test_header_roundtrip(
    width, sample_rate, channels, amplitude_max, amplitude_range, pad_samples, order
):
    meta = Metadata(
        sample_rate=sample_rate,
        channels=channels,
        amplitude_max=amplitude_max,
        amplitude_range=amplitude_range,
        pad_samples=pad_samples,
        order=order,
    )
    assert Metadata.from_header(meta.to_header(width=width)) == meta
