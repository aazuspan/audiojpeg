# AudioJPEG

An experiment at making the worst audio compression algorithm by encoding and decoding through JPEG images. More details in the [blog post](https://www.aazuspan.dev/blog/introducing-audiojpeg/).

![Compression ratios](/output/PinkPanther30_F.jpeg)

## How it Works

### Encoding

1. Load the audio as an array of numbers in the shape `(samples, channels)`.
1. Scale it, pad it, and reshape it to fit in an image array of shape `(rows, columns, bands)`.
1. Encode metadata like sample rate and amplitude as a bit array header row.
1. Write it out as a JPEG with the desired quality level.

### Decoding

1. Load the image as an array of numbers in the shape `(rows, columns, bands)`.
1. Remove and parse the bits of the metadata header, averaging values across bands to compensate for compression artifacts.
1. Reshape, remove the padded pixels, and rescale to the original amplitude.
1. See what it sounds like.

## Results

![Compression ratios](/output/audiojpeg_compression.png)