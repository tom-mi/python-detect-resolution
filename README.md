# detect-resolution

[![PyPI - Version](https://img.shields.io/pypi/v/detect-resolution.svg)](https://pypi.org/project/detect-resolution)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/detect-resolution.svg)](https://pypi.org/project/detect-resolution)

Try to detect if a video has been upscaled.
This tool uses a custom neural network to detect by how much a video might have been upscaled,
then tries to match the result with a list of common video resolutions.

As video frames are often blurry for other reasons, the tool will sample a number of frames and take the
10-th percentile of the upscaling factor to make the result more robust.

## Installation

```console
pip install detect-resolution
```
Then, install pytorch and torchvision as needed for your system.
Please see the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.

## Usage

```console
detect-resolution --help
detect-resolution path/to/some_video.mp4
detect-resolution path/to/some_video.mp4 --n-samples 100
detect-resolution path/to/some_video.mp4 --debug --debug-images
```

## License

`detect-resolution` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
