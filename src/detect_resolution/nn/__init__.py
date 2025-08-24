import numpy.typing as npt
import torchvision
from PIL.Image import Image

from detect_resolution.nn.model import INPUT_SIZE, load_model

_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.CenterCrop(INPUT_SIZE),
    torchvision.transforms.Grayscale(),
])


def preprocess_image(image: npt.NDArray) -> Image:
    """
    Crop image and convert to grayscale for use with `detect_scaling_factor`
    """
    assert image.shape[0] >= INPUT_SIZE
    assert image.shape[1] >= INPUT_SIZE

    return _data_transform(image)


_model = None


def detect_scaling_factor(prepared_image: Image) -> float:
    assert prepared_image.size == (INPUT_SIZE, INPUT_SIZE)
    assert prepared_image.mode == 'L'

    global _model
    if _model is None:
        _model = load_model()

    transformed_tensor = torchvision.transforms.ToTensor()(prepared_image).unsqueeze(0)
    detected_scaling_factor = _model(transformed_tensor).item()
    return detected_scaling_factor
