import abc
import os
import pathlib
import urllib.request
import zipfile
from typing import Callable, Union, Any

import torch
import torchvision.datasets
import torchvision.transforms as T
from PIL import Image

from detect_resolution.nn.model import INPUT_SIZE


class ScaledImageDatasetMixin(abc.ABC):
    def __init__(self, *args, num_samples_per_image=8, **kwargs):
        super().__init__(root='data', *args, **kwargs)
        self.num_samples_per_image = num_samples_per_image
        self.crop = T.CenterCrop(INPUT_SIZE)
        self.downscale_to_ensure_high_detail = T.Resize((INPUT_SIZE, INPUT_SIZE))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return super().__len__() * self.num_samples_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.num_samples_per_image
        image, _ = super().__getitem__(img_idx)  # assumes base returns (image, caption)
        image = self.crop(image)
        # scale factor 1..4 float
        scale_factor = torch.rand((1,)).item() * 3 + 1  # scale factor between 1 and 4
        w, h = image.size
        downscaled = image.resize((int(w // scale_factor), int(h // scale_factor)))
        upscaled = downscaled.resize((w, h))
        upscaled = upscaled.convert('L')  # convert to grayscale
        return self.to_tensor(upscaled), scale_factor


def image_loader(path: Union[str, pathlib.Path]) -> Any:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA")  # some images have transparency


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, url: str, img_dir: str, download: bool = False,
                 loader: Callable[[Union[str, pathlib.Path]], Any] = image_loader,
                 transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.root = os.path.join(os.path.dirname(__file__), root, type(self).__name__.lower())
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        # download if not present
        filename = url.split('/')[-1]
        filepath = os.path.join(self.root, filename)
        if not os.path.isfile(filepath):
            if not download:
                raise RuntimeError(f"Dataset not found at {filepath}. Set download=True to download it.")
            print(f"Downloading {url} to {filepath}...")
            urllib.request.urlretrieve(url, filepath)
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.root)

        self.files = os.listdir(os.path.join(self.root, self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_dir, self.files[idx])
        image = self.loader(img_path)
        label = self.files[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ScaledImageDatasetDTD(ScaledImageDatasetMixin, torchvision.datasets.DTD):
    pass


class ScaledImageCartoonDataset(ScaledImageDatasetMixin, CustomImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            url='https://zenodo.org/records/7267054/files/Figures.zip',
            img_dir='Figures/Cartoon/train_set',
            *args,
            **kwargs)


class ScaledImageDatasetCountry211(ScaledImageDatasetMixin, torchvision.datasets.Country211):
    pass


TrainingDataset = torch.utils.data.ConcatDataset([
    ScaledImageDatasetDTD(download=True),
    ScaledImageCartoonDataset(download=True),
    # ScaledImageDatasetCountry211(download=True),
])
