from abc import ABC, abstractmethod
from typing import List, Union

import pyiqa
import pyiqa.models
import pyiqa.models.inference_model
import torch
from PIL import Image
from pyiqa.models.inference_model import InferenceModel
from torchvision import transforms
from torchvision.transforms import ToTensor
from typing import Tuple
import numpy as np

import torch

def is_cuda_available() -> bool:
    """Checks if CUDA is available on the machine."""
    return torch.cuda.is_available()

def get_torch_device_name() -> str:
    """Return the device name based on whether CUDA is available."""
    return "cuda" if is_cuda_available() else "cpu"

def get_torch_device() -> torch.device:
    """
    Checks if CUDA is available on the machine and returns PyTorch device.
    """
    return torch.device(get_torch_device_name())

import urllib

def is_url(location: str) -> bool:
    """Return True if `location` is a url. False otherwise."""
    return urllib.parse.urlparse(location).scheme in ["http", "https"]

from PIL import Image
def open_image(image_location: str) -> Image.Image:
    """
    Opens image with the Python Imaging Library (PIL).
    """
    image: Image.Image
    if is_url(image_location):
        image = Image.open(requests.get(image_location, stream=True).raw)
    else:
        image = Image.open(image_location)
    return image.convert("RGB")


class BaseMetric(ABC):
    """BaseMetric Class."""

    def __init__(self) -> None:
        self._metric = None
        self._device = get_torch_device()

    def _process_image(
        self,
        rendered_images: List[Union[str, Image.Image]],
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        rendered_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

        img: torch.Tensor = torch.stack(rendered_images_).to(self._device)

        return img

    def _process_images(
        self,
        rendered_images: List[Union[str, Image.Image]],
        reference_image: Union[str, Image.Image],
    ) -> float:
        preprocessing = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        # Handle the reference image input
        if isinstance(reference_image, str):
            reference_image = preprocessing(open_image(reference_image))

        rendered_images_: List[torch.Tensor] = []
        reference_images_: List[torch.Tensor] = []
        for image in rendered_images:
            # Handle the rendered image input
            if isinstance(image, str):
                image = preprocessing(open_image(image))
            else:
                image = preprocessing(image)
            rendered_images_.append(image)

            reference_images_.append(reference_image)

        img1: torch.Tensor = torch.stack(rendered_images_).to(self._device)
        img2: torch.Tensor = torch.stack(reference_images_).to(self._device)

        return img1, img2

    def _process_np_to_tensor(
        self,
        rendered_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img1 = ToTensor()(rendered_image).unsqueeze(0).to(self._device)
        img2 = ToTensor()(reference_image).unsqueeze(0).to(self._device)
        return img1, img2
    
    @abstractmethod
    def _compute_scores(self, *args):
        pass


class IQAPytorchMetric(BaseMetric):
    def __init__(self, metric_name: str) -> None:
        super().__init__()
        self._metric = self._create_metric(metric_name).to(self._device)

    def _create_metric(self, metric: str) -> InferenceModel:
        metric = pyiqa.create_metric(metric)
        return metric
