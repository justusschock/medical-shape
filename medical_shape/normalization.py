from typing import Sequence, Union

import torch


class ShapeNormalization:
    """Context manager to normalizes the shape of a shapeimage to [-1, 1] You can pass the attribute name to
    normalize a different shape than the normal shape Uses the old image size or a new size to undo the
    normalization Makes it possible to use the same translation used to translate an image with a gridsampler for
    the shape Use new_size for example when the output of the gridsampler has a different size than the input."""

    @staticmethod
    def normalize(shape: torch.Tensor, image_size: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
        shape = shape.float()
        if not isinstance(image_size, torch.Tensor):
            image_size = torch.tensor(image_size, device=shape.device, dtype=torch.float)
        image_size = image_size.float()
        return (shape / image_size * 2) - 1

    @staticmethod
    def denormalize(shape: torch.Tensor, image_size: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
        shape = shape.float()
        if not isinstance(image_size, torch.Tensor):
            image_size = torch.tensor(image_size, device=shape.device, dtype=torch.float)
        image_size = image_size.float()

        return (shape + 1) * image_size / 2
