from typing import Optional, Sequence
from shape_image.image import ShapeImage

import torch

class _ShapeNormalization:
    """
    Context manager to normalizes the shape of an shapeimage to [-1, 1]
    You can pass the attribute name to normalize a different shape than the normal shape
    Uses the old image size or a new size to undo the normalization
    Makes it possible to use the same translation used to translate an image with a gridsampler for the shape
    Use new_size for example when the output of the gridsampler has a different size than the input
    """

    def __init__(self, shape_img: ShapeImage,
                 new_size: Optional[Sequence[int]] = None,
                 shape_attribute_name: Optional[str] = "shape"):
        self.shape_img = shape_img
        self.shape_attribute_name = shape_attribute_name
        if new_size is None:
            self.new_size = torch.tensor(list(self.shape_img.image.shape[-self.shape_img.image_ndim:]),
                                         device=self.shape_img.image.device)
        else:
            self.new_size = torch.tensor(new_size, device=self.shape_img.image.device, dtype=torch.float)

    def __enter__(self):
        tmp = getattr(self.shape_img, self.shape_attribute_name) / (
                torch.tensor(list(self.shape_img.image.shape[-self.shape_img.image_ndim:]),
                             device=self.shape_img.image.device, dtype=torch.float) / 2)
        tmp = tmp - 1
        setattr(self.shape_img, self.shape_attribute_name, tmp)
        return tmp

    def __exit__(self, exc_type, exc_value, exc_traceback):
        tmp = getattr(self.shape_img, self.shape_attribute_name)
        tmp = tmp + 1
        tmp = tmp * (self.new_size.to(torch.float) / 2)
        setattr(self.shape_img, self.shape_attribute_name, tmp)
        return super().__exit__(exc_type, exc_value, exc_traceback)
