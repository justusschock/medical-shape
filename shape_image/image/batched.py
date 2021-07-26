import pathlib
from contextlib import contextmanager
from typing import Optional, Union, Generator, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

from shape_image.image.shape_image import ShapeImage


class BatchedShapeImage(ShapeImage):
    """
    Class that represents a batch of images with a shape
    Reimplements the methods for shape and image batches or wraps the none batch methods
    """

    @staticmethod
    def _adjust_image_format(new_images: torch.Tensor) -> torch.Tensor:
        # 2d, no channels
        if new_images.ndim == 3:
            new_images = new_images.unsqueeze(1)
        elif new_images.ndim == 4:
            # check for one, 3, 4 as in greyscale, RGB, RGBA
            if new_images.size(1) not in [1, 3, 4]:
                # 2d, channels at back
                if new_images.size(-1) in [1, 3, 4]:
                    new_images = new_images.permute(0, 3, 1, 2)
                # 3d, no channels
                else:
                    new_images = new_images.unsqueeze(1)

        # must be 3d, check for channels at front or back
        elif new_images.ndim == 5:
            # channels at back
            if new_images.size(1) != 1:
                new_images = new_images.permute(0, 4, 1, 2, 3)
        else:
            raise RuntimeError("Invalid Dimension for image. Got %s")

        return new_images

    @staticmethod
    def _adjust_shape_format(new_shape: torch.Tensor) -> torch.Tensor:
        if new_shape.ndim < 2:
            new_shape = new_shape.view(1, 1, -1)
        elif new_shape.ndim < 3:
            new_shape = new_shape.unsqueeze(0)

        while new_shape.ndim > 3:
            new_shape = new_shape.squeeze(0)

        if new_shape.size(-1) not in (2, 3):
            raise ValueError(
                "Only shapes containing 2D and 3D points supported, "
                "but got {0}".format(new_shape.size(-1))
            )

        return new_shape

    @property
    def image_ndim(self) -> Optional[int]:
        if self.image is not None:
            return self.image.ndim - 2

    @property
    def mask_ndim(self) -> Optional[int]:
        if self.shape is not None:
            return self.shape.ndim - 2

    @property
    def batchsize(self) -> int:
        image = self.image
        batchsize = 1
        if image is not None:
            batchsize = image.size(0)

        elif self.shape is not None:
            batchsize = self.shape.size(0)

        else:
            shape = self.shape
            if shape is not None:
                batchsize = shape.size(0)

        return batchsize

    @staticmethod
    @contextmanager
    def _with_batch_dim(tensor: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        yield tensor

    def save(self, path: Union[str, pathlib.Path]):

        images = self.split()
        for idx, img in enumerate(images):
            img.save(path=str(path) + "_%0{0}d".format(len(str(len(images)))) % idx)

    def split(self) -> list:
        images = self.image
        shapes = self.shape

        if images is None and shapes is not None:
            images = [images] * len(shapes)
        elif shapes is None and images is not None:
            shapes = [shapes] * len(images)
        elif images is None and shapes is None:
            return [ShapeImage()]

        return [
            ShapeImage(image=image, shape=shape) for image, shape in zip(images, shapes)
        ]

    @property
    def shape_bbox(self) -> torch.Tensor:
        mins = self.shape.min(1)[0]
        maxs = self.shape.max(1)[0]
        return torch.stack([mins, maxs], 1)

    def to_gray(self):
        """
        Converts the image to a gray scale image

        """
        if self.image.shape[1] == 3:
            self.image[:, 0] *= 0.2989
            self.image[:, 1] *= 0.5870
            self.image[:, 2] *= 0.1140
            self.image = torch.sum(self.image, dim=1, keepdim=True)
        elif self.image.shape[1] == 4:
            raise NotImplementedError
        return self

    def crop_to_shape(self, proportion: float = 0.0):
        return sum(
            [
                x.crop_to_shape(
                    proportion=proportion,
                )
                for x in self.split()
            ]
        )

    def zoom_and_resize(self, zoom, new_size):
        return sum([x.zoom_and_resize(zoom, new_size) for x in self.split()])

    def crop_to_shape_and_resize(
        self,
        new_size,
        cropping_proportion: float = 0.0,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        keep_image_proportions: bool = False,
    ):
        return sum(
            [
                x.crop_to_shape_and_resize(
                    new_size=new_size,
                    cropping_proportion=cropping_proportion,
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                    keep_image_proportions=keep_image_proportions,
                    align_corners=align_corners,
                )
                for x in self.split()
            ]
        )

    def lmk_pca(
        self,
        center: bool,
        scale_idx: Tuple[int] = (),
        scale_with_eigen_val: bool = True,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        perform PCA on samples' landmarks

        Parameters
        ----------
        scale_with_eigen_val:
            whether or not to scale the principal components with the
            corresponding eigen value
        center:
            whether or not to substract mean before pca
        scale_idx :
            point idx used to scale the shape to the mean shape
            If empty no scaling is applied
        args :
            additional positional arguments (passed to pca)
        **kwargs :
            additional keyword arguments (passed to pca)

        """

        shape = np.asarray(self.shape).copy()

        if center:
            mean = np.mean(shape, axis=1, keepdims=True)
            shape = shape - mean

        mean_shape = np.mean(shape, axis=0, keepdims=True)

        if scale_idx:
            for i in range(shape.shape[0]):
                point_scale_factors = np.absolute(
                    shape[i, scale_idx, :] / mean_shape[:, scale_idx, :]
                )
                scal_factor = np.sum(point_scale_factors) / (len(scale_idx) * 2)
                shape[i] = shape[i] / scal_factor
                # plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1])
                # plt.scatter(mean_shape[0, :, 0], mean_shape[0, :, 1])
                # plt.show()

        landmarks_transposed = shape.transpose((0, 2, 1))

        reshaped = landmarks_transposed.reshape(shape.shape[0], -1)
        pca = PCA(*args, **kwargs)
        pca.fit(reshaped)

        if scale_with_eigen_val:
            components = pca.components_ * pca.singular_values_.reshape(-1, 1)
        else:
            components = pca.components_

        return torch.from_numpy(
            np.array([pca.mean_] + list(components))
            .reshape(components.shape[0] + 1, *landmarks_transposed.shape[1:])
            .transpose(0, 2, 1)
        )
