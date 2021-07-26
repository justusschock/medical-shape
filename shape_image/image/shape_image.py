import numbers
import os
import pathlib
from contextlib import contextmanager
from typing import Optional, Union, Sequence, Generator

import numpy as np
from rising.transforms.functional.intensity import norm_mean_std
import torch
from rising.transforms.functional.affine import (
    affine_point_transform,
    parametrize_matrix,
    create_scale,
    create_translation,
    create_rotation,
    affine_image_transform,
)
from rising.utils.affine import (
    matrix_revert_coordinate_order,
    matrix_to_homogeneous,
)
from rising.transforms.functional import crop


from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin

from shape_fitting.data.io import load_image, pts_importer

import json


class ShapeImage(DeviceDtypeModuleMixin):
    """
    Class that hold the image (3D or 2D) and its shape(landmarks)
    It also provides a set of functions to transform, or save the data
    """

    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        shape: Optional[torch.Tensor] = None,
        spacing: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._image = None
        self.image = image

        self._shape = None
        self.shape = shape

        self._mask = None
        self.mask = mask

        self._spacing = None
        self.spacing = spacing

    @staticmethod
    def _adjust_shape_format(new_shape: torch.Tensor) -> torch.Tensor:
        """
        Brings the dimensions of the shape into the correct order
        Correct shape: [num_landmakrs, 2/3]
        Args:
            new_shape:
                new shape(landmarks)

        Returns:
            The new shape

        """
        if new_shape.ndim < 2:
            new_shape = new_shape.view(1, -1)

        while new_shape.ndim > 2:
            new_shape = new_shape.squeeze(0)

        if new_shape.size(-1) not in (2, 3):
            raise ValueError(
                "Only shapes containing 2D and 3D points supported, "
                "but got {0}".format(new_shape.size(-1))
            )

        return new_shape

    def crop_to_shape(
        self,
        cropping_proportion: float = 0.0,
        output_size: Optional[Sequence[int]] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ):
        """
        Applies a croping to the shape and resize.
        Args:
            output_size:
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            cropping_proportion:
                Sets the margin on every side to the proportion / 2 of the shape size in this dimension.
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Returns:
            self
        """

        from shape_fitting.data.normalization import _ShapeNormalization
        with _ShapeNormalization(self):
            center = self.shape_center
            shape_centralization = create_translation(
                offset=-center,
                image_transform=False,
                batchsize=self.batchsize,
                ndim=self.ndim,
                device=self.device,
            )
            max_size = self.max_shape_size
            scale = (2 / max_size) * 1 / (1 + cropping_proportion)

            old_shape = torch.tensor(self.image.shape[-self.ndim :], dtype=torch.float)
            image_propotion_keeper = old_shape / old_shape.min()
            output_size_t = torch.tensor(output_size, dtype=torch.float)
            image_propotion_keeper2 = output_size_t.min() / output_size_t
            scale = scale * image_propotion_keeper * image_propotion_keeper2
            shape_scale = create_scale(
                scale=scale,
                image_transform=False,
                batchsize=self.batchsize,
                ndim=self.ndim,
                device=self.device,
            )

        affine_lmk_matrix = torch.bmm(shape_scale, shape_centralization)

        affine_img_matrix = affine_lmk_matrix.inverse()
        affine_img_matrix = matrix_revert_coordinate_order(affine_img_matrix)[:, :-1]

        self.apply_affine(
            affine_matrix=affine_img_matrix,
            affine_lmk_matrix=affine_lmk_matrix[:, :-1],
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        return self

    def apply_affine(
        self,
        affine_matrix: torch.Tensor,
        affine_lmk_matrix: Optional[torch.Tensor] = None,
        output_size: Optional[Sequence] = None,
        adjust_size: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        padding_mode: Optional[str] = "zeros",
        align_corners: Optional[bool] = False,
    ):
        """
        Applies a transform specified by an affine matrix to image and shape

        Args:
            affine_matrix :
                The affine matrix for the image.
                Normally the affine matrix is used to transform the grid, not the image
                => The inverse trafo of affine_matrix is applied to the image
            affine_lmk_matrix:
                The affine matrix for the shape.
                If None, the inverse transformation of affine_matrix is used for the shape.
            output_size :
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Warnings:
            When align_corners = True, the grid positions depend on the pixel size
            relative to the input image size, and so the locations sampled by
            grid_sample() will differ for the same input given at different
            resolutions (that is, after being upsampled or downsampled).

        Notes:
            :param:`output_size` and :param:`adjust_size` are mutually exclusive.
            If None of them is set, the resulting image will have the same size
            as the input image

        Returns:
            self
        """

        if affine_lmk_matrix is None:
            affine_lmk_matrix = matrix_to_homogeneous(affine_matrix).inverse()
            affine_lmk_matrix = matrix_revert_coordinate_order(affine_lmk_matrix)[
                :, :-1
            ]

        if not output_size:
            output_size = torch.tensor(
                list(self.image.shape[-self.image_ndim :]), device=self.device
            )

        if self.shape is not None:
            self.apply_affine_to_shapes(affine_lmk_matrix, output_size)

        if self.image is not None:
            with self._with_batch_dim(self.image) as batched_image:
                self.image = affine_image_transform(
                    image_batch=batched_image,
                    matrix_batch=affine_matrix,
                    output_size=output_size,
                    adjust_size=adjust_size,
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )[0]

        if self.mask is not None:
            with self._with_batch_dim(self.mask) as batched_mask:
                self.mask = affine_image_transform(
                    image_batch=batched_mask,
                    matrix_batch=affine_matrix,
                    output_size=output_size,
                    adjust_size=adjust_size,
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )[0]

        return self

    def apply_affine_to_shapes(
        self, affine_lmk_matrix: torch.Tensor, output_size: torch.Tensor
    ):
        """
        Applies a affine transformation to the shape or shapes
        Args:
            affine_lmk_matrix: The affine matrix for the shapes.
            output_size: This will be the resulting image size.

        """
        from shape_fitting.data.normalization import _ShapeNormalization

        with _ShapeNormalization(self, output_size) as shape:
            with self._with_batch_dim(shape) as batched_shape:
                self.shape = affine_point_transform(
                    point_batch=batched_shape, matrix_batch=affine_lmk_matrix
                )
        return self

    def save(self, path: Union[str, pathlib.Path]):
        """
        Saves the image and the shape
        Cases:
            2D: Saves image as png and shapes as pts
            3D: Saves image series as nifti and shapes as pts
        Args:
            path:
                Path of the file without extension
                The function will add the extension
        """
        if self.image is not None:
            if self.ndim == 2:
                from skimage.io import imsave

                # permute channels to back
                imsave(
                    str(path) + "_image.png",
                    self.image.detach().cpu().permute(1, 2, 0).numpy(),
                )
            else:
                import SimpleITK as sitk

                sitk.WriteImage(
                    sitk.GetImageFromArray(self.image.detach().cpu().squeeze().numpy()),
                    str(path) + "_image.nii.gz",
                )

        if self.mask is not None:
            if self.ndim == 2:
                from skimage.io import imsave

                # permute channels to back
                imsave(
                    str(path) + "_mask.png",
                    self.mask.detach().cpu().permute(1, 2, 0).numpy(),
                )
            else:
                import SimpleITK as sitk

                sitk.WriteImage(
                    sitk.GetImWriteImageFromArray(
                        self.mask.detach().cpu().squeeze().numpy()
                    ),
                    str(path) + "_mask.nii.gz",
                )

        if self.shape is not None:
            from shape_fitting.data.io import pts_exporter

            pts_exporter(self.shape, str(path) + ".pts")

    @property
    def image(self) -> Optional[torch.Tensor]:
        """
        Returns the image data
        Returns:
            ShapeImage data
            Arrayformat: [channel, y, x]

        """
        return self._image

    @image.setter
    def image(self, new_image: torch.Tensor):
        """
        Sets the image
        Args:
            new_image
        """
        if new_image is not None:
            new_image = self._adjust_image_format(new_image)

        if not isinstance(new_image, Sequence):
            self.register_buffer("_image", new_image)
        else:
            self._image = new_image

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask

    @mask.setter
    def mask(self, new_mask: torch.Tensor):
        if new_mask is not None:
            new_mask = self._adjust_image_format(new_mask)

        if not isinstance(new_mask, Sequence):
            self.register_buffer("_mask", new_mask)
        else:
            self._mask = new_mask

    @property
    def spacing(self) -> torch.Tensor:
        return self._spacing

    @spacing.setter
    def spacing(self, new_spacing: torch.Tensor):
        if new_spacing is None:
            new_spacing = torch.tensor([1.0] * self.ndim, device=self.device)
        self.register_buffer("_spacing", new_spacing)

    @property
    def shape(self) -> Optional[torch.Tensor]:
        """
        Returns the shape (landmarks) of the image
        Array shape [num_landmakrs, 2/3]

        Returns:
            Shape

        """
        return self._shape

    @shape.setter
    def shape(self, new_shape: Optional[torch.Tensor]):
        if new_shape is not None:
            new_shape = self._adjust_shape_format(new_shape)

        self.register_buffer("_shape", new_shape)

    @property
    def ndim(self):
        """
        Returns:
            Returns the number of the shape_image dimensions
        """
        return self.image_ndim or self.mask_ndim or self.shape_ndim

    @property
    def shape_bbox(self) -> torch.Tensor:
        """
        Calculates the bounding box of the shape
        Returns:
            Two points that mark the corners of the bb
        """
        mins = self.shape.min(0)[0]
        maxs = self.shape.max(0)[0]

        return torch.stack([mins, maxs])

    @property
    def shape_center(self) -> torch.Tensor:
        """
        Calculates the center of the shape. Does not calculate the mean!
        Returns:
            The center point of the shape
        """
        bb = self.shape_bbox

        return bb.sum(dim=-self.ndim) / 2

    @property
    def max_shape_size(self) -> torch.Tensor:
        """
        Calculates the maximum size of the shape.
        Returns:
            The max size of the shape
        """
        bb = self.shape_bbox
        distances = bb[..., 1, :] - bb[..., 0, :]

        return torch.max(distances)

    @staticmethod
    @contextmanager
    def _with_batch_dim(tensor: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        """
        A context manager to use an tensor with code written for batched tensors
        Adds a batch dimension to the tensor while in the this context
        Args:
            tensor:
                The tensor that should get a batch dimension
        """
        yield tensor[None]

    @property
    def batchsize(self) -> int:
        """
        Returns:
            1, because this is just one shape_image!
        """
        return 1

    @property
    def spatial_size(self):
        if self.image is not None:
            return self.image.shape[-self.ndim :]

        return self.mask.shape[-self.ndim :]

    def apply_parametrized_affine(
        self,
        scale,
        rotation,
        translation,
        output_size: tuple = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        degree: bool = True,
    ):
        """

        Args:
        scale : torch.Tensor, int, float
                the scale factor(s). Supported are:
                    * a single parameter (as float or int), which will be
                        replicated for all dimensions and batch samples
                    * a parameter per sample, which will be
                        replicated for all dimensions
                    * a parameter per dimension, which will be replicated for all
                        batch samples
                    * a parameter per sampler per dimension
                None will be treated as a scaling factor of 1
            rotation : torch.Tensor, int, float
                the rotation factor(s). Supported are:
                    * a single parameter (as float or int), which will be
                        replicated for all dimensions and batch samples
                    * a parameter per sample, which will be
                        replicated for all dimensions
                    * a parameter per dimension, which will be replicated for all
                        batch samples
                    * a parameter per sampler per dimension
                None will be treated as a rotation factor of 1
            translation : torch.Tensor, int, float
                the translation offset(s). Supported are:
                    * a single parameter (as float or int), which will be
                        replicated for all dimensions and batch samples
                    * a parameter per sample, which will be
                        replicated for all dimensions
                    * a parameter per dimension, which will be replicated for all
                        batch samples
                    * a parameter per sampler per dimension
                None will be treated as a translation offset of 0
            output_size : Iterable
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            interpolation_mode : str
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.
            degree : bool
                whether all rotation angles are given in degree

        Returns:
            self

        """
        affine_matrix = parametrize_matrix(
            scale=scale,
            rotation=rotation,
            translation=translation,
            batchsize=self.batchsize,
            ndim=self.ndim,
            degree=degree,
            device=self.device,
            dtype=torch.float32,
            image_transform=True,
        )

        self.apply_affine(
            affine_matrix=affine_matrix,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return self

    def to_gray(self):
        """
        Converts the image to a gray scale image
        Returns:
            self
        """
        if self.image.shape[0] == 3:
            self.image[0] *= 0.2989
            self.image[1] *= 0.5870
            self.image[2] *= 0.1140
            self.image = torch.sum(self.image, dim=0, keepdim=True)
        elif self.image.shape[1] == 4:
            raise NotImplementedError
        return self

    def resize(
        self,
        new_size: Union[Sequence[int], int],
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        keep_image_proportions: bool = False,
    ):
        """
        Resizes the image to a new size.
        Args:
            new_size:
                The new size of the image.
                If just one int: Same size is used as new size for every dimension.
            keep_image_proportions:
                If true and new_size is just one int, the sizes of the other dimensions are calculated in a way,
                that the image proportions are preserved.
                The shortest old size will become the new_size.
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.


        Returns:

        """
        if keep_image_proportions and isinstance(new_size, int):
            with self._with_batch_dim(self.image) as bimage:
                curr_size = bimage.shape[-self.image_ndim :]
                min_size = min(curr_size)
                factor = new_size / min_size
                new_size = [int(cs * factor) for cs in list(curr_size)]
        if isinstance(new_size, int):
            new_size = [new_size] * self.image_ndim

        return self.apply_parametrized_affine(
            scale=1,
            rotation=None,
            translation=None,
            output_size=new_size,
            adjust_size=False,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    def zoom_and_resize(self, zoom: float, new_size: tuple):
        """
        Zooms into the image and resizes it.

        Args:
            zoom:
                The scale factor/zoom factor
            new_size:
                The new size of the image.
                If just one int: Same size is used as new size for every dimension.

        Returns:

        """
        img = self.apply_parametrized_affine(
            scale=zoom, output_size=new_size, rotation=0, translation=(0, 0)
        )
        return img

    def scale(
        self,
        scale,
        output_size: tuple = None,
        adjust_size: bool = True,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ):
        """
        Applies a scaling to the shapeimage

        Args:
            scale: The scalefactor
            output_size : Iterable
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Returns:

        """
        return self.apply_parametrized_affine(
            scale=scale,
            rotation=None,
            translation=None,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    def rotate(
        self,
        rotation,
        output_size: tuple = None,
        adjust_size: bool = True,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        degree: bool = True,
    ):
        """
        Applies a rotation to the shapeimage

        Args:
            rotation:
                The rotation angle
            output_size : Iterable
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.
            degree:
                Whether or not the rotation is given in degree

        Returns:

        """
        return self.apply_parametrized_affine(
            scale=None,
            rotation=rotation,
            translation=None,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            degree=degree,
        )

    def shift(
        self,
        translation,
        output_size: tuple = None,
        adjust_size: bool = True,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ):
        """
        Applies a translation to the shapeimage

        Args:
            translation: The translation
            output_size : Iterable
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Returns:

        """
        return self.apply_parametrized_affine(
            scale=None,
            rotation=None,
            translation=translation,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    def crop_to_shape_and_resize(
        self,
        new_size,
        cropping_proportion: float = 0.0,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        keep_image_proportions: bool = False,
    ):
        """
        Crops to the shape of the image and resize it

        Args:
            new_size:
                The new size of the image.
                If just one int: Same size is used as new size for every dimension.
            keep_image_proportions:
                If true and new_size is just one int, the sizes of the other dimensions are calculated in a way,
                that the image proportions are preserved.
                The shortest old size will become the new_size.
            cropping_proportion:
                Sets the margin on every side to the proportion / 2 of the shape size in this dimension.
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Returns:
            self

        """
        img = self.crop_to_shape(proportion=cropping_proportion).resize(
            new_size,
            keep_image_proportions=keep_image_proportions,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        return img

    def trans_rot_crop_resize(
        self,
        translation: tuple,
        rotation: tuple,
        scale_offset: float,
        cropping_proportion: float = 0.0,
        output_size: tuple = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ):
        """
        Applies a transmation, rotation croping to the shape and resize all at onec.
        Args:
            translation:
                The translation
            rotation:
                The rotation
            scale_offset:
                The scale_offset
            output_size:
                if given, this will be the resulting image size.
                Defaults to ``None``
                adjust_size : bool
                if True, the resulting image size will be calculated dynamically
                to ensure that the whole image fits.
            adjust_size :
                Whether to automatically infer the output size
            cropping_proportion:
                Sets the margin on every side to the proportion / 2 of the shape size in this dimension.
            interpolation_mode :
                interpolation mode to calculate output values
                'bilinear' | 'nearest'.
                Default: 'bilinear'm
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners :
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema (-1 and 1)
                are considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Returns:
            self
        """

        from shape_fitting.data.normalization import _ShapeNormalization

        with _ShapeNormalization(self):
            center = self.shape_center
            shape_centralization = create_translation(
                offset=-center,
                image_transform=False,
                batchsize=self.batchsize,
                ndim=self.ndim,
                device=self.image.device,
            )
            max_size = self.max_shape_size
            scale = (2 / max_size) * 1 / (1 + cropping_proportion) * (1 + scale_offset)

            old_shape = torch.tensor(self.image.shape[-self.ndim :], dtype=torch.float)
            image_propotion_keeper = old_shape / old_shape.min()
            output_size_t = torch.tensor(output_size, dtype=torch.float)
            image_propotion_keeper2 = output_size_t.min() / output_size_t
            scale = scale * image_propotion_keeper * image_propotion_keeper2
            shape_scale = create_scale(
                scale=scale,
                image_transform=False,
                batchsize=self.batchsize,
                ndim=self.ndim,
                device=self.image.device,
            )
            trans = create_translation(
                offset=translation,
                image_transform=False,
                batchsize=self.batchsize,
                ndim=self.ndim,
                device=self.image.device,
            )
            rot = create_rotation(
                rotation=rotation,
                batchsize=self.batchsize,
                ndim=self.ndim,
                device=self.image.device,
                degree=True,
            )

        affine_lmk_matrix = torch.bmm(
            torch.bmm(torch.bmm(trans, rot), shape_scale), shape_centralization
        )

        affine_img_matrix = affine_lmk_matrix.inverse()
        affine_img_matrix = matrix_revert_coordinate_order(affine_img_matrix)[:, :-1]

        self.apply_affine(
            affine_matrix=affine_img_matrix,
            affine_lmk_matrix=affine_lmk_matrix[:, :-1],
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        return self

    def resample_to_spacing(self, new_spacing):
        if not isinstance(new_spacing, torch.Tensor):
            new_spacing = torch.tensor(
                new_spacing, device=self.device, dtype=self.dtype
            )

        zoom_factor = self.spacing / new_spacing
        new_size = torch.ceil(
            torch.round(
                zoom_factor
                * torch.tensor(self.spatial_size, dtype=self.dtype, device=self.device)
            )
        )

        self.scale(zoom_factor, output_size=new_size)
        self.spacing = new_spacing
        return self

    def __add__(self, other):
        # to make it use with sum (which initializes with zero)
        if isinstance(other, numbers.Number) and other == 0.0:
            return self
        assert isinstance(other, ShapeImage)

        from shape_fitting.data.shape_image.batched import BatchedShapeImage

        with self._with_batch_dim(self.image) as sb_image:
            with other._with_batch_dim(other.image) as ob_image:
                batched_image = torch.cat([sb_image, ob_image])

        with self._with_batch_dim(self.shape) as sb_shape:
            with other._with_batch_dim(other.shape) as ob_shape:
                batched_shape = torch.cat([sb_shape, ob_shape])

        return BatchedShapeImage(image=batched_image, shape=batched_shape)

    def __radd__(self, other):
        # to make it use with sum (which initializes with zero)
        from shape_fitting.data.shape_image.batched import BatchedShapeImage

        if isinstance(other, numbers.Number) and other == 0.0:
            return BatchedShapeImage(image=self.image, shape=self.shape)

        raise NotImplementedError

    @staticmethod
    def _adjust_image_format(new_image: torch.Tensor) -> torch.Tensor:

        # 2d, no channels
        if new_image.ndim == 2:
            new_image = new_image[None]
        elif new_image.ndim == 3:
            # check for one, 3, 4 as in greyscale, RGB, RGBA
            if new_image.size(0) not in [1, 3, 4]:
                # 2d, channels at back
                if new_image.size(-1) in [1, 3, 4]:
                    new_image = new_image.permute(2, 0, 1)
                # 3d, no channels
                else:
                    new_image = new_image[None]

        # must be 3d, check for channels at front or back
        elif new_image.ndim == 4:
            # channels at back
            if new_image.size(0) != 1:
                new_image = new_image.permute(3, 0, 1, 2)
        else:
            raise RuntimeError("Invalid Dimension for image. Got %s")

        return new_image

    @property
    def image_ndim(self) -> Optional[int]:
        """
        Returns:
            The number of image dimensions
            2 for 2D and 3 for 3D

        """
        if self.image is not None:
            return self.image.ndim - 1

    @property
    def shape_ndim(self) -> Optional[int]:
        """
        Returns:
            The number of shape dimensions
            2 for 2D and 3 for 3D
        """
        if self.shape is not None:
            return self.shape.size(-1)

    @property
    def mask_ndim(self) -> Optional[int]:
        if self.mask is not None:
            return self.mask.ndim - 1

    def pad(self, *padding_dims):
        if self.image is not None:
            self.image = torch.nn.functional.pad(self.image, *padding_dims)

        if self.mask is not None:
            self.mask = torch.nn.functional.pad(self.mask, *padding_dims)

        if self.shape is not None:
            padding_dims_front = padding_dims[::2]
            self.shape = self.shape + torch.tensor(
                padding_dims_front, dtype=self.dtype
            ).unsqueeze(0)
        return self

    def center_crop(self, patch_size):
        start_corner = (torch.tensor(self.spatial_size) - torch.tensor(patch_size)) // 2
        if self.image is not None:
            self.image = crop(self.image, start_corner, patch_size.tolist())
        if self.mask is not None:
            self.mask = crop(self.mask, start_corner, patch_size.tolist())
        if self.shape is not None:
            self.shape = self.shape - start_corner.unsqueeze(0)

        return self

    def pad_or_crop(self, target_size):
        crop_size = torch.where(
            target_size < self.spatial_size, target_size, self.spatial_size
        )

        pads = torch.clamp(
            torch.tensor(target_size - torch.tensor(self.spatial_size)), min=0
        )

        self.center_crop(crop_size)

        total_pads = []
        for p in pads:
            total_pads.extend([p // 2, p // 2 + p % 2])

        self.pad(total_pads)

        return self

    @classmethod
    def from_numpy(
        cls,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        shape: Optional[np.ndarray] = None,
        spacing: Optional[np.ndarray] = None,
    ):
        if image is not None:
            image = torch.from_numpy(image)
        if mask is not None:
            mask = torch.from_numpy(mask)
        if shape is not None:
            shape = torch.from_numpy(shape)
        if spacing is not None:
            spacing = torch.from_numpy(spacing)

        return cls(image=image, shape=shape, mask=mask, spacing=spacing)

    @classmethod
    def from_files(
        cls,
        image_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        shape_path: Optional[str] = None,
    ):
        spacing = None
        if image_path is not None:
            image, _spacing = load_image(image_path, return_spacing=True)
            if spacing is None:
                spacing = _spacing
        else:
            image = None

        if mask_path is not None:
            mask, _spacing = load_image(mask_path, return_spacing=True)
            if spacing is None:
                spacing = _spacing
        else:
            mask = None

        if shape_path is not None:
            shape = pts_importer(shape_path)
        else:
            shape = None
        return cls(image, shape, mask)

    def from_series_dir(cls, path: str):
        with open(os.path.join(path, 'series.json')) as f:
            series_data = json.load(f)

        image_path = series_data.get('image', None)
        mask_path = series_data.get('mask', None)
        shape_path = series_data.get('shape', None)

        if image_path is not None:
            image_path = os.path.join(path, image_path)

        if mask_path is not None:
            mask_path = os.path.join(path, mask_path)
        
        if shape_path is not None:
            shape_path = os.path.join(path, shape_path)

        return cls.from_files(image_path=image_path, mask_path=mask_path, shape_path=shape_path)
