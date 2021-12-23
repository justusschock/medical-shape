from typing import Union

import torch
import torchio as tio
from rising.transforms.functional.affine import (
    affine_point_transform,
    parametrize_matrix,
)

from shape.normalization import ShapeNormalization
from shape.shape import Shape
from shape.subject import ShapeSupportSubject
from shape.transforms.mixin import TransformShapeValidationMixin


class Affine(tio.transforms.augmentation.spatial.Affine, TransformShapeValidationMixin):
    def __init__(
        self,
        scales: tio.typing.TypeTripletFloat,
        degrees: tio.typing.TypeTripletFloat,
        translation: tio.typing.TypeTripletFloat,
        center: str = "image",
        default_pad_value: Union[str, float] = "minimum",
        image_interpolation: str = "linear",
        check_shape: bool = True,
        **kwargs
    ):
        assert (
            center == "image"
        ), "Currently only affines centered on the image are supported"

        super().__init__(
            scales,
            degrees,
            translation,
            center,
            default_pad_value,
            image_interpolation,
            check_shape,
            **kwargs
        )

    # TODO: Add custom class for typing?
    def apply_transformation(self, subject: tio.data.Subject):
        current_shape = subject.spatial_shape
        sub = super().apply_transformation(
            getattr(subject, "get_images_only_subject", lambda: subject)()
        )
        new_size = sub.spatial_shape

        affine_lmk_matrix = parametrize_matrix(
            scale=self.scales,
            rotation=self.degrees,
            translation=self.translation,
            batchsize=1,
            ndim=3,
            degree=True,
            device=torch.device("cpu"),
            dtype=torch.float,
        )
        sub_dict = dict(sub)

        for k, v in getattr(subject, "get_shapes_dict", lambda: {})().items():
            normalized_shape = ShapeNormalization.normalize(v.tensor, current_shape)
            transformed_shape = affine_point_transform(
                point_batch=normalized_shape[None], matrix_batch=affine_lmk_matrix
            )[0]
            transformed_shape = ShapeNormalization.denormalize(
                transformed_shape, new_size
            )
            new_shape = Shape(tensor=transformed_shape, affine=v.affine, path=v.path)
            sub_dict[k] = new_shape
        return type(subject)(sub_dict)
