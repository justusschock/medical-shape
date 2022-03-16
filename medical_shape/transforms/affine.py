from copy import deepcopy

import torch
import torchio as tio
from rising.transforms.functional.affine import affine_point_transform, parametrize_matrix

from medical_shape.normalization import ShapeNormalization
from medical_shape.shape import Shape
from medical_shape.transforms.mixin import TransformShapeValidationMixin


class Affine(tio.transforms.augmentation.spatial.Affine, TransformShapeValidationMixin):
    # def __init__(
    #     self,
    #     scales: tio.typing.TypeTripletFloat,
    #     degrees: tio.typing.TypeTripletFloat,
    #     translation: tio.typing.TypeTripletFloat,
    #     center: str = "image",
    #     default_pad_value: Union[str, float] = "minimum",
    #     image_interpolation: str = "linear",
    #     check_shape: bool = True,
    #     **kwargs
    # ):
    #     assert (
    #         center == "image"
    #     ), "Currently only affines centered on the image are supported"

    #     super().__init__(
    #         scales,
    #         degrees,
    #         translation,
    #         center,
    #         default_pad_value,
    #         image_interpolation,
    #         check_shape,
    #         **kwargs
    #     )

    # TODO: Add custom class for typing?
    def apply_transform(self, subject: tio.data.Subject):
        current_shape = deepcopy(subject.spatial_shape)
        sub = super().apply_transform(getattr(subject, "get_images_only_subject", lambda: subject)())
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
            transformed_shape = ShapeNormalization.denormalize(transformed_shape, new_size)
            new_shape = Shape(
                tensor=transformed_shape, affine=v.affine, path=v.path, point_descriptions=v.point_descriptions
            )
            sub_dict[k] = new_shape
        return type(subject)(sub_dict)


class RandomAffine(tio.transforms.RandomAffine, TransformShapeValidationMixin):
    # duplicate this to use custom affine class
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        scaling_params, rotation_params, translation_params = self.get_params(
            self.scales,
            self.degrees,
            self.translation,
            self.isotropic,
        )
        arguments = {
            "scales": scaling_params.tolist(),
            "degrees": rotation_params.tolist(),
            "translation": translation_params.tolist(),
            "center": self.center,
            "default_pad_value": self.default_pad_value,
            "image_interpolation": self.image_interpolation,
            "label_interpolation": self.label_interpolation,
            "check_shape": self.check_shape,
        }
        transform = Affine(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed
