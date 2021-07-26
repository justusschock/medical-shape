from typing import Sequence
import torch

from shape_image.transforms.base import ShapeImageFunctionTransform


class Affine(ShapeImageFunctionTransform):
    """
    Applies the given affine matrix to every shapeimage feed to a object of this class
    Parametrize the affine transformation by giving the matrix
    """

    def __init__(
        self,
        affine_matrix: torch.Tensor,
        output_size: tuple = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        shapeimage_query_keys: Sequence[str] = ("data",),
    ):
        super().__init__(
            "apply_affine",
            affine_matrix=affine_matrix,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            shapeimage_query_keys=shapeimage_query_keys,
        )


class ParametrizedAffine(ShapeImageFunctionTransform):
    """
    Applies the given affine matrix to every shapeimage feed to a object of this class
    Parametrize the affine transformation by giving the a value for rotation, translation, scale...
    """

    def __init__(
        self,
        scale,
        rotation,
        translation,
        shapeimage_query_keys: tuple = ("data",),
        output_size: tuple = None,
        adjust_size: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        degree: bool = True,
        grad: bool = False,
    ):
        super().__init__(
            "apply_parametrized_affine",
            scale=scale,
            rotation=rotation,
            translation=translation,
            shapeimage_query_keys=shapeimage_query_keys,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            degree=degree,
            grad=grad,
        )
