from typing import Tuple
from shape_image.transforms.base import ShapeImageFunctionTransform
from shape_image.transforms.affine import ParametrizedAffine
from shape_image.image
import torch

class RandomTransRotCropResize(ShapeImageFunctionTransform):
    def __init__(self, translation_range: Tuple[float, float],
                 rotation_range: Tuple[float, float],
                 scale_offset_range: Tuple[float, float] = 0.0,
                 cropping_proportion: float = 0.0,
                 shapeimage_query_keys: tuple = ('data',),
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False):
        super().__init__(
            "trans_rot_crop_resize",
            cropping_proportion=cropping_proportion,
            shapeimage_query_keys=shapeimage_query_keys,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners)
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.scale_offset_range = scale_offset_range

    @staticmethod
    def calc_value(batchsize, min_val, max_val, dimension):
        return torch.rand(batchsize,
                          dimension) * (max_val - min_val) + min_val

    def calc_rotation(self, batchsize, dimension):
        if dimension == 2:
            dimension = 1

        self.func_kwargs['rotation'] = self.calc_value(
            batchsize=batchsize,
            min_val=self.rotation_range[0],
            max_val=self.rotation_range[1],
            dimension=dimension)

    def calc_scal(self, batchsize, dimension):
        if dimension == 2:
            dimension = 1

        self.func_kwargs['scale_offset'] = self.calc_value(
            batchsize=batchsize,
            min_val=self.scale_offset_range[0],
            max_val=self.scale_offset_range[1],
            dimension=dimension).item()

    def calc_translation(self, batchsize, dimension):
        self.func_kwargs['translation'] = self.calc_value(
            batchsize=batchsize,
            min_val=self.translation_range[0],
            max_val=self.translation_range[1],
            dimension=dimension)

    def pre_calc(self, key: str, val: ShapeImage):
        bs = val.batchsize
        dimension = val.ndim
        self.calc_translation(batchsize=bs, dimension=dimension)
        self.calc_rotation(batchsize=bs, dimension=dimension)
        self.calc_scal(batchsize=bs, dimension=dimension)


class RandomParametrizedAffine(ParametrizedAffine):
    def __init__(self,
                 scale_range: tuple,
                 rotation_range: tuple,
                 translation_range: tuple,
                 shapeimage_query_keys: tuple = ('data',),
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 degree: bool = True,
                 grad: bool = False):
        super().__init__(
            scale=None,
            translation=None,
            rotation=None,
            shapeimage_query_keys=shapeimage_query_keys,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            degree=degree,
            grad=grad)
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.rotation_range = rotation_range

    @staticmethod
    def calc_value(batchsize, min_val, max_val, dimension):
        return torch.rand(batchsize,
                          dimension) * (max_val - min_val) + min_val

    def calc_rotation(self, batchsize, dimension):
        if dimension == 2:
            dimension = 1

        self.func_kwargs['rotation'] = self.calc_value(
            batchsize=batchsize,
            min_val=self.rotation_range[0],
            max_val=self.rotation_range[1],
            dimension=dimension)

    def calc_scale(self, batchsize, dimension):
        self.func_kwargs['scale'] = self.calc_value(
            batchsize=batchsize,
            min_val=self.scale_range[0],
            max_val=self.scale_range[1],
            dimension=dimension)

    def calc_translation(self, batchsize, dimension):
        self.func_kwargs['translation'] = self.calc_value(
            batchsize=batchsize,
            min_val=self.translation_range[0],
            max_val=self.translation_range[1],
            dimension=dimension)

    def pre_calc(self, key: str, val: ShapeImage):
        bs = val.batchsize
        dimension = val.ndim
        self.calc_translation(batchsize=bs, dimension=dimension)
        self.calc_scale(batchsize=bs, dimension=dimension)
        self.calc_rotation(batchsize=bs, dimension=dimension)


class RandomRotation(RandomParametrizedAffine):
    def __init__(self,
                 rotation_range: tuple,
                 shapeimage_query_keys: tuple = ('data',),
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 degree: bool = True,
                 grad: bool = False):
        super().__init__(
            scale_range=(None, None),
            rotation_range=rotation_range,
            translation_range=(None, None),
            shapeimage_query_keys=shapeimage_query_keys,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            degree=degree,
            grad=grad)

    def pre_calc(self, key: str, val: ShapeImage):
        bs = val.batchsize
        dimension = val.ndim
        self.calc_rotation(batchsize=bs, dimension=dimension)


class RandomScale(RandomParametrizedAffine):
    def __init__(self,
                 scale_range: tuple,
                 shapeimage_query_keys: tuple = ('data',),
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False):
        super().__init__(
            scale_range=scale_range,
            rotation_range=(None, None),
            translation_range=(None, None),
            shapeimage_query_keys=shapeimage_query_keys,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            degree=True,
            grad=grad)

    def pre_calc(self, key: str, val: ShapeImage):
        bs = val.batchsize
        dimension = val.ndim
        self.calc_scale(batchsize=bs, dimension=dimension)


class RandomTranslation(RandomParametrizedAffine):
    def __init__(self,
                 translation_range: tuple,
                 shapeimage_query_keys: tuple = ('data',),
                 output_size: tuple = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False):
        super().__init__(
            scale_range=(None, None),
            translation_range=translation_range,
            rotation_range=(None, None),
            shapeimage_query_keys=shapeimage_query_keys,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            degree=True,
            grad=grad)

    def pre_calc(self, key: str, val: ShapeImage):
        bs = val.batchsize
        dimension = val.ndim
        self.calc_translation(batchsize=bs, dimension=dimension)
