from shape_image.transforms.base import ShapeImageFunctionTransform


class CropAndResize(ShapeImageFunctionTransform):
    """
    Applies the given crop and resize to every shapeimage feed to a object of this class
    Parametrize the crop and resize by giving a cropping_proportion and a new_size
    """

    def __init__(self, new_size,
                 cropping_proportion: float = 0.0,
                 keep_image_proportions: bool = False,
                 shapeimage_query_keys: tuple = ('data',),
                 interpolation_mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 grad: bool = False):
        super().__init__('crop_to_shape_and_resize',
                         new_size=new_size,
                         keep_image_proportions=keep_image_proportions,
                         cropping_proportion=cropping_proportion,
                         shapeimage_query_keys=shapeimage_query_keys,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         grad=grad)


class ZoomAndResize(ShapeImageFunctionTransform):
    """
    Applies the given zoom and resize to every shapeimage feed to a object of this class
    Parametrize the zoom and resize by giving a zoom ratio and a new_size
    """

    def __init__(self, zoom: float,
                 new_size: tuple, 
                 shapeimage_query_keys: tuple = ('data',)):
        super().__init__('zoom_and_resize',
                         zoom=zoom,
                         new_size=new_size,
                         shapeimage_query_keys=shapeimage_query_keys)


class Crop(ShapeImageFunctionTransform):
    """
    Applies the given crop to every shapeimage feed to a object of this class
    Parametrize the crop by giving a proportion
    """

    def __init__(self,
                 proportion: float = 0.0,
                 shapeimage_query_keys: tuple = ('data',),
                 grad: bool = False):
        super().__init__('crop_to_shape',
                         proportion=proportion,
                         shapeimage_query_keys=shapeimage_query_keys,
                         grad=grad)
