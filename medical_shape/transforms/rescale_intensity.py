from mdlu.transforms import RescaleIntensityPercentiles as _RescaleIntensityPercentiles

from medical_shape.transforms.mixin import TransformShapeValidationMixin


# Need to subclass from TransformShapeValidationMixin as that one fixes kwargs
class RescaleIntensityPercentiles(_RescaleIntensityPercentiles, TransformShapeValidationMixin):
    pass
