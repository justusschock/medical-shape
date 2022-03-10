import torchio as tio

from medical_shape.transforms.mixin import TransformShapeValidationMixin


# Need to subclass from TransformShapeValidationMixin as that one fixes kwargs
class CopyAffine(tio.transforms.CopyAffine, TransformShapeValidationMixin):
    pass
