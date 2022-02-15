import torchio as tio
from shape.transforms.mixin import TransformShapeValidationMixin


# Need to subclass from TransformShapeValidationMixin as that one fixes kwargs
class CopyAffine(tio.transforms.CopyAffine, TransformShapeValidationMixin):
    pass
