from copy import deepcopy

import torchio as tio

from medical_shape.normalization import ShapeNormalization
from medical_shape.subject import ShapeSupportSubject
from medical_shape.transforms.mixin import TransformShapeValidationMixin


# Need to subclass from TransformShapeValidationMixin as that one fixes kwargs
class Resample(tio.transforms.Resample, TransformShapeValidationMixin):
    def apply_transform(self, subject: ShapeSupportSubject) -> ShapeSupportSubject:
        pre_shape = deepcopy(subject.spatial_shape)
        sub = super().apply_transform(getattr(subject, "get_images_only_subject", lambda: subject)())
        sub_dict = dict(sub)
        post_affine = sub.get_first_image().affine
        post_shape = sub.spatial_shape

        for k, v in getattr(subject, "get_shapes_dict", lambda: {})().items():
            v.set_data(ShapeNormalization.denormalize(ShapeNormalization.normalize(v.tensor, pre_shape), post_shape))

            v.affine = post_affine
            sub_dict[k] = v
        return type(subject)(sub_dict)
