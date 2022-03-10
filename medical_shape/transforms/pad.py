from typing import Dict, Union

import nibabel as nib
import torch
import torchio as tio

from medical_shape import ShapeSupportSubject
from medical_shape.shape import Shape
from medical_shape.transforms.mixin import TransformShapeValidationMixin


class Pad(tio.transforms.Pad, TransformShapeValidationMixin):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        if not isinstance(subject, ShapeSupportSubject) or subject.get_images_dict(
            intensity_only=False, include_shapes=False
        ):
            sub = dict(super().apply_transform(getattr(subject, "get_images_only_subject", lambda: subject)()))
        else:
            sub = {}
        shapes: Union[Dict[str, Shape], Dict] = getattr(subject, "get_shapes_dict", lambda: {})()

        index_ini = torch.tensor(self.bounds_parameters[::2], dtype=torch.float)
        for k, v in shapes.items():
            v.set_data(v.tensor + index_ini[None])
            new_affine = v.affine.copy()
            new_origin = nib.affines.apply_affine(v.affine, -index_ini.numpy())
            new_affine[:3, 3] = new_origin
            v.affine = new_affine
            sub[k] = v

        return type(subject)(sub)
