import nibabel as nib
import torch
import torchio as tio

from shape.subject import ShapeSupportSubject
from shape.transforms.mixin import TransformShapeValidationMixin


class Crop(tio.transforms.Crop, TransformShapeValidationMixin):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        sub = dict(
            super().apply_transform(
                getattr(subject, "get_images_only_subject", lambda: subject)()
            )
        )
        shapes = getattr(subject, "get_shapes_dict", lambda: {})()

        index_ini = torch.tensor(self.bounds_parameters[::2], dtype=torch.float)
        for k, v in shapes.items():
            # TODO: Handle out of bounds
            v.set_data(v.tensor - index_ini[None])
            new_affine = v.affine.copy()
            new_origin = nib.affines.apply_affine(v.affine, index_ini.numpy())
            new_affine[:3, 3] = new_origin
            v.affine = new_affine
            sub[k] = v

        return type(subject)(sub)