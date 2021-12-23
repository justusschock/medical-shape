import nibabel as nib
import numpy as np
import torch
import torchio as tio
from rising.transforms.functional.affine import affine_point_transform

from shape.subject import ShapeSupportSubject
from shape.transforms.mixin import TransformShapeValidationMixin


class ToCanonical(
    tio.transforms.preprocessing.ToCanonical, TransformShapeValidationMixin
):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        sub = dict(
            super().apply_transform(
                getattr(subject, "get_images_only_subject", lambda: subject)()
            )
        )

        for k, v in getattr(subject, "get_shapes_dict", lambda: {})().items():
            affine_pre = v.affine
            if nib.aff2axcodes(affine_pre) == tuple("RAS"):
                sub[k] = v
                continue
            # some dummy array to make use of nibabel
            array = np.zeros((2, 2, 2, 1, 1))  # (W, H, D, 1, C)
            nii = nib.Nifti1Image(array, affine_pre)
            reoriented = nib.as_closest_canonical(nii)
            affine_post = reoriented.affine

            trafo = torch.tensor(
                affine_pre, dtype=torch.float
            ).inverse() @ torch.tensor(affine_post, dtype=torch.float)

            transformed_points = affine_point_transform(v.tensor[None], trafo[None])[0]
            v.set_data(transformed_points)
            v.affine = affine_post
            sub[k] = v

        return type(subject)(sub)
