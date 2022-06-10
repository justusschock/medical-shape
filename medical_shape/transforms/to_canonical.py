from copy import deepcopy
from typing import Dict, Optional, Union

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from rising.utils.affine import points_to_cartesian, points_to_homogeneous

from medical_shape.shape import Shape
from medical_shape.subject import ShapeSupportSubject
from medical_shape.transforms.mixin import TransformShapeValidationMixin


class ToCanonical(tio.transforms.preprocessing.ToCanonical, TransformShapeValidationMixin):
    def __init__(self, shape_image_key: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, parse_input=False, **kwargs)
        self.shape_image_key = shape_image_key

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:

        shapes_dict: Union[Dict[str, Shape], Dict] = getattr(subject, "get_shapes_dict", lambda: {})()
        if shapes_dict:
            if isinstance(self.shape_image_key, str):
                shape_image_key = self.shape_image_key
            elif (
                self.shape_image_key is None
                and len(ShapeSupportSubject.exclude_shapes(subject.get_images_dict(intensity_only=False))) == 1
            ):
                shape_image_key = list(subject.get_images_dict(intensity_only=False).keys())[0]
            else:
                raise ValueError(
                    f"shape_image_key must be a string or None if there is only one image in the subject. Got {self.shape_image_key}"
                )

            src_ref_shape = torch.tensor(deepcopy(subject[shape_image_key].spatial_shape))

        new_sub = {}
        try:
            new_sub.update(
                dict(super().apply_transform(getattr(subject, "get_images_only_subject", lambda: subject)()))
            )
        except ValueError:
            new_sub.update(ShapeSupportSubject.exclude_shapes(dict(subject)))

        for k, v in shapes_dict.items():

            affine_pre = v.affine
            if nib.aff2axcodes(affine_pre) == tuple("RAS"):
                new_sub[k] = v
                continue

            else:
                affine_post = affine_pre.dot(
                    nib.orientations.inv_ornt_aff(
                        nib.orientations.io_orientation(affine_pre),
                        src_ref_shape.numpy(),
                    )
                )

                hom_points = points_to_homogeneous(v.tensor[None])[0].numpy()

                trafo = np.linalg.inv(affine_pre) @ affine_post

                points_transformed = (trafo @ hom_points.T).T

                points_transformed = points_to_cartesian(torch.from_numpy(np.array(points_transformed))[None])[0]

                v.set_data(points_transformed.to(v.tensor))
                v.affine = affine_post
                new_sub[k] = v

        return type(subject)(new_sub)
