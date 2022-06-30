from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from rising.utils.affine import points_to_cartesian, points_to_homogeneous

from medical_shape.shape import Shape
from medical_shape.subject import ShapeSupportSubject
from medical_shape.transforms.mixin import TransformShapeValidationMixin


class ToOrientation(TransformShapeValidationMixin):
    def __init__(
        self,
        axcode: Optional[Sequence[str]] = None,
        affine: Optional[np.ndarray] = None,
        shape_trafo_image_size: Optional[Tuple[int, int, int]] = None,
        shape_trafo_image_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if axcode is None and affine is None:
            raise ValueError("Either affine or axcode have to be specified!")

        if axcode is not None and affine is not None:
            raise ValueError("Cannot specify both affine and axcode!")

        if affine is not None:
            axcode = nib.orientations.aff2axcodes(affine)

        self.dst_ornt = nib.orientations.axcodes2ornt(axcode)

        if shape_trafo_image_size is not None and shape_trafo_image_key is not None:
            raise ValueError("Cannot specify both shape_trafo_image_size and shape_trafo_image_key!")
        self.shape_trafo_image_size = deepcopy(shape_trafo_image_size)
        self.shape_trafo_image_key = deepcopy(shape_trafo_image_key)

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:

        shapes_dict: Union[Dict[str, Shape], Dict] = getattr(subject, "get_shapes_dict", lambda: {})()
        shape_trafo_image_size = (0, 0, 0)
        if shapes_dict:
            if self.shape_trafo_image_size is not None:
                shape_trafo_image_size = self.shape_trafo_image_size
            elif self.shape_trafo_image_key is not None:
                shape_trafo_image_size = deepcopy(subject[self.shape_trafo_image_key].spatial_shape)
            elif len(ShapeSupportSubject.exclude_shapes(subject.get_images_dict(intensity_only=False))) == 1:
                shape_trafo_image_size = deepcopy(subject.get_first_image().spatial_shape)
            else:
                raise ValueError(
                    "Either a shape_trafo_image_size or a shape_trafo_image_key "
                    "has to be specified or the subject should only contain a single Image."
                )

        for k, v in subject.get_images_dict(intensity_only=False).items():
            src_ornt = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(v.affine))
            ornt_trafo = nib.orientations.ornt_transform(src_ornt, self.dst_ornt)

            if isinstance(v, Shape):
                trafo_affine = nib.orientations.inv_ornt_aff(ornt_trafo, shape_trafo_image_size)

                hom_points = points_to_homogeneous(v.tensor[None])[0].numpy()
                points_transformed = (trafo_affine @ hom_points.T).T

                points_transformed = points_to_cartesian(torch.from_numpy(np.array(points_transformed))[None])[0]

                v.set_data(points_transformed.to(v.tensor))
                v.affine = v.affine.dot(trafo_affine)
            else:
                # adapted from https://github.com/fepegar/torchio/blob/main/src/torchio/transforms/preprocessing/spatial/to_canonical.py
                array = v.numpy()[np.newaxis]  # (1, C, W, H, D)
                # NIfTI images should have channels in 5th dimension
                array = array.transpose(2, 3, 4, 0, 1)  # (W, H, D, 1, C)
                nii = nib.Nifti1Image(array, v.affine)
                reoriented = nii.as_reoriented(ornt_trafo)
                # https://nipy.org/nibabel/reference/nibabel.dataobj_images.html#nibabel.dataobj_images.DataobjImage.get_data
                array = np.asanyarray(reoriented.dataobj)
                # https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
                array = array.copy()
                array = array.transpose(3, 4, 0, 1, 2)  # (1, C, W, H, D)
                v.set_data(torch.as_tensor(array[0]))
                v.affine = reoriented.affine

            subject[k] = v

        return subject
