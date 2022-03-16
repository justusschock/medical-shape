import numpy as np
import torch
import torchio as tio
from rising.utils.affine import points_to_cartesian, points_to_homogeneous

from medical_shape.shape import Shape
from medical_shape.transforms.mixin import TransformShapeValidationMixin


class ShapeToOrientation(TransformShapeValidationMixin):
    def __init__(self, affine: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.affine = affine

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        for k, v in subject.items():
            if isinstance(v, Shape):

                trafo = np.linalg.inv(v.affine) @ self.affine

                hom_points = points_to_homogeneous(v.tensor[None])[0].numpy()

                points_transformed = (trafo @ hom_points.T).T

                points_transformed = points_to_cartesian(torch.from_numpy(np.array(points_transformed))[None])[0]

                v.set_data(points_transformed.to(v.tensor))
                v.affine = self.affine
                subject[k] = v

        return subject
