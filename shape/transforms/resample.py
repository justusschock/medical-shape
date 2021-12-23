import torch
import torchio as tio
from torchio.transforms.spatial_transform import SpatialTransform

from shape.normalization import ShapeNormalization
from shape.subject import ShapeSupportSubject


class Resample(tio.transforms.Resample):
    def apply_transform(self, subject: ShapeSupportSubject) -> ShapeSupportSubject:
        pre_affine = subject.get_first_image().affine
        pre_shape = subject.spatial_shape
        sub = super().apply_transform(
            getattr(subject, "get_images_only_subject", lambda: subject)()
        )
        sub_dict = dict(sub)
        post_affine = sub.get_first_image().affine
        post_shape = sub.spatial_shape

        for k, v in getattr(subject, "get_shapes_dict", lambda: {})().items():
            v.set_data(ShapeNormalization.denormalize(
                ShapeNormalization.normalize(v.tensor, pre_shape), post_shape
            ))
            trafo = torch.tensor(
                pre_affine, dtype=torch.float
            ).inverse() @ torch.tensor(post_affine, dtype=torch.float)
            v.affine = (
                (torch.tensor(v.affine, dtype=torch.float) @ trafo)
                .numpy()
                .astype(v.affine.dtype)
            )
            sub_dict[k] = v
        return type(subject)(sub_dict)
