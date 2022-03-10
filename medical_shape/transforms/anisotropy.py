import warnings

import torchio as tio

from medical_shape.transforms.mixin import TransformShapeValidationMixin
from medical_shape.transforms.resample import Resample


class RandomAnisotropy(tio.transforms.augmentation.spatial.RandomAnisotropy, TransformShapeValidationMixin):

    # copy paste to use custom resample trafo
    def apply_transform(self, subject):
        is_2d = subject.get_first_image().is_2d()
        if is_2d and 2 in self.axes:
            warnings.warn(
                f'Input image is 2D, but "2" is in axes: {self.axes}',
                RuntimeWarning,
            )
            self.axes = list(self.axes)
            self.axes.remove(2)
        axis, downsampling = self.get_params(
            self.axes,
            self.downsampling_range,
        )
        target_spacing = list(subject.spacing)
        target_spacing[axis] *= downsampling

        arguments = {
            "image_interpolation": "nearest",
            "scalars_only": self.scalars_only,
        }

        downsample = Resample(target=tuple(target_spacing), **self.add_include_exclude(arguments))
        downsampled = downsample(subject)
        image = subject.get_first_image()
        target = image.spatial_shape, image.affine
        upsample = Resample(
            target=target,
            image_interpolation=self.image_interpolation,
            scalars_only=self.scalars_only,
        )
        upsampled = upsample(downsampled)
        return upsampled
