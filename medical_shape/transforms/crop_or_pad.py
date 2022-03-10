import torchio as tio

from medical_shape.transforms.crop import Crop
from medical_shape.transforms.mixin import TransformShapeValidationMixin
from medical_shape.transforms.pad import Pad


class CropOrPad(tio.transforms.CropOrPad, TransformShapeValidationMixin):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {"padding_mode": self.padding_mode}
        if padding_params is not None:
            subject = Pad(padding_params, **padding_kwargs)(subject)
        if cropping_params is not None:
            subject = Crop(cropping_params)(subject)
        return subject
