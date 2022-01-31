import torchio as tio

from shape.transforms.crop_or_pad import CropOrPad


class CropOrPadPerImage(CropOrPad):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        for k, v in subject.get_images_dict(intensity_only=False, include=self.include, exclude=self.exclude).items():
            part_sub = type(subject)({k: v})
            subject[k] = super().apply_transform(part_sub)[k]
        return subject
