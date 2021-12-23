import torchio as tio
from pytorch_lightning.utilities.warnings import WarningCache

_warning_cache = WarningCache()


class TransformShapeValidationMixin(tio.transforms.Transform):
    def add_transform_to_subject_history(self, subject):
        from shape.subject import ShapeSupportSubject

        if not isinstance(subject, ShapeSupportSubject):
            _warning_cache.warn(
                "Using a shape transformation without an explicit ShapeSupportSubject may lead to unexpected behaviour",
                UserWarning,
            )
        return super().add_transform_to_subject_history(subject)
