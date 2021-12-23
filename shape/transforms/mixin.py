import torchio as tio
import warnings

_warning_cache = set()


class TransformShapeValidationMixin(tio.transforms.Transform):
    def add_transform_to_subject_history(self, subject):
        from shape.subject import ShapeSupportSubject

        if not isinstance(subject, ShapeSupportSubject):
            message = "Using a shape transformation without an explicit "
            "ShapeSupportSubject may lead to unexpected behaviour"
            
            if message not in _warning_cache:
                _warning_cache.add(message)
                warnings.warn(message, UserWarning)
        return super().add_transform_to_subject_history(subject)
