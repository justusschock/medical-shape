from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Sequence, Union

import torchio as tio
from pytorch_lightning.utilities.warnings import WarningCache

from shape.shape import Shape

_warning_cache = WarningCache()


class ShapeSupportSubject(tio.data.Subject):
    # TODO: remove after https://github.com/fepegar/torchio/pull/718
    def __copy__(self):
        import copy
        result_dict = {}
        for key, value in self.items():
            if isinstance(value, tio.data.Image):
                value = copy.copy(value)
            else:
                value = copy.deepcopy(value)
            result_dict[key] = value
        new = self.__class__(result_dict)
        new.applied_transforms = self.applied_transforms[:]
        return new

    @staticmethod
    def exclude_shapes(
        collection: Union[Iterable, Mapping]
    ) -> Union[Iterable, Mapping]:
        if isinstance(collection, Mapping):
            return {k: v for k, v in collection.items() if not isinstance(v, Shape)}
        elif isinstance(collection, Iterable):
            return [v for v in collection if not isinstance(v, Shape)]

        raise TypeError(
            "exclude_shapes is only implemented for Mappings and Sequences currently"
        )

    def get_images(
        self,
        intensity_only=True,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[tio.data.Image]:
        return self.exclude_shapes(
            super().get_images(
                intensity_only=intensity_only, include=include, exclude=exclude
            )
        )

    def get_images_dict(
        self,
        intensity_only=True,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> Dict[str, tio.data.Image]:
        return self.exclude_shapes(
            super().get_images_dict(
                intensity_only=intensity_only, include=include, exclude=exclude
            )
        )

    def get_images_only_subject(self):
        return tio.data.Subject(**self.exclude_shapes(self))

    def get_shapes_dict(self):
        return {k: v for k, v in self.items() if isinstance(v, Shape)}

    def add_transform(
        self, transform: tio.transforms.Transform, parameters_dict: Dict[str, Any]
    ) -> None:
        from shape.transforms.mixin import TransformShapeValidationMixin

        if not isinstance(transform, TransformShapeValidationMixin) and bool(
            self.get_shapes_dict()
        ):
            _warning_cache.warn(
                "Using the ShapeSupportSubjects together with one or more Shape instances and "
                "the original torchio transforms can result in unexpected behaior since these "
                "transforms do not support shapes natively!",
                UserWarning,
            )
        return super().add_transform(transform, parameters_dict)
