import os

import pytest
import torch
import torchio as tio

from medical_shape.shape import Shape
from medical_shape.subject import ShapeSupportSubject
from medical_shape.transforms import Crop, CropOrPad, Pad, Resample, ToCanonical


def clear_warning_caches():
    from medical_shape.subject import _warning_cache as cache_subject
    from medical_shape.transforms.mixin import _warning_cache as cache_transform

    cache_subject.clear()
    cache_transform.clear()


def save(subject: ShapeSupportSubject, path, shape_extension: str):
    for k, v in subject.items():
        if isinstance(v, Shape):
            v.save(os.path.join(path, f"{k}{shape_extension}"))
        elif isinstance(v, tio.data.Image):
            v.save(os.path.join(path, f"{k}.nii.gz"))


def create_subject():
    torch.manual_seed(42)
    random_image = torch.rand(1, 160, 384, 384)
    random_shape = torch.stack([torch.randint(0, s, (200,)) for s in random_image.shape[1:]], -1)
    subject = ShapeSupportSubject(
        s=Shape(tensor=random_shape),
        i=tio.data.ScalarImage(tensor=random_image),
    )
    return subject


@pytest.mark.parametrize(
    "shape_extension",
    [
        ".pts",
        ".mjson",
    ],
)
def test_io(tmpdir, shape_extension):
    subject = create_subject()
    save(subject, tmpdir, shape_extension)
    Shape(os.path.join(tmpdir, f"s{shape_extension}"))


@pytest.mark.parametrize(
    "trafo",
    [
        tio.transforms.CopyAffine("i", parse_input=False),
        Crop((2, 2, 2, 2, 2, 2)),
        Pad((2, 2, 2, 2, 2, 2)),
        CropOrPad((155, 384, 390)),
        ToCanonical(),
        Resample([2, 1, 1], parse_input=False),
    ],
)
def test_transforms(trafo):
    subject = create_subject()
    transformed_subject = trafo(subject)
    assert isinstance(transformed_subject, ShapeSupportSubject)

    transformed_subject = trafo(subject.get_images_only_subject())
    assert not isinstance(transformed_subject, ShapeSupportSubject) and isinstance(
        transformed_subject, tio.data.Subject
    )
