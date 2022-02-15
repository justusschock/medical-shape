import os

import pytest
import torch
import torchio as tio
from shape.shape import Shape
from shape.subject import ShapeSupportSubject
from shape.transforms import Crop, CropOrPad, Pad, Resample, ToCanonical


def clear_warning_caches():
    from shape.subject import _warning_cache as cache_subject
    from shape.transforms.mixin import _warning_cache as cache_transform

    cache_subject.clear()
    cache_transform.clear()


def save(subject: ShapeSupportSubject, path):
    for k, v in subject.items():
        if isinstance(v, Shape):
            v.save(os.path.join(path, f"{k}.pts"))
        elif isinstance(v, tio.data.Image):
            v.save(os.path.join(path, f"{k}.nii.gz"))


def create_subject():
    torch.manual_seed(42)
    random_image = torch.rand(1, 160, 384, 384)
    random_shape = torch.stack(
        [torch.randint(0, s, (200,)) for s in random_image.shape[1:]], -1
    )
    subject = ShapeSupportSubject(
        s=Shape(tensor=random_shape),
        i=tio.data.ScalarImage(tensor=random_image),
    )
    return subject


def test_io(tmpdir):
    subject = create_subject()
    save(subject, tmpdir)
    Shape(os.path.join(tmpdir, "s.pts"))


@pytest.mark.parametrize(
    "trafo",
    [
        tio.transforms.CopyAffine("i"),
        Crop((2, 2, 2, 2, 2, 2)),
        Pad((2, 2, 2, 2, 2, 2)),
        CropOrPad((155, 384, 390)),
        ToCanonical(),
        Resample([2, 1, 1]),
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


def test_warning_wrong_subject_type():
    clear_warning_caches()
    trafo = ToCanonical()
    with pytest.warns(
        UserWarning,
        match="Using a shape transformation without an explicit ShapeSupportSubject may lead to unexpected behaviour",
    ):
        # breakpoint()
        trafo(create_subject().get_images_only_subject())


def test_warning_wrong_trafo_type():
    clear_warning_caches()
    trafo = tio.transforms.ToCanonical()
    with pytest.warns(
        UserWarning,
        match="Using the ShapeSupportSubjects together with one or more Shape instances and "
        "the original torchio transforms can result in unexpected behaior since these "
        "transforms do not support shapes natively!",
    ):
        trafo(create_subject())
