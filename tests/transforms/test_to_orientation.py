import os
from copy import deepcopy

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from medical_shape import Shape, ShapeSupportSubject
from medical_shape.transforms.to_orientation import (
    _shape_to_orientation_v1,
    image_to_orientation,
    shape_to_orientation_v2,
    ToOrientation,
)


@pytest.mark.parametrize(
    "input_dict,expected_dst_ornt,expected_trafo_image_size,expected_trafo_image_key",
    [
        ({"axcode": "RAS"}, np.array([[0, 1], [1, 1], [2, 1]]), None, None),
        (
            {"affine": np.eye(4), "shape_trafo_image_size": (1, 2, 3)},
            np.array([[0, 1], [1, 1], [2, 1]]),
            (1, 2, 3),
            None,
        ),
        (
            {
                "affine": np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
                "shape_trafo_image_key": "image",
            },
            np.array([[1, -1], [2, -1], [0, 1]]),
            None,
            "image",
        ),
    ],
)
def test_to_orientation_inputs(input_dict, expected_dst_ornt, expected_trafo_image_size, expected_trafo_image_key):
    trafo = ToOrientation(**input_dict)
    assert np.allclose(trafo.dst_ornt, expected_dst_ornt)
    if expected_trafo_image_size is None:
        assert trafo.shape_trafo_image_size is None
    else:
        assert trafo.shape_trafo_image_size == expected_trafo_image_size

    if expected_trafo_image_key is None:
        assert trafo.shape_trafo_image_key is None
    else:
        assert trafo.shape_trafo_image_key == expected_trafo_image_key


def test_image_to_orientation():
    image = tio.data.ScalarImage(
        tensor=torch.rand(1, 100, 200, 300),
        affine=np.array([[0.0, 0.0, 0.5, 0], [-0.6, 0.0, 0.0, 0.0], [0.0, -0.7, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )

    to_canonical = tio.transforms.ToCanonical()
    canonical_image_tio = to_canonical(tio.data.Subject(image=image))["image"]
    assert canonical_image_tio.orientation == ("R", "A", "S")

    oriented_image = image_to_orientation(image, nib.orientations.axcodes2ornt("RAS"))
    assert oriented_image.orientation == canonical_image_tio.orientation

    assert np.allclose(oriented_image.affine, canonical_image_tio.affine)
    assert torch.allclose(oriented_image.tensor, canonical_image_tio.tensor)


@pytest.mark.xfail(raises=AssertionError)
def test_shapes_trafo_no_images(data_path):
    shape = Shape(os.path.join(data_path, "tests", "test_shape_with_affine_with_descriptions.mjson"))

    transformed_shape_v1 = _shape_to_orientation_v1(
        deepcopy(shape), nib.orientations.axcodes2ornt("LPS"), (100, 200, 300)
    )
    transformed_shape_v2 = shape_to_orientation_v2(
        deepcopy(shape), nib.orientations.axcodes2ornt("LPS"), (100, 200, 300)
    )

    assert np.allclose(transformed_shape_v1.affine, transformed_shape_v2.affine)
    assert torch.allclose(transformed_shape_v1.tensor, transformed_shape_v2.tensor)

    assert transformed_shape_v2.orientation == ("L", "P", "S")
    assert transformed_shape_v1.orientation == ("L", "P", "S")

    assert torch.allclose(transformed_shape_v1.tensor, transformed_shape_v2.tensor)
