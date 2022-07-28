import os
import re
from copy import copy, deepcopy
from typing import Sequence

import numpy as np
import pytest
import torch
import torchio as tio

from medical_shape.shape import POINT_DESCRIPTIONS, Shape, SHAPE



def test_shape_init_tensor_only():
    shape = Shape(tensor=torch.rand(10, 3))
    assert np.allclose(shape.affine, np.eye(4))
    assert shape.tensor.size() == (10, 3)
    assert shape.point_descriptions == None
    assert shape.path == None


def test_shape_init_tensor_affine():
    affine = np.random.rand(4, 4)
    shape = Shape(tensor=torch.rand(10, 3), affine=affine)
    assert np.allclose(affine, shape.affine)
    assert shape.tensor.size() == (10, 3)
    assert shape.point_descriptions == None
    assert shape.path == None


def test_shape_init_pts_file(data_path):
    shape = Shape(path=os.path.join(data_path, "tests", "test_shape.pts"))
    assert shape.tensor.shape == (10, 3)
    assert np.allclose(shape.affine, np.eye(4))
    assert shape.point_descriptions == None
    assert str(shape.path) == str(os.path.join(data_path, "tests", "test_shape.pts"))


@pytest.mark.parametrize("affine,descriptions", [(True, True), (True, False), (False, True), (True, True)])
def test_shape_init_mjson_file(affine, descriptions, data_path):
    file_name = str(os.path.join(data_path, "tests", "test_shape_"))
    if affine:
        file_name += "with_affine"
    else:
        file_name += "without_affine"


    file_name += ""

    if descriptions:
        file_name += "with_descriptions"
    else:
        file_name += "without_descriptions"

    file_name += ".mjson"

    shape = Shape(path=file_name)
    shape.load()

    if affine:
        assert np.allclose(shape.affine, np.arange(16).reshape(4, 4) / 15)
    else:
        assert np.allclose(shape.affine, np.eye(4))

    if descriptions:
        assert shape.point_descriptions == tuple([f"Loaded Point {i:02d}" for i in range(shape.tensor.size(0))])
    else:
        assert shape.point_descriptions == None

    assert torch.allclose(torch.arange(30).view(-1, 3).float(), shape.tensor)


@pytest.mark.parametrize(
    "key,should_load",
    [
        (tio.constants.DATA, True),
        (tio.constants.AFFINE, True),
        (tio.constants.TYPE, False),
        (tio.constants.PATH, False),
        (tio.constants.STEM, False),
    ],
)
def test_getitem_loading(key: str, should_load: bool, data_path):
    shape = Shape(os.path.join(data_path, "tests", "test_shape.pts"))

    assert shape._loaded == False

    shape[key]

    assert shape._loaded == should_load


def test_should_raise_getitem_loading(data_path):
    shape = Shape(os.path.join(data_path, "tests", "test_shape.pts"))
    assert shape._loaded == False

    with pytest.raises(KeyError, match=POINT_DESCRIPTIONS):
        shape[POINT_DESCRIPTIONS]


@pytest.mark.parametrize(
    "load,expected",
    [
        (False, "Shape(path: {shape_file})"),
        (
            True,
            "Shape(shape: (1, 27.0, 27.0, 27.0); spacing: (1.00, 1.00, 1.00); orientation: RAS+; dtype: torch.FloatTensor; memory: 76.9 KiB)",
        ),
    ],
)
def test_repr(load, expected, data_path):
    shape_file = os.path.join(data_path, "tests", "test_shape.pts")
    expected = expected.format(shape_file=shape_file)
    shape = Shape(shape_file)
    if load:
        shape.load()

    assert shape._loaded == load
    assert repr(shape) == expected


@pytest.mark.parametrize("should_load,copy_deep", [(True, True), (True, False), (False, True), (False, False)])
def test_copy(data_path, should_load: bool, copy_deep):
    shape = Shape(os.path.join(data_path, "tests", "test_shape.pts"))

    if should_load:
        shape.load()

    assert shape._loaded == should_load
    if copy_deep:
        copied_shape = deepcopy(shape)
    else:
        copied_shape = copy(shape)

    assert shape._loaded
    assert copied_shape._loaded

    # check that values are equal
    assert torch.allclose(copied_shape.tensor, shape.tensor)
    assert np.allclose(copied_shape.affine, shape.affine)
    assert copied_shape.point_descriptions == shape.point_descriptions
    assert copied_shape.path == shape.path
    assert copied_shape.type == shape.type

    # check that shapes are different objects
    assert copied_shape is not shape

    # with deepcopy, all parts of shape should be different too
    if copy_deep:
        assert copied_shape.tensor is not shape.tensor
        assert copied_shape.affine is not shape.affine
    # with shallow copy, all parts of shape should be the same
    else:
        assert copied_shape.tensor is shape.tensor
        assert copied_shape.affine is shape.affine


@pytest.mark.parametrize(
    "file_name",
    [
        "test_shape.pts",
        "test_shape_with_affine_with_descriptions.mjson",
        "test_shape_with_affine_without_descriptions.mjson",
        "test_shape_without_affine_with_descriptions.mjson",
        "test_shape_without_affine_without_descriptions.mjson",
    ],
)
def test_read_and_check(data_path, file_name):
    # very basic check. More detailed checks wll be done for every single parser and reader
    absolute_file = os.path.join(data_path, "tests", file_name)
    # make this static once the super-method is static
    shape = Shape(absolute_file)

    tensor, affine, descriptions = shape.read_and_check(absolute_file)

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(affine, np.ndarray)
    assert descriptions is None or (
        isinstance(descriptions, Sequence) and all(isinstance(x, str) for x in descriptions)
    )


@pytest.mark.parametrize(
    "file_name",
    [
        "test_shape.pts",
        "test_shape_with_affine_with_descriptions.mjson",
        "test_shape_with_affine_without_descriptions.mjson",
        "test_shape_without_affine_with_descriptions.mjson",
        "test_shape_without_affine_without_descriptions.mjson",
    ],
)
def test_loading(data_path, file_name):
    absolute_path = os.path.join(data_path, "tests", file_name)
    keys_to_check = (POINT_DESCRIPTIONS, tio.constants.AFFINE, tio.constants.DATA)

    shape = Shape(absolute_path)
    assert shape._loaded == False
    for key in keys_to_check:
        assert key not in shape

    shape.load()

    assert shape._loaded == True
    for key in keys_to_check:
        assert key in shape


@pytest.mark.parametrize("point_descriptions", [None, ["Point0", "Point1", "Point2", "Point3"]])
def test_set_data(point_descriptions):
    original_data = torch.randn(4, 3)
    shape = Shape(tensor=original_data, point_descriptions=point_descriptions)
    assert torch.allclose(original_data, shape.tensor)
    new_data = torch.randn(4, 3)
    assert not torch.allclose(original_data, new_data)
    shape.set_data(new_data)
    assert torch.allclose(new_data, shape.tensor)


def test_set_data_error():
    original_data = torch.randn(4, 3)
    point_descriptions = ["Point0", "Point1", "Point2", "Point3"]

    shape = Shape(tensor=original_data, point_descriptions=point_descriptions)
    assert torch.allclose(shape.tensor, original_data)

    new_data = torch.randn(5, 3)
    with pytest.raises(
        ValueError, match=re.escape("Number of point descriptions (4) does not match number of points (5)")
    ):
        shape.set_data(new_data)


def test_affine_loading(data_path):
    shape = Shape(path=os.path.join(data_path, "tests", "test_shape.pts"))

    assert shape._loaded == False
    assert np.allclose(np.eye(4), shape.affine)
    assert shape._loaded == True


def test_affine_setting_is_parsed():
    shape = Shape(tensor=torch.randn(4, 3), affine=np.random.rand(4, 4))
    assert not np.allclose(np.eye(4), shape.affine)
    assert shape.affine.dtype == np.float64

    new_affine = np.eye(4).astype(np.float16)
    shape.affine = new_affine
    assert new_affine.dtype != shape.affine.dtype

    assert np.allclose(new_affine, shape.affine)


@pytest.mark.parametrize(
    "input_affine,output_affine",
    [
        (
            (np.arange(16).reshape(4, 4) / 15).astype(np.float64),
            (np.arange(16).reshape(4, 4) / 15).astype(np.float64),
        ),
        (
            None,
            np.eye(4).astype(np.float64),
        ),
        (
            torch.arange(16, dtype=torch.float32).view(4, 4) / 15,
            (np.arange(16).reshape(4, 4) / 15).astype(np.float64),
        ),
        (
            (np.arange(16).reshape(4, 4) / 15).astype(np.float32),
            (np.arange(16).reshape(4, 4) / 15).astype(np.float64),
        ),
    ],
)
def test_affine_parsing(input_affine, output_affine):
    # TODO: make static once changed in tio
    shape = Shape(tensor=torch.randn(4, 3))
    assert np.allclose(shape._parse_affine(input_affine), output_affine)


@pytest.mark.parametrize("bad_val", [5, np.eye(4).tolist(), "abc", 4.0])
def test_affine_parsing_typeerror(bad_val):
    shape = Shape(tensor=torch.randn(4, 3))
    with pytest.raises(TypeError, match=f"Affine must be a NumPy array, not {type(bad_val)}"):
        shape._parse_affine(bad_val)


@pytest.mark.parametrize("bad_shape", [(3, 3), (3, 4), (4, 3), (700,), (1, 2)])
def test_affine_parsing_valueerror(bad_shape):
    shape = Shape(tensor=torch.randn(4, 3))

    with pytest.raises(ValueError, match=re.escape(f"Affine shape must be (4, 4), not {bad_shape}")):
        shape._parse_affine(np.random.rand(*bad_shape))


@pytest.mark.parametrize(
    "input_tensor,expected",
    [
        (None, None),
        (np.array([[3, 4, 5]]), torch.tensor([[3, 4, 5]], dtype=torch.float32)),
        (torch.tensor([[True, True, True]]), torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)),
    ],
)
def test_parse_tensor(input_tensor, expected):
    shape = Shape(tensor=torch.randn(4, 3))
    output = shape._parse_tensor(input_tensor)

    if input_tensor is None:
        assert output is None
    else:
        # to ensure shapes are correct also without broadcasting
        assert output.ndim == expected.ndim
        assert output.shape == expected.shape
        assert torch.allclose(output, expected)


def test_parse_tensor_none_error():
    shape = Shape(tensor=torch.randn(4, 3))

    with pytest.raises(RuntimeError, match="Input tensor cannot be None"):
        shape._parse_tensor(None, none_ok=False)


@pytest.mark.parametrize("bad_val", [1, 1.0, "abc", [[1, 2, 3]]])
def test_parse_tensor_type_error(bad_val):
    shape = Shape(tensor=torch.randn(4, 3))

    with pytest.raises(
        TypeError,
        match=f"Input tensor must be a PyTorch tensor or NumPy array, but type {type(bad_val)} was found",
    ):
        shape._parse_tensor(bad_val)


@pytest.mark.parametrize("ndim", [1, 3, 4, 5])
def test_parse_tensor_dimensionality_error(ndim):
    shape = Shape(tensor=torch.randn(4, 3))

    dummy_tensor = torch.rand([1 for _ in range(ndim - 1)] + [3])
    with pytest.raises(ValueError, match=f"Input tensor must be 2D, but it is {ndim}D"):
        shape._parse_tensor(dummy_tensor)


@pytest.mark.parametrize("final_size", [1, 4, 5])
def test_parse_tensor_size_error(final_size):
    shape = Shape(tensor=torch.randn(4, 3))

    with pytest.raises(
        ValueError,
        match=re.escape(f"Input Tensor must consist of 2D or 3D points (last dimension), but got {final_size}D points"),
    ):
        shape._parse_tensor(torch.randn(1, final_size))


def test_check_nans():
    shape = Shape(tensor=torch.randn(4, 3), check_nans=True)
    nan_tensor = torch.tensor([[float("nan"), 1.0, 2.0]], dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="NaNs found in tensor"):
        shape._parse_tensor(nan_tensor)


@pytest.mark.parametrize(
    "input_tensor,expected_tensor",
    [
        (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1.0, 2.0, 3.0]])),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0, 1.0]])),
    ],
)
def test_parse_tensor_shape(input_tensor, expected_tensor):
    output = Shape._parse_tensor_shape(input_tensor)
    assert output.ndim == expected_tensor.ndim
    assert output.shape == expected_tensor.shape
    assert torch.allclose(output, expected_tensor)


@pytest.mark.parametrize(
    "file_name",
    [
        "test_shape.pts",
        "test_shape_with_affine_with_descriptions.mjson",
        "test_shape_with_affine_without_descriptions.mjson",
        "test_shape_without_affine_with_descriptions.mjson",
        "test_shape_without_affine_without_descriptions.mjson",
    ],
)
def test_save(file_name, data_path, tmpdir):
    full_file_name = os.path.join(data_path, "tests", file_name)
    full_output_name = os.path.join(tmpdir, file_name)

    original_shape = Shape(path=full_file_name)
    original_shape.save(full_output_name)
    reloaded_shape = Shape(full_output_name)

    assert np.allclose(original_shape.affine, reloaded_shape.affine)
    assert torch.allclose(original_shape.tensor, reloaded_shape.tensor)
    assert original_shape.point_descriptions == reloaded_shape.point_descriptions

    with open(full_file_name, "r") as f:
        original_content = f.read()

    with open(full_output_name, "r") as f:
        saved_content = f.read()

    assert original_content == saved_content


def test_as_sitk():
    shape = Shape(tensor=torch.randn(4, 3))
    with pytest.raises(NotImplementedError):
        shape.as_sitk()


def test_as_pil():
    shape = Shape(tensor=torch.randn(4, 3))
    with pytest.raises(NotImplementedError):
        shape.as_pil()


def test_from_sitk():
    import SimpleITK as sitk

    dummy_image = sitk.GetImageFromArray(np.random.rand(3, 3, 3))
    with pytest.raises(NotImplementedError):
        Shape.from_sitk(dummy_image)


def test_to_gif():
    shape = Shape(tensor=torch.randn(4, 3))
    with pytest.raises(NotImplementedError):
        shape.to_gif()


@pytest.mark.parametrize(
    "point_descriptions, parsed_descriptions",
    [
        (None, None),
        (("a", "b", "c"), ("a", "b", "c")),
        (["a", "b", "c"], ("a", "b", "c")),
    ],
)
def test_parse_point_descriptions(point_descriptions, parsed_descriptions):
    assert Shape._parse_point_descriptions(point_descriptions) == parsed_descriptions


@pytest.mark.parametrize(
    "file_name,has_descriptions",
    [
        ("test_shape.pts", False),
        ("test_shape_with_affine_with_descriptions.mjson", True),
        ("test_shape_with_affine_without_descriptions.mjson", False),
        ("test_shape_without_affine_with_descriptions.mjson", True),
        ("test_shape_without_affine_without_descriptions.mjson", False),
    ],
)
def test_point_descriptions(file_name, has_descriptions, data_path):
    shape = Shape(path=os.path.join(data_path, "tests", file_name))

    if has_descriptions:
        assert isinstance(shape.point_descriptions, tuple)
        assert all(isinstance(x, str) for x in shape.point_descriptions)
        assert len(shape.point_descriptions) == shape.tensor.size(0)

    else:
        assert shape.point_descriptions is None or (isinstance(shape.point_descriptions))


def test_points_description_setter():
    shape = Shape(tensor=torch.randn(4, 3))

    shape.point_descriptions = ["a", "b", "c", "d"]
    assert shape.point_descriptions == ("a", "b", "c", "d")


def test_point_description_setter_different_len():
    shape = Shape(tensor=torch.randn(4, 3))

    with pytest.raises(
        ValueError, match=re.escape("Number of point predictions (10) does not match number of points (4)")
    ):
        shape.point_descriptions = [str(i) for i in range(10)]


def test_get_points_by_description():
    shape_tensor = torch.randn(4, 3)

    point_descriptions = ["a", "b", "c", "d"]

    shape = Shape(tensor=shape_tensor, point_descriptions=point_descriptions)

    for i in range(4):
        assert torch.allclose(shape.get_points_by_description(point_descriptions[i]), shape_tensor[i][None])

    for i in range(4):
        for j in range(4):
            assert torch.allclose(
                torch.stack([shape_tensor[i], shape_tensor[j]]),
                shape.get_points_by_description(point_descriptions[i], point_descriptions[j]),
            )


def test_no_descriptions_get_points_by_description():

    shape = Shape(tensor=torch.randn(4, 3))

    with pytest.raises(ValueError, match="No point descriptions found!"):
        shape.get_points_by_description("a")


def test_wrong_descriptions_get_points_by_description():
    shape = Shape(tensor=torch.randn(4, 3), point_descriptions=["a", "b", "c", "d"])
    assert shape.point_descriptions == ("a", "b", "c", "d")

    # cannot check for match here as pytest checks the first Error, even though it is handled
    with pytest.raises(ValueError):
        shape.get_points_by_description("f")


@pytest.mark.parametrize(
    "affine,expected",
    [
        (
            np.array(
                [
                    [0.6, 0.0, 0.0, 0.0],
                    [0.0, 0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor([[0.6, 0.6, 0.6], [2.4, 1.5, 1.2]]),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 45],
                    [0.0, 1.0, 0.0, 90],
                    [0.0, 0.0, 1.0, 135],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor([[46.0, 92.0, 138.0], [49.0, 95.0, 141.0]]),
        ),
        (
            np.array(
                [
                    [0.6, 0.0, 0.0, 45.0],
                    [0.0, 0.3, 0.0, 90.0],
                    [0.0, 0.0, 0.2, 135.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor([[45.6, 90.6, 135.6], [47.4, 91.5, 136.2]]),
        ),
    ],
)
def test_to_physical_space(affine, expected):
    shape = Shape(tensor=torch.tensor([[1, 2, 3], [4, 5, 6]]), affine=affine, point_descriptions=["a", "b"])

    physical_shape = shape.to_physical_space()
    assert isinstance(physical_shape, Shape)

    assert physical_shape.tensor.shape == shape.tensor.shape
    print(physical_shape.tensor)
    assert torch.allclose(physical_shape.tensor, expected)
    assert np.allclose(physical_shape.affine, np.eye(4))
    assert physical_shape.point_descriptions == shape.point_descriptions
