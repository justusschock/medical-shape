import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from rising.utils.affine import points_to_homogeneous
from torchio.typing import TypeData, TypePath

from medical_shape.io import point_reader, point_writer

SHAPE = "shape"
POINT_DESCRIPTIONS = "point_descriptions"


class Shape(tio.data.Image):
    _loaded: bool

    def __init__(
        self,
        path: Union[TypePath, Sequence[TypePath], None] = None,
        type: str = SHAPE,
        tensor: Optional[TypeData] = None,
        affine: Optional[TypeData] = None,
        point_descriptions: Optional[Sequence[str]] = None,
        check_nans: bool = False,  # removed by ITK by default
        reader: Callable = point_reader,
        **kwargs: Dict[str, Any],
    ):

        super().__init__(
            path=path,
            type=type,
            tensor=tensor,
            affine=affine,
            check_nans=check_nans,
            reader=reader,
            **kwargs,
        )

        self[POINT_DESCRIPTIONS] = point_descriptions

    def __getitem__(self, item):
        if item in (tio.data.image.DATA, tio.data.image.AFFINE, POINT_DESCRIPTIONS):
            if item not in self:
                self.load()
        return super().__getitem__(item)

    def __copy__(self):
        kwargs = {
            "tensor": self.data,
            "affine": self.affine,
            "type": self.type,
            "path": self.path,
            "point_descriptions": self.point_descriptions,
        }
        for key, value in self.items():
            if key in (*tio.data.image.PROTECTED_KEYS, POINT_DESCRIPTIONS):
                continue
            kwargs[key] = value  # should I copy? deepcopy?
        return self.__class__(**kwargs)

    def read_and_check(self, path: TypePath) -> Tuple[torch.Tensor, np.ndarray, Union[None, Sequence[str]]]:
        tensor, affine, point_descriptions = self.reader(path)
        # Make sure the data type is compatible with PyTorch
        tensor = self._parse_tensor_shape(tensor)
        tensor = self._parse_tensor(tensor)
        affine = self._parse_affine(affine)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{path}"', RuntimeWarning)
        return tensor, affine, point_descriptions

    def load(self) -> None:
        r"""Load the image from disk.
        Returns:
            Tuple containing a 4D tensor of size :math:`(C, W, H, D)` and a 2D
            :math:`4 \times 4` affine matrix to convert voxel indices to world
            coordinates.
        """
        if self._loaded:
            return
        paths = self.path if self._is_multipath() else [self.path]
        tensor, affine, point_descriptions = self.read_and_check(paths[0])
        tensors = [tensor]
        descriptions: Optional[List[str]] = (
            point_descriptions if point_descriptions is None else list(point_descriptions)
        )

        for path in paths[1:]:
            new_tensor, new_affine, point_descriptions = self.read_and_check(path)
            if not np.array_equal(affine, new_affine):
                message = (
                    "Files have different affine matrices."
                    f"\nMatrix of {paths[0]}:"
                    f"\n{affine}"
                    f"\nMatrix of {path}:"
                    f"\n{new_affine}"
                )
                warnings.warn(message, RuntimeWarning)
            if not tensor.shape[1:] == new_tensor.shape[1:]:
                message = f"Files shape do not match, found {tensor.shape}" f"and {new_tensor.shape}"
                RuntimeError(message)
            tensors.append(new_tensor)

            if point_descriptions is not None and descriptions is not None:
                descriptions.extend(list(point_descriptions))

        if descriptions is None:
            all_point_descriptions = descriptions
        else:
            if isinstance(descriptions, Iterable):
                all_point_descriptions = tuple(descriptions)
            else:
                raise TypeError(f"Point descriptions must be iterable or None, not {type(descriptions)}")
        tensor = torch.cat(tensors)

        self[POINT_DESCRIPTIONS] = all_point_descriptions
        self.set_data(tensor)
        self.affine = affine
        self._loaded = True

    def set_data(self, tensor: TypeData, check_description_length: bool = True) -> None:
        if check_description_length and POINT_DESCRIPTIONS in self and self[POINT_DESCRIPTIONS] is not None:
            if len(self[POINT_DESCRIPTIONS]) != tensor.size(0):
                raise ValueError(
                    f"Number of point descriptions ({len(self[POINT_DESCRIPTIONS])}) "
                    f"does not match number of points ({tensor.size(0)})"
                )

        return super().set_data(tensor)

    def _parse_affine(self, affine: Optional[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
        if affine is None:
            return np.eye(4)
        if isinstance(affine, torch.Tensor):
            affine = affine.numpy()
        if not isinstance(affine, np.ndarray):
            bad_type = type(affine)
            raise TypeError(f"Affine must be a NumPy array, not {bad_type}")
        if affine.shape != (4, 4):
            bad_shape = affine.shape
            raise ValueError(f"Affine shape must be (4, 4), not {bad_shape}")
        return affine.astype(np.float64)

    def _parse_tensor(self, tensor: Optional[TypeData], none_ok: bool = True) -> Optional[torch.Tensor]:

        if tensor is None:
            if none_ok:
                return None
            else:
                raise RuntimeError("Input tensor cannot be None")
        if isinstance(tensor, np.ndarray):
            tensor = tio.data.io.check_uint_to_int(tensor)
            tensor = torch.as_tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            message = "Input tensor must be a PyTorch tensor or NumPy array," f' but type "{type(tensor)}" was found'
            raise TypeError(message)
        ndim = tensor.ndim
        if ndim != 2:
            raise ValueError(f"Input tensor must be 2D, but it is {ndim}D")

        if tensor.size(-1) not in (2, 3):
            raise ValueError(
                f"Input Tensor must consist of 2D or 3D points (last dimension), but got {tensor.size(-1)}D points"
            )
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.uint8)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f"NaNs found in tensor", RuntimeWarning)
        return tensor.float()

    @staticmethod
    def _parse_tensor_shape(tensor: torch.Tensor) -> TypeData:
        return ensure_3d_points(tensor)

    def save(self, path: TypePath, squeeze: Optional[bool] = None) -> None:
        if squeeze:
            raise ValueError("Squeezing is not supported for shapes")
        point_writer(path, self.tensor, self.affine, self.point_descriptions)

    def as_sitk(self) -> sitk.Image:
        raise NotImplementedError

    @classmethod
    def from_sitk(cls, sitk_image: sitk.Image) -> "Shape":
        raise NotImplementedError

    def as_pil(self, transpose: bool = True) -> None:
        raise NotImplementedError

    def to_gif(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def point_descriptions(self) -> Optional[Tuple[str, ...]]:
        return self[POINT_DESCRIPTIONS]

    @point_descriptions.setter
    def point_descriptions(self, point_descriptions: Optional[Tuple[str, ...]]) -> None:
        if point_descriptions is not None and len(point_descriptions) != self.tensor.size(0):
            raise ValueError(
                f"Number of point predictions ({len(point_descriptions)}) "
                f"does not match number of points ({self.tensor.size(0)})"
            )

        self[POINT_DESCRIPTIONS] = point_descriptions

    def get_points_by_description(self, *point_descriptions: str) -> torch.Tensor:
        points = []

        if self.point_descriptions is None:
            raise ValueError("No point descriptions found!")

        for desc in point_descriptions:
            try:
                index = self.point_descriptions.index(desc)
            except ValueError:
                raise ValueError(f"{desc} not in point_descriptions. Valid options are {self.point_descriptions}!")

            points.append(self.tensor[index])

        return torch.stack(points)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (1, *self.spatial_shape)

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        mins = self.data.min(0)[0].floor()
        maxs = self.data.max(0)[0].ceil()
        shape = tuple((maxs - mins).tolist())
        return shape

    def is_2d(self) -> bool:
        return self.data.size(-1) == 3 and bool((self.data[..., -1] == 0).all())

    @property
    def bounds(self) -> np.ndarray:
        mins = self.data.min(0)[0].floor()
        maxs = self.data.max(0)[0].ceil()
        point_ini = nib.affines.apply_affine(self.affine, mins.detach().cpu().numpy())
        point_fin = nib.affines.apply_affine(self.affine, maxs.detach().cpu().numpy())
        return np.array(point_ini, point_fin)

    def get_bounds(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get minimum and maximum world coordinates occupied by the image."""
        first_index = self.data.min(0)[0].floor() - 0.5
        last_index = self.data.max(0)[0].ceil() - 0.5
        first_point = nib.affines.apply_affine(self.affine, first_index.detach().cpu().numpy())
        last_point = nib.affines.apply_affine(self.affine, last_index.detach().cpu().numpy())
        array = np.array((first_point, last_point))
        bounds_x, bounds_y, bounds_z = array.T.tolist()
        return tuple(bounds_x), tuple(bounds_y), tuple(bounds_z)

    def plot(self, **kwargs: Any) -> None:
        # TODO: Implement plotting logic
        raise NotImplementedError

    def show(self, viewer_path: Optional[TypePath]) -> None:
        # TODO: Implement showing with external software (plotly?)
        raise NotImplementedError


def ensure_3d_points(tensor: TypeData, num_spatial_dims: Optional[int] = None) -> TypeData:
    tensor = torch.as_tensor(tensor)
    num_dimensions = tensor.size(-1)

    if num_dimensions == 2:
        # X, Y
        # TODO: Add zeros instead of to_homogeneous
        tensor = points_to_homogeneous(tensor)

    elif num_dimensions != 3:
        raise ValueError(f"{num_dimensions}D points not supported!")

    return tensor
