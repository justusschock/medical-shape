import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from torchio.typing import TypeData, TypePath, TypeTripletInt

from shape.io import point_reader, point_writer

SHAPE = "shape"


class Shape(tio.data.Image):
    def __init__(
        self,
        path: Union[TypePath, Sequence[TypePath], None] = None,
        type: str = SHAPE,
        tensor: Optional[TypeData] = None,
        affine: Optional[TypeData] = None,
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

    def _parse_tensor(
        self, tensor: Optional[TypeData], none_ok: bool = True
    ) -> Optional[torch.Tensor]:

        if tensor is None:
            if none_ok:
                return None
            else:
                raise RuntimeError("Input tensor cannot be None")
        if isinstance(tensor, np.ndarray):
            tensor = tio.data.io.check_uint_to_int(tensor)
            tensor = torch.as_tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            message = (
                "Input tensor must be a PyTorch tensor or NumPy array,"
                f' but type "{type(tensor)}" was found'
            )
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
        return tensor

    @staticmethod
    def _parse_tensor_shape(tensor: torch.Tensor) -> TypeData:
        return ensure_3d_points(tensor)

    def save(self, path: TypePath, squeeze: Optional[bool] = None) -> None:
        if squeeze:
            raise ValueError("Squeezing is not supported for shapes")
        point_writer(path, self.tensor, self.affine)

    def as_sitk(self):
        raise NotImplementedError

    @classmethod
    def from_sitk(cls, sitk_image: sitk.Image):
        raise NotImplementedError

    def as_pil(self, transpose: bool = True):
        raise NotImplementedError

    def to_gif(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (1, *self.spatial_shape)

    @property
    def spatial_shape(self) -> TypeTripletInt:
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

    def get_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Get minimum and maximum world coordinates occupied by the image."""
        first_index = self.data.min(0)[0].floor() - 0.5
        last_index = self.data.max(0)[0].ceil() - 0.5
        first_point = nib.affines.apply_affine(
            self.affine, first_index.detach().cpu().numpy()
        )
        last_point = nib.affines.apply_affine(
            self.affine, last_index.detach().cpu().numpy()
        )
        array = np.array((first_point, last_point))
        bounds_x, bounds_y, bounds_z = array.T.tolist()
        return tuple(bounds_x), tuple(bounds_y), tuple(bounds_z)

    def plot(self, **kwargs):
        # TODO: Implement plotting logic
        raise NotImplementedError

    def show(self, viewer_path: Optional[TypePath]):
        # TODO: Implement showing with external software (plotly?)
        raise NotImplementedError


def ensure_3d_points(
    tensor: TypeData, num_spatial_dims: Optional[int] = None
) -> TypeData:
    tensor = torch.as_tensor(tensor)
    num_dimensions = tensor.size(-1)

    if num_dimensions == 2:
        # X, Y

        tensor = points_to_homogeneous(tensor)

    elif num_dimensions != 3:
        raise ValueError(f"{num_dimensions}D points not supported!")

    return tensor