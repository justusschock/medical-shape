import json
import logging
import pathlib
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from rising.transforms.functional.affine import (
    matrix_revert_coordinate_order, points_to_homogeneous)
from torchio.typing import TypeData, TypePath

from shape_image.io import pts_exporter, pts_importer

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
        self, tensor: Optional[TypeData], none_ok: bool
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

        if str(path).endswith(".pts"):
            if not torch.allclose(self.affine, torch.eye(4)):
                logging.warning(
                    (
                        "The PTS file format does not support saving affines."
                        + "Omitting it during saving."
                        + "This may result in problems when loading this sample again!"
                    )
                )

            pts_exporter(self.tensor, path, image_origin=False)
            return

        if not (str(path).endswith(".mpts")):
            path = str(path) + ".mpts"

        mpts_exporter(self.tensor, self.affine, path, image_origin=False)

    def as_sitk(self):
        raise NotImplementedError

    @classmethod
    def from_sitk(cls, sitk_image):
        raise NotImplementedError

    def as_pil(self, transpose):
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

    def get_bounds(self) -> TypeBounds:
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
        return bounds_x, bounds_y, bounds_z

    def plot(self, **kwargs):
        # TODO: Implement plotting logic
        raise NotImplementedError

    def show(self, viewer_path: Optional[TypePath]):
        # TODO: Implement showing with external software
        raise NotImplementedError


def point_reader(path: TypePath, **kwargs: Any):
    if str(path).endswith(".pts"):
        points = pts_importer(path, image_origin=False, **kwargs)
        affine = torch.eye(4).to(points)
    elif str(path).endswith(".mpts"):
        points, affine = mpts_importer(path, image_origin=False, **kwargs)

    else:
        raise ValueError(
            f"Could not find file with extension {str(path).rsplit('.')[1]}. Supported extensions are .pts and .mpts"
        )

    return points, affine


def mpts_importer(
    filepath: Union[str, pathlib.Path],
    image_origin: bool = True,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
):
    with open(filepath) as f:
        content = json.load(f, **kwargs)

    points = content["points"]

    points = torch.tensor(points, device=device)

    affine = content.get("affine", torch.eye(4))
    if not isinstance(affine, torch.Tensor):
        affine = torch.tensor(affine)
    affine = affine.to(points)

    if image_origin:
        points = torch.flip(affine, (-1,))
        affine = matrix_revert_coordinate_order(affine[None])[0]
    return points, affine


# TODO: Rename image_origin to flip_coordinate_order
def mpts_exporter(
    points: Union[torch.Tensor, np.ndarray],
    affine: Union[torch.Tensor, np.ndarray],
    filepath: Union[str, pathlib.Path],
    image_origin: bool = False,
):

    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)

    if isinstance(affine, np.ndarray):
        affine = torch.from_numpy(affine)

    if image_origin:
        points = torch.flip(points, (-1,))
        affine = matrix_revert_coordinate_order(affine[None])[0]

    with open(filepath, "w") as f:
        json.dump(
            {"affine": affine.tolist(), "points": points.tolist()},
            f,
            indent=4,
            sort_keys=True,
        )


def ensure_3d_points(tensor: TypeData, num_spatial_dims=None) -> TypeData:
    tensor = torch.as_tensor(tensor)
    num_dimensions = tensor.size(-1)

    if num_dimensions == 2:
        # X, Y

        tensor = points_to_homogeneous(tensor)

    elif num_dimensions != 3:
        raise ValueError(f"{num_dimensions}D points not supported!")

    return tensor
