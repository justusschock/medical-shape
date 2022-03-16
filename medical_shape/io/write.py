import json
import pathlib
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
from rising.utils.affine import matrix_revert_coordinate_order
from torchio.typing import TypePath


def mjson_exporter(
    points: Union[torch.Tensor, np.ndarray],
    affine: Union[torch.Tensor, np.ndarray],
    point_descriptions: Optional[Tuple[str, ...]],
    filepath: Union[str, pathlib.Path],
    flip_coordinate_order: bool = False,
):
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)

    if isinstance(affine, np.ndarray):
        affine = torch.from_numpy(affine)

    assert isinstance(points, torch.Tensor)
    points = points.float()

    if flip_coordinate_order:
        points = torch.flip(points, (-1,))
        affine = matrix_revert_coordinate_order(affine[None])[0]

    with open(filepath, "w") as f:
        json.dump(
            {
                "header": f"version: 1\nn_points: {points.shape[0]}",
                "affine": affine.tolist(),
                "points": points.tolist(),
                "descriptions": point_descriptions,
            },
            f,
            indent=4,
        )


def pts_exporter(
    pts: Union[torch.Tensor, np.ndarray],
    file_handle: Union[str, pathlib.Path],
    flip_coordinate_order: bool = False,
    **kwargs,
):
    """Given a file handle to write in to (which should act like a Python `file` object), write out the landmark
    data.

    No value is returned. Writes out the PTS format which is a very simple format that does not contain any semantic
    labels. We assume that the PTS format has been created using Matlab and so use 1-based indexing and put the image
    x-axis as the first coordinate (which is the second axis within Menpo). Note that the PTS file format is only
    powerful enough to represent a basic pointcloud. Any further specialization is lost.
    """
    # Swap the x and y axes and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based

    if flip_coordinate_order:
        if pts.shape[-1] == 2:
            pts = pts[:, [1, 0]] + 1
        else:
            pts = pts[:, [2, 1, 0]] + 1

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()

    header = f"version: 1\nn_points: {pts.shape[0]}\n{{"
    np.savetxt(
        file_handle,
        pts,
        delimiter=" ",
        header=header,
        footer="}",
        comments="",
        **kwargs,
    )


def point_writer(
    path: TypePath,
    points: torch.Tensor,
    affine: Union[torch.Tensor, np.ndarray, None] = None,
    point_descriptions: Optional[Tuple[str, ...]] = None,
):
    path = str(path)

    if path.endswith(".pts"):
        if affine is None:
            warnings.warn(f"Cannot save affine {affine} to PTS file. Consider using an mjson format instead!")

        if point_descriptions is None:
            warnings.warn(
                f"Cannot save point descriptions {point_descriptions} to PTS file. Consider using an mjson format instead!"
            )

        pts_exporter(points, path)
    elif path.endswith(".mjson"):
        if affine is None:
            affine = torch.eye(4)
        mjson_exporter(points, affine, point_descriptions, path)
    else:
        raise ValueError(f"Cannot identify a suitable file writer for points to file {path}")
