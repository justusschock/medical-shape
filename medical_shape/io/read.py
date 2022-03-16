import json
import pathlib
from typing import Any, Union

import torch
from rising.utils.affine import matrix_revert_coordinate_order
from torchio.typing import TypePath


def pts_importer(
    filepath: Union[str, pathlib.Path],
    flip_coordinate_order: bool = False,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> torch.Tensor:
    """Importer for the PTS file format.

    Assumes version 1 of the format. Implementations of this class should override the :meth:`_build_points` which
    determines the ordering of axes. For example, for images, the `x` and `y` axes are flipped such that the first axis
    is `y` (height in the image domain). Note that PTS has a very loose format definition. Here we make the assumption
    (as is common) that PTS landmarks are 1-based. That is, landmarks on a 480x480 image are in the range [1-480]. As
    Menpo is consistently 0-based, we *subtract 1* off each landmark value automatically. If you want to use PTS
    landmarks that are 0-based, you will have to manually add one back on to landmarks post importing. Landmark set
    label: PTS
    """
    with open(filepath, "r", **kwargs) as f:
        lines = [l.strip() for l in f.readlines()]

    line = lines[0]
    while not line.startswith("{"):
        line = lines.pop(0)

    include_z = len(lines[0].strip("\n").split()) == 3

    xs = []
    ys = []
    zs = []

    for line in lines:
        if not line.strip().startswith("}"):
            splitted = line.split()
            xpos, ypos = splitted[:2]
            xs.append(float(xpos))
            ys.append(float(ypos))

            if include_z:
                zs.append(float(splitted[2]))

    # PTS landmarks are 1-based, need to convert to 0-based (subtract 1)
    xs_tensor = torch.tensor(xs, dtype=torch.float).view((-1, 1)) - 1
    ys_tensor = torch.tensor(ys, dtype=torch.float).view((-1, 1)) - 1

    points = [xs_tensor, ys_tensor]

    if include_z:
        zs_tensor = torch.tensor(zs, dtype=torch.float).view((-1, 1)) - 1
        points.append(zs_tensor)

    if flip_coordinate_order:
        points = list(reversed(points))
    points_tensor = torch.cat(points, 1)
    return points_tensor.to(device)


def mjson_importer(
    filepath: Union[str, pathlib.Path],
    flip_coordinate_order: bool = True,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
):
    with open(filepath) as f:
        content = json.load(f, **kwargs)

    points = content["points"]

    points = torch.tensor(points, device=device, dtype=torch.float)

    affine = content.get("affine", torch.eye(4))
    if not isinstance(affine, torch.Tensor):
        affine = torch.tensor(affine)
    affine = affine.to(points)

    descriptions = content.get("descriptions", None)

    if flip_coordinate_order:
        points = torch.flip(affine, (-1,))
        affine = matrix_revert_coordinate_order(affine[None])[0]
    return points, affine, descriptions


def point_reader(path: TypePath, **kwargs: Any):
    path = str(path)
    if path.endswith(".pts"):
        points = pts_importer(path, flip_coordinate_order=False, **kwargs)
        affine = torch.eye(4).to(points)
        descriptions = None
    elif path.endswith(".mjson"):
        points, affine, descriptions = mjson_importer(path, flip_coordinate_order=False, **kwargs)

    else:
        raise ValueError(
            f"Could not find file with extension {path.rsplit('.')[1]}. Supported extensions are .pts and .mpts"
        )

    return points, affine, descriptions
