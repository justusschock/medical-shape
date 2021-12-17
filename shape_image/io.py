import os
import pathlib
from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np
import SimpleITK as sitk
import torch
from pytorch_lightning.utilities.enums import LightningEnum


def pts_importer(
    filepath: Union[str, pathlib.Path],
    image_origin: bool = False,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    Importer for the PTS file format. Assumes version 1 of the format.
    Implementations of this class should override the :meth:`_build_points`
    which determines the ordering of axes. For example, for images, the
    `x` and `y` axes are flipped such that the first axis is `y` (height
    in the image domain).
    Note that PTS has a very loose format definition. Here we make the
    assumption (as is common) that PTS landmarks are 1-based. That is,
    landmarks on a 480x480 image are in the range [1-480]. As Menpo is
    consistently 0-based, we *subtract 1* off each landmark value
    automatically.
    If you want to use PTS landmarks that are 0-based, you will have to
    manually add one back on to landmarks post importing.
    Landmark set label: PTS

    Args:
        filepath : str
            Absolute filepath of the file.
        image_origin : `bool`, optional
            If ``True``, assume that the landmarks exist within an image and thus
            the origin is the image origin.
        device : str
            the device to put the loaded tensors on
        **kwargs : `dict`, optional
            Any other keyword arguments.
    Returns:
        torch.Tensor
            imported points
    """
    with open(filepath, "r", **kwargs) as f:
        lines = [l.strip() for l in f.readlines()]

    line = lines[0]
    while not line.startswith("{"):
        line = lines.pop(0)

    include_z = len(lines[0].strip("\n").split()) == 3

    xs = []
    ys = []

    if include_z:
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
    xs = torch.tensor(xs, dtype=torch.float).view((-1, 1)) - 1
    ys = torch.tensor(ys, dtype=torch.float).view((-1, 1)) - 1

    points = [xs, ys]

    if include_z:
        zs = torch.tensor(zs, dtype=torch.float).view((-1, 1)) - 1
        points.append(zs)

    if image_origin:
        points = list(reversed(points))
    points = torch.cat(points, 1)
    return points.to(device)


def pts_exporter(
    pts: Union[torch.Tensor, np.ndarray],
    file_handle: Union[str, pathlib.Path],
    image_origin: bool = False,
    **kwargs,
):
    """
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.
    Writes out the PTS format which is a very simple format that does not
    contain any semantic labels. We assume that the PTS format has been created
    using Matlab and so use 1-based indexing and put the image x-axis as the
    first coordinate (which is the second axis within Menpo).
    Note that the PTS file format is only powerful enough to represent a
    basic pointcloud. Any further specialization is lost.

    Args:
        pts : np.ndarray
            points to save
        file_handle : `file`-like object
            The file to write in to

    """
    # Swap the x and y axis and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based

    if image_origin:
        if pts.shape[-1] == 2:
            pts = pts[:, [1, 0]] + 1
        else:
            pts = pts[:, [2, 1, 0]] + 1

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()

    header = "version: 1\nn_points: {}\n{{".format(pts.shape[0])
    np.savetxt(
        file_handle,
        pts,
        delimiter=" ",
        header=header,
        footer="}",
        comments="",
        **kwargs,
    )


def load_image(
    path,
    device: Optional[Union[str, torch.device]] = None,
    return_spacing: bool = False,
) -> torch.Tensor:
    # TODO: Check for nibabel (might be more efficient)
    if os.path.isfile(path):
        image = sitk.ReadImage(path)
    else:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(path))
        image = reader.Execute()

    spacing = list(reversed(image.GetSpacing()))
    image = sitk.GetArrayFromImage(image)
    image = torch.from_numpy(image)
    spacing = torch.from_numpy(spacing)

    if device is not None:
        image = image.to(device, non_blocking=True)

    if return_spacing:
        if device is not None:
            spacing = spacing.to(device, non_blocking=True)
        return image, spacing
    return image


IMG_EXTENSIONS_2D = (".png", ".PNG", ".jpg", ".JPG")

IMG_EXTENSIONS_3D = (".mhd", ".nii.gz", ".dcm")

# TODO: Update LMK Extensions depending on decision for imports in
#  image factory
LMK_EXTENSIONS = (".txt", ".TXT", ".ljson", ".LJSON", ".pts", ".PTS")


def _check_file_extensions(path: str, allowed_extensions: Sequence[str]):
    return os.path.isfile(path) and any(
        path.lower().endswith(ext) for ext in allowed_extensions
    )


def _is_leaf_dir(path: str):
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, x) for x in os.listdir(path))
    )


def _check_for_dir_no_extension_files(path: str):
    return _is_leaf_dir(path) and not any(os.path.extsep in x for x in os.listdir(path))


def combine_conditions(path: str, conditions: Sequence[Callable[[str], bool]]) -> bool:
    return all(condition(path) for condition in conditions)


class _FileTypeConditions(LightningEnum):
    NIFTI = partial(_check_file_extensions, allowed_extensions=(".nii", ".nii.gz"))
    DICOM = partial(
        combine_conditions,
        conditions=(
            _check_for_dir_no_extension_files,
            partial(_check_file_extensions, allowed_extensions=(".dcm", ".dicom")),
        ),
    )
    MHD = partial(_check_file_extensions, allowed_extensions=(".mhd",))
    PNG = partial(_check_file_extensions, allowed_extensions=(".png",))
    JPG = partial(_check_file_extensions, allowed_extensions=(".jpg", ".jpeg"))
    PTS = partial(_check_file_extensions, allowed_extensions=(".pts",))


class _ImageFiles(LightningEnum):
    NIFTI = _FileTypeConditions.NIFTI
    DICOM = _FileTypeConditions.DICOM
    MHD = _FileTypeConditions.MHD
    PNG = _FileTypeConditions.PNG
    JPG = _FileTypeConditions.JPG


class _LandmarkFiles(LightningEnum):
    PTS = _FileTypeConditions.PTS


def is_image_path(path):
    return any(e.value(path) for e in _ImageFiles)


def is_lmk_path(path):
    return any * (e.value(path) for e in _LandmarkFiles)


def get_files_and_leafdirs(path):
    to_process = [path]
    files = []
    leafdirs = []
    while to_process:
        current_path = to_process.pop()
        items_are_all_files = True

        for file in os.listdir(current_path):
            file_path = os.path.join(current_path, file)

            if os.path.isfile(file_path):
                files.append(file_path)

            elif os.path.isdir(file_path):
                to_process.append(file_path)
                items_are_all_files = False

            if items_are_all_files:
                leafdirs.append(current_path)
    return files, leafdirs


def parse_for_images(path: str):
    files, leafdirs = get_files_and_leafdirs(path)
    return filter(is_image_path, files + leafdirs)


def parse_for_lmk_files(path: str):
    files, leafdirs = get_files_and_leafdirs(path)
    return filter(is_lmk_path, files + leafdirs)


def find_corresponding_image(lmk_file: str):
    return parse_for_images(os.path.dirname(lmk_file))


def find_corresponding_landmarks(image_path: str):
    return parse_for_lmk_files(os.path.dirname(image_path))
