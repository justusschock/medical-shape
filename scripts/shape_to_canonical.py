import json

import numpy as np
import torch
import torchio as tio

from medical_shape import Shape, ShapeSupportSubject
from medical_shape.transforms import ToCanonical


def to_canonical(in_path, out_path, image_path=None, out_path_image=None):
    if in_path.endswith(".json"):
        with open(in_path) as f:
            points = torch.tensor(json.load(f))
        points = torch.flip(points, (-1,))
        shape = Shape(tensor=points)
    else:
        shape = Shape(in_path)

    if image_path is not None:
        image = tio.Image(image_path)
        shape.affine = image.affine.copy()
    else:
        image = None
    canonical = ToCanonical("image")(ShapeSupportSubject(shape=shape, image=image))
    shape = canonical["shape"]
    if image is not None and out_path_image is not None:
        canonical["image"].affine = np.eye(4)
        canonical["image"].save(out_path_image)
    shape.save(out_path)
    if image is not None:
        return shape, image
    return shape


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert shape to canonical coordinates.")
    parser.add_argument("in_path", help="Path to input shape file.")
    parser.add_argument("out_path", help="Path to output shape file.")
    parser.add_argument("--image_path", help="Path to image file.")
    parser.add_argument("--out_path_image", help="Path to output image file.")
    args = parser.parse_args()
    res = to_canonical(
        args.in_path,
        args.out_path,
        args.image_path,
        args.out_path_image,
    )
