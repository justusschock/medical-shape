# Medical Shape

[![UnitTest](https://github.com/justusschock/medical-shape/actions/workflows/unittest.yaml/badge.svg)](https://github.com/justusschock/medical-shape/actions/workflows/unittest.yaml) [![Docker](https://github.com/justusschock/medical-shape/actions/workflows/docker_build.yaml/badge.svg)](https://github.com/justusschock/medical-shape/actions/workflows/docker_build.yaml) [![Docker Stable](https://github.com/justusschock/medical-shape/actions/workflows/docker_stable.yaml/badge.svg)](https://github.com/justusschock/medical-shape/actions/workflows/docker_stable.yaml) [![Build Package](https://github.com/justusschock/medical-shape/actions/workflows/package_build.yaml/badge.svg)](https://github.com/justusschock/medical-shape/actions/workflows/package_build.yaml) ![PyPI](https://img.shields.io/pypi/v/medical-shape?color=grene) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/justusschock/medical-shape/main.svg)](https://results.pre-commit.ci/latest/github/justusschock/medical-shape/main)

A [`torchio`](https://github.com/fepegar/torchio) extension for shapes and their processing.

## Usage

`medical_shape` provides 3 major classes for usage:

### `Shape`

`Shape` is a subclass from `torchio.data.Image`. It stores arbitrary pointclouds together with their descriptions and an associated affine matrix.
The pointclouds are stored as 2D `torch.Tensor` in the form _NxD_ where _N_ is the number of points and _D_ is the dimensionality of points (usually _D=3_).

### `ShapeSupportSubject`

The `ShapeSupportSubject` is an extension of `torchio.data.Subject` to allow the inclusion of `Shape`-type objects into the subject. It should be used instead of `torchio.data.Subject`,
whenever a shape is included and is also safe to use without a shape (will behave exactly like `torchio.data.Subject` in that case) as shapes often require special handling.

### `TransformShapeValidationMixin`

This class is a transformation mixin to allow checks whether shape-support is required and to raise warnings if the incorrect Subject-type was used.
All transformations supporting shapes should inherit from it (and as it is derived from `torchio.transforms.Transform`, it is also safe to use this class as a standalone baseclass).

## Installation

This project can be installed either from PyPI or by cloning the repository from GitHub.

For an install of published packages, use the command

```bash
    pip install medical-shape
```

To install from the (cloned) repository, use the command

```bash
    pip install PATH/TO/medical-shape
```

You can also add -e to the command to make an editable install in case you want to modify the code.

You can also install the package directly from GitHub by running

```bash
    pip install git+https://github.com/justusschock/medical-shape.git
```

## Docker Images

We provide a docker image for easy usage of the package and as a base image for other projects.

The file for this image can be found at dockers/Dockerfile. We provide both, a CPU-only and a CUDA-enabled image based on the NVIDIA NGC PyTorch image. These images can be found on [DockerHub](https://hub.docker.com/repository/docker/justusschock/medical-shape).
