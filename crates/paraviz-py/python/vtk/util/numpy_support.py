"""`vtk.util.numpy_support` equivalent, backed by paraviz DataArrayView."""

from paraview._dataset import numpy_to_vtk, vtk_to_numpy  # noqa: F401

__all__ = ["vtk_to_numpy", "numpy_to_vtk"]
