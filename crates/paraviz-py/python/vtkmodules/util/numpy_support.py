"""`vtkmodules.util.numpy_support` equivalent (VTK 9.x), backed by paraviz."""

from paraview._dataset import numpy_to_vtk, vtk_to_numpy  # noqa: F401

__all__ = ["vtk_to_numpy", "numpy_to_vtk"]
