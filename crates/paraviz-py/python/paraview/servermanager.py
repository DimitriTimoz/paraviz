"""Minimal `paraview.servermanager` shim.

Real ParaView implements `Fetch()` through `servermanager`; here both modules
share the same ParaViz-backed implementation.
"""

from ._dataset import (  # noqa: F401
    CellsView,
    DataArrayView,
    DataObject,
    FieldDataView,
    PointsView,
    numpy_to_vtk,
    vtk_to_numpy,
)
from .simple import *  # noqa: F401,F403
from .simple import Fetch  # noqa: F401

__all__ = ["Fetch", "DataObject", "DataArrayView", "FieldDataView", "PointsView", "CellsView",
           "vtk_to_numpy", "numpy_to_vtk"]
