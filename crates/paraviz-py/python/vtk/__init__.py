"""Minimal top-level `vtk` shim shipped by the paraviz wheel.

Only `vtk.util.numpy_support` (vtk_to_numpy / numpy_to_vtk) is provided, which is
what ParaView scripts use after `Fetch()`. Use a dedicated virtualenv — this
shadows a real VTK install if both are present.
"""

from . import util  # noqa: F401

__version__ = "9.3.0-paraviz"
