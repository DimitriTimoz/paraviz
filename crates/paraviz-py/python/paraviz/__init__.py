"""ParaViz — ParaView rewritten in Rust + WebGPU.

The native extension lives in ``paraviz._paraviz``; this package re-exports it.
A `paraview.simple`-compatible API ships alongside in the top-level
``paraview`` package bundled in this same wheel.
"""

from ._paraviz import Session

__version__ = "0.2.6"

__all__ = ["Session", "__version__"]
