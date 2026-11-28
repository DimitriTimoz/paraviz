"""Drop-in ``paraview`` package backed by ParaViz (Rust + WebGPU).

Only the ``paraview.simple`` scripting API is emulated (the subset used by
99% of ParaView Python scripts). ``paraview.servermanager`` and the raw
``vtk`` module are not provided.
"""

__version__ = "5.13.0"


def __getattr__(name):
    if name in ("simple", "servermanager"):
        from . import simple, servermanager

        return {"simple": simple, "servermanager": servermanager}[name]
    if name == "version":
        return __version__
    raise AttributeError(f"module 'paraview' has no attribute {name!r}")


__all__ = ["simple", "servermanager", "__version__"]
