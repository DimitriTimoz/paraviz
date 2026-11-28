"""Client-side dataset objects for the `paraview` shim (the `Fetch()` payload).

Mimics the small subset of the VTK data-object API that ParaView scripts use:

    data = Fetch(reader)
    data.PointData['RTData']            # DataArrayView
    data.GetPoints().GetData()          # DataArrayView (n x 3)
    data.GetNumberOfPoints(), data.GetBounds()
    from vtk.util.numpy_support import vtk_to_numpy
    a = vtk_to_numpy(data.PointData['RTData'])
"""

import json as _json

import numpy as _np

_NUMPY_DTYPES = {
    "float32": _np.float32,
    "float64": _np.float64,
    "int8": _np.int8,
    "int16": _np.int16,
    "int32": _np.int32,
    "int64": _np.int64,
    "uint8": _np.uint8,
    "uint16": _np.uint16,
    "uint32": _np.uint32,
    "uint64": _np.uint64,
}

_DTYPE_NAMES = {v: k for k, v in _NUMPY_DTYPES.items()}


class DataArrayView:
    """VTK `vtkDataArray`-like wrapper: metadata + raw buffer + numpy conversion."""

    def __init__(self, name, num_components, dtype, data, num_tuples=None, rng=None):
        self._name = name
        self._ncomp = int(num_components)
        self._dtype = dtype
        self._data = bytes(data)
        itemsize = _np.dtype(_NUMPY_DTYPES[dtype]).itemsize
        self._num_tuples = int(num_tuples) if num_tuples is not None else len(self._data) // (itemsize * self._ncomp)
        self._range = list(rng) if rng is not None else None

    # -- vtkDataArray API ---------------------------------------------------
    def GetName(self):
        return self._name

    def GetNumberOfComponents(self):
        return self._ncomp

    def GetNumberOfTuples(self):
        return self._num_tuples

    def GetRange(self, component=0):
        if self._range is not None and self._ncomp == 1:
            return list(self._range)
        a = self.to_numpy()
        if a.ndim == 2:
            a = a[:, component]
        return [float(a.min()), float(a.max())]

    @property
    def shape(self):
        return (self._num_tuples, self._ncomp) if self._ncomp > 1 else (self._num_tuples,)

    @property
    def dtype(self):
        return _np.dtype(_NUMPY_DTYPES[self._dtype])

    # -- numpy bridge ---------------------------------------------------------
    def to_numpy(self):
        a = _np.frombuffer(self._data, dtype=self.dtype)
        if self._ncomp > 1:
            a = a.reshape(-1, self._ncomp)
        return a

    def __array__(self, dtype=None):
        a = self.to_numpy()
        return a.astype(dtype) if dtype is not None else a

    def __len__(self):
        return self._num_tuples

    def __repr__(self):
        return f"<DataArrayView '{self._name}' {self._dtype} x{self._ncomp} ({self._num_tuples} tuples)>"


def vtk_to_numpy(arr):
    """`vtk.util.numpy_support.vtk_to_numpy` equivalent for DataArrayView objects."""
    if isinstance(arr, DataArrayView):
        return arr.to_numpy()
    if isinstance(arr, (bytes, bytearray, memoryview)):
        return _np.frombuffer(arr, dtype=_np.float32)
    return _np.asarray(arr)


def numpy_to_vtk(arr, deep=False, array_type=None):
    """`vtk.util.numpy_support.numpy_to_vtk` equivalent: wraps a numpy array as DataArrayView."""
    a = _np.ascontiguousarray(arr)
    if array_type is not None:
        a = a.astype(array_type)
    name = _np.dtype(a.dtype).name
    dtype = _DTYPE_NAMES.get(name)
    if dtype is None:
        a = a.astype(_np.float64)
        dtype = "float64"
    ncomp = a.shape[1] if a.ndim > 1 else 1
    data = a.tobytes() if deep else a.tobytes()
    return DataArrayView("", ncomp, dtype, data, num_tuples=a.shape[0])


class FieldDataView:
    """VTK `vtkFieldData`-like view over one attribute association (POINTS/CELLS)."""

    def __init__(self, sess, node_id, block, assoc, meta):
        self._sess = sess
        self._node_id = node_id
        self._block = block
        self._assoc = assoc  # "POINTS" | "CELLS"
        key = "point_data" if assoc == "POINTS" else "cell_data"
        self._field = meta.get(key, {})
        self._arrays = {a["name"]: a for a in self._field.get("arrays", [])}

    def _fetch(self, name):
        a = self._arrays[name]
        dtype, ncomp, data = self._sess.get_array_bytes(self._node_id, self._assoc, name, self._block)
        return DataArrayView(name, ncomp, dtype, data, a["num_tuples"], a["range"])

    def __getitem__(self, name):
        if name not in self._arrays:
            raise KeyError(f"no {self._assoc} array named '{name}' (have: {sorted(self._arrays)})")
        return self._fetch(name)

    def GetArray(self, name_or_index):
        if isinstance(name_or_index, int):
            name_or_index = list(self._arrays.keys())[name_or_index]
        return self[name_or_index]

    def GetArrayName(self, i):
        return list(self._arrays.keys())[i]

    def GetNumberOfArrays(self):
        return len(self._arrays)

    def keys(self):
        return self._arrays.keys()

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self):
        return len(self._arrays)

    @property
    def active_scalars(self):
        return self._field.get("active_scalars")

    @property
    def active_vectors(self):
        return self._field.get("active_vectors")

    def __repr__(self):
        return f"<FieldDataView {self._assoc} arrays={sorted(self._arrays)}>"


class PointsView:
    def __init__(self, sess, node_id, block, num_points):
        self._sess = sess
        self._node_id = node_id
        self._block = block
        self._n = num_points

    def GetNumberOfPoints(self):
        return self._n

    def GetData(self):
        n, data = self._sess.get_points_bytes(self._node_id, self._block)
        return DataArrayView("Points", 3, "float32", data, num_tuples=n)


class CellsView:
    def __init__(self, sess, node_id, block):
        self._sess = sess
        self._node_id = node_id
        self._block = block

    def _raw(self):
        return self._sess.get_cells_bytes(self._node_id, self._block)

    @property
    def offsets(self):
        return _np.frombuffer(self._raw()[0], dtype=_np.uint32)

    @property
    def connectivity(self):
        return _np.frombuffer(self._raw()[1], dtype=_np.uint32)

    @property
    def types(self):
        return _np.frombuffer(self._raw()[2], dtype=_np.uint8)

    def GetNumberOfCells(self):
        return len(self.types)


class DataObject:
    """VTK `vtkDataObject`-like snapshot of an evaluated pipeline node (`Fetch(proxy)`)."""

    def __init__(self, sess, node_id, block=None):
        self._sess = sess
        self._node_id = node_id
        self._block = block
        self._meta = _json.loads(sess.fetch_meta_json(node_id, block))

    # -- field data -----------------------------------------------------------
    @property
    def PointData(self):
        return FieldDataView(self._sess, self._node_id, self._block, "POINTS", self._meta)

    @property
    def CellData(self):
        return FieldDataView(self._sess, self._node_id, self._block, "CELLS", self._meta)

    def GetPointData(self):
        return self.PointData

    def GetCellData(self):
        return self.CellData

    # -- geometry ---------------------------------------------------------------
    def GetPoints(self):
        return PointsView(self._sess, self._node_id, self._block, self._meta["num_points"])

    def GetCells(self):
        return CellsView(self._sess, self._node_id, self._block)

    def GetNumberOfPoints(self):
        return self._meta["num_points"]

    def GetNumberOfCells(self):
        return self._meta["num_cells"]

    def GetBounds(self):
        return list(self._meta["bounds"])

    # -- image geometry ---------------------------------------------------------
    def GetDimensions(self):
        return list(self._meta.get("dimensions", [0, 0, 0]))

    def GetOrigin(self):
        return list(self._meta.get("origin", [0.0, 0.0, 0.0]))

    def GetSpacing(self):
        return list(self._meta.get("spacing", [1.0, 1.0, 1.0]))

    # -- multi-block -------------------------------------------------------------
    def GetNumberOfBlocks(self):
        return len(self._meta.get("blocks", []))

    def GetBlock(self, i):
        return DataObject(self._sess, self._node_id, block=int(i))

    def GetBlockName(self, i):
        return self._meta["blocks"][int(i)]

    @property
    def meta(self):
        return self._meta

    def __repr__(self):
        return f"<DataObject {self._meta['type']} points={self._meta['num_points']} cells={self._meta['num_cells']}>"
