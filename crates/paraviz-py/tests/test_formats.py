"""Format matrix: hand-written tiny files -> ParaView reader class -> Fetch.

Validates that every reader proxy class maps to the right paraviz DatasetReader
and that parsed data matches the file contents exactly.
"""

import pytest

import paraview.simple as pvs
from paraview.simple import Fetch


@pytest.fixture(autouse=True)
def fresh():
    pvs.Disconnect()
    yield
    pvs.Disconnect()


VTU = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0 0 0 1</DataArray></Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData Scalars="p"><DataArray type="Float32" Name="p" NumberOfComponents="1" format="ascii">{vals}</DataArray></PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

VTP = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0</DataArray></Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>
      </Polys>
      <PointData Scalars="s"><DataArray type="Float32" Name="s" NumberOfComponents="1" format="ascii">5 6 7</DataArray></PointData>
    </Piece>
  </PolyData>
</VTKFile>
"""

VTI = """<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="0 1 0 1 0 1" Origin="0 0 0" Spacing="1 1 1">
    <Piece Extent="0 1 0 1 0 1">
      <PointData Scalars="d"><DataArray type="Float32" Name="d" NumberOfComponents="1" format="ascii">0 1 2 3 4 5 6 7</DataArray></PointData>
    </Piece>
  </ImageData>
</VTKFile>
"""


def test_vtu(tmp_path):
    f = tmp_path / "a.vtu"
    f.write_text(VTU.format(vals="1 2 3 4"))
    r = pvs.XMLUnstructuredGridReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 4
    assert d.GetNumberOfCells() == 1
    assert d.PointData["p"].GetRange() == [1.0, 4.0]
    assert d.GetBounds() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_vtp(tmp_path):
    f = tmp_path / "a.vtp"
    f.write_text(VTP)
    r = pvs.XMLPolyDataReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 3
    assert d.GetNumberOfCells() == 1
    assert d.PointData["s"].GetRange() == [5.0, 7.0]


def test_vti(tmp_path):
    f = tmp_path / "a.vti"
    f.write_text(VTI)
    r = pvs.XMLImageDataReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 8
    assert d.GetDimensions() == [2, 2, 2]
    assert d.PointData["d"].GetRange() == [0.0, 7.0]


def test_vtm(tmp_path):
    (tmp_path / "blk0.vtp").write_text(VTP)
    vtm = tmp_path / "m.vtm"
    vtm.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n'
        "  <vtkMultiBlockDataSet>\n"
        '    <DataSet index="0" file="blk0.vtp"/>\n'
        "  </vtkMultiBlockDataSet>\n</VTKFile>\n"
    )
    r = pvs.XMLMultiBlockDataReader(FileName=[str(vtm)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfBlocks() == 1
    assert d.GetBlock(0).GetNumberOfPoints() == 3
    assert d.GetBlock(0).PointData["s"].GetRange() == [5.0, 7.0]


def test_pvd(tmp_path):
    (tmp_path / "s0.vtu").write_text(VTU.format(vals="1 2 3 4"))
    (tmp_path / "s1.vtu").write_text(VTU.format(vals="10 20 30 40"))
    pvd = tmp_path / "a.pvd"
    pvd.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n"
        '    <DataSet timestep="0.0" group="" part="0" file="s0.vtu"/>\n'
        '    <DataSet timestep="1.0" group="" part="0" file="s1.vtu"/>\n'
        "  </Collection>\n</VTKFile>\n"
    )
    r = pvs.PVDReader(FileName=[str(pvd)])
    r.UpdatePipeline()
    scene = pvs.GetAnimationScene()
    scene.AnimationTime = 0.0
    d0 = Fetch(r)
    assert d0.PointData["p"].GetRange() == [1.0, 4.0]
    scene.AnimationTime = 1.0
    r.UpdatePipeline()
    d1 = Fetch(r)
    assert d1.PointData["p"].GetRange() == [10.0, 40.0]


def test_csv(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x,y,z,pressure\n0,0,0,1.5\n1,0,0,2.5\n0,1,0,3.5\n")
    r = pvs.CSVReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 3
    assert d.PointData["pressure"].GetRange() == [1.5, 3.5]


def test_stl(tmp_path):
    f = tmp_path / "t.stl"
    f.write_text(
        "solid t\nfacet normal 0 0 1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 2 0 0\n  vertex 0 3 0\n"
        " endloop\nendfacet\nendsolid t\n"
    )
    r = pvs.STLReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 3
    assert d.GetNumberOfCells() == 1
    assert d.GetBounds() == [0.0, 2.0, 0.0, 3.0, 0.0, 0.0]


def test_obj(tmp_path):
    f = tmp_path / "t.obj"
    f.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    r = pvs.OBJReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 3
    assert d.GetNumberOfCells() == 1


def test_ply(tmp_path):
    f = tmp_path / "t.ply"
    f.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )
    r = pvs.PLYReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 3
    assert d.GetNumberOfCells() == 1


def test_mha(tmp_path):
    import struct

    f = tmp_path / "v.mha"
    raw = struct.pack("<8f", *range(8))
    f.write_bytes(
        b"ObjectType = Image\nNDims = 3\nDimSize = 2 2 2\nElementType = MET_FLOAT\n"
        b"ElementDataFile = LOCAL\n"
        + raw
    )
    r = pvs.MetaImageReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 8
    assert d.GetDimensions() == [2, 2, 2]


def test_legacy_vtk(tmp_path):
    f = tmp_path / "t.vtk"
    f.write_text(
        "# vtk DataFile Version 3.0\ntest\nASCII\nDATASET POLYDATA\n"
        "POINTS 3 float\n0 0 0 1 0 0 0 1 0\n"
        "POLYGONS 1 4\n3 0 1 2\n"
        "POINT_DATA 3\nSCALARS s float 1\nLOOKUP_TABLE default\n5 6 7\n"
    )
    r = pvs.LegacyVTKReader(FileName=[str(f)])
    r.UpdatePipeline()
    d = Fetch(r)
    assert d.GetNumberOfPoints() == 3
    assert d.PointData["s"].GetRange() == [5.0, 7.0]
