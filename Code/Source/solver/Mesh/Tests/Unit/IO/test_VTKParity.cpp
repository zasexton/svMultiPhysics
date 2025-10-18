/* Optional VTK parity and IO round-trip tests.
 * Built and run only when MESH_HAS_VTK is enabled and VTK is found.
 */

#include "gtest/gtest.h"
#include "Topology/CellTopology.h"
#include "Mesh.h"

#ifdef MESH_HAS_VTK

#include "IO/VTKReader.h"
#include "IO/VTKWriter.h"

// VTK includes (headers only; link comes from svmesh target)
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkTetra.h>
#include <vtkHexahedron.h>
#include <vtkWedge.h>
#include <vtkPyramid.h>
#include <vtkTriangle.h>
#include <vtkQuad.h>

namespace svmp { namespace test {

static std::vector<std::array<real_t,3>> make_coords(CellFamily fam) {
  switch (fam) {
    case CellFamily::Triangle:
      return {{ {0,0,0}, {1,0,0}, {0,1,0} }};
    case CellFamily::Quad:
      return {{ {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} }};
    case CellFamily::Tetra:
      return {{ {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} }};
    case CellFamily::Hex:
      return {{ {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1} }};
    case CellFamily::Wedge:
      return {{ {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,0,1}, {0,1,1} }};
    case CellFamily::Pyramid:
      return {{ {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {0.5,0.5,1} }};
    default:
      return {};
  }
}

static void build_single_cell_mesh(CellFamily fam, MeshBase& mesh) {
  auto coords = make_coords(fam);
  int dim = (fam == CellFamily::Triangle || fam == CellFamily::Quad) ? 2 : 3;

  std::vector<real_t> X;
  X.reserve(coords.size() * dim);
  for (auto& p : coords) {
    X.push_back(p[0]);
    X.push_back(p[1]);
    if (dim == 3) X.push_back(p[2]);
  }

  std::vector<CellShape> cell_shapes(1);
  cell_shapes[0].family = fam;
  cell_shapes[0].order = 1;
  cell_shapes[0].num_corners = (int)coords.size();

  std::vector<offset_t> c2v_off = {0, (offset_t)coords.size()};
  std::vector<index_t>  c2v;
  c2v.reserve(coords.size());
  for (index_t i = 0; i < (index_t)coords.size(); ++i) c2v.push_back(i);

  mesh = MeshBase(dim);
  mesh.build_from_arrays(dim, X, c2v_off, c2v, cell_shapes);
  mesh.finalize();
}

static void collect_vtk_face_sets(CellFamily fam,
                                  const std::vector<std::array<real_t,3>>& pts,
                                  std::vector<std::vector<index_t>>& face_sets) {
  auto vp = vtkSmartPointer<vtkPoints>::New();
  for (auto& p : pts) vp->InsertNextPoint(p[0], p[1], p[2]);

  vtkSmartPointer<vtkCell> cell;
  switch (fam) {
    case CellFamily::Tetra: {
      auto c = vtkSmartPointer<vtkTetra>::New();
      c->GetPointIds()->SetId(0,0); c->GetPointIds()->SetId(1,1);
      c->GetPointIds()->SetId(2,2); c->GetPointIds()->SetId(3,3);
      cell = c; break; }
    case CellFamily::Hex: {
      auto c = vtkSmartPointer<vtkHexahedron>::New();
      for (int i=0;i<8;++i) c->GetPointIds()->SetId(i,i);
      cell = c; break; }
    case CellFamily::Wedge: {
      auto c = vtkSmartPointer<vtkWedge>::New();
      for (int i=0;i<6;++i) c->GetPointIds()->SetId(i,i);
      cell = c; break; }
    case CellFamily::Pyramid: {
      auto c = vtkSmartPointer<vtkPyramid>::New();
      for (int i=0;i<5;++i) c->GetPointIds()->SetId(i,i);
      cell = c; break; }
    default: return; // 2D handled separately
  }

  int nf = cell->GetNumberOfFaces();
  for (int i = 0; i < nf; ++i) {
    auto f = cell->GetFace(i);
    std::vector<index_t> vs;
    for (int j = 0; j < f->GetNumberOfPoints(); ++j) {
      vs.push_back((index_t)f->GetPointId(j));
    }
    std::sort(vs.begin(), vs.end());
    face_sets.push_back(std::move(vs));
  }
  std::sort(face_sets.begin(), face_sets.end());
}

TEST(VTKParity, OrientedCanonicalParityAndRoundTrip) {
  // Families to test
  std::vector<CellFamily> fams = {
    CellFamily::Triangle, CellFamily::Quad,
    CellFamily::Tetra, CellFamily::Hex, CellFamily::Wedge, CellFamily::Pyramid
  };

  for (auto fam : fams) {
    // Build a single-cell mesh
    MeshBase mesh;
    build_single_cell_mesh(fam, mesh);

    // Compare canonical face sets against VTK for 3D families
    if (fam == CellFamily::Tetra || fam == CellFamily::Hex || fam == CellFamily::Wedge || fam == CellFamily::Pyramid) {
      // Our canonical local faces
      auto view = CellTopology::get_boundary_faces_canonical_view(fam);
      std::vector<std::vector<index_t>> ours;
      for (int f = 0; f < view.face_count; ++f) {
        int b = view.offsets[f], e = view.offsets[f+1];
        std::vector<index_t> loc(view.indices + b, view.indices + e);
        // map local->global (identity since we used 0..N-1)
        std::sort(loc.begin(), loc.end());
        ours.push_back(std::move(loc));
      }
      std::sort(ours.begin(), ours.end());

      // Collect VTK faces
      auto coords = make_coords(fam);
      std::vector<std::vector<index_t>> vtks;
      collect_vtk_face_sets(fam, coords, vtks);

      ASSERT_EQ(ours.size(), vtks.size());
      EXPECT_EQ(ours, vtks);
    }

    // IO round-trip via VTK: write -> read
    std::string fname = "vtk_parity_tmp_" + std::to_string((int)fam) + ".vtu";
    MeshIOOptions optsW; optsW.format = "vtu"; optsW.path = fname;
    VTKWriter::write(mesh, optsW);

    MeshIOOptions optsR; optsR.format = "vtu"; optsR.path = fname;
    auto mesh2 = VTKReader::read(optsR);

    ASSERT_EQ(mesh.n_cells(), mesh2.n_cells());
    ASSERT_EQ(mesh.n_vertices(), mesh2.n_vertices());
    // Compare sorted connectivity of the single cell
    auto span1 = mesh.cell_vertices_span(0);
    auto span2 = mesh2.cell_vertices_span(0);
    std::vector<index_t> a(span1.first, span1.first + span1.second);
    std::vector<index_t> b(span2.first, span2.first + span2.second);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b);
  }
}

}} // namespace svmp::test

#endif // MESH_HAS_VTK

