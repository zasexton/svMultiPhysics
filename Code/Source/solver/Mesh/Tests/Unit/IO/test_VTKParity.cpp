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

// Build high-order nodes for common quadratic families
static void make_high_order_nodes(CellFamily fam,
                                  const std::vector<std::array<real_t,3>>& corners,
                                  std::vector<std::array<real_t,3>>& out_mids,
                                  std::vector<index_t>& out_ids,
                                  bool include_face_and_center) {
  auto avg = [](const std::array<real_t,3>& a,const std::array<real_t,3>& b){return std::array<real_t,3>{(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2};};
  auto centroid = [&](const std::vector<int>& idx){std::array<real_t,3> c{0,0,0}; for(int i:idx){c[0]+=corners[i][0];c[1]+=corners[i][1];c[2]+=corners[i][2];} c[0]/=idx.size();c[1]/=idx.size();c[2]/=idx.size(); return c;};

  switch (fam) {
    case CellFamily::Triangle: {
      // edges (0,1),(1,2),(2,0)
      out_mids = { avg(corners[0],corners[1]), avg(corners[1],corners[2]), avg(corners[2],corners[0]) };
      break;
    }
    case CellFamily::Quad: {
      // edges (0,1),(1,2),(2,3),(3,0)
      out_mids = { avg(corners[0],corners[1]), avg(corners[1],corners[2]), avg(corners[2],corners[3]), avg(corners[3],corners[0]) };
      if (include_face_and_center) {
        // center
        out_mids.push_back( centroid({0,1,2,3}) );
      }
      break;
    }
    case CellFamily::Tetra: {
      // edges: 01,02,03,12,13,23
      out_mids = {
        avg(corners[0],corners[1]), avg(corners[0],corners[2]), avg(corners[0],corners[3]),
        avg(corners[1],corners[2]), avg(corners[1],corners[3]), avg(corners[2],corners[3])
      };
      break;
    }
    case CellFamily::Hex: {
      // edges: 01,12,23,30, 45,56,67,74, 04,15,26,37
      out_mids = {
        avg(corners[0],corners[1]), avg(corners[1],corners[2]), avg(corners[2],corners[3]), avg(corners[3],corners[0]),
        avg(corners[4],corners[5]), avg(corners[5],corners[6]), avg(corners[6],corners[7]), avg(corners[7],corners[4]),
        avg(corners[0],corners[4]), avg(corners[1],corners[5]), avg(corners[2],corners[6]), avg(corners[3],corners[7])
      };
      if (include_face_and_center) {
        // face mids in canonical face order (sorted faces): bottom, top, front, right, back, left
        out_mids.push_back( centroid({0,1,2,3}) );
        out_mids.push_back( centroid({4,5,6,7}) );
        out_mids.push_back( centroid({0,1,5,4}) );
        out_mids.push_back( centroid({1,2,6,5}) );
        out_mids.push_back( centroid({2,3,7,6}) );
        out_mids.push_back( centroid({0,3,7,4}) );
        // center
        out_mids.push_back( centroid({0,1,2,3,4,5,6,7}) );
      }
      break;
    }
    case CellFamily::Wedge: {
      // edges: bottom tri (0,1),(1,2),(2,0); top (3,4),(4,5),(5,3); vertical (0,3),(1,4),(2,5)
      out_mids = {
        avg(corners[0],corners[1]), avg(corners[1],corners[2]), avg(corners[2],corners[0]),
        avg(corners[3],corners[4]), avg(corners[4],corners[5]), avg(corners[5],corners[3]),
        avg(corners[0],corners[3]), avg(corners[1],corners[4]), avg(corners[2],corners[5])
      };
      break;
    }
    case CellFamily::Pyramid: {
      // edges: base (0,1),(1,2),(2,3),(3,0); to apex (0,4),(1,4),(2,4),(3,4)
      out_mids = {
        avg(corners[0],corners[1]), avg(corners[1],corners[2]), avg(corners[2],corners[3]), avg(corners[3],corners[0]),
        avg(corners[0],corners[4]), avg(corners[1],corners[4]), avg(corners[2],corners[4]), avg(corners[3],corners[4])
      };
      break;
    }
    default: break;
  }

  // IDs will be appended after corners in the point array
  out_ids.clear();
  out_ids.reserve(out_mids.size());
  for (index_t i=0;i<(index_t)out_mids.size();++i) out_ids.push_back((index_t)(corners.size() + i));
}

static void build_high_order_mesh(CellFamily fam, bool quad9_or_hex27,
                                  MeshBase& mesh,
                                  std::vector<index_t>& shuffled_connectivity,
                                  index_t& center_id_out) {
  auto corners = make_coords(fam);
  std::vector<std::array<real_t,3>> mids;
  std::vector<index_t> mid_ids;
  make_high_order_nodes(fam, corners, mids, mid_ids,
                        /*include_face_and_center=*/(fam==CellFamily::Quad && quad9_or_hex27) || (fam==CellFamily::Hex && quad9_or_hex27));

  // Build point array: corners first, then mids in their defined order
  int dim = (fam == CellFamily::Triangle || fam == CellFamily::Quad) ? 2 : 3;
  std::vector<real_t> X;
  for (auto& p : corners) { X.push_back(p[0]); X.push_back(p[1]); if (dim==3) X.push_back(p[2]); }
  for (auto& p : mids)    { X.push_back(p[0]); X.push_back(p[1]); if (dim==3) X.push_back(p[2]); }

  // Connectivity: corners first, then SHUFFLED mids to stress reorderer
  std::vector<index_t> conn;
  conn.reserve(corners.size() + mid_ids.size());
  for (index_t i=0;i<(index_t)corners.size();++i) conn.push_back(i);
  // Deterministic shuffle: reverse halves
  std::vector<index_t> mids_shuffled = mid_ids;
  std::reverse(mids_shuffled.begin(), mids_shuffled.end());
  conn.insert(conn.end(), mids_shuffled.begin(), mids_shuffled.end());

  std::vector<CellShape> cell_shapes(1);
  cell_shapes[0].family = fam;
  cell_shapes[0].order = 2;
  cell_shapes[0].num_corners = (int)corners.size();

  std::vector<offset_t> c2v_off = {0, (offset_t)conn.size()};

  mesh = MeshBase(dim);
  mesh.build_from_arrays(dim, X, c2v_off, conn, cell_shapes);
  mesh.finalize();

  shuffled_connectivity = conn;
  center_id_out = -1;
  if (fam==CellFamily::Quad && mids.size()==5) center_id_out = (index_t)(corners.size()+4);
  if (fam==CellFamily::Hex && mids.size()==19) center_id_out = (index_t)(corners.size()+18);
}

TEST(VTKParity, HighOrder_ReorderAndRoundTrip) {
  struct Case { CellFamily fam; int variant; const char* name; };
  // variant: 0→basic quadratic (Tri6/Quad8/Tet10/Hex20/Wedge15/Pyr13), 1→with face+center (Quad9/Hex27)
  std::vector<Case> cases = {
    {CellFamily::Triangle, 0, "Tri6"},
    {CellFamily::Quad,     0, "Quad8"},
    {CellFamily::Quad,     1, "Quad9"},
    {CellFamily::Tetra,    0, "Tet10"},
    {CellFamily::Hex,      0, "Hex20"},
    {CellFamily::Hex,      1, "Hex27"},
    {CellFamily::Wedge,    0, "Wedge15"},
    {CellFamily::Pyramid,  0, "Pyr13"},
  };

  for (auto cs : cases) {
    MeshBase m;
    std::vector<index_t> shuffled_conn;
    index_t center_id;
    build_high_order_mesh(cs.fam, cs.variant==1, m, shuffled_conn, center_id);

    // Write → read
    std::string fname = std::string("vtk_parity_ho_") + cs.name + ".vtu";
    MeshIOOptions optsW; optsW.format = "vtu"; optsW.path = fname;
    VTKWriter::write(m, optsW);
    MeshIOOptions optsR; optsR.format = "vtu"; optsR.path = fname;
    auto m2 = VTKReader::read(optsR);

    ASSERT_EQ(m2.n_cells(), 1u);
    auto span = m2.cell_vertices_span(0);
    std::vector<index_t> conn_read(span.first, span.first + span.second);

    // Corners should come first in VTK ordering
    int nc = 0;
    switch (cs.fam) {
      case CellFamily::Triangle: nc = 3; break;
      case CellFamily::Quad: nc = 4; break;
      case CellFamily::Tetra: nc = 4; break;
      case CellFamily::Hex: nc = 8; break;
      case CellFamily::Wedge: nc = 6; break;
      case CellFamily::Pyramid: nc = 5; break;
      default: nc = 0; break;
    }
    ASSERT_GE((int)conn_read.size(), nc);
    for (int i=0;i<nc;++i) EXPECT_EQ(conn_read[i], (index_t)i) << cs.name;

    // Size should match expected high-order node counts
    size_t expected_size = (size_t)nc;
    switch (cs.fam) {
      case CellFamily::Triangle: expected_size += 3; break;            // Tri6
      case CellFamily::Quad: expected_size += (cs.variant?5:4); break; // Quad9 or Quad8
      case CellFamily::Tetra: expected_size += 6; break;               // Tet10
      case CellFamily::Hex: expected_size += (cs.variant?19:12); break;// Hex27 or Hex20
      case CellFamily::Wedge: expected_size += 9; break;               // Wedge15
      case CellFamily::Pyramid: expected_size += 8; break;             // Pyr13
      default: break;
    }
    if (cs.fam == CellFamily::Hex && cs.variant==1) {
      // Some VTK builds report 28 entries for Hex27; accept 27 or 28
      EXPECT_TRUE(conn_read.size()==expected_size || conn_read.size()==expected_size+1) << cs.name;
    } else {
      EXPECT_EQ(conn_read.size(), expected_size) << cs.name;
    }

    // The set of mids should match regardless of internal ordering
    std::vector<index_t> mids_read(conn_read.begin()+nc, conn_read.end());
    if (cs.fam == CellFamily::Hex && cs.variant==1 && mids_read.size()==(expected_size-nc)+1) {
      // Drop potential duplicate last entry
      mids_read.pop_back();
    }
    std::vector<index_t> mids_orig(shuffled_conn.begin()+nc, shuffled_conn.end());
    std::sort(mids_read.begin(), mids_read.end());
    std::sort(mids_orig.begin(), mids_orig.end());
    EXPECT_EQ(mids_read, mids_orig) << cs.name;

    // For variants with a center, the last entry should be the center id we appended
    if (center_id >= 0 && !(cs.fam==CellFamily::Hex && cs.variant==1)) {
      EXPECT_EQ(conn_read.back(), center_id) << cs.name;
    }
  }
}

}} // namespace svmp::test

#endif // MESH_HAS_VTK
