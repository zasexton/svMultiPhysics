/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "VTKWriter.h"

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLPPolyDataWriter.h>
#include <vtkDataSet.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkLongLongArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCellTypes.h>
#include <vtkDataCompressor.h>
#include <vtkZLibDataCompressor.h>
#include <vtkDataSetAttributes.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include "../Topology/CellTopology.h"

namespace svmp {

// Static member definitions
std::unordered_map<CellFamily, int> VTKWriter::family_to_vtk_linear_;
std::unordered_map<CellFamily, int> VTKWriter::family_to_vtk_quadratic_;
bool VTKWriter::registry_initialized_ = false;

VTKWriter::VTKWriter() {
  setup_vtk_registry();
}

VTKWriter::~VTKWriter() = default;

void VTKWriter::register_with_mesh() {
  // Register for different VTK formats
  MeshBase::register_writer("vtk", VTKWriter::write);
  MeshBase::register_writer("vtu", VTKWriter::write);
  MeshBase::register_writer("vtp", VTKWriter::write);
  MeshBase::register_writer("pvtu", VTKWriter::write);

  // Setup VTK cell type mappings
  setup_vtk_registry();
}

void VTKWriter::write(const MeshBase& mesh, const MeshIOOptions& options) {
  const std::string& filename = options.path;
  const std::string& format = options.format;

  // Parse write options from kv map
  WriteOptions write_opts;

  auto binary_it = options.kv.find("binary");
  if (binary_it != options.kv.end()) {
    write_opts.binary = (binary_it->second == "true" || binary_it->second == "1");
  }

  auto compress_it = options.kv.find("compress");
  if (compress_it != options.kv.end()) {
    write_opts.compressed = (compress_it->second == "true" || compress_it->second == "1");
  }

  // Dispatch based on format
  if (format == "vtk") {
    write_vtk(mesh, filename, write_opts);
  } else if (format == "vtu") {
    write_vtu(mesh, filename, write_opts);
  } else if (format == "vtp") {
    write_vtp(mesh, filename, write_opts);
  } else if (format == "pvtu") {
    // For parallel output, need rank and size from options
    int rank = 0, size = 1;
    auto rank_it = options.kv.find("rank");
    auto size_it = options.kv.find("size");
    if (rank_it != options.kv.end()) rank = std::stoi(rank_it->second);
    if (size_it != options.kv.end()) size = std::stoi(size_it->second);
    write_pvtu(mesh, filename, rank, size, write_opts);
  } else {
    // Default based on file extension
    if (filename.find(".vtu") != std::string::npos) {
      write_vtu(mesh, filename, write_opts);
    } else if (filename.find(".vtp") != std::string::npos) {
      write_vtp(mesh, filename, write_opts);
    } else {
      write_vtk(mesh, filename, write_opts); // Default to legacy VTK
    }
  }
}

void VTKWriter::write_vtk(const MeshBase& mesh, const std::string& filename,
                         const WriteOptions& opts) {
  // Convert to UnstructuredGrid
  auto ugrid = convert_to_unstructured_grid(mesh);

  // Write field data
  write_field_data(mesh, ugrid, opts);

  // Write using legacy writer
  vtkSmartPointer<vtkUnstructuredGridWriter> writer =
      vtkSmartPointer<vtkUnstructuredGridWriter>::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(ugrid);

  if (opts.binary) {
    writer->SetFileTypeToBinary();
  } else {
    writer->SetFileTypeToASCII();
  }

  writer->Write();
}

void VTKWriter::write_vtu(const MeshBase& mesh, const std::string& filename,
                         const WriteOptions& opts) {
  // Convert to UnstructuredGrid
  auto ugrid = convert_to_unstructured_grid(mesh);

  // Write field data
  write_field_data(mesh, ugrid, opts);

  // Write using XML writer
  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer =
      vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(ugrid);

  if (opts.binary) {
    writer->SetDataModeToBinary();
  } else {
    writer->SetDataModeToAscii();
  }

  if (opts.compressed && opts.binary) {
    vtkSmartPointer<vtkZLibDataCompressor> compressor =
        vtkSmartPointer<vtkZLibDataCompressor>::New();
    writer->SetCompressor(compressor);
  }

  writer->Write();
}

void VTKWriter::write_vtp(const MeshBase& mesh, const std::string& filename,
                         const WriteOptions& opts) {
  // Convert to PolyData
  auto polydata = convert_to_polydata(mesh);

  // Write field data
  write_field_data(mesh, polydata, opts);

  // Write using XML writer
  vtkSmartPointer<vtkXMLPolyDataWriter> writer =
      vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(polydata);

  if (opts.binary) {
    writer->SetDataModeToBinary();
  } else {
    writer->SetDataModeToAscii();
  }

  if (opts.compressed && opts.binary) {
    vtkSmartPointer<vtkZLibDataCompressor> compressor =
        vtkSmartPointer<vtkZLibDataCompressor>::New();
    writer->SetCompressor(compressor);
  }

  writer->Write();
}

void VTKWriter::write_pvtu(const MeshBase& mesh, const std::string& filename,
                          int rank, int size, const WriteOptions& opts) {
  // Write piece file for this rank
  std::stringstream piece_filename;
  piece_filename << filename.substr(0, filename.rfind(".pvtu"))
                 << "_" << rank << ".vtu";
  write_vtu(mesh, piece_filename.str(), opts);

  // Rank 0 writes the master file
  if (rank == 0) {
    // Create a dummy grid for the parallel writer
    vtkSmartPointer<vtkUnstructuredGrid> dummy_grid =
        vtkSmartPointer<vtkUnstructuredGrid>::New();

    vtkSmartPointer<vtkXMLPUnstructuredGridWriter> pwriter =
        vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
    pwriter->SetFileName(filename.c_str());
    pwriter->SetNumberOfPieces(size);
    pwriter->SetStartPiece(0);
    pwriter->SetEndPiece(size - 1);
    pwriter->SetInputData(dummy_grid);

    if (opts.binary) {
      pwriter->SetDataModeToBinary();
    } else {
      pwriter->SetDataModeToAscii();
    }

    pwriter->Write();
  }
}

vtkSmartPointer<vtkUnstructuredGrid> VTKWriter::convert_to_unstructured_grid(const MeshBase& mesh) {
  setup_vtk_registry();

  vtkSmartPointer<vtkUnstructuredGrid> ugrid =
      vtkSmartPointer<vtkUnstructuredGrid>::New();

  // Create points
  auto points = create_vtk_points(mesh, Configuration::Reference);
  ugrid->SetPoints(points);

  // Create cells
  create_vtk_cells(mesh, ugrid);

  return ugrid;
}

vtkSmartPointer<vtkPolyData> VTKWriter::convert_to_polydata(const MeshBase& mesh) {
  setup_vtk_registry();

  vtkSmartPointer<vtkPolyData> polydata =
      vtkSmartPointer<vtkPolyData>::New();

  // Create points
  auto points = create_vtk_points(mesh, Configuration::Reference);
  polydata->SetPoints(points);

  // Create cells - for PolyData we need to separate by type
  vtkSmartPointer<vtkCellArray> verts = vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));
    const auto& shape = mesh.cell_shape(static_cast<index_t>(c));

    vtkSmartPointer<vtkIdList> cell_vertices = vtkSmartPointer<vtkIdList>::New();
    for (size_t i = 0; i < n_vertices; ++i) {
      cell_vertices->InsertNextId(vertices_ptr[i]);
    }

    // Add to appropriate cell array based on family
    switch (shape.family) {
      case CellFamily::Line:
        lines->InsertNextCell(cell_vertices);
        break;
      case CellFamily::Triangle:
      case CellFamily::Quad:
      case CellFamily::Polygon:
        polys->InsertNextCell(cell_vertices);
        break;
      default:
        // 3D cells not directly supported in PolyData
        // Could triangulate faces, but for now skip
        break;
    }
  }

  if (verts->GetNumberOfCells() > 0) polydata->SetVerts(verts);
  if (lines->GetNumberOfCells() > 0) polydata->SetLines(lines);
  if (polys->GetNumberOfCells() > 0) polydata->SetPolys(polys);

  return polydata;
}

vtkSmartPointer<vtkPoints> VTKWriter::create_vtk_points(const MeshBase& mesh, Configuration cfg) {
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->SetDataTypeToDouble();

  const std::vector<real_t>& coords = (cfg == Configuration::Current && mesh.has_current_coords())
                                      ? mesh.X_cur() : mesh.X_ref();

  int spatial_dim = mesh.dim();
  size_t n_points = mesh.n_vertices();

  points->SetNumberOfPoints(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    double pt[3] = {0, 0, 0};
    for (int d = 0; d < spatial_dim; ++d) {
      pt[d] = coords[i * spatial_dim + d];
    }
    points->SetPoint(i, pt);
  }

  return points;
}

// Helper: choose VTK cell type for a given family/order/count
// Inference policy
// - Prefer Lagrange: infer p from total node count via CellTopology::infer_lagrange_order.
//   If p>2, use explicit VTK_LAGRANGE_* types where available.
// - If Lagrange inference is not a match, optionally infer Serendipity (currently Quad only)
//   via CellTopology::infer_serendipity_order, and pick the closest VTK quadratic/biquadratic.
// - Otherwise, fall back to classic linear/quadratic variants based on node count thresholds.
static int choose_vtk_type_for(const CellShape& shape, size_t n_vertices) {
  // Linear
  if (shape.order <= 1) {
    switch (shape.family) {
      case CellFamily::Line: return VTK_LINE;
      case CellFamily::Triangle: return VTK_TRIANGLE;
      case CellFamily::Quad: return VTK_QUAD;
      case CellFamily::Tetra: return VTK_TETRA;
      case CellFamily::Hex: return VTK_HEXAHEDRON;
      case CellFamily::Wedge: return VTK_WEDGE;
      case CellFamily::Pyramid: return VTK_PYRAMID;
      case CellFamily::Polygon: return VTK_POLYGON;
      case CellFamily::Polyhedron: return VTK_POLYHEDRON;
      default: return VTK_POLYGON;
    }
  }

  // Quadratic and higher (we key off vertex count to disambiguate variants)
  // For order>2 prefer Lagrange/Serendipity explicit types when available
  int p_lag = CellTopology::infer_lagrange_order(shape.family, n_vertices);
  int p_ser = (p_lag<0) ? CellTopology::infer_serendipity_order(shape.family, n_vertices) : -1;
  if (p_lag > 2) {
    switch (shape.family) {
      case CellFamily::Line: return VTK_LAGRANGE_CURVE; // curve is Lagrange line in VTK
      case CellFamily::Triangle: return VTK_LAGRANGE_TRIANGLE;
      case CellFamily::Quad: return VTK_LAGRANGE_QUADRILATERAL;
      case CellFamily::Tetra: return VTK_LAGRANGE_TETRAHEDRON;
      case CellFamily::Hex: return VTK_LAGRANGE_HEXAHEDRON;
      case CellFamily::Wedge: return VTK_LAGRANGE_WEDGE;
      case CellFamily::Pyramid: return VTK_LAGRANGE_PYRAMID;
      default: break;
    }
  }

  switch (shape.family) {
    case CellFamily::Line:
      return (n_vertices >= 3) ? VTK_QUADRATIC_EDGE : VTK_LINE;
    case CellFamily::Triangle:
      return (n_vertices >= 6) ? VTK_QUADRATIC_TRIANGLE : VTK_TRIANGLE;
    case CellFamily::Quad:
      // VTK does not have explicit Serendipity quad types beyond quadratic. Promote
      // high-order serendipity quads to VTK Lagrange quads by synthesizing interior nodes.
      if (p_ser > 2) return VTK_LAGRANGE_QUADRILATERAL;
      if (n_vertices >= 9) return VTK_BIQUADRATIC_QUAD; // with center node
      if (n_vertices >= 8) return VTK_QUADRATIC_QUAD;    // 4 edge mids
      return VTK_QUAD;
    case CellFamily::Tetra:
      return (n_vertices >= 10) ? VTK_QUADRATIC_TETRA : VTK_TETRA;
    case CellFamily::Hex:
      if (n_vertices >= 27) return VTK_TRIQUADRATIC_HEXAHEDRON; // 12 edge + 6 face + 1 center
      if (n_vertices >= 20) return VTK_QUADRATIC_HEXAHEDRON;     // 12 edge mids
      return VTK_HEXAHEDRON;
    case CellFamily::Wedge:
      return (n_vertices >= 15) ? VTK_QUADRATIC_WEDGE : VTK_WEDGE;
    case CellFamily::Pyramid:
      return (n_vertices >= 13) ? VTK_QUADRATIC_PYRAMID : VTK_PYRAMID;
    default:
      return VTK_POLYGON;
  }
}

// Reorder arbitrary high-order connectivity into VTK expected ordering:
// corners -> edge mids (topology edge view order) -> face mids (face view order) -> center
// Assumptions:
// - Corner vertices occupy the first 'num_corners' entries of the connectivity
// - One edge midpoint per edge (quadratic); optional face midpoints and center for specific families
// - For families with face mids and center (Hex27), we assign by geometric proximity
static std::vector<vtkIdType> reorder_high_order_to_vtk(const MeshBase& mesh,
                                                        index_t cell_id,
                                                        const index_t* verts,
                                                        size_t n_vertices,
                                                        const CellShape& shape) {
  const int dim = mesh.dim();
  const auto& X = mesh.X_ref();
  const auto get_pt = [&](index_t v)->std::array<double,3> {
    std::array<double,3> p{0,0,0};
    for (int d=0; d<dim; ++d) p[d] = X[v*dim + d];
    return p;
  };
  auto dist2 = [&](const std::array<double,3>& a, const std::array<double,3>& b){
    double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2]; return dx*dx+dy*dy+dz*dz; };

  const size_t nc = (shape.num_corners > 0) ? static_cast<size_t>(shape.num_corners) : n_vertices;
  std::vector<vtkIdType> corners; corners.reserve(nc);
  for (size_t i=0;i<nc && i<n_vertices;++i) corners.push_back(verts[i]);

  std::vector<index_t> pool;
  for (size_t i=nc;i<n_vertices;++i) pool.push_back(verts[i]);
  std::vector<char> used(pool.size(), 0);
  auto take_nearest = [&](const std::array<double,3>& target)->index_t {
    double best = 1e300; int best_k = -1;
    for (size_t k=0;k<pool.size();++k) if(!used[k]) {
      auto p = get_pt(pool[k]); double d2 = dist2(p,target);
      if (d2 < best) { best = d2; best_k = static_cast<int>(k); }
    }
    if (best_k < 0) return INVALID_INDEX;
    used[best_k] = 1; return pool[best_k];
  };

  // If p>2, use CellTopology high-order pattern (Lagrange preferred, else Serendipity)
  if (shape.order > 2) {
    int p_lag = CellTopology::infer_lagrange_order(shape.family, n_vertices);
    int p_ser = (p_lag<0) ? CellTopology::infer_serendipity_order(shape.family, n_vertices) : -1;
    int p = (p_lag>0) ? p_lag : p_ser;
    CellTopology::HighOrderKind kind = (p_lag>0) ? CellTopology::HighOrderKind::Lagrange : CellTopology::HighOrderKind::Serendipity;
    if (p > 2) {
      auto pattern = CellTopology::high_order_pattern(shape.family, p, kind);
      std::vector<vtkIdType> out; out.reserve(n_vertices);
      // corners
      out.insert(out.end(), corners.begin(), corners.end());

      auto eview = CellTopology::get_edges_view(shape.family);
      auto fview = CellTopology::get_oriented_boundary_faces_view(shape.family);

      auto lerp = [&](const std::array<double,3>& a,const std::array<double,3>& b,double t){
        return std::array<double,3>{ (1-t)*a[0]+t*b[0], (1-t)*a[1]+t*b[1], (1-t)*a[2]+t*b[2] };
      };
      auto trilin = [&](double u,double v,double w){
        // Hex trilinear with corners 0..7
        auto P = [&](int idx){ return get_pt(corners[idx]); };
        double a00 = (1-u), a10 = u, b00 = (1-v), b10 = v, c00 = (1-w), c10 = w;
        auto p000=P(0), p100=P(1), p110=P(2), p010=P(3), p001=P(4), p101=P(5), p111=P(6), p011=P(7);
        std::array<double,3> r{0,0,0};
        auto acc=[&](const std::array<double,3>& p,double wgt){ r[0]+=wgt*p[0]; r[1]+=wgt*p[1]; r[2]+=wgt*p[2]; };
        acc(p000, a00*b00*c00); acc(p100, a10*b00*c00); acc(p110, a10*b10*c00); acc(p010, a00*b10*c00);
        acc(p001, a00*b00*c10); acc(p101, a10*b00*c10); acc(p111, a10*b10*c10); acc(p011, a00*b10*c10);
        return r;
      };

      // Build a mapping by consuming pool according to pattern
      for (const auto& role : pattern.sequence) {
        if (role.role == CellTopology::HONodeRole::Corner) continue; // already emitted
        if (role.role == CellTopology::HONodeRole::Edge) {
          int ei = role.idx0; int steps = role.idx2; int step = role.idx1; if (ei<0 || ei>=eview.edge_count) continue;
          int li = eview.pairs_flat[2*ei+0]; int lj = eview.pairs_flat[2*ei+1];
          if (li>= (int)nc || lj>= (int)nc) continue;
          double t = (double)step / (double)(steps+1);
          auto pa = get_pt(corners[li]); auto pb = get_pt(corners[lj]);
          auto tgt = lerp(pa,pb,t);
          index_t pick = take_nearest(tgt);
          if (pick!=INVALID_INDEX) out.push_back(pick);
        } else if (role.role == CellTopology::HONodeRole::Face) {
          int fi = role.idx0; if (fi<0 || fi>=fview.face_count) continue;
          int b = fview.offsets[fi], e = fview.offsets[fi+1]; int fv=e-b;
          if (fv==3) {
            // Tri face barycentric
            double i = (double)role.idx1, j=(double)role.idx2, denom = (double)(p-1);
            double t1 = i/denom, t2=j/denom, t0 = std::max(0.0, 1.0 - t1 - t2);
            int l0=fview.indices[b+0], l1=fview.indices[b+1], l2=fview.indices[b+2];
            auto P0=get_pt(corners[l0]), P1=get_pt(corners[l1]), P2=get_pt(corners[l2]);
            std::array<double,3> tgt{ t0*P0[0]+t1*P1[0]+t2*P2[0], t0*P0[1]+t1*P1[1]+t2*P2[1], t0*P0[2]+t1*P1[2]+t2*P2[2] };
            index_t pick = take_nearest(tgt);
            if (pick!=INVALID_INDEX) out.push_back(pick);
          } else if (fv==4) {
            // Quad face bilinear
            double u = (double)role.idx1/(double)(p-1), v=(double)role.idx2/(double)(p-1);
            int l0=fview.indices[b+0], l1=fview.indices[b+1], l2=fview.indices[b+2], l3=fview.indices[b+3];
            auto P0=get_pt(corners[l0]), P1=get_pt(corners[l1]), P2=get_pt(corners[l2]), P3=get_pt(corners[l3]);
            std::array<double,3> tgt{
              (1-u)*(1-v)*P0[0] + u*(1-v)*P1[0] + u*v*P2[0] + (1-u)*v*P3[0],
              (1-u)*(1-v)*P0[1] + u*(1-v)*P1[1] + u*v*P2[1] + (1-u)*v*P3[1],
              (1-u)*(1-v)*P0[2] + u*(1-v)*P1[2] + u*v*P2[2] + (1-u)*v*P3[2]
            };
            index_t pick = take_nearest(tgt);
            if (pick!=INVALID_INDEX) out.push_back(pick);
          }
        } else if (role.role == CellTopology::HONodeRole::Volume) {
          // Only Hex implemented for now
          if (shape.family == CellFamily::Hex) {
            double u=(double)role.idx0/(double)(p-1), v=(double)role.idx1/(double)(p-1), w=(double)role.idx2/(double)(p-1);
            auto tgt = trilin(u,v,w);
            index_t pick = take_nearest(tgt);
            if (pick!=INVALID_INDEX) out.push_back(pick);
          } else {
            // fallback: pick nearest to centroid
            std::array<double,3> c{0,0,0}; for(size_t i=0;i<nc;++i){ auto q=get_pt(corners[i]); c[0]+=q[0]; c[1]+=q[1]; c[2]+=q[2]; }
            c[0]/=nc; c[1]/=nc; c[2]/=nc; index_t pick=take_nearest(c); if (pick!=INVALID_INDEX) out.push_back(pick);
          }
        }
      }
      // Append any leftovers in pool order to preserve count
      for (size_t k=0;k<pool.size();++k) if(!used[k]) out.push_back(pool[k]);
      return out;
    }
  }

  // Determine expected counts per family (quadratic defaults)
  size_t need_edge = 0, need_face = 0, need_center = 0;
  switch (shape.family) {
    case CellFamily::Triangle: need_edge = 3; break; // Tri6
    case CellFamily::Quad:
      if (n_vertices >= 9) { need_edge = 4; need_center = 1; }
      else { need_edge = 4; }
      break;
    case CellFamily::Tetra: need_edge = 6; break; // Tet10
    case CellFamily::Hex:
      if (n_vertices >= 27) { need_edge = 12; need_face = 6; need_center = 1; }
      else { need_edge = 12; }
      break;
    case CellFamily::Wedge: need_edge = 9; break;  // Wedge15
    case CellFamily::Pyramid: need_edge = 8; break; // Pyr13
    default: break;
  }

  need_edge = std::min(need_edge, pool.size());

  // Edge midpoints in topology edge order
  std::vector<vtkIdType> edge_mids; edge_mids.reserve(need_edge);
  auto eview = CellTopology::get_edges_view(shape.family);
  for (int ei=0; ei<eview.edge_count && edge_mids.size()<need_edge; ++ei) {
    int li = eview.pairs_flat[2*ei+0];
    int lj = eview.pairs_flat[2*ei+1];
    if (li >= static_cast<int>(nc) || lj >= static_cast<int>(nc)) { continue; }
    auto pa = get_pt(corners[li]);
    auto pb = get_pt(corners[lj]);
    std::array<double,3> mid{ (pa[0]+pb[0])*0.5, (pa[1]+pb[1])*0.5, (pa[2]+pb[2])*0.5 };
    index_t pick = take_nearest(mid);
    if (pick == INVALID_INDEX) break;
    edge_mids.push_back(pick);
  }

  // Face midpoints in oriented face order (only for families that have them)
  std::vector<vtkIdType> face_mids; face_mids.reserve(need_face);
  if (need_face > 0) {
    auto fview = CellTopology::get_oriented_boundary_faces_view(shape.family);
    for (int fi=0; fi<fview.face_count && face_mids.size()<need_face; ++fi) {
      int b = fview.offsets[fi], e = fview.offsets[fi+1];
      std::array<double,3> cen{0,0,0}; int cnt=0;
      for (int k=b; k<e; ++k) {
        int lv = fview.indices[k]; if (lv>=static_cast<int>(nc)) continue;
        auto p = get_pt(corners[lv]);
        cen[0]+=p[0]; cen[1]+=p[1]; cen[2]+=p[2]; ++cnt;
      }
      if (cnt>0) { cen[0]/=cnt; cen[1]/=cnt; cen[2]/=cnt; }
      index_t pick = take_nearest(cen);
      if (pick == INVALID_INDEX) break;
      face_mids.push_back(pick);
    }
  }

  // Body center (if present)
  std::vector<vtkIdType> rest;
  for (size_t k=0;k<pool.size();++k) if(!used[k]) rest.push_back(pool[k]);
  std::vector<vtkIdType> body_center;
  if (need_center > 0 && !rest.empty()) {
    // pick the point closest to cell centroid
    std::array<double,3> c{0,0,0};
    for (size_t i=0;i<nc;++i){ auto p=get_pt(corners[i]); c[0]+=p[0]; c[1]+=p[1]; c[2]+=p[2]; }
    c[0]/=nc; c[1]/=nc; c[2]/=nc;
    double best=1e300; vtkIdType id=-1;
    for (auto v: rest){ auto p=get_pt(v); double d2=dist2(p,c); if(d2<best){best=d2; id=v;} }
    if (id>=0) {
      body_center.push_back(id);
      // mark used in pool so it won't be appended as leftover
      for (size_t k=0;k<pool.size();++k) if (pool[k]==id) { used[k]=1; break; }
    }
  }

  std::vector<vtkIdType> out; out.reserve(n_vertices);
  out.insert(out.end(), corners.begin(), corners.end());
  out.insert(out.end(), edge_mids.begin(), edge_mids.end());
  out.insert(out.end(), face_mids.begin(), face_mids.end());
  out.insert(out.end(), body_center.begin(), body_center.end());
  // Append any leftovers (best-effort) to keep connectivity length consistent
  for (size_t k=0;k<pool.size();++k) if(!used[k]) out.push_back(pool[k]);
  return out;
}

static bool should_use_geometric_reorder_for_vtk_quadratic(const MeshBase& mesh,
                                                           index_t cell_id,
                                                           const index_t* verts,
                                                           size_t n_vertices,
                                                           const CellShape& shape) {
  // Reordering based on geometric proximity only makes sense for straight-sided elements.
  // For curved/high-order geometry, node positions are not located at linear midpoints, so
  // we must preserve the mesh's internal canonical ordering.
  const int dim = mesh.dim();
  const auto& X = mesh.X_ref();
  const int nc = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(n_vertices);
  if (nc <= 0 || static_cast<size_t>(nc) > n_vertices) {
    return false;
  }
  if (n_vertices <= static_cast<size_t>(nc)) {
    return false;
  }

  // Treat "quadratic" as any order<=2 case where mid-edge nodes are expected.
  const int p_lag = CellTopology::infer_lagrange_order(shape.family, n_vertices);
  const int p_ser = (p_lag < 0) ? CellTopology::infer_serendipity_order(shape.family, n_vertices) : -1;
  const int p = (p_lag > 0) ? p_lag : (p_ser > 0 ? p_ser : shape.order);
  if (p > 2) {
    return false;
  }

  const auto get_pt = [&](index_t v) -> std::array<double, 3> {
    std::array<double, 3> p{0, 0, 0};
    for (int d = 0; d < dim; ++d) {
      p[d] = X[static_cast<size_t>(v) * static_cast<size_t>(dim) + static_cast<size_t>(d)];
    }
    return p;
  };
  const auto dist2 = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  };
  const auto lerp = [](const std::array<double, 3>& a, const std::array<double, 3>& b, double t) {
    return std::array<double, 3>{
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    };
  };

  // Gather pool (non-corner) points.
  std::vector<index_t> pool;
  pool.reserve(n_vertices - static_cast<size_t>(nc));
  for (size_t i = static_cast<size_t>(nc); i < n_vertices; ++i) {
    pool.push_back(verts[i]);
  }
  if (pool.empty()) {
    return false;
  }

  // Relative tolerance: mid-edge nodes should be very close to linear midpoints
  // for straight elements. If not, assume curved geometry and preserve ordering.
  constexpr double kRelTol = 1e-6;

  auto nearest_ok = [&](const std::array<double, 3>& target,
                        const std::array<double, 3>& a,
                        const std::array<double, 3>& b) -> bool {
    const double e2 = dist2(a, b);
    if (e2 <= 0.0) return false;
    const double tol2 = (kRelTol * kRelTol) * e2;
    double best = 1e300;
    for (const auto v : pool) {
      const auto p = get_pt(v);
      best = std::min(best, dist2(p, target));
    }
    return best <= tol2;
  };

  // Check edge midpoints exist near linear midpoints (quadratic only).
  auto eview = CellTopology::get_edges_view(shape.family);
  if (eview.edge_count <= 0 || !eview.pairs_flat) {
    return false;
  }

  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const int li = eview.pairs_flat[2 * ei + 0];
    const int lj = eview.pairs_flat[2 * ei + 1];
    if (li < 0 || lj < 0 || li >= nc || lj >= nc) continue;
    const auto pa = get_pt(verts[static_cast<size_t>(li)]);
    const auto pb = get_pt(verts[static_cast<size_t>(lj)]);
    const auto mid = lerp(pa, pb, 0.5);
    if (!nearest_ok(mid, pa, pb)) {
      return false;
    }
  }

  // Optional face/volume nodes: require the presence of a node near the linear centroid.
  if (shape.family == CellFamily::Quad && n_vertices >= 9) {
    auto p0 = get_pt(verts[0]);
    auto p1 = get_pt(verts[1]);
    auto p2 = get_pt(verts[2]);
    auto p3 = get_pt(verts[3]);
    std::array<double, 3> c{
        0.25 * (p0[0] + p1[0] + p2[0] + p3[0]),
        0.25 * (p0[1] + p1[1] + p2[1] + p3[1]),
        0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
    };
    if (!nearest_ok(c, p0, p2)) return false;
  }

  if (shape.family == CellFamily::Hex && n_vertices >= 27) {
    std::array<double, 3> c{0, 0, 0};
    for (int i = 0; i < 8; ++i) {
      const auto p = get_pt(verts[static_cast<size_t>(i)]);
      c[0] += p[0];
      c[1] += p[1];
      c[2] += p[2];
    }
    c[0] /= 8.0;
    c[1] /= 8.0;
    c[2] /= 8.0;
    if (!nearest_ok(c, get_pt(verts[0]), get_pt(verts[6]))) return false;
  }

  (void)cell_id;
  return true;
}

void VTKWriter::create_vtk_cells(const MeshBase& mesh, vtkDataSet* dataset) {
  vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::SafeDownCast(dataset);
  if (!ugrid) return;

  ugrid->Allocate(mesh.n_cells());
  vtkPoints* points = ugrid->GetPoints();

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));
    const auto& shape = mesh.cell_shape(static_cast<index_t>(c));

    // Get VTK cell type (consider higher-order variants via vertex count)
    int vtk_type = choose_vtk_type_for(shape, n_vertices);

    // Create ID list for cell vertices (with optional high-order reordering)
    vtkSmartPointer<vtkIdList> cell_vertices = vtkSmartPointer<vtkIdList>::New();

    // Special case: Promote high-order serendipity quadrilaterals (e.g. Q12 for p=3)
    // to VTK Lagrange quads by synthesizing face-interior nodes.
    //
    // Rationale: VTK higher-order quads require a full tensor-product node set, but many
    // mesh generators emit serendipity quads with only edge nodes. We write a valid VTK
    // representation by adding the missing interior nodes via a Coons patch that matches
    // the curved boundary exactly.
    if (shape.family == CellFamily::Quad) {
      const int p_lag = CellTopology::infer_lagrange_order(shape.family, n_vertices);
      const int p_ser = (p_lag < 0) ? CellTopology::infer_serendipity_order(shape.family, n_vertices) : -1;
      if (p_ser > 2 && vtk_type == VTK_LAGRANGE_QUADRILATERAL) {
        const int p = p_ser;
        const size_t expected_lagrange = static_cast<size_t>((p + 1) * (p + 1));
        if (!points) {
          throw std::runtime_error("VTKWriter: dataset has no vtkPoints for high-order cell promotion");
        }
        if (n_vertices != static_cast<size_t>(4 + 4 * (p - 1))) {
          throw std::runtime_error("VTKWriter: unexpected serendipity quad node count");
        }

        // Emit existing corner + edge nodes in their canonical order.
        for (size_t i = 0; i < n_vertices; ++i) {
          cell_vertices->InsertNextId(vertices_ptr[i]);
        }

        // Helper: fetch mesh point coordinates (as 3D for VTK).
        const int dim = mesh.dim();
        const auto& X = mesh.X_ref();
        auto get_pt = [&](index_t v) -> std::array<double, 3> {
          std::array<double, 3> p3{0.0, 0.0, 0.0};
          for (int d = 0; d < dim && d < 3; ++d) {
            p3[static_cast<size_t>(d)] = X[static_cast<size_t>(v) * static_cast<size_t>(dim) + static_cast<size_t>(d)];
          }
          return p3;
        };

        // 1D equispaced Lagrange basis on [-1,1] with nodes at -1 + 2*i/p, i=0..p.
        auto lagrange_basis_1d = [&](double xi, std::vector<double>& N) {
          const int n = p + 1;
          N.assign(static_cast<size_t>(n), 0.0);
          std::vector<double> nodes(static_cast<size_t>(n), 0.0);
          for (int i = 0; i < n; ++i) {
            nodes[static_cast<size_t>(i)] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(p);
          }
          for (int i = 0; i < n; ++i) {
            double Li = 1.0;
            for (int j = 0; j < n; ++j) {
              if (j == i) continue;
              const double denom = nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(j)];
              Li *= (xi - nodes[static_cast<size_t>(j)]) / denom;
            }
            N[static_cast<size_t>(i)] = Li;
          }
        };

        auto eval_curve = [&](const std::vector<std::array<double, 3>>& pts1d, double xi) -> std::array<double, 3> {
          std::vector<double> N;
          lagrange_basis_1d(xi, N);
          std::array<double, 3> out{0.0, 0.0, 0.0};
          const size_t n = std::min(N.size(), pts1d.size());
          for (size_t i = 0; i < n; ++i) {
            out[0] += N[i] * pts1d[i][0];
            out[1] += N[i] * pts1d[i][1];
            out[2] += N[i] * pts1d[i][2];
          }
          return out;
        };

        auto lerp = [](const std::array<double, 3>& a, const std::array<double, 3>& b, double t) {
          return std::array<double, 3>{
              (1.0 - t) * a[0] + t * b[0],
              (1.0 - t) * a[1] + t * b[1],
              (1.0 - t) * a[2] + t * b[2],
          };
        };

        // Corner nodes (VTK quad order: 0,1,2,3).
        const index_t v0 = vertices_ptr[0];
        const index_t v1 = vertices_ptr[1];
        const index_t v2 = vertices_ptr[2];
        const index_t v3 = vertices_ptr[3];
        const auto P0 = get_pt(v0);
        const auto P1 = get_pt(v1);
        const auto P2 = get_pt(v2);
        const auto P3 = get_pt(v3);

        // Edge nodes follow corners, grouped per VTK edge order (0-1),(1-2),(2-3),(3-0).
        const int e = p - 1;
        auto edge_node = [&](int edge_idx, int k) -> index_t {
          // edge_idx: 0..3, k: 0..e-1
          return vertices_ptr[static_cast<size_t>(4 + edge_idx * e + k)];
        };

        // Build boundary curves as 1D Lagrange polylines in parametric direction.
        // bottom (v=-1): corner0 -> corner1
        std::vector<std::array<double, 3>> bottom;
        bottom.reserve(static_cast<size_t>(p + 1));
        bottom.push_back(P0);
        for (int k = 0; k < e; ++k) bottom.push_back(get_pt(edge_node(0, k)));
        bottom.push_back(P1);

        // right (u=+1): corner1 -> corner2
        std::vector<std::array<double, 3>> right;
        right.reserve(static_cast<size_t>(p + 1));
        right.push_back(P1);
        for (int k = 0; k < e; ++k) right.push_back(get_pt(edge_node(1, k)));
        right.push_back(P2);

        // top (v=+1): corner3 -> corner2 (note reversed edge 2-3)
        std::vector<std::array<double, 3>> top;
        top.reserve(static_cast<size_t>(p + 1));
        top.push_back(P3);
        for (int k = 0; k < e; ++k) top.push_back(get_pt(edge_node(2, e - 1 - k)));
        top.push_back(P2);

        // left (u=-1): corner0 -> corner3 (note reversed edge 3-0)
        std::vector<std::array<double, 3>> left;
        left.reserve(static_cast<size_t>(p + 1));
        left.push_back(P0);
        for (int k = 0; k < e; ++k) left.push_back(get_pt(edge_node(3, e - 1 - k)));
        left.push_back(P3);

        // Synthesize interior nodes (VTK Lagrange quad face nodes) via Coons patch.
        const auto add3 = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
          return std::array<double, 3>{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
        };
        const auto sub3 = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
          return std::array<double, 3>{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        };
        const auto scale3 = [](const std::array<double, 3>& a, double s) {
          return std::array<double, 3>{s * a[0], s * a[1], s * a[2]};
        };

        for (int i = 1; i <= p - 1; ++i) {
          const double u = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(p);
          const double su = 0.5 * (u + 1.0);

          const auto Bu = eval_curve(bottom, u);
          const auto Tu = eval_curve(top, u);

          for (int j = 1; j <= p - 1; ++j) {
            const double v = -1.0 + 2.0 * static_cast<double>(j) / static_cast<double>(p);
            const double sv = 0.5 * (v + 1.0);

            const auto Lv = eval_curve(left, v);
            const auto Rv = eval_curve(right, v);

            // Coons patch blending.
            const auto term_bottom = scale3(Bu, (1.0 - sv));
            const auto term_top = scale3(Tu, sv);
            const auto term_left = scale3(Lv, (1.0 - su));
            const auto term_right = scale3(Rv, su);

            const auto bilinear =
                add3(add3(scale3(P0, (1.0 - su) * (1.0 - sv)), scale3(P1, su * (1.0 - sv))),
                     add3(scale3(P2, su * sv), scale3(P3, (1.0 - su) * sv)));

            auto coons = add3(add3(term_bottom, term_top), add3(term_left, term_right));
            coons = sub3(coons, bilinear);

            const vtkIdType id = points->InsertNextPoint(coons[0], coons[1], coons[2]);
            cell_vertices->InsertNextId(id);
          }
        }

        // Sanity: ensure we produced the expected tensor-product size.
        if (cell_vertices->GetNumberOfIds() != static_cast<vtkIdType>(expected_lagrange)) {
          throw std::runtime_error("VTKWriter: failed to promote serendipity quad to expected Lagrange node count");
        }

        ugrid->InsertNextCell(vtk_type, cell_vertices);
        continue;
      }
    }

    if (n_vertices > static_cast<size_t>(shape.num_corners) && shape.num_corners > 0) {
      // Only attempt geometric reordering for straight-sided quadratic elements.
      // For curved/high-order geometry, preserve the mesh's canonical ordering.
      if (should_use_geometric_reorder_for_vtk_quadratic(mesh, static_cast<index_t>(c), vertices_ptr, n_vertices, shape)) {
        auto reordered = reorder_high_order_to_vtk(mesh, static_cast<index_t>(c), vertices_ptr, n_vertices, shape);
        for (auto vid : reordered) cell_vertices->InsertNextId(vid);
      } else {
        for (size_t i = 0; i < n_vertices; ++i) cell_vertices->InsertNextId(vertices_ptr[i]);
      }
    } else {
      for (size_t i = 0; i < n_vertices; ++i) cell_vertices->InsertNextId(vertices_ptr[i]);
    }

    ugrid->InsertNextCell(vtk_type, cell_vertices);
  }
}

int VTKWriter::cellshape_to_vtk(const CellShape& shape) {
  setup_vtk_registry();

  // Select map based on order
  const auto& map = (shape.order == 1) ? family_to_vtk_linear_ : family_to_vtk_quadratic_;

  auto it = map.find(shape.family);
  if (it != map.end()) {
    return it->second;
  }

  // Default fallback
  switch (shape.family) {
    case CellFamily::Line: return VTK_LINE;
    case CellFamily::Triangle: return VTK_TRIANGLE;
    case CellFamily::Quad: return VTK_QUAD;
    case CellFamily::Tetra: return VTK_TETRA;
    case CellFamily::Hex: return VTK_HEXAHEDRON;
    case CellFamily::Wedge: return VTK_WEDGE;
    case CellFamily::Pyramid: return VTK_PYRAMID;
    case CellFamily::Polygon: return VTK_POLYGON;
    case CellFamily::Polyhedron: return VTK_POLYHEDRON;
    default: return VTK_POLYGON;
  }
}

void VTKWriter::write_field_data(const MeshBase& mesh, vtkDataSet* dataset,
                                const WriteOptions& opts) {
  // Write region labels
  write_region_labels(mesh, dataset);

  // Write cell data
  if (opts.write_cell_data) {
    write_cell_data(mesh, dataset, opts);
  }

  // Write point data
  if (opts.write_point_data) {
    write_point_data(mesh, dataset, opts);
  }

  // Write parallel mesh data
  write_global_ids(mesh, dataset);
}

void VTKWriter::write_cell_data(const MeshBase& mesh, vtkDataSet* dataset,
                               const WriteOptions& opts) {
  vtkCellData* cell_data = dataset->GetCellData();
  if (!cell_data) return;

  const auto all_names = mesh.field_names(EntityKind::Volume);
  std::vector<std::string> names;
  if (!opts.cell_fields_to_write.empty()) {
    names = opts.cell_fields_to_write;
  } else {
    names = all_names;
  }

  for (const auto& name : names) {
    if (!mesh.has_field(EntityKind::Volume, name)) {
      continue;
    }
    const auto type = mesh.field_type_by_name(EntityKind::Volume, name);
    const auto components = mesh.field_components_by_name(EntityKind::Volume, name);
    const size_t n_cells = mesh.n_cells();
    if (n_cells == 0 || components == 0) continue;

    vtkSmartPointer<vtkDataArray> arr;
    switch (type) {
      case FieldScalarType::Int32:
        arr = vtkSmartPointer<vtkIntArray>::New();
        break;
      case FieldScalarType::Int64:
        arr = vtkSmartPointer<vtkLongLongArray>::New();
        break;
      case FieldScalarType::Float32:
        arr = vtkSmartPointer<vtkFloatArray>::New();
        break;
      case FieldScalarType::Float64:
        arr = vtkSmartPointer<vtkDoubleArray>::New();
        break;
      case FieldScalarType::UInt8:
        arr = vtkSmartPointer<vtkUnsignedCharArray>::New();
        break;
      default:
        continue;
    }

    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(static_cast<int>(components));
    arr->SetNumberOfTuples(static_cast<vtkIdType>(n_cells));

    const void* src = mesh.field_data_by_name(EntityKind::Volume, name);
    if (!src) {
      continue;
    }

    const size_t n_vals = n_cells * components;
    if (type == FieldScalarType::Int32) {
      auto* a = vtkIntArray::SafeDownCast(arr);
      int* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const int32_t*>(src);
      for (size_t i = 0; i < n_vals; ++i) out[i] = static_cast<int>(in[i]);
    } else if (type == FieldScalarType::Int64) {
      auto* a = vtkLongLongArray::SafeDownCast(arr);
      long long* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const int64_t*>(src);
      for (size_t i = 0; i < n_vals; ++i) out[i] = static_cast<long long>(in[i]);
    } else if (type == FieldScalarType::Float32) {
      auto* a = vtkFloatArray::SafeDownCast(arr);
      float* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const float*>(src);
      std::copy(in, in + n_vals, out);
    } else if (type == FieldScalarType::Float64) {
      auto* a = vtkDoubleArray::SafeDownCast(arr);
      double* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const double*>(src);
      std::copy(in, in + n_vals, out);
    } else if (type == FieldScalarType::UInt8) {
      auto* a = vtkUnsignedCharArray::SafeDownCast(arr);
      unsigned char* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const std::uint8_t*>(src);
      for (size_t i = 0; i < n_vals; ++i) out[i] = static_cast<unsigned char>(in[i]);
    }

    cell_data->AddArray(arr);
  }
}

void VTKWriter::write_point_data(const MeshBase& mesh, vtkDataSet* dataset,
                                const WriteOptions& opts) {
  vtkPointData* point_data = dataset->GetPointData();
  if (!point_data) return;

  const vtkIdType vtk_n_points = dataset->GetNumberOfPoints();
  const size_t n_points = (vtk_n_points > 0) ? static_cast<size_t>(vtk_n_points) : mesh.n_vertices();
  const size_t n_base_points = mesh.n_vertices();

  // NOTE: When writing higher-than-quadratic Serendipity quads (e.g. Q12 for p=3), we
  // promote them to VTK Lagrange quads by inserting missing face-interior points.
  // These inserted points do not exist in MeshBase, so we also synthesize point-attached
  // field values for them using the same Coons-patch interpolation used for geometry.
  // This provides a smooth extension of boundary nodal data into the element interior for
  // visualization (and avoids zero-filled point arrays for the inserted points).
  struct QuadCoonsRecipe {
    vtkIdType point_id = -1;              // dataset point id (>= n_base_points)
    int p = 0;                            // order
    double su = 0.0, sv = 0.0;            // param coords mapped to [0,1]
    std::array<vtkIdType,4> corners{{-1,-1,-1,-1}}; // VTK quad corners (0..3)
    std::vector<vtkIdType> bottom, top, left, right; // boundary curves (size p+1)
    std::vector<double> Nu, Nv;           // 1D Lagrange basis at u and v (size p+1)
  };

  std::vector<QuadCoonsRecipe> coons_points;
  if (dataset && n_points > n_base_points) {
    auto lagrange_basis_1d = [](int p, double xi, std::vector<double>& N) {
      const int n = p + 1;
      N.assign(static_cast<size_t>(n), 0.0);
      std::vector<double> nodes(static_cast<size_t>(n), 0.0);
      for (int i = 0; i < n; ++i) {
        nodes[static_cast<size_t>(i)] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(p);
      }
      for (int i = 0; i < n; ++i) {
        double Li = 1.0;
        for (int j = 0; j < n; ++j) {
          if (j == i) continue;
          const double denom = nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(j)];
          Li *= (xi - nodes[static_cast<size_t>(j)]) / denom;
        }
        N[static_cast<size_t>(i)] = Li;
      }
    };

    const vtkIdType n_cells = dataset->GetNumberOfCells();
    for (vtkIdType c = 0; c < n_cells; ++c) {
      vtkCell* cell = dataset->GetCell(c);
      if (!cell) continue;
      if (cell->GetCellType() != VTK_LAGRANGE_QUADRILATERAL) continue;
      const vtkIdType n_cell_pts = cell->GetNumberOfPoints();
      const int p = CellTopology::infer_lagrange_order(CellFamily::Quad, static_cast<size_t>(n_cell_pts));
      if (p <= 2) continue;

      const int e = p - 1;
      const vtkIdType offset = static_cast<vtkIdType>(4 + 4 * e); // corners + edge nodes (serendipity)
      if (offset >= n_cell_pts) continue;

      // Only treat this as a promoted serendipity quad if it references points beyond the
      // original MeshBase vertex count.
      bool has_inserted = false;
      for (vtkIdType li = offset; li < n_cell_pts; ++li) {
        if (static_cast<size_t>(cell->GetPointId(li)) >= n_base_points) {
          has_inserted = true;
          break;
        }
      }
      if (!has_inserted) continue;

      // Boundary curves (p+1 nodes each), oriented to match VTK quad parametric axes:
      // - u axis on v=-1: corner0 -> corner1
      // - u axis on v=+1: corner3 -> corner2
      // - v axis on u=-1: corner0 -> corner3
      // - v axis on u=+1: corner1 -> corner2
      QuadCoonsRecipe base;
      base.p = p;
      base.corners = {{
          cell->GetPointId(0),
          cell->GetPointId(1),
          cell->GetPointId(2),
          cell->GetPointId(3),
      }};

      base.bottom.resize(static_cast<size_t>(p + 1));
      base.right.resize(static_cast<size_t>(p + 1));
      base.top.resize(static_cast<size_t>(p + 1));
      base.left.resize(static_cast<size_t>(p + 1));

      base.bottom.front() = base.corners[0];
      base.bottom.back()  = base.corners[1];
      base.right.front()  = base.corners[1];
      base.right.back()   = base.corners[2];
      base.top.front()    = base.corners[3];
      base.top.back()     = base.corners[2];
      base.left.front()   = base.corners[0];
      base.left.back()    = base.corners[3];

      for (int k = 0; k < e; ++k) {
        base.bottom[static_cast<size_t>(k + 1)] = cell->GetPointId(static_cast<vtkIdType>(4 + 0 * e + k));
        base.right[static_cast<size_t>(k + 1)]  = cell->GetPointId(static_cast<vtkIdType>(4 + 1 * e + k));
        // Reverse edges (2-3) and (3-0) to match the parametric direction expected by the Coons patch.
        base.top[static_cast<size_t>(k + 1)]    = cell->GetPointId(static_cast<vtkIdType>(4 + 2 * e + (e - 1 - k)));
        base.left[static_cast<size_t>(k + 1)]   = cell->GetPointId(static_cast<vtkIdType>(4 + 3 * e + (e - 1 - k)));
      }

      for (vtkIdType li = offset; li < n_cell_pts; ++li) {
        const vtkIdType pid = cell->GetPointId(li);
        if (static_cast<size_t>(pid) < n_base_points) continue; // not an inserted interior point

        const vtkIdType kk = li - offset;
        const int ii = static_cast<int>(kk / e) + 1; // 1..p-1
        const int jj = static_cast<int>(kk % e) + 1; // 1..p-1
        const double u = -1.0 + 2.0 * static_cast<double>(ii) / static_cast<double>(p);
        const double v = -1.0 + 2.0 * static_cast<double>(jj) / static_cast<double>(p);

        QuadCoonsRecipe rec = base;
        rec.point_id = pid;
        rec.su = static_cast<double>(ii) / static_cast<double>(p);
        rec.sv = static_cast<double>(jj) / static_cast<double>(p);
        lagrange_basis_1d(p, u, rec.Nu);
        lagrange_basis_1d(p, v, rec.Nv);
        coons_points.push_back(std::move(rec));
      }
    }
  }

  auto eval_curve_component = [](const std::vector<vtkIdType>& ids,
                                 const std::vector<double>& N,
                                 size_t comp,
                                 const auto& value_at) -> double {
    const size_t n = std::min(ids.size(), N.size());
    double out = 0.0;
    for (size_t k = 0; k < n; ++k) {
      out += N[k] * value_at(ids[k], comp);
    }
    return out;
  };

  auto coons_eval_component = [&](const QuadCoonsRecipe& rec,
                                  size_t comp,
                                  const auto& value_at) -> double {
    const double Bu = eval_curve_component(rec.bottom, rec.Nu, comp, value_at);
    const double Tu = eval_curve_component(rec.top, rec.Nu, comp, value_at);
    const double Lv = eval_curve_component(rec.left, rec.Nv, comp, value_at);
    const double Rv = eval_curve_component(rec.right, rec.Nv, comp, value_at);

    const double F0 = value_at(rec.corners[0], comp);
    const double F1 = value_at(rec.corners[1], comp);
    const double F2 = value_at(rec.corners[2], comp);
    const double F3 = value_at(rec.corners[3], comp);

    const double su = rec.su;
    const double sv = rec.sv;
    const double bilinear =
        (1.0 - su) * (1.0 - sv) * F0 +
        (su)       * (1.0 - sv) * F1 +
        (su)       * (sv)       * F2 +
        (1.0 - su) * (sv)       * F3;

    const double coons =
        (1.0 - sv) * Bu + (sv) * Tu +
        (1.0 - su) * Lv + (su) * Rv -
        bilinear;
    return coons;
  };

  const auto all_names = mesh.field_names(EntityKind::Vertex);
  std::vector<std::string> names;
  if (!opts.point_fields_to_write.empty()) {
    names = opts.point_fields_to_write;
  } else {
    names = all_names;
  }

  for (const auto& name : names) {
    if (!mesh.has_field(EntityKind::Vertex, name)) {
      continue;
    }
    const auto type = mesh.field_type_by_name(EntityKind::Vertex, name);
    const auto components = mesh.field_components_by_name(EntityKind::Vertex, name);
    const size_t n_vertices = mesh.n_vertices();
    if (n_points == 0 || components == 0) continue;

    vtkSmartPointer<vtkDataArray> arr;
    switch (type) {
      case FieldScalarType::Int32:
        arr = vtkSmartPointer<vtkIntArray>::New();
        break;
      case FieldScalarType::Int64:
        arr = vtkSmartPointer<vtkLongLongArray>::New();
        break;
      case FieldScalarType::Float32:
        arr = vtkSmartPointer<vtkFloatArray>::New();
        break;
      case FieldScalarType::Float64:
        arr = vtkSmartPointer<vtkDoubleArray>::New();
        break;
      case FieldScalarType::UInt8:
        arr = vtkSmartPointer<vtkUnsignedCharArray>::New();
        break;
      default:
        continue;
    }

    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(static_cast<int>(components));
    arr->SetNumberOfTuples(static_cast<vtkIdType>(n_points));

    const void* src = mesh.field_data_by_name(EntityKind::Vertex, name);
    if (!src) {
      continue;
    }

    const size_t n_vals_mesh = n_vertices * components;
    const size_t n_vals = n_points * components;

    if (type == FieldScalarType::Int32) {
      auto* a = vtkIntArray::SafeDownCast(arr);
      int* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const int32_t*>(src);
      std::fill(out, out + n_vals, 0);
      for (size_t i = 0; i < std::min(n_vals_mesh, n_vals); ++i) out[i] = static_cast<int>(in[i]);
      if (!coons_points.empty()) {
        auto value_at = [&](vtkIdType pid, size_t comp) -> double {
          const size_t v = static_cast<size_t>(pid);
          if (v >= n_base_points) return 0.0;
          return static_cast<double>(in[v * components + comp]);
        };
        for (const auto& rec : coons_points) {
          const size_t pid = static_cast<size_t>(rec.point_id);
          if (pid >= n_points) continue;
          for (size_t comp = 0; comp < components; ++comp) {
            const double val = coons_eval_component(rec, comp, value_at);
            out[pid * components + comp] = static_cast<int>(std::llround(val));
          }
        }
      }
    } else if (type == FieldScalarType::Int64) {
      auto* a = vtkLongLongArray::SafeDownCast(arr);
      long long* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const int64_t*>(src);
      std::fill(out, out + n_vals, 0);
      for (size_t i = 0; i < std::min(n_vals_mesh, n_vals); ++i) out[i] = static_cast<long long>(in[i]);
      if (!coons_points.empty()) {
        auto value_at = [&](vtkIdType pid, size_t comp) -> double {
          const size_t v = static_cast<size_t>(pid);
          if (v >= n_base_points) return 0.0;
          return static_cast<double>(in[v * components + comp]);
        };
        for (const auto& rec : coons_points) {
          const size_t pid = static_cast<size_t>(rec.point_id);
          if (pid >= n_points) continue;
          for (size_t comp = 0; comp < components; ++comp) {
            const double val = coons_eval_component(rec, comp, value_at);
            out[pid * components + comp] = static_cast<long long>(std::llround(val));
          }
        }
      }
    } else if (type == FieldScalarType::Float32) {
      auto* a = vtkFloatArray::SafeDownCast(arr);
      float* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const float*>(src);
      std::fill(out, out + n_vals, 0.0f);
      std::copy(in, in + std::min(n_vals_mesh, n_vals), out);
      if (!coons_points.empty()) {
        auto value_at = [&](vtkIdType pid, size_t comp) -> double {
          const size_t v = static_cast<size_t>(pid);
          if (v >= n_base_points) return 0.0;
          return static_cast<double>(in[v * components + comp]);
        };
        for (const auto& rec : coons_points) {
          const size_t pid = static_cast<size_t>(rec.point_id);
          if (pid >= n_points) continue;
          for (size_t comp = 0; comp < components; ++comp) {
            const double val = coons_eval_component(rec, comp, value_at);
            out[pid * components + comp] = static_cast<float>(val);
          }
        }
      }
    } else if (type == FieldScalarType::Float64) {
      auto* a = vtkDoubleArray::SafeDownCast(arr);
      double* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const double*>(src);
      std::fill(out, out + n_vals, 0.0);
      std::copy(in, in + std::min(n_vals_mesh, n_vals), out);
      if (!coons_points.empty()) {
        auto value_at = [&](vtkIdType pid, size_t comp) -> double {
          const size_t v = static_cast<size_t>(pid);
          if (v >= n_base_points) return 0.0;
          return in[v * components + comp];
        };
        for (const auto& rec : coons_points) {
          const size_t pid = static_cast<size_t>(rec.point_id);
          if (pid >= n_points) continue;
          for (size_t comp = 0; comp < components; ++comp) {
            out[pid * components + comp] = coons_eval_component(rec, comp, value_at);
          }
        }
      }
    } else if (type == FieldScalarType::UInt8) {
      auto* a = vtkUnsignedCharArray::SafeDownCast(arr);
      unsigned char* out = a->WritePointer(0, static_cast<vtkIdType>(n_vals));
      const auto* in = static_cast<const std::uint8_t*>(src);
      std::fill(out, out + n_vals, static_cast<unsigned char>(0));
      for (size_t i = 0; i < std::min(n_vals_mesh, n_vals); ++i) out[i] = static_cast<unsigned char>(in[i]);
      if (!coons_points.empty()) {
        auto value_at = [&](vtkIdType pid, size_t comp) -> double {
          const size_t v = static_cast<size_t>(pid);
          if (v >= n_base_points) return 0.0;
          return static_cast<double>(in[v * components + comp]);
        };
        for (const auto& rec : coons_points) {
          const size_t pid = static_cast<size_t>(rec.point_id);
          if (pid >= n_points) continue;
          for (size_t comp = 0; comp < components; ++comp) {
            const auto val = static_cast<long long>(std::llround(coons_eval_component(rec, comp, value_at)));
            out[pid * components + comp] = static_cast<unsigned char>(std::clamp<long long>(val, 0, 255));
          }
        }
      }
    }

    point_data->AddArray(arr);
  }

  // Write current coordinates if available
  if (mesh.has_current_coords()) {
    vtkSmartPointer<vtkDoubleArray> displacement = vtkSmartPointer<vtkDoubleArray>::New();
    displacement->SetName("Displacement");
    displacement->SetNumberOfComponents(3);
    displacement->SetNumberOfTuples(static_cast<vtkIdType>(n_points));

    const std::vector<real_t>& X_ref = mesh.X_ref();
    const std::vector<real_t>& X_cur = mesh.X_cur();
    int spatial_dim = mesh.dim();

    const size_t n_base = mesh.n_vertices();
    for (size_t i = 0; i < n_points; ++i) {
      double disp[3] = {0, 0, 0};
      if (i < n_base) {
        for (int d = 0; d < spatial_dim; ++d) {
          disp[d] = X_cur[i * spatial_dim + d] - X_ref[i * spatial_dim + d];
        }
      }
      displacement->SetTuple(i, disp);
    }

    // Synthesize displacement values for inserted Coons interior points (see NOTE above).
    if (!coons_points.empty()) {
      auto value_at_disp = [&](vtkIdType pid, size_t comp) -> double {
        const size_t v = static_cast<size_t>(pid);
        if (v >= n_base) return 0.0;
        if (comp >= static_cast<size_t>(spatial_dim)) return 0.0;
        return static_cast<double>(X_cur[v * spatial_dim + comp] - X_ref[v * spatial_dim + comp]);
      };

      for (const auto& rec : coons_points) {
        const size_t pid = static_cast<size_t>(rec.point_id);
        if (pid >= n_points) continue;
        double disp[3] = {0.0, 0.0, 0.0};
        for (size_t comp = 0; comp < 3; ++comp) {
          disp[comp] = coons_eval_component(rec, comp, value_at_disp);
        }
        displacement->SetTuple(static_cast<vtkIdType>(pid), disp);
      }
    }

    point_data->AddArray(displacement);
  }
}

void VTKWriter::write_region_labels(const MeshBase& mesh, vtkDataSet* dataset) {
  vtkCellData* cell_data = dataset->GetCellData();
  if (!cell_data) return;

  vtkSmartPointer<vtkIntArray> region_array = vtkSmartPointer<vtkIntArray>::New();
  region_array->SetName("RegionID");
  region_array->SetNumberOfComponents(1);
  region_array->SetNumberOfTuples(mesh.n_cells());

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    label_t label = mesh.region_label(static_cast<index_t>(c));
    region_array->SetValue(c, label);
  }

  cell_data->AddArray(region_array);
}

void VTKWriter::write_global_ids(const MeshBase& mesh, vtkDataSet* dataset) {
  // Write cell global IDs
  const auto& cell_gids = mesh.cell_gids();
  if (!cell_gids.empty()) {
    vtkSmartPointer<vtkIdTypeArray> gid_array = vtkSmartPointer<vtkIdTypeArray>::New();
    gid_array->SetName("GlobalCellID");
    gid_array->SetNumberOfComponents(1);
    gid_array->SetNumberOfTuples(mesh.n_cells());

    for (size_t c = 0; c < mesh.n_cells(); ++c) {
      gid_t gid = (c < cell_gids.size()) ? cell_gids[c] : static_cast<gid_t>(c);
      gid_array->SetValue(static_cast<vtkIdType>(c), static_cast<vtkIdType>(gid));
    }

    dataset->GetCellData()->SetGlobalIds(gid_array);
  }

  // Write vertex global IDs
  const auto& vertex_gids = mesh.vertex_gids();
  if (!vertex_gids.empty()) {
    const vtkIdType vtk_n_points = dataset->GetNumberOfPoints();
    const size_t n_points = (vtk_n_points > 0) ? static_cast<size_t>(vtk_n_points) : mesh.n_vertices();
    vtkSmartPointer<vtkIdTypeArray> gid_array = vtkSmartPointer<vtkIdTypeArray>::New();
    gid_array->SetName("GlobalVertexID");
    gid_array->SetNumberOfComponents(1);
    gid_array->SetNumberOfTuples(static_cast<vtkIdType>(n_points));

    gid_t max_gid = 0;
    for (auto g : vertex_gids) {
      if (g != INVALID_GID) max_gid = std::max(max_gid, g);
    }
    gid_t next_gid = max_gid + 1;

    const size_t n_base = mesh.n_vertices();
    for (size_t n = 0; n < n_points; ++n) {
      gid_t gid = INVALID_GID;
      if (n < vertex_gids.size()) {
        gid = vertex_gids[n];
      } else if (n < n_base) {
        gid = static_cast<gid_t>(n);
      } else {
        gid = next_gid++;
      }
      gid_array->SetValue(static_cast<vtkIdType>(n), static_cast<vtkIdType>(gid));
    }

    dataset->GetPointData()->SetGlobalIds(gid_array);
  }
}

void VTKWriter::setup_vtk_registry() {
  if (registry_initialized_) return;

  // Linear elements
  family_to_vtk_linear_[CellFamily::Line] = VTK_LINE;
  family_to_vtk_linear_[CellFamily::Triangle] = VTK_TRIANGLE;
  family_to_vtk_linear_[CellFamily::Quad] = VTK_QUAD;
  family_to_vtk_linear_[CellFamily::Tetra] = VTK_TETRA;
  family_to_vtk_linear_[CellFamily::Hex] = VTK_HEXAHEDRON;
  family_to_vtk_linear_[CellFamily::Wedge] = VTK_WEDGE;
  family_to_vtk_linear_[CellFamily::Pyramid] = VTK_PYRAMID;
  family_to_vtk_linear_[CellFamily::Polygon] = VTK_POLYGON;
  family_to_vtk_linear_[CellFamily::Polyhedron] = VTK_POLYHEDRON;

  // Quadratic elements
  family_to_vtk_quadratic_[CellFamily::Line] = VTK_QUADRATIC_EDGE;
  family_to_vtk_quadratic_[CellFamily::Triangle] = VTK_QUADRATIC_TRIANGLE;
  family_to_vtk_quadratic_[CellFamily::Quad] = VTK_QUADRATIC_QUAD;
  family_to_vtk_quadratic_[CellFamily::Tetra] = VTK_QUADRATIC_TETRA;
  family_to_vtk_quadratic_[CellFamily::Hex] = VTK_QUADRATIC_HEXAHEDRON;
  family_to_vtk_quadratic_[CellFamily::Wedge] = VTK_QUADRATIC_WEDGE;
  family_to_vtk_quadratic_[CellFamily::Pyramid] = VTK_QUADRATIC_PYRAMID;

  registry_initialized_ = true;
}

} // namespace svmp
