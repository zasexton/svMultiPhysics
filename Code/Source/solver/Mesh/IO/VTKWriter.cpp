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
#include <vtkIntArray.h>
#include <vtkLongArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkCellTypes.h>
#include <vtkDataCompressor.h>
#include <vtkZLibDataCompressor.h>

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
  switch (shape.family) {
    case CellFamily::Line:
      return (n_vertices >= 3) ? VTK_QUADRATIC_EDGE : VTK_LINE;
    case CellFamily::Triangle:
      return (n_vertices >= 6) ? VTK_QUADRATIC_TRIANGLE : VTK_TRIANGLE;
    case CellFamily::Quad:
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
      auto pattern = CellTopology::vtk_high_order_pattern(shape.family, p, kind);
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

void VTKWriter::create_vtk_cells(const MeshBase& mesh, vtkDataSet* dataset) {
  vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::SafeDownCast(dataset);
  if (!ugrid) return;

  ugrid->Allocate(mesh.n_cells());

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));
    const auto& shape = mesh.cell_shape(static_cast<index_t>(c));

    // Get VTK cell type (consider higher-order variants via vertex count)
    int vtk_type = choose_vtk_type_for(shape, n_vertices);

    // Create ID list for cell vertices (with optional high-order reordering)
    vtkSmartPointer<vtkIdList> cell_vertices = vtkSmartPointer<vtkIdList>::New();
    if (n_vertices > static_cast<size_t>(shape.num_corners) && shape.num_corners > 0) {
      auto reordered = reorder_high_order_to_vtk(mesh, static_cast<index_t>(c), vertices_ptr, n_vertices, shape);
      for (auto vid : reordered) cell_vertices->InsertNextId(vid);
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

  // Iterate through all cell fields
  for (int kind_idx = 0; kind_idx <= 3; ++kind_idx) {
    EntityKind kind = static_cast<EntityKind>(kind_idx);
    if (kind != EntityKind::Volume) continue;

    // This is a simplified approach - in production would need to enumerate fields
    // For now, write quality metrics as an example
    vtkSmartPointer<vtkDoubleArray> quality_array = vtkSmartPointer<vtkDoubleArray>::New();
    quality_array->SetName("Quality");
    quality_array->SetNumberOfComponents(1);
    quality_array->SetNumberOfTuples(mesh.n_cells());

    for (size_t c = 0; c < mesh.n_cells(); ++c) {
      real_t quality = mesh.compute_quality(static_cast<index_t>(c), "aspect_ratio");
      quality_array->SetValue(c, quality);
    }

    cell_data->AddArray(quality_array);

    // Cell measures
    vtkSmartPointer<vtkDoubleArray> measure_array = vtkSmartPointer<vtkDoubleArray>::New();
    measure_array->SetName("CellMeasure");
    measure_array->SetNumberOfComponents(1);
    measure_array->SetNumberOfTuples(mesh.n_cells());

    for (size_t c = 0; c < mesh.n_cells(); ++c) {
      real_t measure = mesh.cell_measure(static_cast<index_t>(c));
      measure_array->SetValue(c, measure);
    }

    cell_data->AddArray(measure_array);
  }
}

void VTKWriter::write_point_data(const MeshBase& mesh, vtkDataSet* dataset,
                                const WriteOptions& opts) {
  vtkPointData* point_data = dataset->GetPointData();
  if (!point_data) return;

  // Write current coordinates if available
  if (mesh.has_current_coords()) {
    vtkSmartPointer<vtkDoubleArray> displacement = vtkSmartPointer<vtkDoubleArray>::New();
    displacement->SetName("Displacement");
    displacement->SetNumberOfComponents(3);
    displacement->SetNumberOfTuples(mesh.n_vertices());

    const std::vector<real_t>& X_ref = mesh.X_ref();
    const std::vector<real_t>& X_cur = mesh.X_cur();
    int spatial_dim = mesh.dim();

    for (size_t i = 0; i < mesh.n_vertices(); ++i) {
      double disp[3] = {0, 0, 0};
      for (int d = 0; d < spatial_dim; ++d) {
        disp[d] = X_cur[i * spatial_dim + d] - X_ref[i * spatial_dim + d];
      }
      displacement->SetTuple(i, disp);
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
    vtkSmartPointer<vtkLongArray> gid_array = vtkSmartPointer<vtkLongArray>::New();
    gid_array->SetName("GlobalCellID");
    gid_array->SetNumberOfComponents(1);
    gid_array->SetNumberOfTuples(mesh.n_cells());

    for (size_t c = 0; c < mesh.n_cells(); ++c) {
      gid_t gid = (c < cell_gids.size()) ? cell_gids[c] : static_cast<gid_t>(c);
      gid_array->SetValue(c, gid);
    }

    dataset->GetCellData()->SetGlobalIds(gid_array);
  }

  // Write vertex global IDs
  const auto& vertex_gids = mesh.vertex_gids();
  if (!vertex_gids.empty()) {
    vtkSmartPointer<vtkLongArray> gid_array = vtkSmartPointer<vtkLongArray>::New();
    gid_array->SetName("GlobalVertexID");
    gid_array->SetNumberOfComponents(1);
    gid_array->SetNumberOfTuples(mesh.n_vertices());

    for (size_t n = 0; n < mesh.n_vertices(); ++n) {
      gid_t gid = (n < vertex_gids.size()) ? vertex_gids[n] : static_cast<gid_t>(n);
      gid_array->SetValue(n, gid);
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
