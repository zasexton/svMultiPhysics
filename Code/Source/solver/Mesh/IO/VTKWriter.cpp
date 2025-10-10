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
    auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(static_cast<index_t>(c));
    const auto& shape = mesh.cell_shape(static_cast<index_t>(c));

    vtkSmartPointer<vtkIdList> cell_nodes = vtkSmartPointer<vtkIdList>::New();
    for (size_t i = 0; i < n_nodes; ++i) {
      cell_nodes->InsertNextId(nodes_ptr[i]);
    }

    // Add to appropriate cell array based on family
    switch (shape.family) {
      case CellFamily::Line:
        lines->InsertNextCell(cell_nodes);
        break;
      case CellFamily::Triangle:
      case CellFamily::Quad:
      case CellFamily::Polygon:
        polys->InsertNextCell(cell_nodes);
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
  size_t n_points = mesh.n_nodes();

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

void VTKWriter::create_vtk_cells(const MeshBase& mesh, vtkDataSet* dataset) {
  vtkUnstructuredGrid* ugrid = vtkUnstructuredGrid::SafeDownCast(dataset);
  if (!ugrid) return;

  ugrid->Allocate(mesh.n_cells());

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(static_cast<index_t>(c));
    const auto& shape = mesh.cell_shape(static_cast<index_t>(c));

    // Get VTK cell type
    int vtk_type = cellshape_to_vtk(shape);

    // Create ID list for cell nodes
    vtkSmartPointer<vtkIdList> cell_nodes = vtkSmartPointer<vtkIdList>::New();
    for (size_t i = 0; i < n_nodes; ++i) {
      cell_nodes->InsertNextId(nodes_ptr[i]);
    }

    ugrid->InsertNextCell(vtk_type, cell_nodes);
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
    if (kind != EntityKind::Cell) continue;

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
    displacement->SetNumberOfTuples(mesh.n_nodes());

    const std::vector<real_t>& X_ref = mesh.X_ref();
    const std::vector<real_t>& X_cur = mesh.X_cur();
    int spatial_dim = mesh.dim();

    for (size_t i = 0; i < mesh.n_nodes(); ++i) {
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

  // Write node global IDs
  const auto& node_gids = mesh.node_gids();
  if (!node_gids.empty()) {
    vtkSmartPointer<vtkLongArray> gid_array = vtkSmartPointer<vtkLongArray>::New();
    gid_array->SetName("GlobalNodeID");
    gid_array->SetNumberOfComponents(1);
    gid_array->SetNumberOfTuples(mesh.n_nodes());

    for (size_t n = 0; n < mesh.n_nodes(); ++n) {
      gid_t gid = (n < node_gids.size()) ? node_gids[n] : static_cast<gid_t>(n);
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