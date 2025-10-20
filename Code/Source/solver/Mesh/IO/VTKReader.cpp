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

#include "VTKReader.h"

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkDataSet.h>
#include <vtkPointSet.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkCellTypes.h>
#include "../Topology/CellTopology.h"

#include <iostream>
#include <stdexcept>
#include <fstream>

namespace svmp {

// Static member definitions
std::unordered_map<int, CellFamily> VTKReader::vtk_to_family_map_;
std::unordered_map<int, int> VTKReader::vtk_to_order_map_;
std::unordered_set<int> VTKReader::vtk_lagrange_types_;
std::unordered_set<int> VTKReader::vtk_serendipity_types_;
bool VTKReader::registry_initialized_ = false;

VTKReader::VTKReader() {
  setup_vtk_registry();
}

VTKReader::~VTKReader() = default;

void VTKReader::register_with_mesh() {
  // Register for different VTK formats
  MeshBase::register_reader("vtk", VTKReader::read);
  MeshBase::register_reader("vtu", VTKReader::read);
  MeshBase::register_reader("vtp", VTKReader::read);

  // Setup VTK cell type mappings
  setup_vtk_registry();
}

MeshBase VTKReader::read(const MeshIOOptions& options) {
  const std::string& filename = options.path;
  const std::string& format = options.format;

  // Check file exists
  std::ifstream file(filename);
  if (!file.good()) {
    throw std::runtime_error("VTKReader: Cannot open file: " + filename);
  }
  file.close();

  // Dispatch based on format
  if (format == "vtk") {
    return read_vtk(filename);
  } else if (format == "vtu") {
    return read_vtu(filename);
  } else if (format == "vtp") {
    return read_vtp(filename);
  } else {
    // Try to determine format from file extension
    if (filename.find(".vtu") != std::string::npos) {
      return read_vtu(filename);
    } else if (filename.find(".vtp") != std::string::npos) {
      return read_vtp(filename);
    } else {
      return read_vtk(filename); // Default to legacy VTK
    }
  }
}

MeshBase VTKReader::read_vtk(const std::string& filename) {
  // Read legacy VTK file
  vtkSmartPointer<vtkUnstructuredGridReader> reader =
      vtkSmartPointer<vtkUnstructuredGridReader>::New();
  reader->SetFileName(filename.c_str());
  reader->Update();

  if (!reader->GetOutput()) {
    // Try as PolyData
    vtkSmartPointer<vtkPolyDataReader> polyReader =
        vtkSmartPointer<vtkPolyDataReader>::New();
    polyReader->SetFileName(filename.c_str());
    polyReader->Update();

    if (!polyReader->GetOutput()) {
      throw std::runtime_error("VTKReader: Failed to read VTK file: " + filename);
    }

    return convert_from_vtk_dataset(polyReader->GetOutput());
  }

  return convert_from_vtk_dataset(reader->GetOutput());
}

MeshBase VTKReader::read_vtu(const std::string& filename) {
  // Read VTU (XML UnstructuredGrid) file
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
      vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(filename.c_str());
  reader->Update();

  if (!reader->GetOutput()) {
    throw std::runtime_error("VTKReader: Failed to read VTU file: " + filename);
  }

  return convert_from_vtk_dataset(reader->GetOutput());
}

MeshBase VTKReader::read_vtp(const std::string& filename) {
  // Read VTP (XML PolyData) file
  vtkSmartPointer<vtkXMLPolyDataReader> reader =
      vtkSmartPointer<vtkXMLPolyDataReader>::New();
  reader->SetFileName(filename.c_str());
  reader->Update();

  if (!reader->GetOutput()) {
    throw std::runtime_error("VTKReader: Failed to read VTP file: " + filename);
  }

  return convert_from_vtk_dataset(reader->GetOutput());
}

MeshBase VTKReader::convert_from_vtk_dataset(vtkDataSet* dataset) {
  if (!dataset) {
    throw std::runtime_error("VTKReader: Null dataset");
  }

  setup_vtk_registry();

  // Cast to vtkPointSet to access GetPoints()
  // All mesh datasets (UnstructuredGrid, PolyData) inherit from vtkPointSet
  vtkPointSet* pointset = vtkPointSet::SafeDownCast(dataset);
  if (!pointset) {
    throw std::runtime_error("VTKReader: Dataset is not a vtkPointSet (cannot extract points)");
  }

  // Extract spatial dimension and coordinates
  int spatial_dim = 0;
  std::vector<real_t> coordinates = extract_coordinates(pointset->GetPoints(), spatial_dim);

  // Extract topology
  std::vector<CellShape> cell_shapes;
  std::vector<offset_t> cell2vertex_offsets;
  std::vector<index_t> cell2vertex;
  std::vector<CellShape> face_shapes;
  std::vector<offset_t> face2vertex_offsets;
  std::vector<index_t> face2vertex;
  std::vector<std::array<index_t,2>> face2cell;
  extract_topology(dataset, cell_shapes, cell2vertex_offsets, cell2vertex,
                   face_shapes, face2vertex_offsets, face2vertex, face2cell);

  // Create mesh
  MeshBase mesh(spatial_dim);
  mesh.build_from_arrays(spatial_dim, coordinates, cell2vertex_offsets, cell2vertex, cell_shapes);

  // Install codim-1 entities if provided (e.g., from VTK polyhedron/polygon)
  if (!face_shapes.empty()) {
    mesh.set_faces_from_arrays(face_shapes, face2vertex_offsets, face2vertex, face2cell);
  }

  // Read field data
  read_field_data(dataset, mesh);

  // Finalize mesh
  mesh.finalize();

  return mesh;
}

void VTKReader::extract_topology(
    vtkDataSet* dataset,
    std::vector<CellShape>& cell_shapes,
    std::vector<offset_t>& cell2vertex_offsets,
    std::vector<index_t>& cell2vertex,
    std::vector<CellShape>& face_shapes,
    std::vector<offset_t>& face2vertex_offsets,
    std::vector<index_t>& face2vertex,
    std::vector<std::array<index_t,2>>& face2cell)
{
  vtkIdType n_cells = dataset->GetNumberOfCells();

  cell_shapes.reserve(n_cells);
  cell2vertex_offsets.reserve(n_cells + 1);
  cell2vertex_offsets.push_back(0);

  // Accumulate codim-1 entities using canonical keys
  struct FaceAcc {
    std::vector<index_t> vertices; // oriented vertices as reported by VTK
    std::array<index_t,2> cells = {{-1,-1}};
    CellShape shape;
  };
  // Hash for vector<index_t> keys (sorted vertex IDs). Marked noexcept to satisfy
  // libstdc++ hashtable requirements for bucket index computation.
  struct VecIndexHash {
    size_t operator()(const std::vector<index_t>& v) const noexcept {
      // 64-bit FNV-1a inspired + hash_combine
      size_t h = 1469598103934665603ull;
      for (index_t x : v) {
        size_t k = std::hash<index_t>{}(x);
        h ^= k + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      }
      return h;
    }
  };
  std::unordered_map<std::vector<index_t>, FaceAcc, VecIndexHash> face_map; // key: sorted vertex ids

  // Process each cell
  for (vtkIdType c = 0; c < n_cells; ++c) {
    vtkCell* cell = dataset->GetCell(c);
    if (!cell) continue;

    int vtk_type = cell->GetCellType();
    CellShape shape = vtk_to_cellshape(vtk_type);
    // For Lagrange/Serendipity, infer order from node count
    vtkIdType n_pts = cell->GetNumberOfPoints();
    if (vtk_lagrange_types_.count(vtk_type)) {
      int p = CellTopology::infer_lagrange_order(shape.family, static_cast<size_t>(n_pts));
      if (p > 2) shape.order = p;
    } else if (vtk_serendipity_types_.count(vtk_type)) {
      int p = CellTopology::infer_serendipity_order(shape.family, static_cast<size_t>(n_pts));
      if (p > 2) shape.order = p;
    }
    cell_shapes.push_back(shape);

    // Get cell vertices
    for (vtkIdType i = 0; i < n_pts; ++i) {
      cell2vertex.push_back(static_cast<index_t>(cell->GetPointId(i)));
    }
    cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));

    // Build codim-1 entities (faces for 3D, edges for 2D)
    const int cell_dim = cell->GetCellDimension();
    if (cell_dim == 3) {
      int nf = cell->GetNumberOfFaces();
      for (int i = 0; i < nf; ++i) {
        vtkCell* f = cell->GetFace(i);
        if (!f) continue;
        vtkIdType fn = f->GetNumberOfPoints();
        std::vector<index_t> fverts(fn);
        for (vtkIdType j = 0; j < fn; ++j) fverts[j] = static_cast<index_t>(f->GetPointId(j));
        // Canonical key: sorted vertices
        std::vector<index_t> key = fverts;
        std::sort(key.begin(), key.end());

        auto& acc = face_map[key];
        if (acc.vertices.empty()) acc.vertices = std::move(fverts);
        if (acc.cells[0] < 0) acc.cells[0] = static_cast<index_t>(c);
        else if (acc.cells[1] < 0 && acc.cells[0] != static_cast<index_t>(c)) acc.cells[1] = static_cast<index_t>(c);

        // Shape family from face cell type
        int ftype = f->GetCellType();
        CellShape fshape;
        if (ftype == VTK_TRIANGLE) fshape.family = CellFamily::Triangle;
        else if (ftype == VTK_QUAD) fshape.family = CellFamily::Quad;
        else fshape.family = CellFamily::Polygon;
        fshape.order = 1;
        fshape.num_corners = static_cast<int>(key.size());
        fshape.is_mixed_order = false;
        acc.shape = fshape;
      }
    } else if (cell_dim == 2) {
      int ne = cell->GetNumberOfEdges();
      for (int i = 0; i < ne; ++i) {
        vtkCell* e = cell->GetEdge(i);
        if (!e) continue;
        vtkIdType en = e->GetNumberOfPoints();
        if (en < 2) continue;
        std::vector<index_t> everts(en);
        for (vtkIdType j = 0; j < en; ++j) everts[j] = static_cast<index_t>(e->GetPointId(j));
        std::vector<index_t> key = everts;
        std::sort(key.begin(), key.end());

        auto& acc = face_map[key];
        if (acc.vertices.empty()) acc.vertices = std::move(everts);
        if (acc.cells[0] < 0) acc.cells[0] = static_cast<index_t>(c);
        else if (acc.cells[1] < 0 && acc.cells[0] != static_cast<index_t>(c)) acc.cells[1] = static_cast<index_t>(c);

        CellShape fshape;
        fshape.family = CellFamily::Line;
        fshape.order = 1;
        fshape.num_corners = 2;
        fshape.is_mixed_order = false;
        acc.shape = fshape;
      }
    }
  }

  // Materialize codim-1 arrays if any
  if (!face_map.empty()) {
    face2vertex_offsets.reserve(face_map.size() + 1);
    face2vertex_offsets.push_back(0);
    face_shapes.reserve(face_map.size());
    face2cell.reserve(face_map.size());

    for (auto& kv : face_map) {
      auto& acc = kv.second;
      // append vertices
      face2vertex.insert(face2vertex.end(), acc.vertices.begin(), acc.vertices.end());
      face2vertex_offsets.push_back(static_cast<offset_t>(face2vertex.size()));
      face_shapes.push_back(acc.shape);
      face2cell.push_back(acc.cells);
    }
  }
}

std::vector<real_t> VTKReader::extract_coordinates(vtkPoints* points, int& spatial_dim) {
  if (!points) {
    throw std::runtime_error("VTKReader: No points in dataset");
  }

  vtkIdType n_points = points->GetNumberOfPoints();
  if (n_points == 0) {
    throw std::runtime_error("VTKReader: Zero points in dataset");
  }

  // Determine spatial dimension from data
  double bounds[6];
  points->GetBounds(bounds);

  const double tol = 1e-10;
  bool x_varies = (bounds[1] - bounds[0]) > tol;
  bool y_varies = (bounds[3] - bounds[2]) > tol;
  bool z_varies = (bounds[5] - bounds[4]) > tol;

  if (z_varies) {
    spatial_dim = 3;
  } else if (y_varies) {
    spatial_dim = 2;
  } else if (x_varies) {
    spatial_dim = 1;
  } else {
    spatial_dim = 3; // Default to 3D for single point
  }

  // Extract coordinates
  std::vector<real_t> coordinates(n_points * spatial_dim);

  for (vtkIdType i = 0; i < n_points; ++i) {
    double pt[3];
    points->GetPoint(i, pt);

    for (int d = 0; d < spatial_dim; ++d) {
      coordinates[i * spatial_dim + d] = static_cast<real_t>(pt[d]);
    }
  }

  return coordinates;
}

CellShape VTKReader::vtk_to_cellshape(int vtk_cell_type) {
  CellShape shape;

  // Look up cell family
  auto family_it = vtk_to_family_map_.find(vtk_cell_type);
  if (family_it != vtk_to_family_map_.end()) {
    shape.family = family_it->second;
  } else {
    // Default to polygon for unknown types
    shape.family = CellFamily::Polygon;
  }

  // Look up order
  auto order_it = vtk_to_order_map_.find(vtk_cell_type);
  if (order_it != vtk_to_order_map_.end()) {
    shape.order = order_it->second;
  } else {
    shape.order = 1; // Default to linear
  }

  // Set number of corners based on VTK type
  switch (vtk_cell_type) {
    case VTK_LINE: shape.num_corners = 2; break;
    case VTK_TRIANGLE: shape.num_corners = 3; break;
    case VTK_QUAD: shape.num_corners = 4; break;
    case VTK_TETRA: shape.num_corners = 4; break;
    case VTK_HEXAHEDRON: shape.num_corners = 8; break;
    case VTK_WEDGE: shape.num_corners = 6; break;
    case VTK_PYRAMID: shape.num_corners = 5; break;
    case VTK_QUADRATIC_EDGE: shape.num_corners = 2; break;
    case VTK_QUADRATIC_TRIANGLE: shape.num_corners = 3; break;
    case VTK_QUADRATIC_QUAD: shape.num_corners = 4; break;
    case VTK_QUADRATIC_TETRA: shape.num_corners = 4; break;
    case VTK_QUADRATIC_HEXAHEDRON: shape.num_corners = 8; break;
    case VTK_QUADRATIC_WEDGE: shape.num_corners = 6; break;
    case VTK_QUADRATIC_PYRAMID: shape.num_corners = 5; break;
    default: shape.num_corners = -1; // Variable for polygons
  }

  shape.is_mixed_order = false;

  // Register this mapping with CellShapeRegistry for VTK format
  CellShapeRegistry::register_shape("vtk", vtk_cell_type, shape);

  return shape;
}

void VTKReader::read_field_data(vtkDataSet* dataset, MeshBase& mesh) {
  // Read cell data
  read_cell_data(dataset, mesh);

  // Read point data
  read_point_data(dataset, mesh);
}

void VTKReader::read_cell_data(vtkDataSet* dataset, MeshBase& mesh) {
  vtkCellData* cell_data = dataset->GetCellData();
  if (!cell_data) return;

  int n_arrays = cell_data->GetNumberOfArrays();

  for (int i = 0; i < n_arrays; ++i) {
    vtkDataArray* array = cell_data->GetArray(i);
    if (!array) continue;

    std::string name = array->GetName() ? array->GetName() : "field_" + std::to_string(i);
    int n_components = array->GetNumberOfComponents();
    vtkIdType n_tuples = array->GetNumberOfTuples();

    // Special handling for region labels
    if (name == "RegionLabel" || name == "MaterialID" || name == "Region") {
      if (n_components == 1 && array->GetDataType() == VTK_INT) {
        vtkIntArray* int_array = vtkIntArray::SafeDownCast(array);
        if (int_array) {
          for (vtkIdType c = 0; c < n_tuples; ++c) {
            mesh.set_region_label(static_cast<index_t>(c),
                                 static_cast<label_t>(int_array->GetValue(c)));
          }
        }
      }
      continue;
    }

    // Determine field type
    FieldScalarType field_type;
    switch (array->GetDataType()) {
      case VTK_INT: field_type = FieldScalarType::Int32; break;
      case VTK_FLOAT: field_type = FieldScalarType::Float32; break;
      case VTK_DOUBLE: field_type = FieldScalarType::Float64; break;
      default: field_type = FieldScalarType::Float64;
    }

    // Attach field to mesh
    auto handle = mesh.attach_field(EntityKind::Volume, name, field_type, n_components);

    // Copy data
    void* field_data = mesh.field_data(handle);
    if (field_type == FieldScalarType::Float64) {
      double* data_ptr = static_cast<double*>(field_data);
      for (vtkIdType j = 0; j < n_tuples * n_components; ++j) {
        data_ptr[j] = array->GetComponent(j / n_components, j % n_components);
      }
    } else if (field_type == FieldScalarType::Float32) {
      float* data_ptr = static_cast<float*>(field_data);
      for (vtkIdType j = 0; j < n_tuples * n_components; ++j) {
        data_ptr[j] = static_cast<float>(array->GetComponent(j / n_components, j % n_components));
      }
    } else if (field_type == FieldScalarType::Int32) {
      int* data_ptr = static_cast<int*>(field_data);
      vtkIntArray* int_array = vtkIntArray::SafeDownCast(array);
      if (int_array) {
        for (vtkIdType j = 0; j < n_tuples * n_components; ++j) {
          data_ptr[j] = int_array->GetValue(j);
        }
      }
    }
  }
}

void VTKReader::read_point_data(vtkDataSet* dataset, MeshBase& mesh) {
  vtkPointData* point_data = dataset->GetPointData();
  if (!point_data) return;

  int n_arrays = point_data->GetNumberOfArrays();

  for (int i = 0; i < n_arrays; ++i) {
    vtkDataArray* array = point_data->GetArray(i);
    if (!array) continue;

    std::string name = array->GetName() ? array->GetName() : "field_" + std::to_string(i);
    int n_components = array->GetNumberOfComponents();
    vtkIdType n_tuples = array->GetNumberOfTuples();

    // Determine field type
    FieldScalarType field_type;
    switch (array->GetDataType()) {
      case VTK_INT: field_type = FieldScalarType::Int32; break;
      case VTK_FLOAT: field_type = FieldScalarType::Float32; break;
      case VTK_DOUBLE: field_type = FieldScalarType::Float64; break;
      default: field_type = FieldScalarType::Float64;
    }

    // Attach field to mesh
    auto handle = mesh.attach_field(EntityKind::Vertex, name, field_type, n_components);

    // Copy data
    void* field_data = mesh.field_data(handle);
    if (field_type == FieldScalarType::Float64) {
      double* data_ptr = static_cast<double*>(field_data);
      for (vtkIdType j = 0; j < n_tuples * n_components; ++j) {
        data_ptr[j] = array->GetComponent(j / n_components, j % n_components);
      }
    } else if (field_type == FieldScalarType::Float32) {
      float* data_ptr = static_cast<float*>(field_data);
      for (vtkIdType j = 0; j < n_tuples * n_components; ++j) {
        data_ptr[j] = static_cast<float>(array->GetComponent(j / n_components, j % n_components));
      }
    } else if (field_type == FieldScalarType::Int32) {
      int* data_ptr = static_cast<int*>(field_data);
      vtkIntArray* int_array = vtkIntArray::SafeDownCast(array);
      if (int_array) {
        for (vtkIdType j = 0; j < n_tuples * n_components; ++j) {
          data_ptr[j] = int_array->GetValue(j);
        }
      }
    }
  }
}

void VTKReader::setup_vtk_registry() {
  if (registry_initialized_) return;

  // Linear elements
  vtk_to_family_map_[VTK_LINE] = CellFamily::Line;
  vtk_to_order_map_[VTK_LINE] = 1;

  vtk_to_family_map_[VTK_TRIANGLE] = CellFamily::Triangle;
  vtk_to_order_map_[VTK_TRIANGLE] = 1;

  vtk_to_family_map_[VTK_QUAD] = CellFamily::Quad;
  vtk_to_order_map_[VTK_QUAD] = 1;

  vtk_to_family_map_[VTK_TETRA] = CellFamily::Tetra;
  vtk_to_order_map_[VTK_TETRA] = 1;

  vtk_to_family_map_[VTK_HEXAHEDRON] = CellFamily::Hex;
  vtk_to_order_map_[VTK_HEXAHEDRON] = 1;

  vtk_to_family_map_[VTK_WEDGE] = CellFamily::Wedge;
  vtk_to_order_map_[VTK_WEDGE] = 1;

  vtk_to_family_map_[VTK_PYRAMID] = CellFamily::Pyramid;
  vtk_to_order_map_[VTK_PYRAMID] = 1;

  vtk_to_family_map_[VTK_POLYGON] = CellFamily::Polygon;
  vtk_to_order_map_[VTK_POLYGON] = 1;

  vtk_to_family_map_[VTK_POLYHEDRON] = CellFamily::Polyhedron;
  vtk_to_order_map_[VTK_POLYHEDRON] = 1;

  // Quadratic elements
  vtk_to_family_map_[VTK_QUADRATIC_EDGE] = CellFamily::Line;
  vtk_to_order_map_[VTK_QUADRATIC_EDGE] = 2;

  vtk_to_family_map_[VTK_QUADRATIC_TRIANGLE] = CellFamily::Triangle;
  vtk_to_order_map_[VTK_QUADRATIC_TRIANGLE] = 2;

  vtk_to_family_map_[VTK_QUADRATIC_QUAD] = CellFamily::Quad;
  vtk_to_order_map_[VTK_QUADRATIC_QUAD] = 2;

  vtk_to_family_map_[VTK_BIQUADRATIC_QUAD] = CellFamily::Quad;
  vtk_to_order_map_[VTK_BIQUADRATIC_QUAD] = 2;

  vtk_to_family_map_[VTK_QUADRATIC_TETRA] = CellFamily::Tetra;
  vtk_to_order_map_[VTK_QUADRATIC_TETRA] = 2;

  vtk_to_family_map_[VTK_QUADRATIC_HEXAHEDRON] = CellFamily::Hex;
  vtk_to_order_map_[VTK_QUADRATIC_HEXAHEDRON] = 2;

  vtk_to_family_map_[VTK_TRIQUADRATIC_HEXAHEDRON] = CellFamily::Hex;
  vtk_to_order_map_[VTK_TRIQUADRATIC_HEXAHEDRON] = 2;

  vtk_to_family_map_[VTK_QUADRATIC_WEDGE] = CellFamily::Wedge;
  vtk_to_order_map_[VTK_QUADRATIC_WEDGE] = 2;

  vtk_to_family_map_[VTK_QUADRATIC_PYRAMID] = CellFamily::Pyramid;
  vtk_to_order_map_[VTK_QUADRATIC_PYRAMID] = 2;

  // Lagrange families (order >= 2)
  vtk_lagrange_types_.insert(VTK_LAGRANGE_CURVE);
  vtk_lagrange_types_.insert(VTK_LAGRANGE_TRIANGLE);
  vtk_lagrange_types_.insert(VTK_LAGRANGE_QUADRILATERAL);
  vtk_lagrange_types_.insert(VTK_LAGRANGE_TETRAHEDRON);
  vtk_lagrange_types_.insert(VTK_LAGRANGE_HEXAHEDRON);
  vtk_lagrange_types_.insert(VTK_LAGRANGE_WEDGE);
  vtk_lagrange_types_.insert(VTK_LAGRANGE_PYRAMID);

  // Serendipity families (2D/3D subsets)
  // Serendipity families: not available in this VTK; leave empty (pattern scaffolding only)

  registry_initialized_ = true;
}

} // namespace svmp
