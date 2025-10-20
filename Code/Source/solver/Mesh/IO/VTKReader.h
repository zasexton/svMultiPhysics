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

#ifndef SVMP_VTK_READER_H
#define SVMP_VTK_READER_H

#include "../Mesh.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Forward declarations for VTK classes
class vtkUnstructuredGrid;
class vtkPolyData;
class vtkDataSet;
class vtkCellArray;
class vtkPoints;
class vtkDataArray;

namespace svmp {

/**
 * @brief VTK mesh reader supporting VTK, VTU, VTP formats
 *
 * This class reads mesh data from VTK format files and converts them
 * to the svmp::MeshBase format. It supports:
 * - Legacy VTK files (.vtk)
 * - VTK XML Unstructured Grid files (.vtu)
 * - VTK XML PolyData files (.vtp)
 */
class VTKReader {
public:
  VTKReader();
  ~VTKReader();

  /**
   * @brief Read a mesh from a VTK format file
   * @param options IO options including file path and format hints
   * @return Loaded mesh
   * @throws std::runtime_error if file cannot be read or format is unsupported
   */
  static MeshBase read(const MeshIOOptions& options);

  /**
   * @brief Register VTK reader with MeshBase IO registry
   *
   * Registers readers for "vtk", "vtu", and "vtp" formats
   */
  static void register_with_mesh();

  /**
   * @brief Read mesh from legacy VTK file
   */
  static MeshBase read_vtk(const std::string& filename);

  /**
   * @brief Read mesh from VTU (XML UnstructuredGrid) file
   */
  static MeshBase read_vtu(const std::string& filename);

  /**
   * @brief Read mesh from VTP (XML PolyData) file
   */
  static MeshBase read_vtp(const std::string& filename);

private:
  /**
   * @brief Convert VTK dataset to MeshBase
   */
  static MeshBase convert_from_vtk_dataset(vtkDataSet* dataset);

  /**
   * @brief Extract topology from VTK dataset
   */
  static void extract_topology(
      vtkDataSet* dataset,
      std::vector<CellShape>& cell_shapes,
      std::vector<offset_t>& cell2vertex_offsets,
      std::vector<index_t>& cell2vertex,
      // Optional outputs for codim-1 entities (faces in 3D, edges in 2D)
      std::vector<CellShape>& face_shapes,
      std::vector<offset_t>& face2vertex_offsets,
      std::vector<index_t>& face2vertex,
      std::vector<std::array<index_t,2>>& face2cell);

  /**
   * @brief Extract coordinates from VTK points
   */
  static std::vector<real_t> extract_coordinates(vtkPoints* points, int& spatial_dim);

  /**
   * @brief Convert VTK cell type to CellShape
   */
  static CellShape vtk_to_cellshape(int vtk_cell_type);

  /**
   * @brief Read field data from VTK dataset
   */
  static void read_field_data(vtkDataSet* dataset, MeshBase& mesh);

  /**
   * @brief Read cell data arrays
   */
  static void read_cell_data(vtkDataSet* dataset, MeshBase& mesh);

  /**
   * @brief Read point data arrays
   */
  static void read_point_data(vtkDataSet* dataset, MeshBase& mesh);

  /**
   * @brief Setup VTK cell type registry
   */
  static void setup_vtk_registry();

  /**
   * @brief Map from VTK cell type ID to CellFamily
   */
  static std::unordered_map<int, CellFamily> vtk_to_family_map_;

  /**
   * @brief Map from VTK cell type ID to element order
   */
  static std::unordered_map<int, int> vtk_to_order_map_;
  static std::unordered_set<int> vtk_lagrange_types_;
  static std::unordered_set<int> vtk_serendipity_types_;

  /**
   * @brief Flag indicating if VTK registry has been initialized
   */
  static bool registry_initialized_;
};

} // namespace svmp

#endif // SVMP_VTK_READER_H
