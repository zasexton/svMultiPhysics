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

#ifndef SVMP_VTK_WRITER_H
#define SVMP_VTK_WRITER_H

#include "../Mesh.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <vtkSmartPointer.h>

// Forward declarations for VTK classes
class vtkUnstructuredGrid;
class vtkPolyData;
class vtkDataSet;
class vtkCellArray;
class vtkPoints;
class vtkDataArray;
class vtkXMLWriter;
class vtkWriter;

namespace svmp {

/**
 * @brief VTK mesh writer supporting VTK, VTU, VTP formats
 *
 * This class writes mesh data from svmp::MeshBase format to VTK files.
 * It supports:
 * - Legacy VTK files (.vtk)
 * - VTK XML Unstructured Grid files (.vtu)
 * - VTK XML PolyData files (.vtp)
 * - Parallel formats (.pvtu, .pvtp) when used with DistributedMesh
 */
class VTKWriter {
public:
  /**
   * @brief Configuration options for VTK writer
   */
  struct WriteOptions {
    WriteOptions() = default;  ///< Default constructor
    bool binary = false;          ///< Write binary format (vs ASCII)
    bool compressed = false;      ///< Compress data (XML formats only)
    bool write_cell_data = true;  ///< Write cell-attached fields
    bool write_point_data = true; ///< Write point-attached fields
    bool write_ghost_cells = false; ///< Include ghost cells in output
    std::vector<std::string> cell_fields_to_write;  ///< Specific fields (empty = all)
    std::vector<std::string> point_fields_to_write; ///< Specific fields (empty = all)
  };

  VTKWriter();
  ~VTKWriter();

  /**
   * @brief Write a mesh to a VTK format file
   * @param mesh The mesh to write
   * @param options IO options including file path and format hints
   * @throws std::runtime_error if file cannot be written
   */
  static void write(const MeshBase& mesh, const MeshIOOptions& options);

  /**
   * @brief Register VTK writer with MeshBase IO registry
   *
   * Registers writers for "vtk", "vtu", and "vtp" formats
   */
  static void register_with_mesh();

  /**
   * @brief Write mesh to legacy VTK file
   */
  static void write_vtk(const MeshBase& mesh, const std::string& filename,
                       const WriteOptions& opts);

  /**
   * @brief Write mesh to VTU (XML UnstructuredGrid) file
   */
  static void write_vtu(const MeshBase& mesh, const std::string& filename,
                       const WriteOptions& opts);

  /**
   * @brief Write mesh to VTP (XML PolyData) file
   */
  static void write_vtp(const MeshBase& mesh, const std::string& filename,
                       const WriteOptions& opts);

  /**
   * @brief Write parallel VTU file set (for DistributedMesh)
   */
  static void write_pvtu(const MeshBase& mesh, const std::string& filename,
                        int rank, int size, const WriteOptions& opts);

private:
  /**
   * @brief Convert MeshBase to VTK UnstructuredGrid
   */
  static vtkSmartPointer<vtkUnstructuredGrid> convert_to_unstructured_grid(const MeshBase& mesh);

  /**
   * @brief Convert MeshBase to VTK PolyData
   */
  static vtkSmartPointer<vtkPolyData> convert_to_polydata(const MeshBase& mesh);

  /**
   * @brief Create VTK points from mesh coordinates
   */
  static vtkSmartPointer<vtkPoints> create_vtk_points(const MeshBase& mesh, Configuration cfg);

  /**
   * @brief Create VTK cells from mesh topology
   */
  static void create_vtk_cells(const MeshBase& mesh, vtkDataSet* dataset);

  /**
   * @brief Convert CellShape to VTK cell type ID
   */
  static int cellshape_to_vtk(const CellShape& shape);

  /**
   * @brief Write field data to VTK dataset
   */
  static void write_field_data(const MeshBase& mesh, vtkDataSet* dataset,
                              const WriteOptions& opts);

  /**
   * @brief Write cell data arrays
   */
  static void write_cell_data(const MeshBase& mesh, vtkDataSet* dataset,
                             const WriteOptions& opts);

  /**
   * @brief Write point data arrays
   */
  static void write_point_data(const MeshBase& mesh, vtkDataSet* dataset,
                              const WriteOptions& opts);

  /**
   * @brief Write region labels as cell data
   */
  static void write_region_labels(const MeshBase& mesh, vtkDataSet* dataset);

  /**
   * @brief Write boundary labels as cell data
   */
  static void write_boundary_labels(const MeshBase& mesh, vtkDataSet* dataset);

  /**
   * @brief Write ownership information for parallel meshes
   */
  static void write_ownership_data(const MeshBase& mesh, vtkDataSet* dataset);

  /**
   * @brief Write global IDs for parallel meshes
   */
  static void write_global_ids(const MeshBase& mesh, vtkDataSet* dataset);

  /**
   * @brief Setup VTK cell type registry
   */
  static void setup_vtk_registry();

  /**
   * @brief Map from CellFamily to VTK cell type for linear elements
   */
  static std::unordered_map<CellFamily, int> family_to_vtk_linear_;

  /**
   * @brief Map from CellFamily to VTK cell type for quadratic elements
   */
  static std::unordered_map<CellFamily, int> family_to_vtk_quadratic_;

  /**
   * @brief Flag indicating if VTK registry has been initialized
   */
  static bool registry_initialized_;
};

} // namespace svmp

#endif // SVMP_VTK_WRITER_H