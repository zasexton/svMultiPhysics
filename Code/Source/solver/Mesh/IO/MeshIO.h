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

#ifndef SVMP_MESH_IO_H
#define SVMP_MESH_IO_H

#include "../Core/MeshTypes.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Mesh I/O registry and utilities
 *
 * This class manages:
 * - Format-agnostic I/O registry
 * - File format detection
 * - Reader/writer registration
 * - I/O configuration and options
 * - Format conversion utilities
 */
class MeshIO {
public:
  // Type aliases for reader/writer functions
  using ReaderFunc = std::function<std::unique_ptr<MeshBase>(const MeshIOOptions&)>;
  using WriterFunc = std::function<void(const MeshBase&, const MeshIOOptions&)>;

  // ---- Registry management ----

  /**
   * @brief Register a mesh reader for a format
   * @param format Format identifier (e.g., "vtk", "gmsh", "exodus")
   * @param reader Reader function
   */
  static void register_reader(const std::string& format, ReaderFunc reader);

  /**
   * @brief Register a mesh writer for a format
   * @param format Format identifier
   * @param writer Writer function
   */
  static void register_writer(const std::string& format, WriterFunc writer);

  /**
   * @brief Unregister a reader
   * @param format Format identifier
   */
  static void unregister_reader(const std::string& format);

  /**
   * @brief Unregister a writer
   * @param format Format identifier
   */
  static void unregister_writer(const std::string& format);

  /**
   * @brief Check if reader is registered
   * @param format Format identifier
   * @return True if reader exists
   */
  static bool has_reader(const std::string& format);

  /**
   * @brief Check if writer is registered
   * @param format Format identifier
   * @return True if writer exists
   */
  static bool has_writer(const std::string& format);

  /**
   * @brief List all registered readers
   * @return Vector of format names
   */
  static std::vector<std::string> registered_readers();

  /**
   * @brief List all registered writers
   * @return Vector of format names
   */
  static std::vector<std::string> registered_writers();

  // ---- Main I/O interface ----

  /**
   * @brief Load mesh from file
   * @param options I/O options including file path
   * @return Loaded mesh
   */
  static std::unique_ptr<MeshBase> load(const MeshIOOptions& options);

  /**
   * @brief Load mesh from file with auto-detected format
   * @param filename File path
   * @return Loaded mesh
   */
  static std::unique_ptr<MeshBase> load(const std::string& filename);

  /**
   * @brief Save mesh to file
   * @param mesh Mesh to save
   * @param options I/O options including file path
   */
  static void save(const MeshBase& mesh, const MeshIOOptions& options);

  /**
   * @brief Save mesh to file with auto-detected format
   * @param mesh Mesh to save
   * @param filename File path
   */
  static void save(const MeshBase& mesh, const std::string& filename);

  // ---- Format detection ----

  /**
   * @brief Detect format from file extension
   * @param filename File name or path
   * @return Format identifier (empty if unknown)
   */
  static std::string detect_format_from_extension(const std::string& filename);

  /**
   * @brief Detect format from file content
   * @param filename File path
   * @return Format identifier (empty if unknown)
   */
  static std::string detect_format_from_content(const std::string& filename);

  /**
   * @brief Get file extensions for format
   * @param format Format identifier
   * @return Vector of extensions (e.g., {".vtk", ".vtu"})
   */
  static std::vector<std::string> get_format_extensions(const std::string& format);

  /**
   * @brief Get format description
   * @param format Format identifier
   * @return Human-readable format description
   */
  static std::string get_format_description(const std::string& format);

  // ---- Format capabilities ----

  /**
   * @brief Format capability flags
   */
  struct FormatCapabilities {
    bool supports_3d = true;
    bool supports_2d = false;
    bool supports_1d = false;
    bool supports_mixed_cells = false;
    bool supports_high_order = false;
    bool supports_parallel = false;
    bool supports_fields = false;
    bool supports_labels = false;
    bool supports_compression = false;
    bool supports_binary = false;
    bool supports_ascii = true;
  };

  /**
   * @brief Register format capabilities
   * @param format Format identifier
   * @param caps Format capabilities
   */
  static void register_capabilities(const std::string& format,
                                   const FormatCapabilities& caps);

  /**
   * @brief Get format capabilities
   * @param format Format identifier
   * @return Format capabilities
   */
  static FormatCapabilities get_capabilities(const std::string& format);

  // ---- Batch I/O ----

  /**
   * @brief Load multiple meshes
   * @param filenames File paths
   * @return Vector of loaded meshes
   */
  static std::vector<std::unique_ptr<MeshBase>> load_batch(
      const std::vector<std::string>& filenames);

  /**
   * @brief Save multiple meshes
   * @param meshes Meshes to save
   * @param filenames File paths
   */
  static void save_batch(const std::vector<const MeshBase*>& meshes,
                        const std::vector<std::string>& filenames);

  // ---- Format conversion ----

  /**
   * @brief Convert mesh file format
   * @param input_file Input file path
   * @param output_file Output file path
   * @param output_format Target format (auto-detect if empty)
   */
  static void convert(const std::string& input_file,
                     const std::string& output_file,
                     const std::string& output_format = "");

  /**
   * @brief Batch convert mesh files
   * @param input_files Input file paths
   * @param output_dir Output directory
   * @param output_format Target format
   */
  static void convert_batch(const std::vector<std::string>& input_files,
                          const std::string& output_dir,
                          const std::string& output_format);

  // ---- I/O configuration ----

  /**
   * @brief Create default options for format
   * @param format Format identifier
   * @return Default I/O options
   */
  static MeshIOOptions default_options(const std::string& format);

  /**
   * @brief Validate I/O options
   * @param options I/O options
   * @param format Format identifier
   * @return True if options are valid
   */
  static bool validate_options(const MeshIOOptions& options,
                              const std::string& format);

  // ---- Parallel I/O ----

  /**
   * @brief Load distributed mesh
   * @param options I/O options with rank/size info
   * @return Loaded distributed mesh
   */
  static std::unique_ptr<MeshBase> load_parallel(const MeshIOOptions& options);

  /**
   * @brief Save distributed mesh
   * @param mesh Distributed mesh
   * @param options I/O options with rank/size info
   */
  static void save_parallel(const MeshBase& mesh, const MeshIOOptions& options);

  // ---- Error handling ----

  /**
   * @brief I/O error information
   */
  struct IOError {
    std::string message;
    std::string filename;
    std::string format;
    int line_number = -1;
  };

  /**
   * @brief Get last I/O error
   * @return Last error information
   */
  static IOError get_last_error();

  /**
   * @brief Clear error state
   */
  static void clear_error();

  // ---- Format registration helpers ----

  /**
   * @brief Register all built-in formats
   */
  static void register_builtin_formats();

  /**
   * @brief Register VTK formats
   */
  static void register_vtk_formats();

  /**
   * @brief Register Gmsh formats
   */
  static void register_gmsh_formats();

  /**
   * @brief Register Exodus formats
   */
  static void register_exodus_formats();

  /**
   * @brief Register CGNS formats
   */
  static void register_cgns_formats();

private:
  // Registry storage
  static std::unordered_map<std::string, ReaderFunc>& readers();
  static std::unordered_map<std::string, WriterFunc>& writers();
  static std::unordered_map<std::string, FormatCapabilities>& capabilities();
  static std::unordered_map<std::string, std::vector<std::string>>& extensions();

  // Error state
  static IOError& last_error();
};

} // namespace svmp

#endif // SVMP_MESH_IO_H