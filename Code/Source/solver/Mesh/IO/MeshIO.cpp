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

#include "MeshIO.h"
#include "../Core/MeshBase.h"

#ifdef MESH_HAS_VTK
#include "VTKReader.h"
#include "VTKWriter.h"
#endif

#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace svmp {

// ---- Static registry storage ----

std::unordered_map<std::string, MeshIO::ReaderFn>& MeshIO::readers() {
  static std::unordered_map<std::string, ReaderFn> registry;
  return registry;
}

std::unordered_map<std::string, MeshIO::WriterFn>& MeshIO::writers() {
  static std::unordered_map<std::string, WriterFn> registry;
  return registry;
}

// ---- Reader/Writer registration ----

void MeshIO::register_reader(const std::string& format, ReaderFn reader) {
  std::string fmt = normalize_format(format);
  readers()[fmt] = reader;
}

void MeshIO::register_writer(const std::string& format, WriterFn writer) {
  std::string fmt = normalize_format(format);
  writers()[fmt] = writer;
}

bool MeshIO::has_reader(const std::string& format) {
  std::string fmt = normalize_format(format);
  return readers().find(fmt) != readers().end();
}

bool MeshIO::has_writer(const std::string& format) {
  std::string fmt = normalize_format(format);
  return writers().find(fmt) != writers().end();
}

std::vector<std::string> MeshIO::available_readers() {
  std::vector<std::string> formats;
  for (const auto& [fmt, _] : readers()) {
    formats.push_back(fmt);
  }
  std::sort(formats.begin(), formats.end());
  return formats;
}

std::vector<std::string> MeshIO::available_writers() {
  std::vector<std::string> formats;
  for (const auto& [fmt, _] : writers()) {
    formats.push_back(fmt);
  }
  std::sort(formats.begin(), formats.end());
  return formats;
}

// ---- Main I/O functions ----

MeshBase MeshIO::read(const std::string& filename, const MeshIOOptions& options) {
  // Check file exists
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File does not exist: " + filename);
  }

  // Determine format
  std::string format = options.format;
  if (format.empty()) {
    format = detect_format(filename);
  }
  format = normalize_format(format);

  // Find reader
  auto it = readers().find(format);
  if (it == readers().end()) {
    throw std::runtime_error("No reader registered for format: " + format);
  }

  // Set filename in options
  MeshIOOptions opts = options;
  opts.filename = filename;
  opts.format = format;

  // Call reader
  return it->second(opts);
}

void MeshIO::write(const MeshBase& mesh, const std::string& filename,
                  const MeshIOOptions& options) {
  // Determine format
  std::string format = options.format;
  if (format.empty()) {
    format = detect_format(filename);
  }
  format = normalize_format(format);

  // Find writer
  auto it = writers().find(format);
  if (it == writers().end()) {
    throw std::runtime_error("No writer registered for format: " + format);
  }

  // Set filename in options
  MeshIOOptions opts = options;
  opts.filename = filename;
  opts.format = format;

  // Call writer
  it->second(mesh, opts);
}

// ---- Format detection ----

std::string MeshIO::detect_format(const std::string& filename) {
  // Get file extension
  std::filesystem::path path(filename);
  std::string ext = path.extension().string();

  if (ext.empty()) {
    // Try to detect from file content
    return detect_format_from_content(filename);
  }

  // Remove leading dot and convert to lowercase
  if (!ext.empty() && ext[0] == '.') {
    ext = ext.substr(1);
  }
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  // Map common extensions to formats
  if (ext == "vtk") return "vtk";
  if (ext == "vtu") return "vtu";
  if (ext == "vtp") return "vtp";
  if (ext == "pvtu") return "pvtu";
  if (ext == "pvtp") return "pvtp";
  if (ext == "stl") return "stl";
  if (ext == "ply") return "ply";
  if (ext == "obj") return "obj";
  if (ext == "off") return "off";
  if (ext == "mesh") return "medit";
  if (ext == "msh") return "gmsh";
  if (ext == "nas" || ext == "bdf") return "nastran";
  if (ext == "inp") return "abaqus";
  if (ext == "exo" || ext == "e" || ext == "ex2") return "exodus";
  if (ext == "cgns") return "cgns";
  if (ext == "h5" || ext == "hdf5") return "hdf5";

  // Default to extension if not recognized
  return ext;
}

std::string MeshIO::detect_format_from_content(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return "";  // Cannot detect
  }

  // Read first few lines
  std::string line;
  std::vector<std::string> lines;
  for (int i = 0; i < 10 && std::getline(file, line); ++i) {
    lines.push_back(line);
  }
  file.close();

  // Check for format signatures
  for (const auto& line : lines) {
    // VTK formats
    if (line.find("# vtk DataFile Version") != std::string::npos) {
      return "vtk";
    }
    if (line.find("<VTKFile") != std::string::npos) {
      if (line.find("UnstructuredGrid") != std::string::npos) return "vtu";
      if (line.find("PolyData") != std::string::npos) return "vtp";
      if (line.find("StructuredGrid") != std::string::npos) return "vts";
      if (line.find("RectilinearGrid") != std::string::npos) return "vtr";
    }

    // STL format
    if (line.find("solid") == 0) {
      return "stl";
    }

    // PLY format
    if (line == "ply") {
      return "ply";
    }

    // OFF format
    if (line == "OFF" || line == "COFF" || line == "NOFF") {
      return "off";
    }

    // Gmsh format
    if (line.find("$MeshFormat") != std::string::npos) {
      return "gmsh";
    }

    // OBJ format
    if (line.find("v ") == 0 || line.find("vn ") == 0 || line.find("f ") == 0) {
      // Weak detection for OBJ
      return "obj";
    }
  }

  return "";  // Unknown format
}

std::string MeshIO::normalize_format(const std::string& format) {
  std::string fmt = format;
  std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);

  // Remove dots if present
  if (!fmt.empty() && fmt[0] == '.') {
    fmt = fmt.substr(1);
  }

  // Map aliases to canonical names
  if (fmt == "unstructuredgrid") return "vtu";
  if (fmt == "polydata") return "vtp";
  if (fmt == "nastran" || fmt == "nas" || fmt == "bdf") return "nastran";
  if (fmt == "abaqus" || fmt == "inp") return "abaqus";
  if (fmt == "exodus" || fmt == "exo" || fmt == "e" || fmt == "ex2") return "exodus";
  if (fmt == "gmsh" || fmt == "msh") return "gmsh";
  if (fmt == "hdf" || fmt == "h5") return "hdf5";

  return fmt;
}

// ---- Metadata functions ----

MeshIOInfo MeshIO::query_file_info(const std::string& filename) {
  MeshIOInfo info;
  info.filename = filename;

  // Check file exists
  if (!std::filesystem::exists(filename)) {
    info.valid = false;
    return info;
  }

  // Get file size
  info.file_size = std::filesystem::file_size(filename);

  // Detect format
  info.format = detect_format(filename);

  // Try to get basic mesh info without fully loading
  // This would need format-specific implementations
  try {
    info.valid = true;
    // Format-specific quick readers would go here
    // For now, we'd need to actually read the file
  } catch (...) {
    info.valid = false;
  }

  return info;
}

bool MeshIO::can_read_format(const std::string& format) {
  return has_reader(format);
}

bool MeshIO::can_write_format(const std::string& format) {
  return has_writer(format);
}

// ---- Conversion utilities ----

void MeshIO::convert(const std::string& input_file,
                    const std::string& output_file,
                    const MeshIOOptions& read_options,
                    const MeshIOOptions& write_options) {
  // Read mesh
  MeshBase mesh = read(input_file, read_options);

  // Write mesh
  write(mesh, output_file, write_options);
}

// ---- Batch operations ----

std::vector<MeshBase> MeshIO::read_multiple(const std::vector<std::string>& filenames,
                                           const MeshIOOptions& options) {
  std::vector<MeshBase> meshes;
  meshes.reserve(filenames.size());

  for (const auto& filename : filenames) {
    meshes.push_back(read(filename, options));
  }

  return meshes;
}

void MeshIO::write_multiple(const std::vector<MeshBase>& meshes,
                           const std::vector<std::string>& filenames,
                           const MeshIOOptions& options) {
  if (meshes.size() != filenames.size()) {
    throw std::invalid_argument("Number of meshes and filenames must match");
  }

  for (size_t i = 0; i < meshes.size(); ++i) {
    write(meshes[i], filenames[i], options);
  }
}

// ---- Built-in format registration ----

void MeshIO::register_builtin_formats() {
  static bool registered = false;
  if (registered) return;
  registered = true;

#ifdef MESH_HAS_VTK
  // Register VTK formats
  register_reader("vtk", [](const MeshIOOptions& opts) {
    return VTKReader::read_vtk(opts.filename);
  });
  register_writer("vtk", [](const MeshBase& mesh, const MeshIOOptions& opts) {
    VTKWriter::write_vtk(mesh, opts.filename, opts.ascii);
  });

  register_reader("vtu", [](const MeshIOOptions& opts) {
    return VTKReader::read_vtu(opts.filename);
  });
  register_writer("vtu", [](const MeshBase& mesh, const MeshIOOptions& opts) {
    VTKWriter::write_vtu(mesh, opts.filename, opts.compression);
  });

  register_reader("vtp", [](const MeshIOOptions& opts) {
    return VTKReader::read_vtp(opts.filename);
  });
  register_writer("vtp", [](const MeshBase& mesh, const MeshIOOptions& opts) {
    VTKWriter::write_vtp(mesh, opts.filename, opts.compression);
  });

  // Parallel VTK formats
  register_reader("pvtu", [](const MeshIOOptions& opts) {
    return VTKReader::read_parallel(opts.filename);
  });
  register_writer("pvtu", [](const MeshBase& mesh, const MeshIOOptions& opts) {
    VTKWriter::write_parallel(mesh, opts.filename, opts.n_parts, opts.compression);
  });
#endif

  // Register other formats as they are implemented
  // STL, PLY, OBJ, OFF, etc.
}

// ---- Static initialization ----

struct MeshIOInitializer {
  MeshIOInitializer() {
    MeshIO::register_builtin_formats();
  }
};

static MeshIOInitializer initializer;

} // namespace svmp