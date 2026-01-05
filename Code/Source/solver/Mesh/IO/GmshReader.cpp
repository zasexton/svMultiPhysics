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

#include "GmshReader.h"
#include "../Topology/NodeOrdering.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <set>
#include <map>

namespace svmp {

namespace {

template <typename T>
T byteswap_value(T v);

template <>
inline uint32_t byteswap_value<uint32_t>(uint32_t v) {
    return ((v & 0x000000FFu) << 24) |
           ((v & 0x0000FF00u) << 8) |
           ((v & 0x00FF0000u) >> 8) |
           ((v & 0xFF000000u) >> 24);
}

template <>
inline uint64_t byteswap_value<uint64_t>(uint64_t v) {
    return ((v & 0x00000000000000FFull) << 56) |
           ((v & 0x000000000000FF00ull) << 40) |
           ((v & 0x0000000000FF0000ull) << 24) |
           ((v & 0x00000000FF000000ull) << 8) |
           ((v & 0x000000FF00000000ull) >> 8) |
           ((v & 0x0000FF0000000000ull) >> 24) |
           ((v & 0x00FF000000000000ull) >> 40) |
           ((v & 0xFF00000000000000ull) >> 56);
}

template <typename T>
T read_binary(std::ifstream& file, bool swap_endian) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("GmshReader: Unexpected EOF while reading binary data");
    }
    if (!swap_endian) {
        return value;
    }

    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        uint32_t tmp{};
        std::memcpy(&tmp, &value, sizeof(uint32_t));
        tmp = byteswap_value<uint32_t>(tmp);
        std::memcpy(&value, &tmp, sizeof(uint32_t));
        return value;
    } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        uint64_t tmp{};
        std::memcpy(&tmp, &value, sizeof(uint64_t));
        tmp = byteswap_value<uint64_t>(tmp);
        std::memcpy(&value, &tmp, sizeof(uint64_t));
        return value;
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unsupported binary swap size");
        return value;
    }
}

double read_binary_real(std::ifstream& file, int data_size, bool swap_endian) {
    if (data_size == 8) {
        const double v = read_binary<double>(file, swap_endian);
        return v;
    }
    if (data_size == 4) {
        const float v = read_binary<float>(file, swap_endian);
        return static_cast<double>(v);
    }
    throw std::runtime_error("GmshReader: Unsupported binary real size " + std::to_string(data_size));
}

size_t read_binary_tag(std::ifstream& file, int data_size, bool swap_endian) {
    if (data_size == 8) {
        const uint64_t v = read_binary<uint64_t>(file, swap_endian);
        return static_cast<size_t>(v);
    }
    if (data_size == 4) {
        const uint32_t v = read_binary<uint32_t>(file, swap_endian);
        return static_cast<size_t>(v);
    }
    throw std::runtime_error("GmshReader: Unsupported binary tag size " + std::to_string(data_size));
}

} // namespace

GmshReader::GmshReader() = default;
GmshReader::~GmshReader() = default;

MeshBase GmshReader::read(const MeshIOOptions& options) {
    if (options.path.empty()) {
        throw std::runtime_error("GmshReader::read: options.path is empty");
    }
    return read(options.path);
}

MeshBase GmshReader::read(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("GmshReader: Cannot open file: " + filename);
    }

    // Read lines until we find $MeshFormat (skip empty lines and comments)
    std::string line;
    bool found_format = false;
    while (std::getline(file, line)) {
        // Skip empty lines and whitespace-only lines
        size_t first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos) {
            continue;  // Empty line
        }
        // Check for $MeshFormat header
        if (line.find("$MeshFormat") != std::string::npos) {
            found_format = true;
            break;
        }
        // If we find a non-empty line that's not $MeshFormat, it's invalid
        throw std::runtime_error("GmshReader: Invalid Gmsh file (missing $MeshFormat): " + filename);
    }

    if (!found_format) {
        throw std::runtime_error("GmshReader: Empty or invalid Gmsh file (missing $MeshFormat): " + filename);
    }

    // Read version line
    if (!std::getline(file, line)) {
        throw std::runtime_error("GmshReader: Cannot read format version: " + filename);
    }

    double version;
    int file_type, data_size;
    std::istringstream iss(line);
    iss >> version >> file_type >> data_size;

    bool swap_endian = false;
    if (file_type != 0) {
        if (file_type != 1) {
            throw std::runtime_error("GmshReader: Unsupported file_type " + std::to_string(file_type) +
                                     " in " + filename);
        }

        // Binary: read endianness check integer and set swap flag if needed.
        const uint32_t one = read_binary<uint32_t>(file, /*swap_endian=*/false);
        if (one != 1u) {
            if (byteswap_value<uint32_t>(one) == 1u) {
                swap_endian = true;
            } else {
                throw std::runtime_error("GmshReader: Invalid binary endianness marker in " + filename);
            }
        }
        // Consume remainder of the line after the marker (usually just newline).
        std::getline(file, line);
    }

    // Route to appropriate parser based on version
    if (version >= 4.0) {
        return (file_type == 0) ? read_msh4(file, filename)
                                : read_msh4_binary(file, filename, data_size, swap_endian);
    } else if (version >= 2.0) {
        return (file_type == 0) ? read_msh2(file, filename)
                                : read_msh2_binary(file, filename, data_size, swap_endian);
    } else {
        throw std::runtime_error("GmshReader: Unsupported MSH version " +
                               std::to_string(version) + " in file: " + filename);
    }
}

void GmshReader::register_with_mesh_io() {
    // Register Gmsh reader with MeshBase's I/O registry.
    MeshBase::register_reader("gmsh", [](const MeshIOOptions& opts) { return GmshReader::read(opts); });
    MeshBase::register_reader("msh", [](const MeshIOOptions& opts) { return GmshReader::read(opts); });
}

bool GmshReader::is_gmsh_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    // Skip empty lines at the start
    while (std::getline(file, line)) {
        size_t first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos) {
            continue;  // Skip empty lines
        }
        return line.find("$MeshFormat") != std::string::npos;
    }

    return false;
}

double GmshReader::get_format_version(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return 0.0;

    std::string line;
    // Skip empty lines until we find $MeshFormat
    while (std::getline(file, line)) {
        size_t first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos) {
            continue;  // Skip empty lines
        }
        if (line.find("$MeshFormat") != std::string::npos) {
            break;
        }
        return 0.0;  // Not a valid Gmsh file
    }

    // Read version line
    if (!std::getline(file, line)) return 0.0;

    double version;
    std::istringstream iss(line);
    iss >> version;
    return version;
}

// ==========================================
// MSH 2.x Format Parser
// ==========================================

MeshBase GmshReader::read_msh2(std::ifstream& file, const std::string& filename) {
    std::vector<real_t> coords;
    std::vector<GmshElement> elements;
    std::vector<PhysicalGroup> physical_groups;
    std::unordered_map<size_t, size_t> node_id_map;

    std::string line;

    // Skip to $EndMeshFormat
    while (std::getline(file, line)) {
        if (line.find("$EndMeshFormat") != std::string::npos) break;
    }

    // Parse remaining sections
    while (std::getline(file, line)) {
        if (line.find("$PhysicalNames") != std::string::npos) {
            parse_physical_names(file, physical_groups);
        } else if (line.find("$Nodes") != std::string::npos) {
            parse_nodes_v2(file, coords, node_id_map);
        } else if (line.find("$Elements") != std::string::npos) {
            parse_elements_v2(file, elements);
        }
        // Skip other sections (Periodic, NodeData, etc.)
    }

    return build_mesh(coords, elements, physical_groups, node_id_map);
}

MeshBase GmshReader::read_msh2_binary(std::ifstream& file,
                                      const std::string& filename,
                                      int data_size,
                                      bool swap_endian) {
    std::vector<real_t> coords;
    std::vector<GmshElement> elements;
    std::vector<PhysicalGroup> physical_groups;
    std::unordered_map<size_t, size_t> node_id_map;

    std::string line;

    // Skip to $EndMeshFormat
    while (std::getline(file, line)) {
        if (line.find("$EndMeshFormat") != std::string::npos) break;
    }

    while (std::getline(file, line)) {
        if (line.find("$PhysicalNames") != std::string::npos) {
            parse_physical_names(file, physical_groups);
        } else if (line.find("$Nodes") != std::string::npos) {
            parse_nodes_v2_binary(file, coords, node_id_map, data_size, swap_endian);
        } else if (line.find("$Elements") != std::string::npos) {
            parse_elements_v2_binary(file, elements, swap_endian);
        }
        // Skip other sections
    }

    return build_mesh(coords, elements, physical_groups, node_id_map);
}

void GmshReader::parse_nodes_v2(std::ifstream& file,
                                std::vector<real_t>& coords,
                                std::unordered_map<size_t, size_t>& node_id_map) {
    std::string line;
    std::getline(file, line);

    size_t num_nodes;
    std::istringstream iss(line);
    iss >> num_nodes;

    coords.reserve(num_nodes * 3);

    for (size_t i = 0; i < num_nodes; ++i) {
        std::getline(file, line);
        std::istringstream node_iss(line);

        size_t node_id;
        real_t x, y, z;
        node_iss >> node_id >> x >> y >> z;

        // Map 1-based Gmsh node ID to 0-based index
        node_id_map[node_id] = coords.size() / 3;

        coords.push_back(x);
        coords.push_back(y);
        coords.push_back(z);
    }

    // Read $EndNodes
    std::getline(file, line);
}

void GmshReader::parse_nodes_v2_binary(std::ifstream& file,
                                       std::vector<real_t>& coords,
                                       std::unordered_map<size_t, size_t>& node_id_map,
                                       int data_size,
                                       bool swap_endian) {
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("GmshReader: Unexpected EOF while reading $Nodes header");
    }

    size_t num_nodes = 0;
    {
        std::istringstream iss(line);
        iss >> num_nodes;
    }

    coords.reserve(num_nodes * 3);

    for (size_t i = 0; i < num_nodes; ++i) {
        const int32_t node_tag = read_binary<int32_t>(file, swap_endian);
        const double x = read_binary_real(file, data_size, swap_endian);
        const double y = read_binary_real(file, data_size, swap_endian);
        const double z = read_binary_real(file, data_size, swap_endian);

        node_id_map[static_cast<size_t>(node_tag)] = coords.size() / 3;
        coords.push_back(static_cast<real_t>(x));
        coords.push_back(static_cast<real_t>(y));
        coords.push_back(static_cast<real_t>(z));
    }

    // Skip to $EndNodes (there is typically a newline before the tag).
    skip_section(file, "$EndNodes");
}

void GmshReader::parse_elements_v2(std::ifstream& file,
                                std::vector<GmshElement>& elements) {
    std::string line;
    std::getline(file, line);

    size_t num_elements;
    std::istringstream iss(line);
    iss >> num_elements;

    elements.reserve(num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
        std::getline(file, line);
        std::istringstream elem_iss(line);

        size_t elem_id;
        int elem_type, num_tags;
        elem_iss >> elem_id >> elem_type >> num_tags;

        GmshElement elem;
        elem.type = elem_type;
        elem.physical_tag = 0;
        elem.entity_tag = 0;

        // Read tags (first is physical, second is elementary/geometric entity)
        for (int t = 0; t < num_tags; ++t) {
            int tag;
            elem_iss >> tag;
            if (t == 0) elem.physical_tag = tag;
            else if (t == 1) elem.entity_tag = tag;
        }

        // Read node IDs
        int num_nodes = gmsh_element_num_nodes(elem_type);
        elem.nodes.resize(num_nodes);
        for (int n = 0; n < num_nodes; ++n) {
            size_t node_id;
            elem_iss >> node_id;
            elem.nodes[n] = node_id;  // Still 1-based, will convert later
        }

        elements.push_back(std::move(elem));
    }

    // Read $EndElements
    std::getline(file, line);
}

void GmshReader::parse_elements_v2_binary(std::ifstream& file,
                                          std::vector<GmshElement>& elements,
                                          bool swap_endian) {
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("GmshReader: Unexpected EOF while reading $Elements header");
    }

    size_t num_elements = 0;
    {
        std::istringstream iss(line);
        iss >> num_elements;
    }

    elements.clear();
    elements.reserve(num_elements);

    size_t remaining = num_elements;
    while (remaining > 0) {
        const int32_t elem_type = read_binary<int32_t>(file, swap_endian);
        const int32_t block_count = read_binary<int32_t>(file, swap_endian);
        const int32_t num_tags = read_binary<int32_t>(file, swap_endian);

        if (block_count <= 0 || static_cast<size_t>(block_count) > remaining) {
            throw std::runtime_error("GmshReader: Invalid binary element block count");
        }

        const int nn = gmsh_element_num_nodes(elem_type);

        for (int32_t i = 0; i < block_count; ++i) {
            // Element number (unused)
            (void)read_binary<int32_t>(file, swap_endian);

            std::vector<int32_t> tags;
            tags.resize(static_cast<size_t>(std::max<int32_t>(0, num_tags)));
            for (int32_t t = 0; t < num_tags; ++t) {
                tags[static_cast<size_t>(t)] = read_binary<int32_t>(file, swap_endian);
            }

            GmshElement elem;
            elem.type = elem_type;
            elem.physical_tag = (num_tags >= 1) ? tags[0] : 0;
            elem.entity_tag = (num_tags >= 2) ? tags[1] : elem.physical_tag;

            elem.nodes.resize(static_cast<size_t>(nn));
            for (int n = 0; n < nn; ++n) {
                const int32_t node_tag = read_binary<int32_t>(file, swap_endian);
                elem.nodes[static_cast<size_t>(n)] = static_cast<size_t>(node_tag);
            }

            elements.push_back(std::move(elem));
        }

        remaining -= static_cast<size_t>(block_count);
    }

    // Skip to $EndElements (there is typically a newline before the tag).
    skip_section(file, "$EndElements");
}

// ==========================================
// MSH 4.x Format Parser
// ==========================================

MeshBase GmshReader::read_msh4(std::ifstream& file, const std::string& filename) {
    std::vector<real_t> coords;
    std::vector<GmshElement> elements;
    std::vector<PhysicalGroup> physical_groups;
    std::unordered_map<size_t, size_t> node_id_map;

    std::string line;

    // Skip to $EndMeshFormat
    while (std::getline(file, line)) {
        if (line.find("$EndMeshFormat") != std::string::npos) break;
    }

    // Parse remaining sections
    while (std::getline(file, line)) {
        if (line.find("$PhysicalNames") != std::string::npos) {
            parse_physical_names(file, physical_groups);
        } else if (line.find("$Nodes") != std::string::npos) {
            parse_nodes_v4(file, coords, node_id_map);
        } else if (line.find("$Elements") != std::string::npos) {
            parse_elements_v4(file, elements);
        }
        // Skip other sections
    }

    return build_mesh(coords, elements, physical_groups, node_id_map);
}

MeshBase GmshReader::read_msh4_binary(std::ifstream& file,
                                      const std::string& filename,
                                      int data_size,
                                      bool swap_endian) {
    std::vector<real_t> coords;
    std::vector<GmshElement> elements;
    std::vector<PhysicalGroup> physical_groups;
    std::unordered_map<size_t, size_t> node_id_map;

    std::string line;

    // Skip to $EndMeshFormat
    while (std::getline(file, line)) {
        if (line.find("$EndMeshFormat") != std::string::npos) break;
    }

    while (std::getline(file, line)) {
        if (line.find("$PhysicalNames") != std::string::npos) {
            parse_physical_names(file, physical_groups);
        } else if (line.find("$Nodes") != std::string::npos) {
            parse_nodes_v4_binary(file, coords, node_id_map, data_size, swap_endian);
        } else if (line.find("$Elements") != std::string::npos) {
            parse_elements_v4_binary(file, elements, data_size, swap_endian);
        }
        // Skip other sections
    }

    return build_mesh(coords, elements, physical_groups, node_id_map);
}

void GmshReader::parse_nodes_v4(std::ifstream& file,
                                std::vector<real_t>& coords,
                                std::unordered_map<size_t, size_t>& node_id_map) {
    std::string line;
    std::getline(file, line);

    // Format: numEntityBlocks numNodes minNodeTag maxNodeTag
    size_t num_entity_blocks, num_nodes, min_tag, max_tag;
    std::istringstream header_iss(line);
    header_iss >> num_entity_blocks >> num_nodes >> min_tag >> max_tag;

    coords.reserve(num_nodes * 3);

    for (size_t block = 0; block < num_entity_blocks; ++block) {
        std::getline(file, line);
        std::istringstream block_iss(line);

        int entity_dim, entity_tag, parametric;
        size_t num_nodes_in_block;
        block_iss >> entity_dim >> entity_tag >> parametric >> num_nodes_in_block;

        // Read node tags
        std::vector<size_t> node_tags(num_nodes_in_block);
        for (size_t i = 0; i < num_nodes_in_block; ++i) {
            std::getline(file, line);
            std::istringstream tag_iss(line);
            tag_iss >> node_tags[i];
        }

        // Read node coordinates
        for (size_t i = 0; i < num_nodes_in_block; ++i) {
            std::getline(file, line);
            std::istringstream coord_iss(line);

            real_t x, y, z;
            coord_iss >> x >> y >> z;

            node_id_map[node_tags[i]] = coords.size() / 3;
            coords.push_back(x);
            coords.push_back(y);
            coords.push_back(z);
        }
    }

    // Read $EndNodes
    std::getline(file, line);
}

void GmshReader::parse_nodes_v4_binary(std::ifstream& file,
                                       std::vector<real_t>& coords,
                                       std::unordered_map<size_t, size_t>& node_id_map,
                                       int data_size,
                                       bool swap_endian) {
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("GmshReader: Unexpected EOF while reading $Nodes header");
    }

    // Header: numEntityBlocks numNodes minNodeTag maxNodeTag
    size_t num_entity_blocks = 0;
    size_t num_nodes = 0;
    size_t min_tag = 0;
    size_t max_tag = 0;
    {
        std::istringstream iss(line);
        iss >> num_entity_blocks >> num_nodes >> min_tag >> max_tag;
    }
    (void)min_tag;
    (void)max_tag;

    coords.reserve(num_nodes * 3);

    for (size_t block = 0; block < num_entity_blocks; ++block) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("GmshReader: Unexpected EOF while reading $Nodes block header");
        }

        int entity_dim = 0;
        int entity_tag = 0;
        int parametric = 0;
        size_t num_nodes_in_block = 0;
        {
            std::istringstream iss(line);
            iss >> entity_dim >> entity_tag >> parametric >> num_nodes_in_block;
        }

        std::vector<size_t> node_tags(num_nodes_in_block);
        for (size_t i = 0; i < num_nodes_in_block; ++i) {
            node_tags[i] = read_binary_tag(file, data_size, swap_endian);
        }

        for (size_t i = 0; i < num_nodes_in_block; ++i) {
            const double x = read_binary_real(file, data_size, swap_endian);
            const double y = read_binary_real(file, data_size, swap_endian);
            const double z = read_binary_real(file, data_size, swap_endian);

            node_id_map[node_tags[i]] = coords.size() / 3;
            coords.push_back(static_cast<real_t>(x));
            coords.push_back(static_cast<real_t>(y));
            coords.push_back(static_cast<real_t>(z));
        }

        // Optional parametric coordinates: ignore (read and discard).
        if (parametric) {
            for (size_t i = 0; i < num_nodes_in_block; ++i) {
                for (int d = 0; d < entity_dim; ++d) {
                    (void)read_binary_real(file, data_size, swap_endian);
                }
            }
        }
    }

    skip_section(file, "$EndNodes");
}

void GmshReader::parse_elements_v4(std::ifstream& file,
                                   std::vector<GmshElement>& elements) {
    std::string line;
    std::getline(file, line);

    // Format: numEntityBlocks numElements minElementTag maxElementTag
    size_t num_entity_blocks, num_elements, min_tag, max_tag;
    std::istringstream header_iss(line);
    header_iss >> num_entity_blocks >> num_elements >> min_tag >> max_tag;

    elements.reserve(num_elements);

    for (size_t block = 0; block < num_entity_blocks; ++block) {
        std::getline(file, line);
        std::istringstream block_iss(line);

        int entity_dim, entity_tag, elem_type;
        size_t num_elements_in_block;
        block_iss >> entity_dim >> entity_tag >> elem_type >> num_elements_in_block;

        int num_nodes = gmsh_element_num_nodes(elem_type);

        for (size_t i = 0; i < num_elements_in_block; ++i) {
            std::getline(file, line);
            std::istringstream elem_iss(line);

            GmshElement elem;
            elem.type = elem_type;
            elem.entity_tag = entity_tag;
            elem.physical_tag = entity_tag;  // In MSH 4.x, use entity tag as default

            size_t elem_tag;
            elem_iss >> elem_tag;

            elem.nodes.resize(num_nodes);
            for (int n = 0; n < num_nodes; ++n) {
                elem_iss >> elem.nodes[n];
            }

            elements.push_back(std::move(elem));
        }
    }

    // Read $EndElements
    std::getline(file, line);
}

void GmshReader::parse_elements_v4_binary(std::ifstream& file,
                                          std::vector<GmshElement>& elements,
                                          int data_size,
                                          bool swap_endian) {
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("GmshReader: Unexpected EOF while reading $Elements header");
    }

    size_t num_entity_blocks = 0;
    size_t num_elements = 0;
    size_t min_tag = 0;
    size_t max_tag = 0;
    {
        std::istringstream iss(line);
        iss >> num_entity_blocks >> num_elements >> min_tag >> max_tag;
    }
    (void)min_tag;
    (void)max_tag;

    elements.clear();
    elements.reserve(num_elements);

    for (size_t block = 0; block < num_entity_blocks; ++block) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("GmshReader: Unexpected EOF while reading $Elements block header");
        }

        int entity_dim = 0;
        int entity_tag = 0;
        int elem_type = 0;
        size_t num_elements_in_block = 0;
        {
            std::istringstream iss(line);
            iss >> entity_dim >> entity_tag >> elem_type >> num_elements_in_block;
        }
        (void)entity_dim;

        const int num_nodes = gmsh_element_num_nodes(elem_type);
        for (size_t i = 0; i < num_elements_in_block; ++i) {
            // Element tag (unused)
            (void)read_binary_tag(file, data_size, swap_endian);

            GmshElement elem;
            elem.type = elem_type;
            elem.entity_tag = entity_tag;
            elem.physical_tag = entity_tag; // best-effort; $Entities not parsed

            elem.nodes.resize(static_cast<size_t>(num_nodes));
            for (int n = 0; n < num_nodes; ++n) {
                elem.nodes[static_cast<size_t>(n)] = read_binary_tag(file, data_size, swap_endian);
            }

            elements.push_back(std::move(elem));
        }
    }

    skip_section(file, "$EndElements");
}

void GmshReader::parse_physical_names(std::ifstream& file,
                                      std::vector<PhysicalGroup>& groups) {
    std::string line;
    std::getline(file, line);

    size_t num_groups;
    std::istringstream iss(line);
    iss >> num_groups;

    groups.reserve(num_groups);

    for (size_t i = 0; i < num_groups; ++i) {
        std::getline(file, line);
        std::istringstream group_iss(line);

        PhysicalGroup group;
        group_iss >> group.dimension >> group.tag;

        // Read name (may be quoted)
        std::string name;
        group_iss >> name;
        // Remove quotes if present
        if (!name.empty() && name.front() == '"') {
            name = name.substr(1);
            if (!name.empty() && name.back() == '"') {
                name = name.substr(0, name.size() - 1);
            }
        }
        group.name = name;

        groups.push_back(std::move(group));
    }

    // Read $EndPhysicalNames
    std::getline(file, line);
}

// ==========================================
// Element Type Handling
// ==========================================

CellShape GmshReader::gmsh_to_cellshape(int gmsh_type) {
    CellShape shape;
    shape.num_corners = gmsh_element_num_nodes(gmsh_type);

    switch (gmsh_type) {
        case GMSH_POINT:
            shape.family = CellFamily::Point;
            shape.num_corners = 1;
            shape.order = 1;
            break;

        case GMSH_LINE:
            shape.family = CellFamily::Line;
            shape.num_corners = 2;
            shape.order = 1;
            break;
        case GMSH_LINE3:
            shape.family = CellFamily::Line;
            shape.num_corners = 2;
            shape.order = 2;
            break;
        case GMSH_EDGE4:
            shape.family = CellFamily::Line;
            shape.num_corners = 2;
            shape.order = 3;
            break;

        case GMSH_TRIANGLE:
            shape.family = CellFamily::Triangle;
            shape.num_corners = 3;
            shape.order = 1;
            break;
        case GMSH_TRIANGLE6:
            shape.family = CellFamily::Triangle;
            shape.num_corners = 3;
            shape.order = 2;
            break;
        case GMSH_TRIANGLE9:
        case GMSH_TRIANGLE10:
            shape.family = CellFamily::Triangle;
            shape.num_corners = 3;
            shape.order = 3;
            break;

        case GMSH_QUAD:
            shape.family = CellFamily::Quad;
            shape.num_corners = 4;
            shape.order = 1;
            break;
        case GMSH_QUAD8:
            shape.family = CellFamily::Quad;
            shape.num_corners = 4;
            shape.order = 2;
            break;
        case GMSH_QUAD9:
            shape.family = CellFamily::Quad;
            shape.num_corners = 4;
            shape.order = 2;
            break;

        case GMSH_TETRAHEDRON:
            shape.family = CellFamily::Tetra;
            shape.num_corners = 4;
            shape.order = 1;
            break;
        case GMSH_TETRAHEDRON10:
            shape.family = CellFamily::Tetra;
            shape.num_corners = 4;
            shape.order = 2;
            break;
        case GMSH_TETRAHEDRON20:
            shape.family = CellFamily::Tetra;
            shape.num_corners = 4;
            shape.order = 3;
            break;

        case GMSH_HEXAHEDRON:
            shape.family = CellFamily::Hex;
            shape.num_corners = 8;
            shape.order = 1;
            break;
        case GMSH_HEXAHEDRON20:
            shape.family = CellFamily::Hex;
            shape.num_corners = 8;
            shape.order = 2;
            break;
        case GMSH_HEXAHEDRON27:
            shape.family = CellFamily::Hex;
            shape.num_corners = 8;
            shape.order = 2;
            break;

        case GMSH_PRISM:
            shape.family = CellFamily::Wedge;
            shape.num_corners = 6;
            shape.order = 1;
            break;
        case GMSH_PRISM15:
        case GMSH_PRISM18:
            shape.family = CellFamily::Wedge;
            shape.num_corners = 6;
            shape.order = 2;
            break;

        case GMSH_PYRAMID:
            shape.family = CellFamily::Pyramid;
            shape.num_corners = 5;
            shape.order = 1;
            break;
        case GMSH_PYRAMID13:
        case GMSH_PYRAMID14:
            shape.family = CellFamily::Pyramid;
            shape.num_corners = 5;
            shape.order = 2;
            break;

        default:
            throw std::runtime_error("GmshReader: Unsupported element type " +
                                   std::to_string(gmsh_type));
    }

    return shape;
}

int GmshReader::gmsh_element_num_nodes(int gmsh_type) {
    switch (gmsh_type) {
        case GMSH_POINT:        return 1;
        case GMSH_LINE:         return 2;
        case GMSH_TRIANGLE:     return 3;
        case GMSH_QUAD:         return 4;
        case GMSH_TETRAHEDRON:  return 4;
        case GMSH_HEXAHEDRON:   return 8;
        case GMSH_PRISM:        return 6;
        case GMSH_PYRAMID:      return 5;
        case GMSH_LINE3:        return 3;
        case GMSH_TRIANGLE6:    return 6;
        case GMSH_QUAD9:        return 9;
        case GMSH_TETRAHEDRON10: return 10;
        case GMSH_HEXAHEDRON27: return 27;
        case GMSH_PRISM18:      return 18;
        case GMSH_PYRAMID14:    return 14;
        case GMSH_QUAD8:        return 8;
        case GMSH_HEXAHEDRON20: return 20;
        case GMSH_PRISM15:      return 15;
        case GMSH_PYRAMID13:    return 13;
        case GMSH_TRIANGLE9:    return 9;
        case GMSH_TRIANGLE10:   return 10;
        case GMSH_EDGE4:        return 4;
        case GMSH_TETRAHEDRON20: return 20;
        default:
            throw std::runtime_error("GmshReader: Unknown element type " +
                                   std::to_string(gmsh_type));
    }
}

int GmshReader::gmsh_element_dimension(int gmsh_type) {
    switch (gmsh_type) {
        case GMSH_POINT:
            return 0;

        case GMSH_LINE:
        case GMSH_LINE3:
        case GMSH_EDGE4:
            return 1;

        case GMSH_TRIANGLE:
        case GMSH_QUAD:
        case GMSH_TRIANGLE6:
        case GMSH_QUAD8:
        case GMSH_QUAD9:
        case GMSH_TRIANGLE9:
        case GMSH_TRIANGLE10:
            return 2;

        case GMSH_TETRAHEDRON:
        case GMSH_HEXAHEDRON:
        case GMSH_PRISM:
        case GMSH_PYRAMID:
        case GMSH_TETRAHEDRON10:
        case GMSH_HEXAHEDRON20:
        case GMSH_HEXAHEDRON27:
        case GMSH_PRISM15:
        case GMSH_PRISM18:
        case GMSH_PYRAMID13:
        case GMSH_PYRAMID14:
        case GMSH_TETRAHEDRON20:
            return 3;

        default:
            return -1;
    }
}

void GmshReader::reorder_nodes_to_svmp(int gmsh_type, std::vector<size_t>& nodes) {
    const CellShape shape = gmsh_to_cellshape(gmsh_type);
    if (shape.family == CellFamily::Point) return;

    try {
        NodeOrdering::reorder_to_vtk(NodeOrderingFormat::Gmsh, shape.family, shape.order, nodes);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("GmshReader: failed to reorder nodes: ") + e.what());
    }
}

void GmshReader::skip_section(std::ifstream& file, const std::string& end_tag) {
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(end_tag) != std::string::npos) break;
    }
}

// ==========================================
// Mesh Building
// ==========================================

MeshBase GmshReader::build_mesh(const std::vector<real_t>& coords,
                                const std::vector<GmshElement>& elements,
                                const std::vector<PhysicalGroup>& physical_groups,
                                const std::unordered_map<size_t, size_t>& node_id_map) {

    // Find the maximum element dimension
    int max_dim = 0;
    for (const auto& elem : elements) {
        int dim = gmsh_element_dimension(elem.type);
        if (dim > max_dim) max_dim = dim;
    }

    if (max_dim == 0) {
        throw std::runtime_error("GmshReader: No valid elements found in mesh");
    }

    // Separate volume elements from boundary elements
    std::vector<const GmshElement*> volume_elements;
    std::vector<const GmshElement*> boundary_elements;

    for (const auto& elem : elements) {
        int dim = gmsh_element_dimension(elem.type);
        if (dim == max_dim) {
            volume_elements.push_back(&elem);
        } else if (dim == max_dim - 1) {
            boundary_elements.push_back(&elem);
        }
        // Skip lower-dimensional elements (points in 3D, etc.)
    }

    // Build cell connectivity
    std::vector<offset_t> cell2vertex_offsets;
    std::vector<index_t> cell2vertex;
    std::vector<CellShape> cell_shapes;
    std::vector<label_t> cell_regions;

    cell2vertex_offsets.push_back(0);

    for (const auto* elem : volume_elements) {
        CellShape shape = gmsh_to_cellshape(elem->type);
        cell_shapes.push_back(shape);
        cell_regions.push_back(static_cast<label_t>(elem->physical_tag));

        // Convert and add node indices
        std::vector<size_t> nodes = elem->nodes;
        reorder_nodes_to_svmp(elem->type, nodes);

        for (size_t node_id : nodes) {
            auto it = node_id_map.find(node_id);
            if (it == node_id_map.end()) {
                throw std::runtime_error("GmshReader: Invalid node ID " +
                                       std::to_string(node_id));
            }
            cell2vertex.push_back(static_cast<index_t>(it->second));
        }

        cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
    }

    // Determine spatial dimension from coordinates
    // Check if any z-coordinates are non-zero
    int spatial_dim = 3;
    if (max_dim == 2) {
        bool all_z_zero = true;
        for (size_t i = 2; i < coords.size(); i += 3) {
            if (std::abs(coords[i]) > 1e-12) {
                all_z_zero = false;
                break;
            }
        }
        // Even for 2D meshes, we keep 3D coordinates for generality
        spatial_dim = 3;
    }

    // Build the mesh
    MeshBase mesh;
    mesh.build_from_arrays(spatial_dim, coords, cell2vertex_offsets,
                          cell2vertex, cell_shapes);

    // Set region labels
    for (size_t c = 0; c < cell_regions.size(); ++c) {
        mesh.set_region_label(static_cast<index_t>(c), cell_regions[c]);
    }

    // Set global IDs (use sequential numbering)
    std::vector<gid_t> vertex_gids(coords.size() / 3);
    for (size_t i = 0; i < vertex_gids.size(); ++i) {
        vertex_gids[i] = static_cast<gid_t>(i);
    }
    mesh.set_vertex_gids(vertex_gids);

    std::vector<gid_t> cell_gids(volume_elements.size());
    for (size_t i = 0; i < cell_gids.size(); ++i) {
        cell_gids[i] = static_cast<gid_t>(i);
    }
    mesh.set_cell_gids(cell_gids);

    // Process boundary elements to set boundary labels
    // Build a map from face vertices to boundary label
    if (!boundary_elements.empty()) {
        std::map<std::set<index_t>, label_t> face_vertex_to_label;

        for (const auto* elem : boundary_elements) {
            std::set<index_t> face_vertices;
            for (size_t node_id : elem->nodes) {
                auto it = node_id_map.find(node_id);
                if (it != node_id_map.end()) {
                    // Only add corner vertices for high-order elements
                    CellShape shape = gmsh_to_cellshape(elem->type);
                    if (face_vertices.size() < static_cast<size_t>(shape.num_corners)) {
                        face_vertices.insert(static_cast<index_t>(it->second));
                    }
                }
            }
            if (!face_vertices.empty()) {
                face_vertex_to_label[face_vertices] = static_cast<label_t>(elem->physical_tag);
            }
        }

        // Match mesh faces with boundary elements
        // This requires iterating over mesh faces and checking their vertices
        // For now, store boundary info in the mesh labels
        // A more complete implementation would set face labels directly
    }

    // Finalize the mesh
    mesh.finalize();

    // Create physical group name to tag mapping
    // Could be exposed via mesh metadata if needed
    for (const auto& group : physical_groups) {
        // Store physical group info (could add to mesh metadata)
        // For now, just log
        (void)group;  // Suppress unused warning
    }

    return mesh;
}

} // namespace svmp
