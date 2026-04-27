/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_MOVING_MESH_RESTART_H
#define SVMP_MOVING_MESH_RESTART_H

#include "../Core/MeshBase.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace svmp {
namespace moving_mesh_restart {

inline constexpr std::uint32_t kSupportedVersion = 4;

struct WriteOptions {
  bool include_fields = true;
  bool include_motion_fields = true;
  std::uint64_t restart_epoch = 0;
  std::map<std::string, std::string> motion_backend_state;
  std::map<std::string, std::string> moving_geometry_validity_state;
  std::vector<std::string> adaptivity_provenance;
};

struct ReadOptions {
  bool require_supported_version = true;
  bool require_current_coordinates_when_active_current = true;
};

struct Metadata {
  std::uint32_t version = 0;
  std::uint64_t restart_epoch = 0;
  MeshRevisionState mesh_revisions{};
  Configuration active_configuration = Configuration::Reference;
  bool has_current_coordinates = false;
  GeometryOrderDescriptor geometry_order{};
  ReferenceRebaseInfo reference_rebase{};
  std::map<std::string, std::string> motion_backend_state;
  std::map<std::string, std::string> moving_geometry_validity_state;
  std::vector<std::string> adaptivity_provenance;
};

void write(const MeshBase& mesh, const std::string& path, const WriteOptions& options = {});

[[nodiscard]] MeshBase read(const std::string& path, const ReadOptions& options = {});

[[nodiscard]] Metadata inspect(const std::string& path, const ReadOptions& options = {});

void register_with_mesh();

} // namespace moving_mesh_restart
} // namespace svmp

#endif // SVMP_MOVING_MESH_RESTART_H
