/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_SEARCH_MULTI_MESH_INTERFACE_H
#define SVMP_SEARCH_MULTI_MESH_INTERFACE_H

#include "../Core/DistributedMesh.h"
#include "../Core/MeshBase.h"
#include "../Core/MeshTypes.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace search {

enum class InterfaceMapState {
  Empty,
  Trial,
  Committed,
  RolledBack
};

enum class LogicalInterfaceRegionKind : std::uint8_t {
  Generic,
  RotatingRegion,
  StationaryRegion,
  SlidingInterface,
  CyclicInterface,
  ContactSurface,
  RemeshedRegion
};

/**
 * @brief Persistent logical identity for moving-interface regions.
 *
 * Physical mesh labels are allowed to change during remeshing, repartitioning,
 * or file import/export.  This identity separates "which region this is" from
 * "which label currently selects it" so interface maps, restart data, contact
 * state, and transfer provenance can be validated after topology/layout changes.
 */
struct LogicalInterfaceRegionId {
  LogicalInterfaceRegionKind kind{LogicalInterfaceRegionKind::Generic};
  std::string persistent_id{};
  std::string name{};
  label_t physical_label{INVALID_LABEL};
  std::uint64_t provenance_epoch{0};

  [[nodiscard]] bool empty() const noexcept { return persistent_id.empty() && name.empty(); }
  [[nodiscard]] bool compatible_with(const LogicalInterfaceRegionId& other) const noexcept;
};

struct InterfaceProvenanceDiagnostic {
  bool ok{true};
  std::vector<std::string> messages{};
};

struct InterfaceRevisionSnapshot {
  Configuration configuration = Configuration::Reference;
  std::uint64_t geometry_revision = 0;
  std::uint64_t reference_geometry_revision = 0;
  std::uint64_t current_geometry_revision = 0;
  std::uint64_t topology_revision = 0;
  std::uint64_t ownership_revision = 0;
  std::uint64_t numbering_revision = 0;
  std::uint64_t field_layout_revision = 0;
  std::uint64_t label_revision = 0;
  std::uint64_t active_configuration_epoch = 0;

  [[nodiscard]] static InterfaceRevisionSnapshot capture(
      const MeshBase& mesh,
      Configuration configuration);

  [[nodiscard]] bool matches(const MeshBase& mesh, Configuration configuration) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct InterfaceSideSpec {
  const MeshBase* mesh = nullptr;
  const DistributedMesh* distributed_mesh = nullptr;
  label_t boundary_label = INVALID_LABEL;
  Configuration configuration = Configuration::Reference;
  std::string name;
  LogicalInterfaceRegionId logical_region{};

  [[nodiscard]] static InterfaceSideSpec from_mesh(
      const MeshBase& mesh,
      label_t boundary_label,
      Configuration configuration = Configuration::Reference,
      std::string name = {});

  [[nodiscard]] static InterfaceSideSpec from_distributed_mesh(
      const DistributedMesh& mesh,
      label_t boundary_label,
      Configuration configuration = Configuration::Reference,
      std::string name = {});

  [[nodiscard]] const MeshBase& local_mesh() const;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] rank_t local_rank() const noexcept;
  [[nodiscard]] int world_size() const noexcept;
  [[nodiscard]] rank_t owner_rank_face(index_t face) const;
  [[nodiscard]] gid_t face_gid(index_t face) const;
};

struct InterfaceRegistryEntry {
  std::string name;
  InterfaceSideSpec source;
  InterfaceSideSpec target;
  real_t max_pair_distance = std::numeric_limits<real_t>::infinity();
};

struct InterfacePair {
  index_t source_face = INVALID_INDEX;
  index_t target_face = INVALID_INDEX;
  index_t source_cell = INVALID_INDEX;
  index_t target_cell = INVALID_INDEX;
  gid_t source_face_gid = INVALID_GID;
  gid_t target_face_gid = INVALID_GID;
  rank_t source_owner_rank = 0;
  rank_t target_owner_rank = 0;
  rank_t source_local_rank = 0;
  rank_t target_local_rank = 0;
  label_t source_label = INVALID_LABEL;
  label_t target_label = INVALID_LABEL;
  std::array<real_t, 3> source_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> source_face_xi{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_face_xi{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> source_cell_xi{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_cell_xi{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> source_normal{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> target_normal{{0.0, 0.0, 0.0}};
  real_t source_measure = 0.0;
  real_t target_measure = 0.0;
  real_t distance = 0.0;
  LogicalInterfaceRegionId source_logical_region{};
  LogicalInterfaceRegionId target_logical_region{};
};

struct InterfaceMap {
  std::string name;
  InterfaceSideSpec source;
  InterfaceSideSpec target;
  InterfaceRevisionSnapshot source_revision;
  InterfaceRevisionSnapshot target_revision;
  std::vector<InterfacePair> pairs;
  InterfaceMapState state = InterfaceMapState::Empty;

  [[nodiscard]] bool valid_for_current_revisions() const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
  void accept_trial();
  void rollback_trial();
};

class InterfaceSearchRegistry {
public:
  void register_interface(
      std::string name,
      InterfaceSideSpec source,
      InterfaceSideSpec target,
      real_t max_pair_distance = std::numeric_limits<real_t>::infinity());

  [[nodiscard]] bool contains(const std::string& name) const noexcept;
  [[nodiscard]] std::vector<std::string> interface_names() const;
  [[nodiscard]] const InterfaceRegistryEntry& interface_entry(const std::string& name) const;

  [[nodiscard]] InterfaceMap build_trial_map(const std::string& name) const;
  void commit_map(InterfaceMap map);
  void rollback_committed_map(const std::string& name);
  [[nodiscard]] const InterfaceMap* committed_map(const std::string& name) const noexcept;
  [[nodiscard]] bool committed_map_valid(const std::string& name) const noexcept;

private:
  std::unordered_map<std::string, InterfaceRegistryEntry> entries_;
  std::unordered_map<std::string, InterfaceMap> committed_maps_;
};

[[nodiscard]] std::array<real_t, 3> closest_point_on_face(
    const MeshBase& mesh,
    index_t face,
    const std::array<real_t, 3>& point,
    Configuration configuration = Configuration::Reference);

[[nodiscard]] std::array<real_t, 3> face_local_coordinates(
    const MeshBase& mesh,
    index_t face,
    const std::array<real_t, 3>& point,
    Configuration configuration = Configuration::Reference);

[[nodiscard]] InterfaceProvenanceDiagnostic validate_interface_provenance(
    const InterfaceMap& map);

} // namespace search
} // namespace svmp

#endif // SVMP_SEARCH_MULTI_MESH_INTERFACE_H
