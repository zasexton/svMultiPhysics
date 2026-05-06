/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_APPLICATION_CORE_MESH_COLLECTION_H
#define SVMP_APPLICATION_CORE_MESH_COLLECTION_H

#include "Mesh/Mesh.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class MeshParameters;

namespace application {
namespace core {

struct MeshParticipant {
  std::string name{};
  std::optional<int> domain_id{};
  std::shared_ptr<svmp::Mesh> mesh{};
  std::map<std::string, svmp::label_t> face_labels{};
  const MeshParameters* source_parameters{nullptr};

  [[nodiscard]] static MeshParticipant fromLoadedMesh(
      const MeshParameters& parameters,
      std::shared_ptr<svmp::Mesh> mesh);
};

struct MeshFaceOwner {
  std::string participant_name{};
  svmp::label_t label{svmp::INVALID_LABEL};
};

class MeshCollection {
public:
  void clear();
  void addParticipant(MeshParticipant participant);

  [[nodiscard]] bool empty() const noexcept { return participants_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return participants_.size(); }
  [[nodiscard]] const std::vector<MeshParticipant>& participants() const noexcept
  {
    return participants_;
  }

  [[nodiscard]] const MeshParticipant* participantByName(std::string_view name) const noexcept;
  [[nodiscard]] const MeshParticipant* participantByDomain(int domain_id) const noexcept;
  [[nodiscard]] const MeshParticipant* participantOwningFace(std::string_view face_name) const noexcept;
  [[nodiscard]] std::optional<MeshFaceOwner> faceOwner(std::string_view face_name) const noexcept;

  [[nodiscard]] std::map<std::string, std::shared_ptr<svmp::Mesh>> asMeshMap() const;

  void validateDomainIdsRequired(std::string_view context) const;

private:
  std::vector<MeshParticipant> participants_{};
  std::unordered_map<std::string, std::size_t> name_to_index_{};
  std::unordered_map<int, std::size_t> domain_to_index_{};
  std::unordered_map<std::string, MeshFaceOwner> face_to_owner_{};
};

} // namespace core
} // namespace application

#endif // SVMP_APPLICATION_CORE_MESH_COLLECTION_H
