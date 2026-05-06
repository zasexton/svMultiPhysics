/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Application/Core/MeshCollection.h"

#include "Parameters.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace application {
namespace core {

namespace {

std::string meshNameForMessage(const std::string& name)
{
  return name.empty() ? std::string{"<unnamed>"} : name;
}

} // namespace

MeshParticipant MeshParticipant::fromLoadedMesh(
    const MeshParameters& parameters,
    std::shared_ptr<svmp::Mesh> mesh)
{
  MeshParticipant participant;
  participant.name = parameters.name.value();
  if (parameters.domain_id.defined()) {
    participant.domain_id = parameters.domain_id.value();
  }
  participant.mesh = std::move(mesh);
  participant.source_parameters = &parameters;

  if (!participant.mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                             meshNameForMessage(participant.name) +
                             "\"> produced a null mesh.");
  }

  std::set<std::string> local_face_names;
  for (const auto* face : parameters.face_parameters) {
    if (!face) {
      continue;
    }
    const auto face_name = face->name.value();
    if (face_name.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                               meshNameForMessage(participant.name) +
                               "\"> contains an <Add_face> with an empty name.");
    }
    if (!local_face_names.insert(face_name).second) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                               meshNameForMessage(participant.name) +
                               "\"> contains duplicate <Add_face name=\"" +
                               face_name + "\"> entries.");
    }

    const auto label = participant.mesh->base().label_from_name(face_name);
    if (label == svmp::INVALID_LABEL) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                               meshNameForMessage(participant.name) +
                               "\"> face '" + face_name +
                               "' was not registered on the loaded mesh.");
    }
    participant.face_labels.emplace(face_name, label);
  }

  return participant;
}

void MeshCollection::clear()
{
  participants_.clear();
  name_to_index_.clear();
  domain_to_index_.clear();
  face_to_owner_.clear();
}

void MeshCollection::addParticipant(MeshParticipant participant)
{
  if (participant.name.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh> is missing required name attribute.");
  }
  if (!participant.mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                             participant.name + "\"> has no loaded mesh.");
  }
  if (name_to_index_.count(participant.name) != 0) {
    throw std::runtime_error("[svMultiPhysics::Application] Duplicate <Add_mesh name=\"" +
                             participant.name + "\"> detected.");
  }

  if (participant.domain_id.has_value()) {
    const auto existing = domain_to_index_.find(*participant.domain_id);
    if (existing != domain_to_index_.end()) {
      const auto& other = participants_.at(existing->second);
      throw std::runtime_error("[svMultiPhysics::Application] Duplicate <Domain> value " +
                               std::to_string(*participant.domain_id) +
                               " for meshes '" + other.name + "' and '" +
                               participant.name + "'.");
    }
  }

  for (const auto& [face_name, label] : participant.face_labels) {
    if (face_name.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                               participant.name +
                               "\"> contains an empty face name.");
    }
    if (label == svmp::INVALID_LABEL) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh name=\"" +
                               participant.name + "\"> face '" + face_name +
                               "' has an invalid label.");
    }
    const auto existing = face_to_owner_.find(face_name);
    if (existing != face_to_owner_.end()) {
      throw std::runtime_error("[svMultiPhysics::Application] Duplicate face name '" +
                               face_name + "' across meshes '" +
                               existing->second.participant_name + "' and '" +
                               participant.name + "'. Use globally unique face names "
                               "for new-path multi-mesh translation.");
    }
  }

  const auto index = participants_.size();
  name_to_index_.emplace(participant.name, index);
  if (participant.domain_id.has_value()) {
    domain_to_index_.emplace(*participant.domain_id, index);
  }
  for (const auto& [face_name, label] : participant.face_labels) {
    face_to_owner_.emplace(face_name, MeshFaceOwner{participant.name, label});
  }
  participants_.push_back(std::move(participant));
}

const MeshParticipant* MeshCollection::participantByName(std::string_view name) const noexcept
{
  const auto it = name_to_index_.find(std::string(name));
  if (it == name_to_index_.end()) {
    return nullptr;
  }
  return &participants_[it->second];
}

const MeshParticipant* MeshCollection::participantByDomain(int domain_id) const noexcept
{
  const auto it = domain_to_index_.find(domain_id);
  if (it == domain_to_index_.end()) {
    return nullptr;
  }
  return &participants_[it->second];
}

const MeshParticipant* MeshCollection::participantOwningFace(std::string_view face_name) const noexcept
{
  const auto owner = faceOwner(face_name);
  if (!owner.has_value()) {
    return nullptr;
  }
  return participantByName(owner->participant_name);
}

std::optional<MeshFaceOwner> MeshCollection::faceOwner(std::string_view face_name) const noexcept
{
  const auto it = face_to_owner_.find(std::string(face_name));
  if (it == face_to_owner_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::map<std::string, std::shared_ptr<svmp::Mesh>> MeshCollection::asMeshMap() const
{
  std::map<std::string, std::shared_ptr<svmp::Mesh>> meshes;
  for (const auto& participant : participants_) {
    meshes.emplace(participant.name, participant.mesh);
  }
  return meshes;
}

void MeshCollection::validateDomainIdsRequired(std::string_view context) const
{
  for (const auto& participant : participants_) {
    if (!participant.domain_id.has_value()) {
      throw std::runtime_error(std::string("[svMultiPhysics::Application] ") +
                               std::string(context) +
                               " requires <Add_mesh name=\"" + participant.name +
                               "\"> to define <Domain>.");
    }
  }
}

} // namespace core
} // namespace application
