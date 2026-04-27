/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MovingConstraintMetadata.h"

#include "../Core/MeshBase.h"

#include <cmath>
#include <sstream>
#include <utility>

namespace svmp {
namespace constraints {

namespace {

[[nodiscard]] std::array<real_t, 3> face_centroid(const MeshBase& mesh,
                                                  index_t face,
                                                  Configuration cfg)
{
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  const auto span = mesh.face_vertices_span(face);
  const auto* verts = span.first;
  const auto nverts = span.second;
  if (!verts || nverts == 0) {
    return centroid;
  }

  for (std::size_t i = 0; i < nverts; ++i) {
    const auto x = mesh.geometry_dof_coords(verts[i], cfg);
    centroid[0] += x[0];
    centroid[1] += x[1];
    centroid[2] += x[2];
  }
  const auto inv = real_t(1) / static_cast<real_t>(nverts);
  centroid[0] *= inv;
  centroid[1] *= inv;
  centroid[2] *= inv;
  return centroid;
}

[[nodiscard]] bool bc_applies_to_label(const motion::MotionDirichletBC& bc,
                                       label_t label) noexcept
{
  return bc.boundary_label == INVALID_LABEL || bc.boundary_label == label;
}

[[nodiscard]] std::array<real_t, 3> displacement_for_label(
    const MeshBase& mesh,
    label_t label,
    const std::vector<motion::MotionDirichletBC>& bcs,
    double dt,
    double step_scale,
    std::array<bool, 3>& constrained_components,
    bool& found)
{
  std::array<real_t, 3> x{{0.0, 0.0, 0.0}};
  bool have_face = false;
  const auto& labels = mesh.face_boundary_ids();
  for (index_t f = 0; f < static_cast<index_t>(labels.size()); ++f) {
    if (static_cast<label_t>(labels[static_cast<std::size_t>(f)]) == label) {
      const auto cfg = mesh.has_current_coords() ? Configuration::Current : Configuration::Reference;
      x = face_centroid(mesh, f, cfg);
      have_face = true;
      break;
    }
  }

  std::array<real_t, 3> displacement{{0.0, 0.0, 0.0}};
  constrained_components = {{false, false, false}};
  found = false;
  if (!have_face) {
    return displacement;
  }

  for (const auto& bc : bcs) {
    if (!bc.value || !bc_applies_to_label(bc, label)) {
      continue;
    }
    found = true;
    const auto value = bc.value(x, dt, step_scale);
    for (int d = 0; d < 3; ++d) {
      if (bc.component_mask[static_cast<std::size_t>(d)]) {
        displacement[static_cast<std::size_t>(d)] = value[static_cast<std::size_t>(d)];
        constrained_components[static_cast<std::size_t>(d)] = true;
      }
    }
  }
  return displacement;
}

void merge(MeshConstraintRevisionDependencies& dst,
           const MeshConstraintRevisionDependencies& src) noexcept
{
  dst.geometry = dst.geometry || src.geometry;
  dst.topology = dst.topology || src.topology;
  dst.ownership = dst.ownership || src.ownership;
  dst.numbering = dst.numbering || src.numbering;
  dst.field_layout = dst.field_layout || src.field_layout;
  dst.labels = dst.labels || src.labels;
  dst.active_configuration = dst.active_configuration || src.active_configuration;
}

} // namespace

bool MeshConstraintRevisionDependencies::any() const noexcept
{
  return geometry || topology || ownership || numbering || field_layout || labels || active_configuration;
}

MeshConstraintRevisionDependencies
MeshConstraintRevisionDependencies::geometry_only() noexcept
{
  MeshConstraintRevisionDependencies deps;
  deps.geometry = true;
  deps.active_configuration = true;
  return deps;
}

MeshConstraintRevisionDependencies
MeshConstraintRevisionDependencies::topology_labels() noexcept
{
  MeshConstraintRevisionDependencies deps;
  deps.topology = true;
  deps.numbering = true;
  deps.labels = true;
  return deps;
}

MeshConstraintRevisionDependencies
MeshConstraintRevisionDependencies::moving_boundary_relation() noexcept
{
  MeshConstraintRevisionDependencies deps;
  deps.geometry = true;
  deps.topology = true;
  deps.ownership = true;
  deps.numbering = true;
  deps.labels = true;
  deps.active_configuration = true;
  return deps;
}

MeshConstraintRevisionSnapshot
MeshConstraintRevisionSnapshot::capture(const MeshBase& mesh)
{
  const auto rev = mesh.revision_state();
  MeshConstraintRevisionSnapshot out;
  out.valid = true;
  out.geometry = rev.geometry;
  out.topology = rev.topology;
  out.ownership = rev.ownership;
  out.numbering = rev.numbering;
  out.field_layout = rev.field_layout;
  out.labels = rev.labels;
  out.active_configuration = rev.active_configuration;
  return out;
}

bool dependency_changed(const MeshConstraintRevisionDependencies& deps,
                        const MeshConstraintRevisionSnapshot& cached,
                        const MeshConstraintRevisionSnapshot& current) noexcept
{
  if (!cached.valid || !current.valid) {
    return deps.any();
  }
  return (deps.geometry && cached.geometry != current.geometry) ||
         (deps.topology && cached.topology != current.topology) ||
         (deps.ownership && cached.ownership != current.ownership) ||
         (deps.numbering && cached.numbering != current.numbering) ||
         (deps.field_layout && cached.field_layout != current.field_layout) ||
         (deps.labels && cached.labels != current.labels) ||
         (deps.active_configuration && cached.active_configuration != current.active_configuration);
}

void MovingMeshConstraintRegistry::clear()
{
  metadata_.clear();
}

void MovingMeshConstraintRegistry::add(MovingMeshConstraintMetadata metadata)
{
  if (!metadata.dependencies.any()) {
    switch (metadata.kind) {
      case MovingConstraintKind::PeriodicBoundary:
      case MovingConstraintKind::TiedBoundary:
        metadata.dependencies = MeshConstraintRevisionDependencies::moving_boundary_relation();
        break;
      case MovingConstraintKind::GeometricContinuity:
        metadata.dependencies = MeshConstraintRevisionDependencies::topology_labels();
        metadata.dependencies.geometry = true;
        metadata.dependencies.active_configuration = true;
        break;
    }
  }
  metadata_.push_back(std::move(metadata));
}

MeshConstraintRevisionDependencies
MovingMeshConstraintRegistry::combined_dependencies() const noexcept
{
  MeshConstraintRevisionDependencies out;
  for (const auto& item : metadata_) {
    merge(out, item.dependencies);
  }
  return out;
}

MotionConstraintValidationResult
MovingMeshConstraintRegistry::validate_prescribed_motion(
    const MeshBase& mesh,
    const std::vector<motion::MotionDirichletBC>& bcs,
    double dt,
    double step_scale,
    real_t tolerance) const
{
  MotionConstraintValidationResult result;
  if (metadata_.empty() || bcs.empty()) {
    return result;
  }

  for (const auto& item : metadata_) {
    label_t slave = INVALID_LABEL;
    label_t master = INVALID_LABEL;
    bool preserve_relative_motion = false;

    if (item.kind == MovingConstraintKind::PeriodicBoundary) {
      slave = item.periodic.slave_label;
      master = item.periodic.master_label;
      preserve_relative_motion = item.periodic.preserve_relative_motion;
    } else if (item.kind == MovingConstraintKind::TiedBoundary) {
      slave = item.tied.slave_label;
      master = item.tied.master_label;
      preserve_relative_motion = true;
    } else {
      continue;
    }

    if (!preserve_relative_motion || slave == INVALID_LABEL || master == INVALID_LABEL) {
      continue;
    }

    std::array<bool, 3> slave_mask{{false, false, false}};
    std::array<bool, 3> master_mask{{false, false, false}};
    bool slave_found = false;
    bool master_found = false;
    const auto slave_disp =
        displacement_for_label(mesh, slave, bcs, dt, step_scale, slave_mask, slave_found);
    const auto master_disp =
        displacement_for_label(mesh, master, bcs, dt, step_scale, master_mask, master_found);

    if (slave_found != master_found) {
      result.ok = false;
      std::ostringstream oss;
      oss << "moving constraint '" << item.name
          << "' prescribes motion on only one side of a tied/periodic pair"
          << " (slave label=" << slave << ", master label=" << master << ")";
      result.message = oss.str();
      result.diagnostics.push_back(result.message);
      return result;
    }

    if (!slave_found && !master_found) {
      continue;
    }

    for (int d = 0; d < 3; ++d) {
      const auto idx = static_cast<std::size_t>(d);
      if (slave_mask[idx] != master_mask[idx]) {
        result.ok = false;
        std::ostringstream oss;
        oss << "moving constraint '" << item.name
            << "' has mismatched constrained motion components on paired labels"
            << " (component=" << d << ")";
        result.message = oss.str();
        result.diagnostics.push_back(result.message);
        return result;
      }
      if (slave_mask[idx] && std::abs(slave_disp[idx] - master_disp[idx]) > tolerance) {
        result.ok = false;
        std::ostringstream oss;
        oss << "moving constraint '" << item.name
            << "' has incompatible prescribed motion on paired labels"
            << " (component=" << d
            << ", slave=" << slave_disp[idx]
            << ", master=" << master_disp[idx]
            << ", tolerance=" << tolerance << ")";
        result.message = oss.str();
        result.diagnostics.push_back(result.message);
        return result;
      }
    }
  }

  return result;
}

MovingMeshConstraintMetadata make_periodic_boundary_metadata(
    std::string name,
    label_t slave_label,
    label_t master_label,
    std::array<real_t, 3> slave_to_master_translation)
{
  MovingMeshConstraintMetadata out;
  out.name = std::move(name);
  out.kind = MovingConstraintKind::PeriodicBoundary;
  out.dependencies = MeshConstraintRevisionDependencies::moving_boundary_relation();
  out.periodic.slave_label = slave_label;
  out.periodic.master_label = master_label;
  out.periodic.slave_to_master_translation = slave_to_master_translation;
  return out;
}

MovingMeshConstraintMetadata make_tied_boundary_metadata(
    std::string name,
    label_t slave_label,
    label_t master_label)
{
  MovingMeshConstraintMetadata out;
  out.name = std::move(name);
  out.kind = MovingConstraintKind::TiedBoundary;
  out.dependencies = MeshConstraintRevisionDependencies::moving_boundary_relation();
  out.tied.slave_label = slave_label;
  out.tied.master_label = master_label;
  return out;
}

MovingMeshConstraintMetadata make_geometric_continuity_metadata(
    std::string name,
    label_t boundary_label,
    int geometry_order)
{
  MovingMeshConstraintMetadata out;
  out.name = std::move(name);
  out.kind = MovingConstraintKind::GeometricContinuity;
  out.dependencies = MeshConstraintRevisionDependencies::topology_labels();
  out.dependencies.geometry = true;
  out.dependencies.active_configuration = true;
  out.continuity.boundary_label = boundary_label;
  out.continuity.geometry_order = geometry_order;
  return out;
}

} // namespace constraints
} // namespace svmp
