/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Application/Core/FreeSurfaceSemiImplicitReference.h"

#include "Application/Core/LevelSetPdeVelocityExtension.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/SystemState.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace application::core {
namespace {

namespace ls = svmp::FE::level_set;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::INVALID_FIELD_ID;

[[noreturn]] void rejectSemiImplicit(const std::string& reason)
{
  throw std::runtime_error(
      "[svMultiPhysics::Application] Surface_tension_semi_implicit="
      "NormalIncrement " +
      reason);
}

bool allRanks(bool local, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
    int value = local ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MIN, comm.native());
    return global != 0;
  }
#else
  (void)comm;
#endif
  return local;
}

std::string referenceFieldName()
{
  return std::string(svmp::Physics::formulations::navier_stokes::
                         kFreeSurfaceSemiImplicitReferenceVelocityFieldName);
}

// The level set must move with the velocity whose increment the term
// linearizes.  Returns an empty string when it does, else the reason.
std::string transportMismatch(const FreeSurfaceSemiImplicitTransport& transport,
                              const std::string& velocity_name)
{
  // A velocity extension (prescribed coupling, or an algebraic unknown with
  // monolithic coupling) qualifies only when it equals the fluid velocity on
  // every vertex of the retained interface cells at the refresh, which the
  // PDE extensions guarantee by construction.
  if (!transport.extension_method.empty() ||
      !transport.extension_source_velocity_field_name.empty()) {
    if (transport.extension_method.empty()) {
      return "advects the level set with an extension of unknown method";
    }
    if (!pdeVelocityExtensionOperatorFromToken(transport.extension_method)
             .has_value()) {
      return "advects the level set with the '" + transport.extension_method +
             "' extension, whose frozen algebraic map is not the fluid "
             "velocity of the refreshed iterate; use coupled_field or a PDE "
             "extension (pde_harmonic, pde_normal)";
    }
    if (transport.extension_source_velocity_field_name != velocity_name) {
      return "extends velocity '" +
             transport.extension_source_velocity_field_name +
             "', not the free-surface velocity '" + velocity_name + "'";
    }
    return {};
  }
  switch (transport.velocity_source) {
  case ls::LevelSetVelocitySource::CoupledField:
    if (transport.advection_velocity_field_name == velocity_name) {
      return {};
    }
    return "advects the level set with coupled field '" +
           transport.advection_velocity_field_name +
           "', not the free-surface velocity '" + velocity_name + "'";
  case ls::LevelSetVelocitySource::PrescribedData:
    return "advects the level set with prescribed data '" +
           transport.advection_velocity_field_name +
           "' that is not an extension of the fluid velocity";
  case ls::LevelSetVelocitySource::ConstantVector:
    return "advects the level set with a constant vector";
  case ls::LevelSetVelocitySource::MaterialInterfacePhasePair:
    return "advects the level set with a material-interface phase pair";
  }
  return "uses an unsupported level-set velocity source";
}

} // namespace

bool hasFreeSurfaceSemiImplicitReference(
    const svmp::FE::systems::FESystem& system)
{
  return system.findFieldByName(referenceFieldName()) != INVALID_FIELD_ID;
}

std::optional<FreeSurfaceSemiImplicitReferenceBinding>
bindFreeSurfaceSemiImplicitReference(
    const svmp::FE::systems::FESystem& system,
    std::span<const FreeSurfaceSemiImplicitTransport> transports,
    bool transient_solve,
    bool outer_fixed_point,
    const svmp::MeshComm& comm)
{
  const auto reference_name = referenceFieldName();
  const auto reference_field = system.findFieldByName(reference_name);
  if (reference_field == INVALID_FIELD_ID) {
    return std::nullopt;
  }
  if (!transient_solve) {
    rejectSemiImplicit(
        "requires a transient solve; the term is defined by the effective "
        "time step of the time integrator");
  }
  if (!outer_fixed_point) {
    rejectSemiImplicit(
        "requires the generated-state outer fixed point, which refreshes "
        "the reference velocity with the interface geometry");
  }
  const auto& reference_record = system.fieldRecord(reference_field);
  if (reference_record.source_kind !=
          svmp::FE::systems::FieldSourceKind::PrescribedData ||
      system.fieldParticipatesInUnknownVector(reference_field)) {
    rejectSemiImplicit("requires prescribed reference field '" +
                       reference_name + "'");
  }

  // The owning module declares one discrete functional per unfitted
  // SurfaceStress/KAG free surface; all of them name the same velocity.
  FieldId velocity_field = INVALID_FIELD_ID;
  std::set<FieldId> level_set_fields;
  for (const auto& declaration :
       system.freeSurfaceDiscreteFunctionalDeclarations()) {
    if (declaration.velocity_field == INVALID_FIELD_ID) {
      continue;
    }
    if (velocity_field != INVALID_FIELD_ID &&
        velocity_field != declaration.velocity_field) {
      rejectSemiImplicit(
          "requires all unfitted free surfaces to share one velocity field");
    }
    velocity_field = declaration.velocity_field;
    if (declaration.level_set_field != INVALID_FIELD_ID) {
      level_set_fields.insert(declaration.level_set_field);
    }
  }
  if (velocity_field == INVALID_FIELD_ID || level_set_fields.empty()) {
    rejectSemiImplicit(
        "found no declared unfitted free-surface velocity and level set");
  }
  const auto& velocity_record = system.fieldRecord(velocity_field);
  if (!system.fieldParticipatesInUnknownVector(velocity_field) ||
      velocity_record.components != reference_record.components) {
    rejectSemiImplicit("requires the free-surface velocity '" +
                       velocity_record.name +
                       "' to be an unknown with the reference layout");
  }

  for (const auto level_set_field : level_set_fields) {
    const auto& level_set_name = system.fieldRecord(level_set_field).name;
    bool transported = false;
    for (const auto& transport : transports) {
      if (transport.level_set_field_name != level_set_name) {
        continue;
      }
      transported = true;
      const auto mismatch = transportMismatch(transport, velocity_record.name);
      if (!mismatch.empty()) {
        rejectSemiImplicit("cannot use level set '" + level_set_name +
                           "': its level_set equation " + mismatch);
      }
    }
    if (!transported) {
      rejectSemiImplicit("requires level set '" + level_set_name +
                         "' to be transported by a level_set equation");
    }
  }

  // u_ref and u share the velocity space; their field DOF maps must agree so
  // that the refresh is an exact coefficient copy.
  const auto& velocity_map =
      system.fieldDofHandler(velocity_field).getDofMap();
  const auto& reference_map =
      system.fieldDofHandler(reference_field).getDofMap();
  const auto velocity_offsets = velocity_map.getOffsets();
  const auto reference_offsets = reference_map.getOffsets();
  const auto velocity_indices = velocity_map.getDofIndices();
  const auto reference_indices = reference_map.getDofIndices();
  const bool local_layout_matches =
      velocity_map.getNumDofs() == reference_map.getNumDofs() &&
      std::equal(velocity_offsets.begin(), velocity_offsets.end(),
                 reference_offsets.begin(), reference_offsets.end()) &&
      std::equal(velocity_indices.begin(), velocity_indices.end(),
                 reference_indices.begin(), reference_indices.end());
  if (!allRanks(local_layout_matches, comm)) {
    rejectSemiImplicit("requires the reference field '" + reference_name +
                       "' to share the DOF map of velocity '" +
                       velocity_record.name + "'");
  }

  FreeSurfaceSemiImplicitReferenceBinding binding;
  binding.reference_field = reference_field;
  binding.velocity_field = velocity_field;
  binding.velocity_field_name = velocity_record.name;
  binding.reference_coefficient_count = reference_map.getNumDofs();
  binding.field_dofs.assign(velocity_indices.begin(), velocity_indices.end());
  std::sort(binding.field_dofs.begin(), binding.field_dofs.end());
  binding.field_dofs.erase(
      std::unique(binding.field_dofs.begin(), binding.field_dofs.end()),
      binding.field_dofs.end());
  const auto offset = system.fieldDofOffset(velocity_field);
  binding.state_dofs.reserve(binding.field_dofs.size());
  for (const auto dof : binding.field_dofs) {
    if (dof < 0 || dof >= binding.reference_coefficient_count) {
      rejectSemiImplicit("found a velocity DOF outside the field layout");
    }
    binding.state_dofs.push_back(offset + dof);
  }
  return binding;
}

void refreshFreeSurfaceSemiImplicitReference(
    svmp::FE::systems::FESystem& system,
    const FreeSurfaceSemiImplicitReferenceBinding& binding,
    const svmp::FE::systems::SystemStateView& state)
{
  std::vector<svmp::FE::Real> values(binding.state_dofs.size(),
                                     svmp::FE::Real{0.0});
  if (state.u_vector != nullptr) {
    // Read through the backend view, as assembly does, so ghosted and
    // permuted layouts resolve exactly as for the velocity unknown.
    auto* vector =
        const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    const auto view = vector->createAssemblyView();
    if (!view) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Semi-implicit reference refresh "
          "could not read the velocity state.");
    }
    view->getVectorEntries(
        std::span<const GlobalIndex>(binding.state_dofs.data(),
                                     binding.state_dofs.size()),
        std::span<svmp::FE::Real>(values.data(), values.size()));
  } else {
    for (std::size_t i = 0; i < binding.state_dofs.size(); ++i) {
      const auto index = static_cast<std::size_t>(binding.state_dofs[i]);
      if (index >= state.u.size()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Semi-implicit reference refresh "
            "received a state smaller than the FE layout.");
      }
      values[i] = state.u[index];
    }
  }
  std::vector<svmp::FE::Real> coefficients(
      static_cast<std::size_t>(binding.reference_coefficient_count),
      svmp::FE::Real{0.0});
  for (std::size_t i = 0; i < binding.field_dofs.size(); ++i) {
    coefficients[static_cast<std::size_t>(binding.field_dofs[i])] = values[i];
  }
  system.setPrescribedFieldCoefficients(
      binding.reference_field,
      std::span<const svmp::FE::Real>(coefficients.data(),
                                      coefficients.size()));
}

} // namespace application::core
