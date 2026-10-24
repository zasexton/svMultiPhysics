/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/MeshBoundaryKinematics.h"

#include "Analysis/BoundaryConditionDescriptor.h"
#include "Systems/MeshNormalConstraintStageMeasurements.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

void declareMeshNormalKinematics(
    FESystem& system,
    const std::string& operator_tag,
    std::vector<MeshNormalBoundaryConstraintDeclaration> declarations,
    const std::function<MeshNormalKinematicBinding(int boundary_marker)>&
        binding_for)
{
    for (auto& declaration : declarations) {
        const auto marker = declaration.boundary_marker;
        const auto displacement_field =
            declaration.mesh_displacement_field;
        auto binding = binding_for(marker);
        system.declareMeshNormalBoundaryConstraint(
            std::move(declaration));
        system.bindMeshNormalBoundaryConstraintConsumer(
            displacement_field,
            marker,
            operator_tag,
            std::move(binding.mesh_row_source),
            std::move(binding.fluid_row_source));
        meshNormalConstraintStageMeasurements(system)
            .registerFittedALENormalOperatorStageMeasurement(
                displacement_field,
                marker,
                binding.prescribed_tangential_mesh_velocity);
    }
}

const FieldRecord& requireMeshDisplacementRecord(
    const FESystem& system,
    const ALEBinding& ale_binding,
    const std::string& missing_unknown_message,
    const std::string& missing_space_message)
{
    if (ale_binding.mesh_displacement_field == INVALID_FIELD_ID) {
        throw std::invalid_argument(missing_unknown_message);
    }
    const auto& rec = system.fieldRecord(ale_binding.mesh_displacement_field);
    if (!rec.space) {
        throw std::invalid_argument(missing_space_message);
    }
    return rec;
}

void declareMeshBoundaryRowDescriptors(
    FESystem& system,
    const std::string& operator_tag,
    const MeshBoundaryRows& rows,
    bool rows_installed)
{
    if (!rows_installed) {
        analysis::BoundaryConditionDescriptor descriptor;
        descriptor.primary_variable =
            analysis::VariableKey::field(rows.mesh_displacement_field);
        descriptor.boundary_marker = rows.boundary_marker;
        descriptor.trace_kind = analysis::TraceKind::TangentialComponent;
        descriptor.enforcement_kind =
            analysis::EnforcementKind::WeakConsistent;
        descriptor.source = rows.tangential_source;
        system.addBoundaryConditionDescriptor(
            std::move(descriptor), operator_tag);
        system.bindMeshTangentialBoundaryPolicyConsumer(
            rows.mesh_displacement_field,
            rows.boundary_marker,
            rows.tangential_policy,
            operator_tag,
            rows.tangential_source);
        return;
    }
    if (rows.normal_relation) {
        if (rows.velocity_field == INVALID_FIELD_ID) {
            throw std::invalid_argument(rows.missing_velocity_message);
        }
        analysis::BoundaryConditionDescriptor mesh_descriptor;
        mesh_descriptor.primary_variable =
            analysis::VariableKey::field(rows.mesh_displacement_field);
        mesh_descriptor.related_variables.push_back(
            analysis::VariableKey::field(rows.velocity_field));
        mesh_descriptor.boundary_marker = rows.boundary_marker;
        mesh_descriptor.trace_kind = analysis::TraceKind::NormalComponent;
        mesh_descriptor.enforcement_kind =
            analysis::EnforcementKind::WeakPenalty;
        mesh_descriptor.source = rows.mesh_normal_source;
        system.addBoundaryConditionDescriptor(
            std::move(mesh_descriptor), operator_tag);
        analysis::BoundaryConditionDescriptor fluid_descriptor;
        fluid_descriptor.primary_variable =
            analysis::VariableKey::field(rows.velocity_field);
        fluid_descriptor.related_variables.push_back(
            analysis::VariableKey::field(rows.mesh_displacement_field));
        fluid_descriptor.boundary_marker = rows.boundary_marker;
        fluid_descriptor.trace_kind = analysis::TraceKind::NormalComponent;
        // The fluid keeps its natural dynamic condition.
        fluid_descriptor.enforcement_kind =
            analysis::EnforcementKind::WeakConsistent;
        fluid_descriptor.source = rows.fluid_normal_source;
        system.addBoundaryConditionDescriptor(
            std::move(fluid_descriptor), operator_tag);
    }
    analysis::BoundaryConditionDescriptor descriptor;
    descriptor.primary_variable =
        analysis::VariableKey::field(rows.mesh_displacement_field);
    descriptor.boundary_marker = rows.boundary_marker;
    descriptor.trace_kind = analysis::TraceKind::TangentialComponent;
    descriptor.enforcement_kind =
        rows.tangential_prescribed
            ? analysis::EnforcementKind::WeakPenalty
            : analysis::EnforcementKind::WeakConsistent;
    descriptor.source = rows.tangential_source;
    system.addBoundaryConditionDescriptor(
        std::move(descriptor), operator_tag);
    system.bindMeshTangentialBoundaryPolicyConsumer(
        rows.mesh_displacement_field,
        rows.boundary_marker,
        rows.tangential_policy,
        operator_tag,
        rows.tangential_source);
}

} // namespace systems
} // namespace FE
} // namespace svmp
