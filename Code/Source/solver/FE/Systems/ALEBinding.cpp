/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ALEBinding.h"

#include "Forms/FormCompiler.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FieldRegistry.h"
#include "Systems/FormsInstaller.h"
#include "Systems/GeometricNonlinearity.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {
namespace {

void validateVectorSpace(const std::shared_ptr<const spaces::FunctionSpace>& space,
                         int dimension,
                         const char* label)
{
    if (!space) {
        throw std::invalid_argument(std::string("resolveALEBinding: null ") + label);
    }
    if (space->field_type() != FieldType::Vector) {
        throw std::invalid_argument(
            std::string("resolveALEBinding: ") + label + " must be vector-valued");
    }
    if (space->value_dimension() != dimension) {
        throw std::invalid_argument(
            std::string("resolveALEBinding: ") + label +
            " dimension must match the moving-domain spatial dimension");
    }
}

FieldId findBoundOrNamedField(FESystem& system,
                              MeshMotionFieldRole role,
                              const std::string& field_name)
{
    if (const auto bound = system.meshMotionField(role)) {
        return *bound;
    }
    return system.findFieldByName(field_name);
}

FieldId resolveCoupledDisplacementField(FESystem& system,
                                        const std::string& field_name)
{
    const auto bound = system.meshMotionField(MeshMotionFieldRole::Displacement);
    const auto named = system.findFieldByName(field_name);
    if (bound && named != INVALID_FIELD_ID && *bound != named) {
        throw std::invalid_argument(
            "resolveALEBinding: coupled ALE mesh displacement field name '" +
            field_name + "' resolves to a different field than the bound "
            "mesh-motion displacement; use the same displacement field for "
            "the mesh-motion equation, ALE geometry, and derived mesh velocity");
    }
    if (bound) {
        return *bound;
    }
    return named;
}

void requireDerivedVelocityFromDisplacement(FESystem& system,
                                            FieldId velocity,
                                            FieldId displacement)
{
    const auto& rec = system.fieldRecord(velocity);
    if (rec.source_kind != FieldSourceKind::DerivedFromUnknown ||
        rec.derived.source_field != displacement ||
        rec.derived.role != DerivedFieldRole::TimeDerivative ||
        rec.derived.derivative_order != 1) {
        throw std::invalid_argument(
            "resolveALEBinding: coupled ALE requires mesh velocity field '" +
            rec.name + "' to be DerivedFromUnknown(mesh_displacement)");
    }
}

void enableCoupledGeometryPolicy(FESystem& system)
{
    auto policy = system.geometricNonlinearityPolicy();
    policy.enabled = true;
    policy.update_current_coordinates_on_trial = true;
    policy.rollback_geometry_on_line_search_reject = true;
    system.setGeometricNonlinearityPolicy(policy);
}

} // namespace

void ALEBinding::configureInstallOptions(FormInstallOptions& install) const
{
    if (!coupled()) {
        return;
    }
    if (mesh_displacement_field == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "ALEBinding::configureInstallOptions: coupled ALE requires a mesh displacement field");
    }

    install.compiler_options.geometry_sensitivity.mode =
        forms::GeometrySensitivityMode::MeshMotionUnknowns;
    install.compiler_options.geometry_sensitivity.mesh_motion_field =
        mesh_displacement_field;
    install.compiler_options.geometry_tangent_path =
        geometry_tangent_path;

    if (std::find(install.extra_trial_fields.begin(),
                  install.extra_trial_fields.end(),
                  mesh_displacement_field) == install.extra_trial_fields.end()) {
        install.extra_trial_fields.push_back(mesh_displacement_field);
    }
}

ALEBinding resolveALEBinding(FESystem& system, ALEBindingOptions options)
{
    ALEBinding binding;
    binding.enabled = options.enabled;
    binding.mesh_velocity_source = options.mesh_velocity_source;
    binding.geometry_tangent_path = options.geometry_tangent_path;

    if (!options.enabled) {
        return binding;
    }
    if (options.dimension < 1 || options.dimension > 3) {
        throw std::invalid_argument(
            "resolveALEBinding: moving-domain spatial dimension must be in [1, 3]");
    }
    if (options.mesh_velocity_field_name.empty()) {
        throw std::invalid_argument(
            "resolveALEBinding: mesh velocity field name must be non-empty");
    }

    validateVectorSpace(options.mesh_velocity_space,
                        options.dimension,
                        "mesh_velocity_space");

    FieldId mesh_velocity_id =
        findBoundOrNamedField(system,
                              MeshMotionFieldRole::Velocity,
                              options.mesh_velocity_field_name);

    if (options.mesh_velocity_source == ALEMeshVelocitySource::CoupledDisplacement) {
        if (options.mesh_displacement_field_name.empty()) {
            throw std::invalid_argument(
                "resolveALEBinding: coupled ALE requires a non-empty mesh displacement field name");
        }

        FieldId mesh_displacement_id =
            resolveCoupledDisplacementField(system,
                                            options.mesh_displacement_field_name);
        if (mesh_displacement_id != INVALID_FIELD_ID) {
            system.bindMeshMotionField(MeshMotionFieldRole::Displacement,
                                       mesh_displacement_id);
        }

        if (mesh_displacement_id == INVALID_FIELD_ID &&
            options.auto_register_mesh_displacement_field) {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
            auto displacement_space = options.mesh_displacement_space
                                        ? std::move(options.mesh_displacement_space)
                                        : options.mesh_velocity_space;
            validateVectorSpace(displacement_space,
                                options.dimension,
                                "mesh_displacement_space");
            mesh_displacement_id = system.addMeshDisplacementUnknown(
                options.mesh_displacement_field_name,
                std::move(displacement_space),
                options.dimension);
#else
            throw std::invalid_argument(
                "resolveALEBinding: automatic mesh displacement registration "
                "requires Mesh-backed FE support");
#endif
        }

        if (mesh_displacement_id == INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "resolveALEBinding: coupled ALE requires a mesh displacement "
                "unknown bound as mesh-motion displacement or named '" +
                options.mesh_displacement_field_name + "'");
        }

        const auto& d_rec = system.fieldRecord(mesh_displacement_id);
        if (d_rec.source_kind != FieldSourceKind::Unknown) {
            throw std::invalid_argument(
                "resolveALEBinding: coupled ALE mesh displacement field '" +
                d_rec.name + "' must be an Unknown");
        }
        validateVectorSpace(d_rec.space,
                            options.dimension,
                            "mesh displacement field space");

        if (mesh_velocity_id == INVALID_FIELD_ID) {
            mesh_velocity_id = system.addDerivedMeshVelocityField(
                options.mesh_velocity_field_name,
                d_rec.space,
                mesh_displacement_id,
                options.dimension);
        } else {
            requireDerivedVelocityFromDisplacement(system,
                                                   mesh_velocity_id,
                                                   mesh_displacement_id);
        }

        system.bindMeshMotionField(MeshMotionFieldRole::Displacement,
                                   mesh_displacement_id);
        binding.mesh_displacement_field = mesh_displacement_id;
        enableCoupledGeometryPolicy(system);
    } else {
        if (mesh_velocity_id == INVALID_FIELD_ID) {
            if (!options.auto_register_mesh_velocity_field) {
                throw std::invalid_argument(
                    "resolveALEBinding: ALE is enabled but mesh velocity field '" +
                    options.mesh_velocity_field_name + "' is not registered");
            }
            mesh_velocity_id = system.addMeshMotionDataField(
                options.mesh_velocity_field_name,
                std::move(options.mesh_velocity_space),
                options.dimension);
        }

        const auto& w_rec = system.fieldRecord(mesh_velocity_id);
        if (w_rec.source_kind != FieldSourceKind::PrescribedData) {
            throw std::invalid_argument(
                "resolveALEBinding: prescribed ALE requires mesh velocity field '" +
                w_rec.name + "' to be registered as PrescribedData");
        }
        validateVectorSpace(w_rec.space,
                            options.dimension,
                            "mesh velocity field space");
    }

    system.bindMeshMotionField(MeshMotionFieldRole::Velocity, mesh_velocity_id);
    binding.mesh_velocity_field = mesh_velocity_id;
    return binding;
}

} // namespace systems
} // namespace FE
} // namespace svmp
