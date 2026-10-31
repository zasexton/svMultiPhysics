/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_ALEBINDING_H
#define SVMP_FE_SYSTEMS_ALEBINDING_H

#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <memory>
#include <string>

namespace svmp {
namespace FE {

namespace spaces {
class FunctionSpace;
}

namespace systems {

class FESystem;
struct FormInstallOptions;

enum class ALEMeshVelocitySource {
    PrescribedData,
    CoupledDisplacement
};

struct ALEBindingOptions {
    bool enabled{false};
    int dimension{0};
    ALEMeshVelocitySource mesh_velocity_source{ALEMeshVelocitySource::PrescribedData};
    forms::GeometryTangentPath geometry_tangent_path{forms::GeometryTangentPath::SymbolicRequired};
    std::string mesh_velocity_field_name{"mesh_velocity"};
    std::string mesh_displacement_field_name{"mesh_displacement"};
    std::shared_ptr<const spaces::FunctionSpace> mesh_velocity_space{};
    std::shared_ptr<const spaces::FunctionSpace> mesh_displacement_space{};
    bool auto_register_mesh_velocity_field{true};
    bool auto_register_mesh_displacement_field{false};
};

struct ALEBinding {
    bool enabled{false};
    ALEMeshVelocitySource mesh_velocity_source{ALEMeshVelocitySource::PrescribedData};
    forms::GeometryTangentPath geometry_tangent_path{forms::GeometryTangentPath::SymbolicRequired};
    FieldId mesh_velocity_field{INVALID_FIELD_ID};
    FieldId mesh_displacement_field{INVALID_FIELD_ID};

    [[nodiscard]] bool coupled() const noexcept
    {
        return enabled && mesh_velocity_source == ALEMeshVelocitySource::CoupledDisplacement;
    }

    void configureInstallOptions(FormInstallOptions& install) const;
};

void validateALEBinding(const FESystem& system,
                        const ALEBindingOptions& options);

[[nodiscard]] ALEBinding resolveALEBinding(FESystem& system, ALEBindingOptions options);

/// A system's domain moving with a mesh displacement unknown (movingDomain()).
struct MovingDomain {
    FieldId displacement{INVALID_FIELD_ID};
    FieldId velocity{INVALID_FIELD_ID}; ///< the derived mesh velocity, dd/dt
    forms::GeometryTangentPath geometry_tangent_path{forms::GeometryTangentPath::SymbolicRequired};

    /// The coupled ALE binding of the displacement and its velocity.
    [[nodiscard]] ALEBinding binding() const noexcept;
};

struct MovingDomainOptions {
    forms::GeometryTangentPath geometry_tangent_path{forms::GeometryTangentPath::SymbolicRequired};
    std::string mesh_velocity_field_name{"mesh_velocity"};
};

/**
 * @brief Declares that the domain of @p system moves with the mesh
 *        displacement unknown @p displacement: x = X + d
 *
 * Stated once, before setup: the current configuration becomes the assembly
 * frame (plain dx, grad and normal() mean the moving domain), @p displacement
 * and its derived mesh velocity are bound as the system's mesh motion
 * (resolveALEBinding with a coupled displacement), and every later
 * installFormulation on the system differentiates its geometry in
 * @p displacement (MeshMotionUnknowns), with the displacement added to its
 * trial fields, as ALEBinding::configureInstallOptions does for one install.
 * An install that keeps its geometry constant, such as a mesh-motion equation
 * on the reference mesh, sets FormInstallOptions::moving_domain_geometry to
 * false (FormulationBuilder::geometryConstant()).  Throws after setup, for a
 * second declaration, and for a displacement that is not a vector unknown.
 */
[[nodiscard]] MovingDomain movingDomain(FESystem& system, FieldId displacement, MovingDomainOptions options = {});

/// The moving domain declared on @p system, or null.
[[nodiscard]] const MovingDomain* findMovingDomain(const FESystem& system) noexcept;

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_ALEBINDING_H
