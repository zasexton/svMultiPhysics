/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_ALEBINDING_H
#define SVMP_FE_SYSTEMS_ALEBINDING_H

#include "Core/Types.h"

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
    FieldId mesh_velocity_field{INVALID_FIELD_ID};
    FieldId mesh_displacement_field{INVALID_FIELD_ID};

    [[nodiscard]] bool coupled() const noexcept
    {
        return enabled && mesh_velocity_source == ALEMeshVelocitySource::CoupledDisplacement;
    }

    void configureInstallOptions(FormInstallOptions& install) const;
};

[[nodiscard]] ALEBinding resolveALEBinding(FESystem& system, ALEBindingOptions options);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_ALEBINDING_H
