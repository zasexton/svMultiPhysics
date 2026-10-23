/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_TWO_FLUID_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_TWO_FLUID_MODULE_H

/**
 * @file IncompressibleTwoFluidModule.h
 * @brief Monolithic two-field-pair incompressible material-interface module.
 *
 * Its options (IncompressibleTwoFluidOptions) are declared with the other
 * option types in IncompressibleNavierStokesVMSModule.h.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/** Validate parser-visible semantics shared by preflight and registration. */
void validateIncompressibleTwoFluidConfigurationSemantics(
    const IncompressibleTwoFluidOptions& options);

class IncompressibleTwoFluidModule final : public PhysicsModule {
public:
    IncompressibleTwoFluidModule(
        std::shared_ptr<const FE::spaces::FunctionSpace>
            negative_velocity_space,
        std::shared_ptr<const FE::spaces::FunctionSpace>
            negative_pressure_space,
        std::shared_ptr<const FE::spaces::FunctionSpace>
            positive_velocity_space,
        std::shared_ptr<const FE::spaces::FunctionSpace>
            positive_pressure_space,
        IncompressibleTwoFluidOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;
    void applyInitialConditions(const FE::systems::FESystem& system,
                                FE::backends::GenericVector& u0) const override;

    [[nodiscard]] std::optional<EffectiveConfigurationArtifact>
    effectiveConfigurationArtifact() const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace>
        negative_velocity_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace>
        negative_pressure_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace>
        positive_velocity_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace>
        positive_pressure_space_{};
    IncompressibleTwoFluidOptions options_{};
    mutable std::optional<EffectiveConfigurationArtifact>
        effective_configuration_artifact_{};
};

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_TWO_FLUID_MODULE_H
