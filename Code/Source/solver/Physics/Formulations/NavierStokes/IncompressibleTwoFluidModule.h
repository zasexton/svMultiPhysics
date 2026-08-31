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

struct IncompressibleTwoFluidPhaseOptions {
    std::string velocity_field_name{};
    std::string pressure_field_name{};
    FE::Real density{1.0};
    FE::Real viscosity{1.0};
    std::vector<IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        velocity_dirichlet{};
};

/**
 * @brief Qualified initial envelope for a sharp incompressible two-fluid pair
 *
 * The current owner intentionally admits only fixed-Eulerian affine C0 P1
 * Triangle3/Tetra4 spaces, constant phase materials, one shared generated
 * interface, CutVolume phase restriction, phasewise VMS/PSPG and ghost
 * stabilization, and phasewise small-cut aggregation. Compressible gas,
 * contact, phase change, turbulence, variable material laws, and moving-mesh
 * combinations remain outside this contract.
 */
struct IncompressibleTwoFluidOptions {
    IncompressibleTwoFluidPhaseOptions negative_phase{
        .velocity_field_name = "u_negative",
        .pressure_field_name = "p_negative",
        .density = 1000.0,
        .viscosity = 0.001,
    };
    IncompressibleTwoFluidPhaseOptions positive_phase{
        .velocity_field_name = "u_positive",
        .pressure_field_name = "p_positive",
        .density = 1.0,
        .viscosity = 0.00001,
    };

    std::string operator_tag{"equations"};
    std::string level_set_field_name{"level_set"};
    std::string generated_interface_domain_id{"two_fluid_interface"};
    std::string generated_interface_geometry{"LinearCorner"};
    std::string geometry_tangent_policy{"RefreshedFrozenQuadrature"};
    int interface_marker{-1};
    FE::Real level_set_isovalue{0.0};

    FE::Real surface_tension{0.0};
    FE::Real interface_nitsche_gamma{20.0};
    bool include_transient_interface_penalty{true};

    std::array<FE::Real, 3> body_force{0.0, 0.0, 0.0};
    bool enable_convection{true};
    bool enable_vms{true};
    FE::Real ct_m{1.0};
    FE::Real ct_c{36.0};
    FE::Real stabilization_epsilon{1.0e-12};

    FE::Real pressure_gradient_penalty{1.0};
    bool use_cut_metadata_scale{false};
    std::optional<FE::Real> cut_metadata_scale_cap{};
    IncompressibleNavierStokesVMSOptions::FreeSurfaceSmallCutAggregationGuards
        small_cut_aggregation_guards{};
    core::PhysicsJITPolicy jit_policy{};
};

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
