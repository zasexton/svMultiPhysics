#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief FE-system installer for level-set transport residuals.
 */

#include "LevelSet/LevelSetOptions.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

[[nodiscard]] systems::CoupledResidualKernels installLevelSetTransport(
    systems::FESystem& system,
    std::shared_ptr<const spaces::FunctionSpace> level_set_space,
    const LevelSetTransportOptions& options,
    const systems::FormInstallOptions& install_options = {});

struct LevelSetTransportSafetyResult {
    bool success{false};
    bool courant_satisfied{false};
    bool impermeable_boundaries_satisfied{false};
    Real maximum_courant{0.0};
    Real minimum_cell_length{0.0};
    Real maximum_speed{0.0};
    Real maximum_boundary_normal_velocity{0.0};
    Real maximum_boundary_normal_velocity_ratio{0.0};
    std::size_t cells_checked{0u};
    std::size_t impermeable_boundary_faces_checked{0u};
    int worst_boundary_marker{-1};
    std::string diagnostic{};
};

/**
 * @brief Evaluate the one-ring CFL contract and undeclared-wall flux contract.
 *
 * Boundary markers present in @p boundaries.inflow or @p boundaries.outflow
 * are excluded from the impermeable-wall check.  Every other mesh boundary is
 * checked.  Constant and registered prescribed/coupled velocity fields are
 * supported.
 */
[[nodiscard]] LevelSetTransportSafetyResult
evaluateLevelSetTransportSafety(
    const systems::FESystem& system,
    const LevelSetVelocityOptions& velocity,
    const LevelSetBoundaryOptions& boundaries,
    const LevelSetBoundPreservingOptions& options,
    const systems::SystemStateView& state,
    Real dt);

struct LevelSetBoundPreservingResult {
    bool success{false};
    bool applied{false};
    bool bounds_satisfied{false};
    bool sign_preservation_satisfied{false};
    Real observed_courant{0.0};
    Real previous_minimum{0.0};
    Real previous_maximum{0.0};
    Real candidate_minimum{0.0};
    Real candidate_maximum{0.0};
    Real limited_minimum{0.0};
    Real limited_maximum{0.0};
    Real maximum_unrelaxed_bound_violation{0.0};
    Real maximum_bound_violation{0.0};
    Real maximum_correction{0.0};
    std::size_t field_dofs{0u};
    std::size_t limited_dofs{0u};
    std::size_t positive_patch_sign_flips_prevented{0u};
    std::size_t negative_patch_sign_flips_prevented{0u};
    std::string diagnostic{};
};

/**
 * @brief Project a P1 H1 level-set candidate into previous one-ring bounds.
 *
 * Literal inflow values enlarge the bounds on their boundary DOFs.  MPI ranks
 * synchronize patch minima/maxima with the field DofHandler communicator.
 * The output is a complete system vector so callers can replace only after a
 * successful, globally consistent result.
 */
[[nodiscard]] LevelSetBoundPreservingResult
applyLevelSetBoundPreservingLimiter(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetBoundaryOptions& boundaries,
    const LevelSetBoundPreservingOptions& options,
    std::span<const Real> previous_solution,
    std::span<const Real> candidate_solution,
    Real observed_courant,
    std::vector<Real>& limited_solution);

} // namespace svmp::FE::level_set
