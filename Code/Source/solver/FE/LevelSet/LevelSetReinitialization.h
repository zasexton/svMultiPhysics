#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Signed-distance repair utilities for level-set fields.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"
#include "LevelSet/LevelSetOptions.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetSignedDistanceRepairResult {
    bool success{false};
    LevelSetReinitializationMethod method{LevelSetReinitializationMethod::Projection};
    std::size_t repaired_dofs{0};
    std::size_t interface_fragments{0};
    std::size_t cut_cells{0};
    Real max_abs_update{0.0};
    Real max_distance{0.0};
    std::string diagnostic{};
};

/**
 * Repair nodal level-set coefficients by projecting mesh vertices to generated
 * linear interface primitives and preserving the original coefficient signs.
 *
 * Supported cuts are linear Line2/Line3, Triangle3/Triangle6, Quad4/Quad8/Quad9,
 * and Tetra4/Tetra10 corner cuts. Other element types are skipped. This utility
 * does not solve a Hamilton-Jacobi reinitialization PDE and does not reconstruct
 * higher-order curved signed-distance fields.
 */
[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients);

[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_solution,
    std::vector<Real>& repaired_solution);

} // namespace svmp::FE::level_set
