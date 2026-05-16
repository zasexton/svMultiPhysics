#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Cell-local evaluator for scalar level-set finite-element fields.
 */

#include "Core/Types.h"
#include "Dofs/DofHandler.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <array>
#include <span>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetCellEvaluation {
    Real value{0.0};
    std::array<Real, 3> reference_gradient{{0.0, 0.0, 0.0}};
    int interpolation_order{0};
    int implicit_geometry_order{0};
};

class LevelSetCellEvaluator {
public:
    LevelSetCellEvaluator(const spaces::FunctionSpace& space,
                          const dofs::DofHandler& dof_handler,
                          std::span<const Real> field_coefficients);

    [[nodiscard]] int interpolationOrder(GlobalIndex cell_id) const noexcept;

    [[nodiscard]] std::vector<Real> gatherCellCoefficients(
        GlobalIndex cell_id) const;

    [[nodiscard]] LevelSetCellEvaluation evaluate(
        GlobalIndex cell_id,
        const std::array<Real, 3>& parent_coordinate) const;

private:
    [[nodiscard]] const std::vector<Real>& cachedCellCoefficients(
        GlobalIndex cell_id) const;

    const spaces::FunctionSpace* space_{nullptr};
    const dofs::DofHandler* dof_handler_{nullptr};
    std::span<const Real> field_coefficients_{};
    mutable bool cached_cell_valid_{false};
    mutable GlobalIndex cached_cell_id_{static_cast<GlobalIndex>(-1)};
    mutable std::vector<Real> cached_cell_coefficients_{};
};

[[nodiscard]] LevelSetCellEvaluator makeLevelSetCellEvaluator(
    const systems::FESystem& system,
    FieldId field,
    std::span<const Real> solution);

} // namespace svmp::FE::level_set
