#include "LevelSet/LevelSetCellEvaluator.h"

#include <cstddef>
#include <stdexcept>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] spaces::FunctionSpace::Value toSpacePoint(
    const std::array<Real, 3>& parent_coordinate) noexcept
{
    spaces::FunctionSpace::Value point{};
    point[0] = parent_coordinate[0];
    point[1] = parent_coordinate[1];
    point[2] = parent_coordinate[2];
    return point;
}

void validateScalarC0Space(const spaces::FunctionSpace& space)
{
    if (space.field_type() != FieldType::Scalar ||
        space.value_dimension() != 1 ||
        space.continuity() != Continuity::C0) {
        throw std::invalid_argument(
            "level-set cell evaluator requires a scalar C0 finite-element space");
    }
}

} // namespace

LevelSetCellEvaluator::LevelSetCellEvaluator(
    const spaces::FunctionSpace& space,
    const dofs::DofHandler& dof_handler,
    std::span<const Real> field_coefficients)
    : space_(&space)
    , dof_handler_(&dof_handler)
    , field_coefficients_(field_coefficients)
{
    validateScalarC0Space(space);
    if (dof_handler.getNumDofs() < 0 ||
        static_cast<std::size_t>(dof_handler.getNumDofs()) >
            field_coefficients_.size()) {
        throw std::invalid_argument(
            "level-set cell evaluator received too few field coefficients");
    }
}

int LevelSetCellEvaluator::interpolationOrder(GlobalIndex cell_id) const noexcept
{
    return space_->polynomial_order(cell_id);
}

std::vector<Real> LevelSetCellEvaluator::gatherCellCoefficients(
    GlobalIndex cell_id) const
{
    const auto dofs = dof_handler_->getCellDofs(cell_id);
    const auto expected = space_->dofs_per_element(cell_id);
    if (dofs.size() != expected) {
        throw std::invalid_argument(
            "level-set cell evaluator found a cell DOF count that does not match the field space");
    }

    std::vector<Real> coefficients;
    coefficients.reserve(dofs.size());
    for (const auto dof : dofs) {
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= field_coefficients_.size()) {
            throw std::invalid_argument(
                "level-set cell evaluator found a cell DOF outside the coefficient span");
        }
        coefficients.push_back(field_coefficients_[static_cast<std::size_t>(dof)]);
    }
    return coefficients;
}

LevelSetCellEvaluation LevelSetCellEvaluator::evaluate(
    GlobalIndex cell_id,
    const std::array<Real, 3>& parent_coordinate) const
{
    const auto coefficients = gatherCellCoefficients(cell_id);
    const auto point = toSpacePoint(parent_coordinate);
    const auto gradient = space_->evaluate_gradient(point, coefficients);

    LevelSetCellEvaluation evaluation;
    evaluation.value = space_->evaluate_scalar(point, coefficients);
    evaluation.reference_gradient = {{gradient[0], gradient[1], gradient[2]}};
    evaluation.interpolation_order = interpolationOrder(cell_id);
    evaluation.implicit_geometry_order = evaluation.interpolation_order;
    return evaluation;
}

LevelSetCellEvaluator makeLevelSetCellEvaluator(
    const systems::FESystem& system,
    FieldId field,
    std::span<const Real> solution)
{
    const auto& record = system.fieldRecord(field);
    if (record.components != 1 || !record.space) {
        throw std::invalid_argument(
            "level-set cell evaluator requires a registered scalar field");
    }
    validateScalarC0Space(*record.space);

    const auto& field_dofs = system.fieldDofHandler(field);
    const auto n_field_dofs =
        static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set cell evaluator received an incompatible system solution span");
    }
    return LevelSetCellEvaluator(
        *record.space,
        field_dofs,
        solution.subspan(offset, n_field_dofs));
}

} // namespace svmp::FE::level_set
