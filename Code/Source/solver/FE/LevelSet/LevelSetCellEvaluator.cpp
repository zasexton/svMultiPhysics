#include "LevelSet/LevelSetCellEvaluator.h"

#include "Basis/BasisTraits.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/SerendipityBasis.h"
#include "Elements/Element.h"

#include <cstddef>
#include <stdexcept>
#include <string>

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

[[nodiscard]] ElementType linearCornerElementType(ElementType type) noexcept
{
    switch (type) {
    case ElementType::Line2:
    case ElementType::Line3:
        return ElementType::Line2;
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return ElementType::Triangle3;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return ElementType::Quad4;
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return ElementType::Tetra4;
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return ElementType::Hex8;
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return ElementType::Wedge6;
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return ElementType::Pyramid5;
    default:
        return ElementType::Unknown;
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
    const auto linear_type = linearCornerElementType(space.element_type());
    if (linear_type != ElementType::Unknown) {
        linear_corner_basis_ =
            std::make_shared<basis::LagrangeBasis>(linear_type, 1);
    }
}

int LevelSetCellEvaluator::interpolationOrder(GlobalIndex cell_id) const noexcept
{
    return space_->polynomial_order(cell_id);
}

bool LevelSetCellEvaluator::usesCompleteTensorLagrangeBasis(
    GlobalIndex cell_id) const noexcept
{
    const auto& element =
        space_->getElement(space_->element_type(), cell_id);
    return basis::is_tensor_product(element.element_type()) &&
           dynamic_cast<const basis::LagrangeBasis*>(&element.basis()) != nullptr;
}

bool LevelSetCellEvaluator::usesTensorSerendipityBasis(
    GlobalIndex cell_id) const noexcept
{
    const auto& element =
        space_->getElement(space_->element_type(), cell_id);
    return basis::is_tensor_product(element.element_type()) &&
           dynamic_cast<const basis::SerendipityBasis*>(&element.basis()) !=
               nullptr;
}

std::vector<Real> LevelSetCellEvaluator::gatherCellCoefficients(
    GlobalIndex cell_id) const
{
    return cachedCellCoefficients(cell_id);
}

const std::vector<Real>& LevelSetCellEvaluator::cachedCellCoefficients(
    GlobalIndex cell_id) const
{
    if (cached_cell_valid_ && cached_cell_id_ == cell_id) {
        return cached_cell_coefficients_;
    }

    const auto dofs = dof_handler_->getCellDofs(cell_id);
    const auto expected = space_->dofs_per_element(cell_id);
    if (dofs.size() != expected) {
        throw std::invalid_argument(
            "level-set cell evaluator found a cell DOF count that does not match the field space");
    }

    cached_cell_coefficients_.clear();
    cached_cell_coefficients_.reserve(dofs.size());
    for (const auto dof : dofs) {
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= field_coefficients_.size()) {
            throw std::invalid_argument(
                "level-set cell evaluator found a cell DOF outside the coefficient span");
        }
        cached_cell_coefficients_.push_back(
            field_coefficients_[static_cast<std::size_t>(dof)]);
    }
    cached_cell_id_ = cell_id;
    cached_cell_valid_ = true;
    return cached_cell_coefficients_;
}

LevelSetCellEvaluation LevelSetCellEvaluator::evaluate(
    GlobalIndex cell_id,
    const std::array<Real, 3>& parent_coordinate) const
{
    try {
        const auto& coefficients = cachedCellCoefficients(cell_id);
        const auto point = toSpacePoint(parent_coordinate);
        const auto gradient = space_->evaluate_gradient(point, coefficients);

        LevelSetCellEvaluation evaluation;
        evaluation.value = space_->evaluate_scalar(point, coefficients);
        evaluation.reference_gradient = {{gradient[0], gradient[1], gradient[2]}};
        evaluation.interpolation_order = interpolationOrder(cell_id);
        evaluation.implicit_geometry_order = evaluation.interpolation_order;
        return evaluation;
    } catch (const std::exception& ex) {
        throw std::invalid_argument(
            "level-set cell evaluator could not evaluate cell " +
            std::to_string(cell_id) + ": " + ex.what());
    }
}

LevelSetCellEvaluation LevelSetCellEvaluator::evaluateLinearCorner(
    GlobalIndex cell_id,
    const std::array<Real, 3>& parent_coordinate) const
{
    try {
        if (!linear_corner_basis_) {
            throw std::invalid_argument(
                "the field element has no supported LinearCorner topology");
        }
        const auto& coefficients = cachedCellCoefficients(cell_id);
        const auto corner_count = linear_corner_basis_->size();
        if (coefficients.size() < corner_count) {
            throw std::invalid_argument(
                "the cell has fewer coefficients than corner vertices");
        }

        const auto point = toSpacePoint(parent_coordinate);
        std::vector<Real> values;
        std::vector<basis::Gradient> gradients;
        linear_corner_basis_->evaluate_values(point, values);
        linear_corner_basis_->evaluate_gradients(point, gradients);
        if (values.size() != corner_count || gradients.size() != corner_count) {
            throw std::invalid_argument(
                "the LinearCorner basis returned an inconsistent size");
        }

        LevelSetCellEvaluation evaluation;
        for (std::size_t i = 0; i < corner_count; ++i) {
            evaluation.value += coefficients[i] * values[i];
            for (std::size_t d = 0; d < 3u; ++d) {
                evaluation.reference_gradient[d] +=
                    coefficients[i] * gradients[i][d];
            }
        }
        evaluation.interpolation_order = interpolationOrder(cell_id);
        evaluation.implicit_geometry_order = 1;
        return evaluation;
    } catch (const std::exception& ex) {
        throw std::invalid_argument(
            "level-set LinearCorner evaluator could not evaluate cell " +
            std::to_string(cell_id) + ": " + ex.what());
    }
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
    if (field_dofs.getEntityDofMap() == nullptr) {
        throw std::invalid_argument(
            "level-set cell evaluator requires a field entity DOF map for cell-local evaluation");
    }
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
