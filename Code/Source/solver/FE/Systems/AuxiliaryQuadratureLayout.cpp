#include "Systems/AuxiliaryQuadratureLayout.h"

#include "Assembly/Assembler.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"

namespace svmp {
namespace FE {
namespace systems {

std::shared_ptr<const quadrature::QuadratureRule>
resolveAuxiliaryCellQuadratureRule(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id)
{
    const auto cell_type = mesh.getCellType(cell_id);
    const auto& element = space.getElement(cell_type, cell_id);

    const int basis_order = element.polynomial_order();
    if (basis_order <= 1 && cell_type == ElementType::Tetra4) {
        const auto legacy_modifier =
            quadrature::QuadratureFactory::default_legacy_modifier(cell_type);
        return quadrature::QuadratureFactory::create_legacy_compatible(
            cell_type, legacy_modifier);
    }

    auto quad_rule = element.quadrature();
    if (!quad_rule) {
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            basis_order, false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }
    return quad_rule;
}

std::size_t numAuxiliaryCellQuadraturePoints(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id)
{
    const auto quad_rule = resolveAuxiliaryCellQuadratureRule(mesh, space, cell_id);
    return quad_rule ? quad_rule->num_points() : 0u;
}

std::vector<std::size_t> buildAuxiliaryCellQuadratureOffsets(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    std::span<const std::size_t> covered_cells)
{
    std::vector<std::size_t> offsets;
    offsets.reserve(covered_cells.size() + 1u);
    offsets.push_back(0u);
    for (const auto cell : covered_cells) {
        offsets.push_back(
            offsets.back() +
            numAuxiliaryCellQuadraturePoints(mesh, space,
                                             static_cast<GlobalIndex>(cell)));
    }
    return offsets;
}

} // namespace systems
} // namespace FE
} // namespace svmp
