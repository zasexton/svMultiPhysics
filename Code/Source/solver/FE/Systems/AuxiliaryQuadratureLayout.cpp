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
    GlobalIndex cell_id, int minimum_order)
{
    const auto cell_type = mesh.getCellType(cell_id);
    const auto& element = space.getElement(cell_type, cell_id);

    return quadrature::QuadratureFactory::cell_rule(cell_type, element, minimum_order);
}

std::size_t numAuxiliaryCellQuadraturePoints(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id, int minimum_order)
{
    const auto quad_rule = resolveAuxiliaryCellQuadratureRule(mesh, space, cell_id, minimum_order);
    return quad_rule ? quad_rule->num_points() : 0u;
}

std::vector<std::size_t> buildAuxiliaryCellQuadratureOffsets(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    std::span<const std::size_t> covered_cells, int minimum_order)
{
    std::vector<std::size_t> offsets;
    offsets.reserve(covered_cells.size() + 1u);
    offsets.push_back(0u);
    for (const auto cell : covered_cells) {
        offsets.push_back(
            offsets.back() +
            numAuxiliaryCellQuadraturePoints(mesh, space,
                                             static_cast<GlobalIndex>(cell), minimum_order));
    }
    return offsets;
}

} // namespace systems
} // namespace FE
} // namespace svmp
