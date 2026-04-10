#ifndef SVMP_FE_SYSTEMS_AUXILIARY_QUADRATURE_LAYOUT_H
#define SVMP_FE_SYSTEMS_AUXILIARY_QUADRATURE_LAYOUT_H

#include "Core/Types.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class IMeshAccess;
}

namespace quadrature {
class QuadratureRule;
}

namespace spaces {
class FunctionSpace;
}

namespace systems {

[[nodiscard]] std::shared_ptr<const quadrature::QuadratureRule>
resolveAuxiliaryCellQuadratureRule(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id);

[[nodiscard]] std::size_t numAuxiliaryCellQuadraturePoints(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id);

[[nodiscard]] std::vector<std::size_t> buildAuxiliaryCellQuadratureOffsets(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    std::span<const std::size_t> covered_cells);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_QUADRATURE_LAYOUT_H
