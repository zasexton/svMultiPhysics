#pragma once

#include "FE/Geometry/CutQuadrature.h"
#include "Mesh/Mesh.h"

#include <cstddef>
#include <string>
#include <vector>

namespace application {
namespace core {

std::vector<double> collectWetVolumeFractions(
    std::size_t n_cells,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

std::size_t writeWetVolumeFractionField(
    svmp::Mesh& mesh,
    const std::string& field_name,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

} // namespace core
} // namespace application
