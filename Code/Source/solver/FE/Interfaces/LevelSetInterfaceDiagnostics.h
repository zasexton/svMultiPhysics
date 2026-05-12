/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_LEVELSETINTERFACEDIAGNOSTICS_H
#define SVMP_FE_INTERFACES_LEVELSETINTERFACEDIAGNOSTICS_H

/**
 * @file LevelSetInterfaceDiagnostics.h
 * @brief Summary diagnostics for generated level-set interface domains.
 */

#include "Interfaces/LevelSetInterfaceDomain.h"

#include <cstddef>

namespace svmp {
namespace FE {
namespace interfaces {

struct LevelSetInterfaceSummaryStatistics {
    int interface_marker{-1};
    std::size_t fragment_count{0};
    std::size_t active_fragment_count{0};
    std::size_t degenerate_fragment_count{0};
    std::size_t quadrature_point_count{0};
    std::size_t cut_cell_count{0};
    Real total_interface_measure{0.0};
    bool enclosed_measure_available{false};
    Real enclosed_measure{0.0};
};

[[nodiscard]] LevelSetInterfaceSummaryStatistics summarizeLevelSetInterface(
    const LevelSetInterfaceDomain& domain);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_LEVELSETINTERFACEDIAGNOSTICS_H
