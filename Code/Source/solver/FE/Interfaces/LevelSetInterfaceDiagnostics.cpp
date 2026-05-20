/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/LevelSetInterfaceDiagnostics.h"

namespace svmp {
namespace FE {
namespace interfaces {

LevelSetInterfaceSummaryStatistics summarizeLevelSetInterface(
    const LevelSetInterfaceDomain& domain)
{
    const auto domain_summary = domain.summary();
    LevelSetInterfaceSummaryStatistics statistics;
    statistics.interface_marker = domain_summary.interface_marker;
    statistics.fragment_count = domain_summary.fragment_count;
    statistics.active_fragment_count = domain_summary.active_fragment_count;
    statistics.degenerate_fragment_count = domain_summary.degenerate_fragment_count;
    statistics.quadrature_point_count = domain_summary.quadrature_point_count;
    statistics.volume_quadrature_point_count =
        domain_summary.volume_quadrature_point_count;
    statistics.total_quadrature_point_count =
        domain_summary.total_quadrature_point_count;
    statistics.cut_cell_count = domain.cutCells().size();
    statistics.total_interface_measure = domain_summary.measure;
    return statistics;
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
