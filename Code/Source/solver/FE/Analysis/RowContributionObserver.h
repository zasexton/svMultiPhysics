/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ROW_CONTRIBUTION_OBSERVER_H
#define SVMP_FE_ANALYSIS_ROW_CONTRIBUTION_OBSERVER_H

/**
 * @file RowContributionObserver.h
 * @brief The contributions of diagnostic operators to the rows of one field
 *
 * A formulation states, as a RowContributionDeclaration in the system's
 * diagnostic registry, the diagnostic operators whose contributions to the
 * rows of one field are sampled during a nonlinear solve, and the names of
 * that field and of the field its rows couple to.
 */

#include "Systems/DiagnosticRegistry.h"

#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * The operators sampled, in order (a declared operator that is not installed
 * is reported as skipped), and the names of the row field and of the field
 * its rows couple to.
 */
struct RowContributionDeclaration {
    std::vector<systems::DiagnosticOperatorHandle> operators{};
    std::string row_field_name{};
    std::string coupling_field_name{};

    friend bool operator==(const RowContributionDeclaration&, const RowContributionDeclaration&) = default;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ROW_CONTRIBUTION_OBSERVER_H
