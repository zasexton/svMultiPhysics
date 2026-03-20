/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORM_CONTRIBUTION_LOWERER_H
#define SVMP_FE_ANALYSIS_FORM_CONTRIBUTION_LOWERER_H

/**
 * @file FormContributionLowerer.h
 * @brief Lower FormulationRecord into normalized ContributionDescriptors
 *
 * @see ContributionDescriptor for the target type
 * @see FormulationRecord for the source type
 */

#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormulationRecord.h"

#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Lower a FormulationRecord into one or more ContributionDescriptors
 *
 * For each block in the formulation, analyzes the FormExpr DAG to determine
 * the block role (diagonal, off-diagonal, constraint, stabilization) and
 * operator traits (symmetric, PSD, second-order, etc.).
 *
 * @param rec  The formulation record (must have residual_expr populated)
 * @return     One ContributionDescriptor per block
 */
[[nodiscard]] std::vector<ContributionDescriptor>
lowerFormulation(const FormulationRecord& rec);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORM_CONTRIBUTION_LOWERER_H
