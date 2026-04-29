/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGGRAPH_H
#define SVMP_FE_COUPLING_COUPLINGGRAPH_H

/**
 * @file CouplingGraph.h
 * @brief Declaration and finalized setup graph for coupling validation.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingDeclaration.h"

#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingGraph {
public:
    [[nodiscard]] CouplingValidationResult buildDeclarationGraph(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations);

    [[nodiscard]] CouplingValidationResult buildFinalizedGraph(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations,
        std::span<const CouplingFormAnalysisMetadata> installed_forms);

    [[nodiscard]] const std::vector<CouplingContractDeclaration>& declarations() const noexcept;
    [[nodiscard]] const std::vector<CouplingFormAnalysisMetadata>&
    installedFormMetadata() const noexcept;

private:
    std::vector<CouplingContractDeclaration> declarations_{};
    std::vector<CouplingFormAnalysisMetadata> installed_forms_{};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGGRAPH_H
