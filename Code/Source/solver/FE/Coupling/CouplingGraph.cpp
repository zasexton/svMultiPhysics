/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingGraph.h"

namespace svmp {
namespace FE {
namespace coupling {

CouplingValidationResult CouplingGraph::buildDeclarationGraph(
    const CouplingContext&,
    std::span<const CouplingContractDeclaration> declarations)
{
    declarations_.assign(declarations.begin(), declarations.end());

    CouplingValidationResult result;
    for (const auto& declaration : declarations_) {
        result.append(validateContractDeclarationShape(declaration));
    }
    return result;
}

const std::vector<CouplingContractDeclaration>& CouplingGraph::declarations() const noexcept
{
    return declarations_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
