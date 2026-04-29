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
    for (std::size_t i = 0; i < declarations_.size(); ++i) {
        const auto& declaration = declarations_[i];
        result.append(validateContractDeclarationShape(declaration));
        for (std::size_t j = i + 1u; j < declarations_.size(); ++j) {
            if (!declaration.contract_name.empty() &&
                declaration.contract_name == declarations_[j].contract_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message = "duplicate coupling contract instance name",
                });
            }
        }
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
