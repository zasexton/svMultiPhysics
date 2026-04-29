/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingDeclaration.h"

namespace svmp {
namespace FE {
namespace coupling {

CouplingValidationResult validateContractDeclarationShape(
    const CouplingContractDeclaration& declaration)
{
    CouplingValidationResult result;
    if (declaration.contract_type.empty()) {
        result.addError("coupling contract declaration requires a contract type");
    }
    if (declaration.contract_name.empty()) {
        result.addError("coupling contract declaration requires a configured contract name");
    }
    return result;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
