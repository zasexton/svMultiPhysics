/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGPAYLOADDETANGLER_H
#define SVMP_FE_COUPLING_COUPLINGPAYLOADDETANGLER_H

/**
 * @file CouplingPayloadDetangler.h
 * @brief Forms-backed monolithic-to-partitioned payload extraction.
 */

#include "Coupling/CouplingDeclaration.h"

#include <span>
#include <string>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingPayloadDetangler {
public:
    [[nodiscard]] CouplingPayloadExtractionResult extract(
        const CouplingContext& ctx,
        std::span<const CouplingFormContribution> contributions,
        std::span<const CouplingPayloadExtractionRequest> requests,
        const std::string& contract_name) const;
};

[[nodiscard]] CouplingDiagnostic
diagnosticFromPayloadExtraction(
    const CouplingPayloadExtractionDiagnostic& diagnostic);

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGPAYLOADDETANGLER_H
