/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_MONOLITHICCOUPLINGBUILDER_H
#define SVMP_FE_COUPLING_MONOLITHICCOUPLINGBUILDER_H

/**
 * @file MonolithicCouplingBuilder.h
 * @brief Pre-setup lifecycle owner for monolithic coupling contributions.
 */

#include "Coupling/CouplingContract.h"

#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class MonolithicCouplingInstallContext {
public:
    MonolithicCouplingInstallContext() = default;
};

class MonolithicCouplingBuilder {
public:
    [[nodiscard]] CouplingValidationResult validateDeclarations(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations) const;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_MONOLITHICCOUPLINGBUILDER_H
