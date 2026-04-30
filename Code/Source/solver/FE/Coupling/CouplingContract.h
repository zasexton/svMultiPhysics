/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGCONTRACT_H
#define SVMP_FE_COUPLING_COUPLINGCONTRACT_H

/**
 * @file CouplingContract.h
 * @brief Common FE-facing contract interface for coupling modules.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingDeclaration.h"

#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingFormBuilder;
class MonolithicCouplingInstallContext;

class CouplingContract {
public:
    virtual ~CouplingContract() = default;

    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual CouplingContractDeclaration declare() const = 0;

    virtual void validate(const CouplingContext& ctx) const;

    [[nodiscard]] virtual bool supportsMonolithicLowering() const;
    [[nodiscard]] virtual bool supportsPartitionedLowering() const;
    [[nodiscard]] bool supportsCouplingMode(CouplingMode mode) const;

    [[nodiscard]] virtual std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext& ctx,
        const CouplingFormBuilder& forms) const;

    [[nodiscard]] virtual std::vector<CouplingInstallMetadata> installMonolithicTerms(
        MonolithicCouplingInstallContext& install,
        const CouplingContext& ctx);

    [[nodiscard]] virtual std::vector<CouplingExchangeDeclaration>
    buildPartitionedExchangeDeclarations(const CouplingContext& ctx) const;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGCONTRACT_H
