/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_DEFINITIONBACKEDCOUPLINGCONTRACT_H
#define SVMP_FE_COUPLING_DEFINITIONBACKEDCOUPLINGCONTRACT_H

/**
 * @file DefinitionBackedCouplingContract.h
 * @brief Coupling contract adapter for compact definition-backed modules.
 */

#include "Coupling/CouplingContract.h"
#include "Coupling/CouplingDefinitionBuilder.h"

namespace svmp {
namespace FE {
namespace coupling {

class DefinitionBackedCouplingContract : public CouplingContract {
public:
    [[nodiscard]] CouplingContractDeclaration declare() const override;
    void validate(const CouplingContext& ctx) const override;

    [[nodiscard]] bool supportsMonolithicLowering() const override;
    [[nodiscard]] bool supportsPartitionedLowering() const override;

    [[nodiscard]] std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext& ctx,
        const CouplingFormBuilder& forms) const override;

    [[nodiscard]] std::vector<CouplingExchangeDeclaration>
    buildPartitionedExchangeDeclarations(const CouplingContext& ctx) const override;

protected:
    [[nodiscard]] virtual std::string contractInstanceName() const;
    virtual void define(CouplingDefinitionBuilder& builder) const = 0;

private:
    [[nodiscard]] CouplingDefinitionBuilder buildDefinition() const;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_DEFINITIONBACKEDCOUPLINGCONTRACT_H
