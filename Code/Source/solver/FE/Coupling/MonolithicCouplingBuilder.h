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

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class MonolithicCouplingInstallContext {
public:
    MonolithicCouplingInstallContext() = default;

    void recordInstalledContribution() noexcept;
    [[nodiscard]] std::size_t installedContributionCount() const noexcept;

private:
    std::size_t installed_contribution_count_{0};
};

class CouplingGraph;

class MonolithicCouplingBuilder {
public:
    [[nodiscard]] CouplingContext buildInitialContext(
        std::span<const CouplingParticipantRef> participants,
        std::span<const CouplingFieldRef> fields,
        std::span<const CouplingRegionRef> regions = {},
        std::span<const SharedRegionRef> shared_regions = {}) const;

    [[nodiscard]] std::vector<CouplingContractDeclaration> collectDeclarations(
        std::span<const CouplingContract*> contracts) const;

    [[nodiscard]] CouplingValidationResult validateDeclarations(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations) const;

    [[nodiscard]] CouplingValidationResult validateTemporalPolicy(
        const CouplingGraph& graph,
        const std::optional<CouplingTemporalAvailability>& availability) const;

    [[nodiscard]] std::vector<ResolvedCouplingAdditionalFieldDeclaration>
    resolveAdditionalFields(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations) const;

    [[nodiscard]] std::vector<ResolvedCouplingAdditionalFieldDeclaration>
    registerAdditionalFields(
        const CouplingContext& context,
        std::span<const CouplingContractDeclaration> declarations) const;

    [[nodiscard]] CouplingContext refreshContextWithAdditionalFields(
        const CouplingContext& context,
        std::span<const ResolvedCouplingAdditionalFieldDeclaration>
            additional_fields) const;

    [[nodiscard]] ResolvedCouplingFormContribution resolveFormContribution(
        const CouplingContext& context,
        const CouplingFormContribution& contribution) const;

    [[nodiscard]] CouplingFormAnalysisMetadata installResolvedFormContribution(
        systems::FESystem& system,
        const ResolvedCouplingFormContribution& contribution) const;

    [[nodiscard]] std::vector<CouplingFormAnalysisMetadata> installFormContributions(
        systems::FESystem& system,
        const CouplingContext& context,
        std::span<const CouplingFormContribution> contributions) const;

    [[nodiscard]] std::vector<CouplingFormAnalysisMetadata> installExpertTerms(
        MonolithicCouplingInstallContext& install,
        const CouplingContext& context,
        std::span<CouplingContract*> contracts) const;

    [[nodiscard]] static CouplingFormAnalysisMetadata adaptFormAnalysisMetadata(
        const analysis::FormContributionAnalysisMetadata& metadata);

    [[nodiscard]] static CouplingFormAnalysisMetadata adaptInstallMetadata(
        const CouplingInstallMetadata& metadata);

    [[nodiscard]] static std::vector<CouplingFormAnalysisMetadata>
    adaptInstallMetadataRecords(std::span<const CouplingInstallMetadata> metadata);
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_MONOLITHICCOUPLINGBUILDER_H
