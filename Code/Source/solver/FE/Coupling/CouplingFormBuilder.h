/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H
#define SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H

/**
 * @file CouplingFormBuilder.h
 * @brief Adapter from coupling participant names to public Forms vocabulary.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingDeclaration.h"
#include "Coupling/CouplingGeometryRequirements.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingInterfaceSideView;
class CouplingSharedInterfaceView;

class CouplingFormBuilder {
public:
    explicit CouplingFormBuilder(const CouplingContext& context);

    [[nodiscard]] const CouplingContext& context() const noexcept;

    [[nodiscard]] forms::FormExpr state(std::string_view participant,
                                        std::string_view field,
                                        std::string symbol) const;

    [[nodiscard]] forms::FormExpr test(std::string_view participant,
                                       std::string_view field,
                                       std::string symbol) const;

    [[nodiscard]] forms::FormExpr timeDerivative(std::string_view participant,
                                                 std::string_view field,
                                                 std::string symbol,
                                                 int order = 1) const;

    [[nodiscard]] forms::FormExpr previousSolution(std::string_view participant,
                                                   std::string_view field,
                                                   int steps_back = 1) const;

    [[nodiscard]] forms::FormExpr time() const;
    [[nodiscard]] forms::FormExpr timeStep() const;
    [[nodiscard]] forms::FormExpr effectiveTimeStep() const;

    [[nodiscard]] forms::FormExpr meshDisplacement(
        const CouplingGeometryTerminalScope& scope) const;
    [[nodiscard]] forms::FormExpr meshVelocity(
        const CouplingGeometryTerminalScope& scope) const;
    [[nodiscard]] forms::FormExpr meshAcceleration(
        const CouplingGeometryTerminalScope& scope) const;
    [[nodiscard]] forms::FormExpr previousMeshVelocity(
        const CouplingGeometryTerminalScope& scope) const;
    [[nodiscard]] forms::FormExpr predictedMeshVelocity(
        const CouplingGeometryTerminalScope& scope) const;
    [[nodiscard]] forms::FormExpr geometryTerminal(
        CouplingGeometryTerminalQuantity quantity,
        const CouplingGeometryTerminalScope& scope) const;

    [[nodiscard]] std::vector<CouplingFormTerminalProvenanceDeclaration>
    terminalProvenanceFor(const forms::FormExpr& residual) const;

    [[nodiscard]] CouplingFormContribution
    attachTerminalProvenance(CouplingFormContribution contribution) const;

    [[nodiscard]] forms::FormExpr integrate(const forms::FormExpr& integrand,
                                            const CouplingRegionRef& region) const;
    [[nodiscard]] forms::FormExpr integrate(const forms::FormExpr& integrand,
                                            std::string_view participant,
                                            std::string_view region) const;
    [[nodiscard]] forms::FormExpr integrateShared(const forms::FormExpr& integrand,
                                                  std::string_view shared_region,
                                                  std::string_view participant) const;

    [[nodiscard]] CouplingSharedInterfaceView sharedInterface(
        std::string_view name) const;

    [[nodiscard]] CouplingFieldRef field(std::string_view participant,
                                         std::string_view field) const;
    [[nodiscard]] CouplingRegionRef region(std::string_view participant,
                                           std::string_view region) const;
    [[nodiscard]] CouplingRegionRef sharedRegion(std::string_view name,
                                                 std::string_view participant) const;
    [[nodiscard]] SharedRegionRef sharedRegionGroup(std::string_view name) const;

private:
    struct RecordedTerminalProvenance {
        std::weak_ptr<forms::FormExprNode> node;
        CouplingFormTerminalProvenanceDeclaration declaration;
    };

    [[nodiscard]] forms::FormExpr recordTerminal(
        forms::FormExpr expr,
        CouplingFormTerminalProvenanceDeclaration declaration) const;

    const CouplingContext* context_{nullptr};
    mutable std::vector<RecordedTerminalProvenance> recorded_terminals_;
};

class CouplingInterfaceSideView {
public:
    CouplingInterfaceSideView(const CouplingFormBuilder& builder,
                              std::string shared_region_name,
                              CouplingRegionRef region);

    [[nodiscard]] std::string_view sharedRegionName() const noexcept;
    [[nodiscard]] std::string_view participantName() const noexcept;
    [[nodiscard]] const CouplingRegionRef& region() const noexcept;

    [[nodiscard]] forms::FormExpr state(std::string_view field,
                                        std::string symbol) const;
    [[nodiscard]] forms::FormExpr test(std::string_view field,
                                       std::string symbol) const;
    [[nodiscard]] forms::FormExpr dt(std::string_view field,
                                     std::string symbol,
                                     int order = 1) const;
    [[nodiscard]] forms::FormExpr geometryTerminal(
        CouplingGeometryTerminalQuantity quantity) const;
    [[nodiscard]] forms::FormExpr normal() const;

private:
    const CouplingFormBuilder* builder_{nullptr};
    std::string shared_region_name_;
    CouplingRegionRef region_;
};

class CouplingSharedInterfaceView {
public:
    CouplingSharedInterfaceView(const CouplingFormBuilder& builder,
                                std::string shared_region_name);

    [[nodiscard]] std::string_view name() const noexcept;
    [[nodiscard]] SharedRegionRef group() const;
    [[nodiscard]] CouplingInterfaceSideView side(
        std::string_view participant) const;
    [[nodiscard]] forms::FormExpr integral(
        const forms::FormExpr& integrand,
        std::string_view integration_participant) const;

private:
    const CouplingFormBuilder* builder_{nullptr};
    std::string shared_region_name_;
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H
