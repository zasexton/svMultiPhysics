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
#include "Coupling/CouplingGeometryRequirements.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"

#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace coupling {

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

    [[nodiscard]] forms::FormExpr integrate(const forms::FormExpr& integrand,
                                            const CouplingRegionRef& region) const;
    [[nodiscard]] forms::FormExpr integrate(const forms::FormExpr& integrand,
                                            std::string_view participant,
                                            std::string_view region) const;
    [[nodiscard]] forms::FormExpr integrateShared(const forms::FormExpr& integrand,
                                                  std::string_view shared_region,
                                                  std::string_view participant) const;

    [[nodiscard]] CouplingFieldRef field(std::string_view participant,
                                         std::string_view field) const;
    [[nodiscard]] CouplingRegionRef region(std::string_view participant,
                                           std::string_view region) const;
    [[nodiscard]] CouplingRegionRef sharedRegion(std::string_view name,
                                                 std::string_view participant) const;
    [[nodiscard]] SharedRegionRef sharedRegionGroup(std::string_view name) const;

private:
    const CouplingContext* context_{nullptr};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H
