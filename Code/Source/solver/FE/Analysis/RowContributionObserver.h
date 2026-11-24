/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ROW_CONTRIBUTION_OBSERVER_H
#define SVMP_FE_ANALYSIS_ROW_CONTRIBUTION_OBSERVER_H

/**
 * @file RowContributionObserver.h
 * @brief The contributions of diagnostic operators to the rows of one field
 *
 * A formulation states, as a RowContributionDeclaration in the system's
 * diagnostic registry, the diagnostic operators whose contributions to the
 * rows of one field are sampled during a nonlinear solve, and the names of
 * that field and of the field its rows couple to.  RowContributionObserver
 * samples them at every residual assembly of a Newton solve; the
 * FieldResidualObserver samples the constrained residual itself.  Both are
 * switched on by the environment (fieldResidualObserverSwitchedOn,
 * rowContributionObserverSwitchedOn) and installed with the other analysis
 * extensions of a solve (buildNewtonAnalysisExtensions), and their log
 * lines are those the Newton solver printed before they left it.
 */

#include "Systems/DiagnosticRegistry.h"
#include "TimeStepping/NewtonSolver.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * The operators sampled, in order (a declared operator that is not installed
 * is reported as skipped), and the names of the row field and of the field
 * its rows couple to.
 */
struct RowContributionDeclaration {
    std::vector<systems::DiagnosticOperatorHandle> operators{};
    std::string row_field_name{};
    std::string coupling_field_name{};

    friend bool operator==(const RowContributionDeclaration&, const RowContributionDeclaration&) = default;
};

/// Logs the norm, extremes and samples of one field's rows of the
/// constrained residual (the field SVMP_NEWTON_FIELD_RESIDUAL_FIELD, by
/// default Pressure; samples SVMP_NEWTON_FIELD_RESIDUAL_SAMPLE_DOFS).
class FieldResidualObserver final : public timestepping::NewtonObserver {
public:
    [[nodiscard]] std::string_view id() const override { return "field_residual"; }
    void observe(timestepping::NewtonObservation& observation) override;
};

/// Assembles each declared operator into a scratch vector (and, with a
/// matrix switch, a scratch matrix of the Jacobian pattern) and logs its
/// contribution to the rows of the declared field before and after the
/// constrained rows are zeroed, with matrix support and summary samples.
/// Without a declaration it observes nothing.
class RowContributionObserver final : public timestepping::NewtonObserver {
public:
    /// The observer of the rows @p declaration states (none when empty).
    explicit RowContributionObserver(std::optional<RowContributionDeclaration> declaration)
        : declaration_(std::move(declaration))
    {
    }

    [[nodiscard]] std::string_view id() const override { return "row_contribution"; }
    [[nodiscard]] int storageOrder() const noexcept override { return 1; }
    [[nodiscard]] std::unique_ptr<timestepping::NewtonExtensionStorage> allocate(
        const timestepping::NewtonStorageContext& context) const override;
    void refreshStorage(const timestepping::NewtonStorageContext& context,
                        timestepping::NewtonExtensionStorage* storage,
                        bool after_production) const override;
    void observe(timestepping::NewtonObservation& observation) override;

private:
    std::optional<RowContributionDeclaration> declaration_;
};

/// Whether the environment switches on the field-residual observer
/// (SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC or a row-contribution switch).
[[nodiscard]] bool fieldResidualObserverSwitchedOn() noexcept;

/// Whether the environment switches on the row-contribution observer
/// (SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC or
/// SVMP_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC).
[[nodiscard]] bool rowContributionObserverSwitchedOn() noexcept;

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ROW_CONTRIBUTION_OBSERVER_H
