/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/PartitionedCouplingPlan.h"

namespace svmp {
namespace FE {
namespace coupling {

const char* toString(CouplingPayloadKind kind) noexcept
{
    switch (kind) {
    case CouplingPayloadKind::CoefficientExpression:
        return "coefficient_expression";
    case CouplingPayloadKind::PrimalValue:
        return "primal_value";
    case CouplingPayloadKind::DualResidualVector:
        return "dual_residual_vector";
    case CouplingPayloadKind::ResidualRecipe:
        return "residual_recipe";
    case CouplingPayloadKind::ConstraintResidual:
        return "constraint_residual";
    case CouplingPayloadKind::DriverOwned:
        return "driver_owned";
    }
    return "unknown";
}

const char* toString(CouplingPayloadFallbackPolicy policy) noexcept
{
    switch (policy) {
    case CouplingPayloadFallbackPolicy::Error:
        return "error";
    case CouplingPayloadFallbackPolicy::WarnAndUseDualResidual:
        return "warn_and_use_dual_residual";
    case CouplingPayloadFallbackPolicy::WarnAndUseResidualRecipe:
        return "warn_and_use_residual_recipe";
    case CouplingPayloadFallbackPolicy::WarnAndSplitSymmetric:
        return "warn_and_split_symmetric";
    case CouplingPayloadFallbackPolicy::WarnAndUseConstraintResidual:
        return "warn_and_use_constraint_residual";
    case CouplingPayloadFallbackPolicy::WarnAndUseDriverOwned:
        return "warn_and_use_driver_owned";
    }
    return "unknown";
}

const char* toString(CouplingPayloadExtractionReason reason) noexcept
{
    switch (reason) {
    case CouplingPayloadExtractionReason::Exact:
        return "exact";
    case CouplingPayloadExtractionReason::NoConsumerTest:
        return "no_consumer_test";
    case CouplingPayloadExtractionReason::MultipleConsumerTestsInTerm:
        return "multiple_consumer_tests_in_term";
    case CouplingPayloadExtractionReason::NonlinearInConsumerTest:
        return "nonlinear_in_consumer_test";
    case CouplingPayloadExtractionReason::BothSideStateDependency:
        return "both_side_state_dependency";
    case CouplingPayloadExtractionReason::SymmetricWeakEnforcement:
        return "symmetric_weak_enforcement";
    case CouplingPayloadExtractionReason::StabilizedTraceOperator:
        return "stabilized_trace_operator";
    case CouplingPayloadExtractionReason::ConstraintResidualNotLoad:
        return "constraint_residual_not_load";
    case CouplingPayloadExtractionReason::MissingDirection:
        return "missing_direction";
    case CouplingPayloadExtractionReason::MissingTransferPolicy:
        return "missing_transfer_policy";
    case CouplingPayloadExtractionReason::UnsupportedRuntimeProvider:
        return "unsupported_runtime_provider";
    case CouplingPayloadExtractionReason::ContributionNotFound:
        return "contribution_not_found";
    }
    return "unknown";
}

} // namespace coupling
} // namespace FE
} // namespace svmp
