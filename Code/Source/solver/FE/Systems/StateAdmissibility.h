/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_STATEADMISSIBILITY_H
#define SVMP_FE_SYSTEMS_STATEADMISSIBILITY_H

#include "Analysis/AnalysisSummaryTypes.h"
#include "Core/Types.h"
#include "Systems/FieldRegistry.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

enum class StateAdmissibilityTargetKind : std::uint8_t {
    Field,
    StateGroup
};

enum class StateAdmissibilityCheckKind : std::uint8_t {
    LowerBound,
    UpperBound,
    Interval,
    SumEquality,
    SumInequality,
    Finite,
    CallbackResidual
};

enum class StateAdmissibilityScope : std::uint8_t {
    NodalValues,
    CellAverages,
    QuadratureValues,
    RegionAggregate,
    GlobalAggregate
};

enum class StateAdmissibilityFailurePolicy : std::uint8_t {
    WarnOnly,
    HardFailure
};

struct StateAdmissibilityTarget {
    StateAdmissibilityTargetKind kind{StateAdmissibilityTargetKind::Field};
    FieldId field{INVALID_FIELD_ID};
    StateGroupId state_group{INVALID_STATE_GROUP_ID};

    static StateAdmissibilityTarget fieldTarget(FieldId field_id) noexcept;
    static StateAdmissibilityTarget stateGroupTarget(StateGroupId group_id) noexcept;
};

using StateAdmissibilityResidualCallback =
    std::function<Real(std::span<const Real> values)>;

struct StateAdmissibilityDescriptor {
    std::string id;
    StateAdmissibilityTarget target{};
    StateAdmissibilityCheckKind check{StateAdmissibilityCheckKind::Finite};
    StateAdmissibilityScope scope{StateAdmissibilityScope::NodalValues};
    StateAdmissibilityFailurePolicy failure_policy{StateAdmissibilityFailurePolicy::HardFailure};
    std::optional<Real> lower_bound{};
    std::optional<Real> upper_bound{};
    std::optional<Real> sum_target{};
    std::optional<Real> sum_lower_bound{};
    std::optional<Real> sum_upper_bound{};
    Real tolerance{0.0};
    StateAdmissibilityResidualCallback residual_callback{};
};

struct StateAdmissibilitySample {
    std::vector<Real> values;
    std::string label;
};

struct StateAdmissibilityViolation {
    std::size_t sample_index{0};
    Real residual{0.0};
    std::string label;
};

struct StateAdmissibilityResult {
    bool admissible{true};
    Real max_residual{0.0};
    std::uint64_t checked_state_count{0};
    std::uint64_t violation_count{0};
    std::vector<StateAdmissibilityViolation> violations;
};

class StateAdmissibilityEvaluator {
public:
    static StateAdmissibilityResult evaluate(
        const StateAdmissibilityDescriptor& descriptor,
        std::span<const StateAdmissibilitySample> samples,
        std::size_t max_recorded_violations = 8);

    static analysis::InvariantDomainSummary makeInvariantDomainSummary(
        const StateAdmissibilityDescriptor& descriptor,
        const StateAdmissibilityResult& result,
        const FieldRegistry* registry = nullptr);

private:
    static void validateDescriptor(const StateAdmissibilityDescriptor& descriptor);
    static Real sampleResidual(const StateAdmissibilityDescriptor& descriptor,
                               std::span<const Real> values);
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_STATEADMISSIBILITY_H
