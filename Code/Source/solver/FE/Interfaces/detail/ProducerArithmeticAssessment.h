/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_DETAIL_PRODUCERARITHMETICASSESSMENT_H
#define SVMP_FE_INTERFACES_DETAIL_PRODUCERARITHMETICASSESSMENT_H

#include "Core/Types.h"

#include <array>

namespace svmp::FE::interfaces::detail {

using AssessmentPoint = std::array<Real, 3>;

enum class ArithmeticFailure {
    None,
    UnsupportedRepresentation,
    InvalidInput,
    ArithmeticRange,
    UnresolvedBand,
    UnresolvedDenominator,
    ChangedBranch,
    UnresolvedInterior,
    UnknownOrigin,
    UnresolvedDistance,
    DistinctOriginMerge,
    RetainedRepeat
};

struct ArithmeticInterval {
    Real lower{0};
    Real upper{0};
};

struct IntervalAssessment {
    ArithmeticInterval interval{};
    ArithmeticFailure failure{ArithmeticFailure::InvalidInput};

    [[nodiscard]] bool available() const noexcept
    {
        return failure == ArithmeticFailure::None;
    }
};

enum class IntervalOperation { Add, Subtract, Multiply, Divide };

[[nodiscard]] IntervalAssessment assessIntervalOperation(
    IntervalOperation operation,
    ArithmeticInterval left,
    ArithmeticInterval right) noexcept;

[[nodiscard]] IntervalAssessment assessIntervalSquare(
    ArithmeticInterval value) noexcept;

[[nodiscard]] IntervalAssessment assessIntervalSqrt(
    ArithmeticInterval value) noexcept;

struct PointAssessment {
    std::array<ArithmeticInterval, 3> ideal{};
    AssessmentPoint radius{};
    ArithmeticFailure failure{ArithmeticFailure::UnknownOrigin};

    [[nodiscard]] bool available() const noexcept
    {
        return failure == ArithmeticFailure::None;
    }
};

struct OriginalEdgeObservation {
    AssessmentPoint a{};
    AssessmentPoint b{};
    AssessmentPoint emitted{};
    Real phi_a{0};
    Real phi_b{0};
    Real isovalue{0};
    Real signed_band{0};
    Real actual_signed_a{0};
    Real actual_signed_b{0};
    Real actual_denominator{0};
    Real actual_quotient{0};
    Real actual_clamped{0};
    bool canonicalization_changed{false};
    bool helper_denominator_guard{false};
    bool division_taken{false};
};

[[nodiscard]] PointAssessment assessOriginalCorner(
    const AssessmentPoint& original,
    const AssessmentPoint& emitted) noexcept;

[[nodiscard]] PointAssessment assessOriginalEdge(
    const OriginalEdgeObservation& input) noexcept;

enum class OriginRelation { Unknown, SameOriginal, DistinctOriginal };

struct DistanceObservation {
    bool executed{false};
    Real distance{0};
    bool comparison_result{false};
    bool removed{false};
};

struct DistanceAssessment {
    ArithmeticInterval hull{};
    ArithmeticFailure failure{ArithmeticFailure::UnresolvedDistance};

    [[nodiscard]] bool available() const noexcept
    {
        return failure == ArithmeticFailure::None;
    }
};

[[nodiscard]] DistanceAssessment assessDistance(
    const PointAssessment& a,
    const AssessmentPoint& emitted_a,
    const PointAssessment& b,
    const AssessmentPoint& emitted_b,
    Real tolerance,
    unsigned dimension,
    OriginRelation relation,
    DistanceObservation observation) noexcept;

} // namespace svmp::FE::interfaces::detail

#endif // SVMP_FE_INTERFACES_DETAIL_PRODUCERARITHMETICASSESSMENT_H
