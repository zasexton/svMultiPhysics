/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_POINT_EVALUATOR_H
#define SVMP_FE_FORMS_POINT_EVALUATOR_H

/**
 * @file PointEvaluator.h
 * @brief Value-only evaluation of a scalar FormExpr at a physical point
 *
 * This lightweight evaluator is intended for tasks such as lowering strong
 * Dirichlet boundary conditions to algebraic constraints, where a scalar value
 * must be evaluated at a set of physical DOF coordinates (and time).
 *
 * Supported node subset:
 * - Constant, Coefficient (spatial + time-aware), Coordinate (via component access),
 *   Time, TimeStep
 * - Scalar algebra: +, -, *, /, pow, min/max
 * - Scalar functions: abs, sign, sqrt, exp, log
 * - Comparisons + conditional (returns 1.0/0.0 for comparisons)
 *
 * Unsupported nodes (throw):
 * - Test/Trial/DiscreteField/StateField
 * - Differential operators (grad/div/curl/H/dt)
 * - Integrals (dx/ds/dS)
 * - ReferenceCoordinate (not available without a reference-space point)
 */

#include "Forms/FormExpr.h"

#include <array>
#include <span>

namespace svmp {
namespace FE {
namespace forms {

struct PointEvalContext {
    std::array<Real, 3> x{0.0, 0.0, 0.0};
    Real time{0.0};
    Real dt{0.0};

    // Optional JIT/coupled scalar arrays (slot-indexed).
    std::span<const Real> jit_constants{};
    std::span<const Real> coupled_integrals{};
    std::span<const Real> coupled_aux{};
};

/**
 * @brief Evaluate a scalar expression at (x,t)
 *
 * @throws FEException / std::invalid_argument for unsupported expressions.
 */
Real evaluateScalarAt(const FormExpr& expr, const PointEvalContext& ctx);

/**
 * @brief True if expr depends on time (contains Time/TimeStep or time-aware coefficients)
 */
bool isTimeDependent(const FormExpr& expr);

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_POINT_EVALUATOR_H
