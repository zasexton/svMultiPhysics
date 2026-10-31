/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_GLOBALROWKERNEL_H
#define SVMP_FE_SYSTEMS_GLOBALROWKERNEL_H

/**
 * @file GlobalRowKernel.h
 * @brief The residual entry of the row of a global test function
 *
 * installFormulation splits the residual by test function.  The row of a test
 * function mu on a GlobalSpace,
 *
 *   g = sum of the measured terms (mu * f).dx(), (mu * h).ds(m), ...
 *     + sum of the point terms (mu * c).dGlobal(),
 *
 * is lowered in two parts:
 *
 *   - the row's Jacobian comes from the ordinary element kernels of the
 *     measured terms, installed matrix-only (the row and its columns follow
 *     from the cell loops, and off-rank contributions take the usual routing);
 *   - its residual entry comes from this kernel: the measured terms with
 *     mu = 1 are a measured 0-form, evaluated over each rank's owned cells and
 *     faces into exact accumulators and combined once
 *     (evaluateFunctionalFormsWithQuadratureOf), so the entry does not change
 *     with the rank count or the partition.  The point terms with mu = 1 are
 *     evaluated once, and the owner rank of the multiplier's DOF adds the sum.
 *
 * The point terms' derivatives with respect to global fields (a stabilized
 * multiplier's D block) are added to the matrix by the owner as well.
 *
 * A point term may hold dt(X, k) of a global field X (in-matrix lowering): it
 * is evaluated with the stencil of the time integrator, sum_j a_j X^{n-j},
 * from the state and its history, its derivative adds a_0 to X's column, and
 * each term of the row's point sum takes the term weight of the integrator
 * for its highest derivative order, as the terms of an element kernel do.
 * Without dt the point sum is evaluated as one expression, unweighted.  The
 * bordered lowering refuses time derivatives at setup.
 *
 * Evaluation is interpreted (never the JIT).  The kernel is collective: every
 * rank assembles it with the same request.
 */

#include "Forms/FormExpr.h"
#include "Systems/GlobalKernel.h"

#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class GlobalRowKernel final : public GlobalKernel {
public:
    /**
     * @param test_field the global field whose test function the row has
     * @param test_field_name its name (for messages and the manifest)
     * @param measured_terms the row's terms with a spatial measure (still with
     *        the test function), or an invalid expression
     * @param point_integrand the sum of the row's dGlobal integrands with their
     *        signs (still with the test function), or an invalid expression
     */
    GlobalRowKernel(FieldId test_field,
                    std::string test_field_name,
                    forms::FormExpr measured_terms,
                    forms::FormExpr point_integrand);

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] std::string manifestTypeName() const override { return "GlobalRowKernel"; }

    void addSparsityCouplings(const FESystem& system,
                              sparsity::SparsityPattern& pattern) const override;

    [[nodiscard]] assembly::AssemblyResult assemble(const FESystem& system,
                                                    const AssemblyRequest& request,
                                                    const SystemStateView& state,
                                                    assembly::GlobalSystemView* matrix_out,
                                                    assembly::GlobalSystemView* vector_out) override;

    [[nodiscard]] FieldId testField() const noexcept { return test_field_; }
    [[nodiscard]] const std::string& testFieldName() const noexcept { return test_field_name_; }

    /**
     * @brief Whether the row fixes the constant mode of the scalar field @p field
     *
     * True when the field enters the row's measured terms without a
     * derivative and a constant shift of it changes the row (the derivative
     * of the measured 0-form in the direction 1 is not identically zero), as
     * for the mean-value constraint (mu * p).dx().  The gauge registry takes
     * this as the field's anchoring evidence.
     */
    [[nodiscard]] bool anchorsConstantModeOf(FieldId field) const;

    /// The global fields the dGlobal terms depend on (a D block when non-empty)
    [[nodiscard]] std::vector<FieldId> pointTermFields() const;

    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override
    {
        return max_time_derivative_order_;
    }

    /// The global fields under dt() in the dGlobal terms.
    [[nodiscard]] std::vector<FieldId> timeDerivativeFields() const override;

    /// The measured 0-form of the residual entry (the measured terms with mu = 1)
    [[nodiscard]] const forms::FormExpr& measuredValueForm() const noexcept { return measured_value_form_; }

    /// The point value of the residual entry (the dGlobal integrands with mu = 1)
    [[nodiscard]] const forms::FormExpr& pointValue() const noexcept { return point_value_; }

    /**
     * @brief Refuse a dGlobal integrand with anything but global symbols
     *
     * Allowed: constants, parameters, time and the time step, the state of
     * global fields and their time derivatives dt(X, k), the row's test
     * function, and scalar algebra (+, -, *, /, negation, power, min, max,
     * abs, sign, sqrt, exp, log).  Throws InvalidArgumentException naming the
     * first other node.
     */
    static void validatePointIntegrand(const forms::FormExpr& integrand,
                                       const std::string& test_field_name);

private:
    /// dt(field, order) of the point terms, written as the global symbol @p symbol.
    struct TimeDerivativeSymbol {
        FieldId field{INVALID_FIELD_ID};
        int order{1};
        FieldId symbol{INVALID_FIELD_ID};
    };
    /// One term of the point sum of a row with time derivatives.
    struct PointTerm {
        forms::FormExpr value{};
        int order{0}; ///< highest time-derivative order in the term
        /// d(value)/d(field or symbol), for each global field and symbol of the term
        std::vector<std::pair<FieldId, forms::FormExpr>> derivatives{};
    };

    FieldId test_field_{INVALID_FIELD_ID};
    std::string test_field_name_;
    forms::FormExpr measured_value_form_{};
    forms::FormExpr point_value_{};
    /// d(point value)/d(global field), for each global field of the point terms
    std::vector<std::pair<FieldId, forms::FormExpr>> point_derivatives_{};
    /// With time derivatives: the symbols and the point sum term by term
    /// (point_value_ and point_derivatives_ stay unused).
    std::vector<TimeDerivativeSymbol> time_symbols_{};
    std::vector<PointTerm> point_terms_{};
    std::vector<FieldId> point_fields_{};
    int max_time_derivative_order_{0};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_GLOBALROWKERNEL_H
