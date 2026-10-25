/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FUNCTIONAL_FORM_EVALUATOR_H
#define SVMP_FE_SYSTEMS_FUNCTIONAL_FORM_EVALUATOR_H

/**
 * @file FunctionalFormEvaluator.h
 * @brief Values of measured 0-forms, the same at any rank count
 *
 * A functional is a measured 0-form (FunctionalForms.h): a sum (+, -, unary
 * -) of integrals of scalar integrands without a test or trial function.
 * The evaluator integrates each integral of such a form at one state with
 * the form interpreter (never the JIT, never an operator kernel):
 *
 *   dx                           every locally owned cell, with its standard
 *                                quadrature;
 *   ds(m), full-physical ds      the boundary faces of marker m (all markers
 *                                for m = -1) of locally owned cells;
 *   dCutVolume(m, side)          the cut context's volume rules of (m, side);
 *   dI(m), dL(contact line),     the cut context's interface rules of marker
 *   generated-active-boundary ds   m: a level-set interface, a contact line
 *                                or a wetted (active) boundary;
 *
 * the cut rules on locally owned parent cells.  dS, and dI and dCutVolume
 * without a marker, are rejected.  For a free surface the cut rules are those
 * of the authoritative geometry snapshot imported into the cut context; the
 * cut loops assert that every snapshot is current for the mesh.
 *
 * Every quadrature term w_q f_q of every rank goes into an exact accumulator
 * (ExactAccumulator), and the ranks' accumulators are combined and rounded
 * once (collectiveExactSums).  The value of a form therefore depends only on
 * the state and the rules: not on the order of the terms, the partition or
 * the number of ranks.  The absolute mass, the sum of |w_q f_q|, is reduced
 * the same way; it bounds the round-off of any other summation of the same
 * terms.
 *
 * Every function here is collective over its communicator: every rank passes
 * the same forms in the same order.  A rank-local failure is rethrown on that
 * rank and reported on the others before any reduction.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class CutIntegrationContext;
class FunctionalAssembler;
} // namespace assembly

namespace systems {

class CollectiveCommunicator;
class FESystem;
struct SystemStateView;

/// The value of one measured 0-form.
struct FunctionalFormValue {
    /// The sum of w_q f_q over every integral of the form, correctly rounded.
    Real value{0.0};
    /// The sum of |w_q f_q| over the same terms, correctly rounded.
    Real absolute_mass{0.0};
};

/**
 * @brief Evaluates @p forms with @p assembler.
 *
 * The assembler carries the mesh, the primary space and DOF map, the field
 * bindings, the solution and the time and parameter state; its own results
 * are not used.  @p cut_context holds the rules of the cut measures and may
 * be null when no form has one.  Throws InvalidArgumentException, on every
 * rank alike, for a form that is not a measured 0-form, for an unsupported
 * measure and for a cut measure whose marker no rank's cut context knows.
 */
[[nodiscard]] std::vector<FunctionalFormValue> evaluateFunctionalForms(
    assembly::FunctionalAssembler& assembler,
    std::span<const forms::FormExpr> forms,
    const assembly::CutIntegrationContext* cut_context,
    const CollectiveCommunicator& comm);

/**
 * @brief Evaluates @p forms on a set-up @p system at @p state.
 *
 * Every volume and global unknown field of the system is bound (the first
 * non-global one in DOF-map order gives the cell quadrature), as for the
 * system's boundary functionals.  The ghost values of the state's backend
 * vectors are refreshed first.  A null @p cut_context means the system's
 * published cut integration context.
 */
[[nodiscard]] std::vector<FunctionalFormValue> evaluateFunctionalForms(
    const FESystem& system,
    std::span<const forms::FormExpr> forms,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context = nullptr);

/**
 * @brief evaluateFunctionalForms with the quadrature of @p quadrature_field
 *
 * The volume field @p quadrature_field is the primary field: its element's
 * rule integrates the cells, and the face rule of its order the faces.  The
 * rows of a global test function use the field of the largest quadrature
 * order, whose rule the global space binds on the cells.
 */
[[nodiscard]] std::vector<FunctionalFormValue> evaluateFunctionalFormsWithQuadratureOf(
    const FESystem& system,
    FieldId quadrature_field,
    std::span<const forms::FormExpr> forms,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context = nullptr);

/// evaluateFunctionalForms of every functional declared on @p system
/// (functionalForms(system)), in declaration order.
[[nodiscard]] std::vector<FunctionalFormValue> evaluateDeclaredFunctionalForms(
    const FESystem& system,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context = nullptr);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FUNCTIONAL_FORM_EVALUATOR_H
