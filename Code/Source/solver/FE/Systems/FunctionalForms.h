/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FUNCTIONAL_FORMS_H
#define SVMP_FE_SYSTEMS_FUNCTIONAL_FORMS_H

/**
 * @file FunctionalForms.h
 * @brief Named functionals of a formulation, stated as measured 0-forms
 *
 * A functional is a scalar of the state, an energy, a power, a dissipation
 * rate or a measure, written with the same language as a residual: a sum of
 * integrals of a scalar integrand that has no test or trial function,
 *
 *   E = (0.5 * rho * inner(u, u)).dx(Omega) + gamma.dI(L.interface()),
 *
 * with any measure (dx, ds, dS, dI, dL, dCutVolume and the typed domains
 * that lower to them).  A formulation declares its functionals on the FE
 * system (an attachment, in declaration order), usually through
 * FormulationBuilder::functional; an evaluator of functionals reads the
 * declarations (D49 step E1).  Declaring a functional changes no operator,
 * kernel or constraint of the system.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class FESystem;

/// What a declared functional measures.
enum class FunctionalKind : std::uint8_t {
    Stored,          ///< an energy of the state
    Power,           ///< a rate of work done on the system
    DissipationRate, ///< a nonnegative rate of energy loss
    Measure          ///< a length, area or volume
};

/// "stored", "power", "dissipation_rate", "measure" (or "unknown").
[[nodiscard]] const char* functionalKindName(FunctionalKind kind) noexcept;

/// One named functional: a measured 0-form and the owner that declared it.
struct FunctionalFormDeclaration {
    std::string name{};
    FunctionalKind kind{FunctionalKind::Stored};
    forms::FormExpr form{};
    std::string owner{};
};

/// The functionals declared on an FE system (its attachment), in order.
struct FunctionalFormStore {
    std::vector<FunctionalFormDeclaration> declarations{};
};

/**
 * @brief Declares one functional on @p system.
 *
 * Throws InvalidArgumentException, before the system is set up, when the
 * name or the owner is empty, the name is already declared, or the form is
 * not a measured 0-form: a valid sum of integrals (dx, ds, dS, dI,
 * dCutVolume) without a test or trial function.  Throws after setup.
 */
void declareFunctionalForm(FESystem& system, FunctionalFormDeclaration declaration);

/// The functionals declared on @p system, in declaration order (empty when none).
[[nodiscard]] std::span<const FunctionalFormDeclaration> functionalForms(const FESystem& system) noexcept;

/// Whether @p form is a sum (+, -, unary -) of integrals.
[[nodiscard]] bool isSumOfIntegrals(const forms::FormExpr& form);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FUNCTIONAL_FORMS_H
