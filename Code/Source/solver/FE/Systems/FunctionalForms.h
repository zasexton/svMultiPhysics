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
 * with the measures an evaluator integrates (dx, ds, dI, dL, dCutVolume and
 * the typed domains that lower to them; not dS, the interior facets).  A formulation declares its functionals on the FE
 * system (an attachment, in declaration order), usually through
 * FormulationBuilder::functional; an evaluator of functionals reads the
 * declarations (D49 step E1).  Declaring a functional changes no operator,
 * kernel or constraint of the system.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
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
 * not a measured 0-form: a valid sum of integrals (dx, ds, dI, dL,
 * dCutVolume; an interior-facet integral, dS, is rejected because it is not
 * evaluated) without a test or trial function.  Throws after setup.
 */
void declareFunctionalForm(FESystem& system, FunctionalFormDeclaration declaration);

/// The functionals declared on @p system, in declaration order (empty when none).
[[nodiscard]] std::span<const FunctionalFormDeclaration> functionalForms(const FESystem& system) noexcept;

/// Whether @p form is a sum (+, -, unary -) of integrals.
[[nodiscard]] bool isSumOfIntegrals(const forms::FormExpr& form);

/// The generated domains of a wall that a wall functional is integrated over.
enum class WallMeasure : std::uint8_t {
    ContactLines, ///< the contact lines of the interface with the wall (dL)
    WettedPart    ///< the part of the wall on the liquid side of the interface (ds)
};

/**
 * @brief A functional of every wall an interface meets, stated by its integrand.
 *
 * Which walls an interface meets, and their contact lines and wetted parts,
 * follow the interface, so a formulation cannot name these domains when it
 * is set up.  It states the integrand instead (a scalar without a test or
 * trial function and without a measure) and which domains of a wall it is
 * integrated over; an evaluator integrates it over those domains of each
 * wall in the interface's geometry snapshot (wallFunctionalForm).  The
 * functional of wall b is named "<prefix>wall.<b>.<quantity>".
 */
struct WallFunctionalFormDeclaration {
    std::string prefix{};   ///< the name of the interface's functionals, e.g. "free_surface.<marker>."
    std::string quantity{};
    FunctionalKind kind{FunctionalKind::Stored};
    WallMeasure measure{WallMeasure::ContactLines};
    forms::FormExpr integrand{};                            ///< of every wall not in by_wall (none when invalid)
    std::vector<std::pair<int, forms::FormExpr>> by_wall{}; ///< the integrand of a wall, by boundary marker
    std::string owner{};
};

/// The wall functionals declared on an FE system (its attachment), in order.
struct WallFunctionalFormStore {
    std::vector<WallFunctionalFormDeclaration> declarations{};
};

/**
 * @brief Declares one wall functional on @p system.
 *
 * Throws InvalidArgumentException, before the system is set up, when the
 * prefix, quantity or owner is empty, the pair (prefix, quantity) is already
 * declared, no integrand is given, a wall is listed twice, or an integrand
 * is not a scalar integrand of a 0-form (it must not contain a measure, a
 * test or a trial function).  Throws after setup.
 */
void declareWallFunctionalForm(FESystem& system, WallFunctionalFormDeclaration declaration);

/// The wall functionals declared on @p system, in declaration order (empty when none).
[[nodiscard]] std::span<const WallFunctionalFormDeclaration> wallFunctionalForms(const FESystem& system) noexcept;

/// "<prefix>wall.<wall>.<quantity>".
[[nodiscard]] std::string wallFunctionalName(const WallFunctionalFormDeclaration& declaration, int wall);

/**
 * @brief The functional of one wall as a measured 0-form.
 *
 * The integrand of @p wall (its by_wall entry, else the integrand of every
 * wall) integrated over @p contact_lines (dL) or @p wetted_parts (ds), as the
 * declaration's measure says, summed in the order given.  Invalid when the
 * wall has no integrand or no domain of that kind.
 */
[[nodiscard]] forms::FormExpr wallFunctionalForm(const WallFunctionalFormDeclaration& declaration,
                                                 int wall,
                                                 std::span<const forms::ContactLineDomain> contact_lines,
                                                 std::span<const forms::BoundaryDomain> wetted_parts);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FUNCTIONAL_FORMS_H
