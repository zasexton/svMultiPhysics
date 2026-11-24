/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMMANIFEST_H
#define SVMP_FE_SYSTEMS_SYSTEMMANIFEST_H

/**
 * @file SystemManifest.h
 * @brief Ordered manifests of the kernels and constraints installed in an FESystem
 *
 * The manifests are text lines that identify what a formulation installed,
 * so that a refactoring of the formulation code can be shown to leave the
 * installed system unchanged (compare the lines before and after).
 *
 * Kernel manifest: for every operator (sorted by tag) and every registered
 * term (cell, boundary, interior-face, interface-face, cut-volume, in
 * registration order) one "kernel" line with the operator, the slot, the
 * marker and side of the registration, the test and trial fields and the
 * kernel type (for a global kernel, its declared manifestTypeName() when it
 * has one), followed by the compiled forms of that kernel.  Each form
 * prints one "term" line per integral term (domain, markers, sides,
 * time-derivative order, exterior-boundary measure and three hashes of the
 * integrand) and one "group" line per JIT kernel group (terms grouped by
 * domain, marker and side, as the JIT compiler groups them; listed in order
 * of first appearance).
 *
 * The integrand hashes:
 * - tree: the expression tree, ordered (operand order and association
 *   matter), with exact constant bits and the spelling of every leaf;
 * - str:  FormExpr::toString() of the integrand (all node payloads);
 * - kir:  KernelIR::stableHash64() of the optimized KernelIR, the per-term
 *   hash that enters the JIT cache key (insensitive to the operand order of
 *   the operations the KernelIR treats as commutative: +, inner, double
 *   contraction, min, max, == and !=; sensitive to the order of the factors
 *   of a product, which may be a tensor product, and to association).
 * A group line prints the combined IR hash and the JIT cache key computed as
 * JITCompiler does, but without the target-dependent inputs (target triple,
 * CPU, LLVM version, hardware profile, specialization); between two builds on
 * one machine an equal key therefore means an equal JIT kernel.  Groups of
 * kernels that are not JIT-wrapped print key=-.
 *
 * Constraint manifest: see FESystem::constraintManifestLines().
 *
 * Both functions read definition-time data only and change no state.  They
 * are meant for tests and for diagnostic output at the debug log level.
 *
 * Installed row: installedRow() renders one residual row of an operator as
 * its author can read it, the integrands of its compiled terms and the
 * fields it depends on, for tests and reports; it writes no log line.
 */

#include "Core/Types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
class FormExpr;
}
namespace systems {

class FESystem;

/// The ordered tree hash of the manifest ("tree"): node types, operand order
/// and association, exact constant bits, markers, sides and leaf spellings.
/// Equal hashes mean the same expression tree (up to hash collisions); used
/// by the IR-identity tests of the typed measures and helpers.
[[nodiscard]] std::uint64_t formExprTreeHash(const forms::FormExpr& expr);

[[nodiscard]] std::vector<std::string> kernelManifestLines(const FESystem& system);

[[nodiscard]] std::vector<std::string> constraintManifestLines(const FESystem& system);

/// A field of an installed row: its id and registered name.
struct InstalledRowField {
    FieldId id{INVALID_FIELD_ID};
    std::string name{};

    [[nodiscard]] friend bool operator==(const InstalledRowField&, const InstalledRowField&) = default;
};

/**
 * @brief One installed residual row of an operator and its field dependencies
 *
 * The row of a field is what the operator's kernels with that test field
 * assemble.  Read from those kernels (the integral terms of their compiled
 * forms, after installation: the residual of a nonlinear kernel, the
 * bilinear and linear forms of a linear one, a global row's measured and
 * dGlobal terms) and from the operator's formulation records (the Jacobian
 * blocks the installer recorded).
 */
struct InstalledRow {
    std::string operator_tag{};
    InstalledRowField field{};
    /// The trial fields of the row's Jacobian blocks (the kernels' blocks and
    /// the records' block couplings), in field order.
    std::vector<InstalledRowField> jacobian_fields{};
    /// The fields the terms read that have no Jacobian block in the row
    /// (prescribed data, lagged and previous values), in field order.
    std::vector<InstalledRowField> data_fields{};
    /// One line per term, in registration order:
    /// "<slot> trial=<field> <form>: <measure>[ dt^k] : <integrand>", the
    /// measure as the form IR dump prints it (dx, ds(3), dS, dI(7),
    /// dCutVolume(7,Negative), ...).
    std::vector<std::string> terms{};

    /// The row as text: "row <field> of operator <op>", "jacobian: ...",
    /// "data: ..." and the term lines, one per line.
    [[nodiscard]] std::string text() const;
};

/**
 * @brief The row of @p field in operator @p op
 *
 * Reads definition-time data only and changes no state; available before
 * and after setup.  Throws std::invalid_argument for an unknown operator or a
 * field with no term in it.
 */
[[nodiscard]] InstalledRow installedRow(const FESystem& system, const std::string& op, FieldId field);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMMANIFEST_H
