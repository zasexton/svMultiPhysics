#ifndef SVMP_FE_FORMS_FORM_COMPILER_H
#define SVMP_FE_FORMS_FORM_COMPILER_H

/**
 * @file FormCompiler.h
 * @brief Compiler that lowers a FormExpr into a FormIR (integral terms + metadata)
 */

#include "Forms/FormExpr.h"
#include "Forms/FormIR.h"
#include "Forms/MixedFormIR.h"

#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

class BlockBilinearForm;
class BlockLinearForm;

/**
 * @brief Compile a weak-form expression into a normalized integral-term list
 */
class FormCompiler {
public:
    FormCompiler();
    explicit FormCompiler(SymbolicOptions options);
    ~FormCompiler();

    FormCompiler(FormCompiler&&) noexcept;
    FormCompiler& operator=(FormCompiler&&) noexcept;

    FormCompiler(const FormCompiler&) = delete;
    FormCompiler& operator=(const FormCompiler&) = delete;

    [[nodiscard]] FormIR compileLinear(const FormExpr& form);
    [[nodiscard]] FormIR compileBilinear(const FormExpr& form);

    /**
     * @brief Compile a nonlinear residual form F(u;v)
     *
     * The resulting FormIR is tagged as FormKind::Residual; evaluation kernels
     * interpret TrialFunction as the current solution (provided by the assembler).
     */
    [[nodiscard]] FormIR compileResidual(const FormExpr& residual_form);

    // =========================================================================
    // Block compilation (manual decomposition path)
    //
    // These overloads accept pre-decomposed block containers. For the preferred
    // automatic decomposition path, use compileMixed() below.
    // =========================================================================

    /**
     * @brief Compile a block linear form (one FormExpr per test field)
     *
     * Manual path: each block is compiled independently, preserving the
     * single-test/single-trial constraints of the core compiler while enabling
     * multi-field Systems via block assembly.
     *
     * Empty/invalid blocks return `std::nullopt`.
     */
    [[nodiscard]] std::vector<std::optional<FormIR>> compileLinear(const BlockLinearForm& blocks);

    /**
     * @brief Compile a block bilinear form (one FormExpr per (test,trial) block)
     *
     * Empty/invalid blocks return `std::nullopt`.
     */
    [[nodiscard]] std::vector<std::vector<std::optional<FormIR>>> compileBilinear(const BlockBilinearForm& blocks);

    /**
     * @brief Compile a block residual decomposition
     *
     * This matches the current multi-field residual strategy:
     * each block is a residual contribution with exactly one TrialFunction and
     * one TestFunction (corresponding to a Jacobian block in Systems).
     */
    [[nodiscard]] std::vector<std::vector<std::optional<FormIR>>> compileResidual(const BlockBilinearForm& blocks);

    // =========================================================================
    // Auto-detecting compilation (preferred entry point)
    // =========================================================================

    /**
     * @brief Compile a form, auto-detecting single vs mixed
     *
     * This is the preferred high-level entry point. It inspects the expression
     * for multiple test/trial spaces and automatically routes to the appropriate
     * compiler path:
     *
     *   - 1 test space, ≤1 trial space → single-field path (returns 1×1 MixedFormIR)
     *   - 2+ test or trial spaces → mixed decomposition path
     *
     * Always returns a MixedFormIR, even for single-field forms (wrapped as 1×1).
     *
     * Supports all FormKind values:
     *   - Bilinear: test + trial decomposition
     *   - Residual: Jacobian-block decomposition (test-only terms not classified)
     *   - Linear: per-test decomposition into synthetic trial column
     *
     * @param form   The weak-form expression (single-field or mixed)
     * @param kind   FormKind (Bilinear, Residual, or Linear)
     * @return Block-sparse MixedFormIR
     * @throws std::invalid_argument if form is invalid or has no test functions
     */
    [[nodiscard]] MixedFormIR compile(const FormExpr& form, FormKind kind = FormKind::Bilinear);

    // =========================================================================
    // Explicit mixed-form compilation
    // =========================================================================

    /**
     * @brief Compile a mixed weak-form expression containing multiple test/trial spaces
     *
     * Explicit entry point — equivalent to compile() but makes the intent clear
     * when the caller knows the form is mixed. Also handles single-field forms
     * by delegating to the single-field path.
     *
     * Zero blocks (no terms matching a particular test/trial pair) are represented
     * as std::nullopt and can be skipped during assembly.
     *
     * Supports all FormKind values (Bilinear, Residual, Linear).
     *
     * @param form   The mixed weak-form expression
     * @param kind   FormKind (Bilinear, Residual, or Linear)
     * @return Block-sparse MixedFormIR
     * @throws std::invalid_argument if form has no test functions
     */
    [[nodiscard]] MixedFormIR compileMixed(const FormExpr& form, FormKind kind = FormKind::Bilinear);

    /**
     * @brief Set compile-time options (simplification, caching, etc.)
     */
    void setOptions(SymbolicOptions options);
    [[nodiscard]] const SymbolicOptions& options() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    [[nodiscard]] FormIR compileImpl(const FormExpr& form, FormKind kind);
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_COMPILER_H
