#ifndef SVMP_FE_FORMS_FORM_COMPILER_H
#define SVMP_FE_FORMS_FORM_COMPILER_H

/**
 * @file FormCompiler.h
 * @brief Compiler that lowers a FormExpr into a FormIR (integral terms + metadata)
 */

#include "Forms/FormExpr.h"
#include "Forms/FormIR.h"

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
    // Block/mixed compilation helpers (compile per-block)
    // =========================================================================

    /**
     * @brief Compile a block linear form (one FormExpr per test field)
     *
     * Each block is compiled independently, preserving the single-test/single-trial
     * constraints of the core compiler while enabling multi-field Systems via
     * block assembly.
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
