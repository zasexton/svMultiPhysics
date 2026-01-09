#ifndef SVMP_FE_FORMS_JIT_VALIDATION_H
#define SVMP_FE_FORMS_JIT_VALIDATION_H

/**
 * @file JITValidation.h
 * @brief Pre-checks for LLVM JIT compatibility of FE/Forms expressions
 *
 * This module performs structural validation only. It does NOT generate LLVM IR.
 */

#include "Forms/FormIR.h"

#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

enum class Strictness : std::uint8_t {
    /**
     * @brief Only allow expressions that lower to straight-line math + loads
     *
     * Disallows runtime callbacks (e.g., `FormExprType::Coefficient`) and
     * unresolved name-based terminals (`ParameterSymbol`, coupled symbols).
     */
    Strict,

    /**
     * @brief Allow external calls/trampolines (still JIT-compatible, not inlinable)
     *
     * Coefficients and constitutive models are treated as external calls. Such
     * expressions are typically considered non-cacheable.
     */
    AllowExternalCalls,
};

struct ValidationOptions {
    Strictness strictness{Strictness::Strict};
};

struct ValidationIssue {
    FormExprType type{FormExprType::Constant};
    std::string message{};
    std::string subexpr{};
};

struct ValidationResult {
    bool ok{true};
    bool cacheable{true};
    std::optional<ValidationIssue> first_issue{};
};

/**
 * @brief Validate a single integrand expression (no measures)
 */
ValidationResult canCompile(const FormExpr& integrand,
                            const ValidationOptions& options = {});

/**
 * @brief Validate all integrands in a compiled FormIR
 */
ValidationResult canCompile(const FormIR& ir,
                            const ValidationOptions& options = {});

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_VALIDATION_H

