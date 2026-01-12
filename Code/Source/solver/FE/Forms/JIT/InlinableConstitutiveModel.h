/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_JIT_INLINABLE_CONSTITUTIVE_MODEL_H
#define SVMP_FE_FORMS_JIT_INLINABLE_CONSTITUTIVE_MODEL_H

/**
 * @file InlinableConstitutiveModel.h
 * @brief Opt-in interface for constitutive models that can be expanded to FormExpr
 *
 * This is a setup-time facility to remove `FormExprType::Constitutive` as a
 * virtual call boundary in "JIT-fast" mode. It deliberately contains no LLVM
 * dependencies.
 *
 * Key idea:
 * - A model may provide a symbolic expansion (FormExpr) of its outputs and
 *   (optionally) a side-effect program that updates material state.
 * - Systems/Forms can rewrite constitutive calls into plain FormExpr + explicit
 *   state reads/writes during setup.
 */

#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace constitutive {
class StateLayout;
}

namespace forms {

/**
 * @brief Material-state access mode for inlinable constitutive expansions
 */
enum class MaterialStateAccess : std::uint8_t {
    None,
    ReadOnly,
    ReadWrite,
};

/**
 * @brief Setup-time context for constitutive expansion
 */
struct InlinableConstitutiveContext {
    int dim{0};

    // Base offset for this model's per-qpt state block within the kernel's
    // flattened material state buffer.
    std::uint32_t state_base_offset_bytes{0u};

    // Optional structured state layout metadata (byte offsets within the model's
    // state block). When provided, helpers can map field names to offsets.
    const constitutive::StateLayout* state_layout{nullptr};
};

/**
 * @brief Single state write: store a scalar Real into state_work at a byte offset
 *
 * The write target is the kernel's per-qpt state block (AssemblyContext::materialStateWork(q)).
 */
struct MaterialStateUpdateOp {
    std::uint32_t offset_bytes{0u};
    FormExpr value{};
};

/**
 * @brief Result of inlining a constitutive call
 */
struct InlinedConstitutiveExpansion {
    std::vector<FormExpr> outputs{};
    std::vector<MaterialStateUpdateOp> state_updates{};
};

/**
 * @brief Opt-in contract for models that can be expanded into FormExpr
 *
 * This is intentionally separate from runtime evaluation (`ConstitutiveModel`)
 * so that the expansion can be performed entirely at setup-time.
 */
class InlinableConstitutiveModel {
public:
    virtual ~InlinableConstitutiveModel() = default;

    /**
     * @brief Stable kind id for caching/hashing (must not depend on pointer identity)
     */
    [[nodiscard]] virtual std::uint64_t kindId() const noexcept = 0;

    /**
     * @brief State access classification (pure vs stateful)
     */
    [[nodiscard]] virtual MaterialStateAccess stateAccess() const noexcept = 0;

    /**
     * @brief Expand model outputs into lowerable FormExpr + explicit state updates
     *
     * @param inputs Symbolic inputs (already in FormExpr form).
     * @param ctx Setup-time context (dimension, state base offset/layout).
     */
    [[nodiscard]] virtual InlinedConstitutiveExpansion
    inlineExpand(std::span<const FormExpr> inputs,
                 const InlinableConstitutiveContext& ctx) const = 0;

    // ---------------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------------

    [[nodiscard]] static constexpr std::uint64_t fnv1a64(std::string_view s) noexcept
    {
        std::uint64_t h = 14695981039346656037ULL;
        for (const unsigned char c : s) {
            h ^= static_cast<std::uint64_t>(c);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_INLINABLE_CONSTITUTIVE_MODEL_H

