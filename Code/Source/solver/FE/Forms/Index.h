#ifndef SVMP_FE_FORMS_INDEX_H
#define SVMP_FE_FORMS_INDEX_H

/**
 * @file Index.h
 * @brief Basic index objects for UFL-like Einstein notation in FE/Forms
 *
 * This file defines a minimal Index/IndexSet vocabulary for building indexed
 * expressions (e.g., `A(i,j)`).
 *
 * Default path: call `forms::einsum(expr)` to lower indexed expressions into an
 * explicit component-wise sum before compilation/assembly.
 *
 * LLVM JIT tensor-calculus path: when enabled, fully-contracted indexed
 * expressions may be lowered to compact loop-based kernels without scalar-term
 * expansion (see `Forms/JIT/LLVM_JIT_IMPLEMENTATION_CHECKLIST.md`).
 */

#include <string>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Finite index set (range) for Einstein summation
 *
 * An extent of 0 indicates an "auto" extent that should be inferred from the
 * form's bound function-space topological dimension during lowering (falling
 * back to 3 if unknown).
 */
class IndexSet {
public:
    explicit IndexSet(int extent = 0) : extent_(extent) {}

    [[nodiscard]] int extent() const noexcept { return extent_; }

private:
    int extent_{0};
};

/**
 * @brief Symbolic index for Einstein-style notation
 */
class Index {
public:
    explicit Index(std::string name = {}, IndexSet set = IndexSet{});

    [[nodiscard]] int id() const noexcept { return id_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] int extent() const noexcept { return set_.extent(); }

private:
    static int nextId() noexcept;

    int id_{0};
    std::string name_{};
    IndexSet set_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_INDEX_H
