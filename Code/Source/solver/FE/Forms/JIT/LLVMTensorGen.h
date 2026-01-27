#ifndef SVMP_FE_FORMS_JIT_LLVM_TENSOR_GEN_H
#define SVMP_FE_FORMS_JIT_LLVM_TENSOR_GEN_H

/**
 * @file LLVMTensorGen.h
 * @brief LLVM emission helpers for tensor-calculus TensorIR (loop-nest lowering)
 *
 * This module is only compiled/used when `SVMP_FE_ENABLE_LLVM_JIT` is enabled.
 */

#include "Forms/Tensor/TensorIR.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

namespace llvm {
class Function;
class Value;
} // namespace llvm
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

struct LLVMTensorGenOptions {
    bool vectorize{true};
    bool enable_polly{false};
};

#if SVMP_FE_ENABLE_LLVM_JIT
/**
 * @brief LLVM codegen for loop-based tensor evaluation programs.
 *
 * The generator is intentionally agnostic to FE/Forms terminals (Test/Trial,
 * fields, coefficients). Those are provided via callbacks that evaluate scalar
 * expressions and load input-tensor elements.
 */
class LLVMTensorGen final {
public:
    using ScalarEvalFn = std::function<llvm::Value*(const FormExpr& scalar_expr)>;

    using LoadInputTensorElementFn =
        std::function<llvm::Value*(int tensor_id,
                                   const forms::tensor::TensorSpec& spec,
                                   const std::vector<llvm::Value*>& index_env)>;

    LLVMTensorGen(llvm::LLVMContext& ctx,
                  llvm::IRBuilder<>& builder,
                  llvm::Function& fn,
                  LLVMTensorGenOptions options);

    /**
     * @brief Emit code for a scalar-output TensorIR program.
     *
     * Requirements:
     * - `ir.program.ok == true`
     * - `ir.program.output.rank == 0`
     */
    [[nodiscard]] llvm::Value* emitScalar(const forms::tensor::TensorIR& ir,
                                          const ScalarEvalFn& eval_scalar,
                                          const LoadInputTensorElementFn& load_input) const;

private:
    llvm::LLVMContext* ctx_{nullptr};
    llvm::IRBuilder<>* builder_{nullptr};
    llvm::Function* fn_{nullptr};
    LLVMTensorGenOptions options_{};
};
#endif

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_LLVM_TENSOR_GEN_H

