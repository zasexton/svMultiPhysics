#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_INTERPRETER_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_INTERPRETER_H

/**
 * @file TensorInterpreter.h
 * @brief Lightweight interpreter for loop-based TensorIR (non-JIT fallback/testing)
 *
 * This module evaluates `forms::tensor::TensorIR` by executing its `LoopNestProgram`
 * using in-memory temporary buffers described by the `TensorAllocationPlan`.
 *
 * It is intended for:
 * - interpreter-mode fallback for tensor calculus (avoids scalar `einsum()` expansion),
 * - correctness validation against scalar-expanded evaluation,
 * - micro-benchmarks (no LLVM dependency).
 *
 * Current contract:
 * - Only scalar outputs (`program.output.rank == 0`) are supported.
 */

#include "Forms/FormExpr.h"
#include "Forms/Tensor/TensorIR.h"
#include "Forms/Value.h"

#include <functional>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct TensorInterpreterCallbacks {
    // Evaluate a tensor-valued base expression (the child of IndexedAccess) to a concrete Value.
    std::function<Value<Real>(const FormExpr&)> eval_value{};

    // Evaluate a scalar expression (scalar prefactors) to a Real.
    std::function<Real(const FormExpr&)> eval_scalar{};
};

struct TensorInterpreterResult {
    bool ok{true};
    std::string message{};
    Real value{0.0};
};

/**
 * @brief Evaluate a TensorIR that produces a scalar value.
 *
 * @param ir Loop-based tensor IR (must have a valid allocation plan).
 * @param cb Callbacks for evaluating base tensors and scalar prefactors.
 */
[[nodiscard]] TensorInterpreterResult evalTensorIRScalar(const TensorIR& ir,
                                                         const TensorInterpreterCallbacks& cb);

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_INTERPRETER_H

