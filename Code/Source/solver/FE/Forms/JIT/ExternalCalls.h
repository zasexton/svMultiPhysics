#ifndef SVMP_FE_FORMS_JIT_EXTERNAL_CALLS_H
#define SVMP_FE_FORMS_JIT_EXTERNAL_CALLS_H

/**
 * @file ExternalCalls.h
 * @brief Stable C ABIs for LLVM JIT relaxed-mode external calls
 *
 * The JIT backend may lower opaque operations (e.g., std::function coefficients,
 * non-inlinable constitutive models) by emitting calls to these symbols.
 *
 * This header contains no LLVM dependencies.
 */

#include <cstdint>
#include <type_traits>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace external {

inline constexpr std::uint32_t kExternalCallsABIVersionV1 = 1u;
inline constexpr std::uint64_t kExternalCallsMagicV1 = 0x3154494a504d5653ULL; // "SVMPJIT1"

enum class ValueKindV1 : std::uint32_t {
    Scalar = 0u,
    Vector = 1u,
    Matrix = 2u,
    Tensor3 = 3u,
    Tensor4 = 4u,
};

enum class TraceSideV1 : std::uint32_t {
    Minus = 0u,
    Plus = 1u,
};

struct ValueViewV1 {
    std::uint32_t kind{static_cast<std::uint32_t>(ValueKindV1::Scalar)};
    std::uint32_t length{1u};
    const double* data{nullptr};
};

struct ValueBatchViewV1 {
    std::uint32_t kind{static_cast<std::uint32_t>(ValueKindV1::Scalar)};
    std::uint32_t length{1u};
    const double* data{nullptr};
    std::uint32_t stride{0u};
    std::uint32_t reserved{0u};
};

using EvalCoefficientFnV1 = bool (*)(const void* context,
                                     const void* side_ptr,
                                     std::uint32_t q,
                                     const void* coeff_node_ptr,
                                     std::uint32_t out_kind,
                                     double* out_values,
                                     std::uint32_t out_len) noexcept;

using EvalConstitutiveFnV1 = bool (*)(const void* context,
                                      const void* side_ptr,
                                      std::uint32_t q,
                                      std::uint32_t trace_side,
                                      const void* constitutive_node_ptr,
                                      std::uint32_t output_index,
                                      const ValueViewV1* inputs,
                                      std::uint32_t num_inputs,
                                      std::uint32_t out_kind,
                                      double* out_values,
                                      std::uint32_t out_len) noexcept;

struct ExternalCallTableV1 {
    std::uint64_t magic{kExternalCallsMagicV1};
    std::uint32_t abi_version{kExternalCallsABIVersionV1};
    std::uint32_t reserved{0u};

    const void* context{nullptr};
    EvalCoefficientFnV1 eval_coefficient{nullptr};
    EvalConstitutiveFnV1 eval_constitutive{nullptr};
};

static_assert(std::is_standard_layout_v<ValueViewV1>);
static_assert(std::is_trivially_copyable_v<ValueViewV1>);
static_assert(std::is_standard_layout_v<ValueBatchViewV1>);
static_assert(std::is_trivially_copyable_v<ValueBatchViewV1>);
static_assert(std::is_standard_layout_v<ExternalCallTableV1>);
static_assert(std::is_trivially_copyable_v<ExternalCallTableV1>);

} // namespace external

extern "C" double svmp_fe_jit_coeff_eval_scalar_v1(const void* side_ptr,
                                                   std::uint32_t q,
                                                   const void* coeff_node_ptr) noexcept;

extern "C" void svmp_fe_jit_coeff_eval_vector_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 const void* coeff_node_ptr,
                                                 double* out3) noexcept;

extern "C" void svmp_fe_jit_coeff_eval_matrix_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 const void* coeff_node_ptr,
                                                 double* out9) noexcept;

extern "C" void svmp_fe_jit_coeff_eval_tensor3_v1(const void* side_ptr,
                                                  std::uint32_t q,
                                                  const void* coeff_node_ptr,
                                                  double* out27) noexcept;

extern "C" void svmp_fe_jit_coeff_eval_tensor4_v1(const void* side_ptr,
                                                  std::uint32_t q,
                                                  const void* coeff_node_ptr,
                                                  double* out81) noexcept;

extern "C" void svmp_fe_jit_constitutive_eval_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 std::uint32_t trace_side,
                                                 const void* constitutive_node_ptr,
                                                 std::uint32_t output_index,
                                                 const external::ValueViewV1* inputs,
                                                 std::uint32_t num_inputs,
                                                 std::uint32_t out_kind,
                                                 double* out_values,
                                                 std::uint32_t out_len) noexcept;

extern "C" void svmp_fe_jit_constitutive_eval_batch_v1(const void* side_ptr,
                                                       const std::uint32_t* q_indices,
                                                       std::uint32_t num_q,
                                                       std::uint32_t trace_side,
                                                       const void* constitutive_node_ptr,
                                                       std::uint32_t output_index,
                                                       const external::ValueBatchViewV1* inputs,
                                                       std::uint32_t num_inputs,
                                                       std::uint32_t out_kind,
                                                       double* out_values,
                                                       std::uint32_t out_len,
                                                       std::uint32_t out_stride) noexcept;

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_EXTERNAL_CALLS_H

