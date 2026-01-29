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

// ---------------------------------------------------------------------------
// Versioned numeric helpers for new-physics operators (strict-mode compatible)
// ---------------------------------------------------------------------------

// Symmetric eigendecomposition (2x2 / 3x3), row-major input/output.
// - Eigenvalues are sorted in descending order.
// - Eigenvectors are stored as columns of the output matrix (row-major layout).
extern "C" void svmp_fe_jit_eig_sym_2x2_v1(const double* A, double* eigvals2, double* eigvecs4) noexcept;
extern "C" void svmp_fe_jit_eig_sym_3x3_v1(const double* A, double* eigvals3, double* eigvecs9) noexcept;

// General real-matrix eigendecomposition (3x3), row-major input.
// - Eigenvalues are returned as separate real/imag arrays (length 3).
// - Eigenvectors are stored as columns (row-major layout) in separate real/imag arrays (length 9).
extern "C" void svmp_fe_jit_eig_general_3x3_v1(const double* A,
                                              double* eigvals_real3,
                                              double* eigvals_imag3,
                                              double* eigvecs_real9,
                                              double* eigvecs_imag9) noexcept;

// Matrix functions for small dense matrices, row-major input/output.
// - expm is supported for symmetric 2x2/3x3.
// - logm/sqrtm/powm are restricted to SPD inputs (failure -> NaNs in output).
extern "C" void svmp_fe_jit_matrix_exp_2x2_v1(const double* A, double* expA4) noexcept;
extern "C" void svmp_fe_jit_matrix_exp_3x3_v1(const double* A, double* expA9) noexcept;
extern "C" void svmp_fe_jit_matrix_log_2x2_v1(const double* A, double* logA4) noexcept;
extern "C" void svmp_fe_jit_matrix_log_3x3_v1(const double* A, double* logA9) noexcept;
extern "C" void svmp_fe_jit_matrix_sqrt_2x2_v1(const double* A, double* sqrtA4) noexcept;
extern "C" void svmp_fe_jit_matrix_sqrt_3x3_v1(const double* A, double* sqrtA9) noexcept;
extern "C" void svmp_fe_jit_matrix_pow_2x2_v1(const double* A, double p, double* Ap4) noexcept;
extern "C" void svmp_fe_jit_matrix_pow_3x3_v1(const double* A, double p, double* Ap9) noexcept;

// Directional derivatives (Fréchet derivatives) of matrix functions.
// - expm is supported for symmetric 2x2/3x3.
// - logm/sqrtm/powm are restricted to SPD inputs (failure -> NaNs in output).
extern "C" void svmp_fe_jit_matrix_exp_dd_2x2_v1(const double* A, const double* dA, double* dExpA4) noexcept;
extern "C" void svmp_fe_jit_matrix_exp_dd_3x3_v1(const double* A, const double* dA, double* dExpA9) noexcept;
extern "C" void svmp_fe_jit_matrix_log_dd_2x2_v1(const double* A, const double* dA, double* dLogA4) noexcept;
extern "C" void svmp_fe_jit_matrix_log_dd_3x3_v1(const double* A, const double* dA, double* dLogA9) noexcept;
extern "C" void svmp_fe_jit_matrix_sqrt_dd_2x2_v1(const double* A, const double* dA, double* dSqrtA4) noexcept;
extern "C" void svmp_fe_jit_matrix_sqrt_dd_3x3_v1(const double* A, const double* dA, double* dSqrtA9) noexcept;
extern "C" void svmp_fe_jit_matrix_pow_dd_2x2_v1(const double* A, const double* dA, double p, double* dAp4) noexcept;
extern "C" void svmp_fe_jit_matrix_pow_dd_3x3_v1(const double* A, const double* dA, double p, double* dAp9) noexcept;

// Directional derivatives of symmetric eigen operators.
extern "C" void svmp_fe_jit_eigvec_sym_dd_2x2_v1(const double* A, const double* dA, std::int32_t which, double* dv2) noexcept;
extern "C" void svmp_fe_jit_eigvec_sym_dd_3x3_v1(const double* A, const double* dA, std::int32_t which, double* dv3) noexcept;
extern "C" void svmp_fe_jit_spectral_decomp_dd_2x2_v1(const double* A, const double* dA, double* dQ4) noexcept;
extern "C" void svmp_fe_jit_spectral_decomp_dd_3x3_v1(const double* A, const double* dA, double* dQ9) noexcept;

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_EXTERNAL_CALLS_H
