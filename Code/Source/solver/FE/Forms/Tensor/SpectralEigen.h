#ifndef SVMP_FE_FORMS_TENSOR_SPECTRAL_EIGEN_H
#define SVMP_FE_FORMS_TENSOR_SPECTRAL_EIGEN_H

/**
 * @file SpectralEigen.h
 * @brief Small symmetric eigenvalue helpers with a stable C ABI for LLVM JIT.
 *
 * Notes:
 * - Matrices are provided in row-major order.
 * - `which` is 0-based and refers to eigenvalues sorted in descending order.
 * - These helpers are intended for strict-mode JIT support and interpreter parity.
 */

#include <cstdint>

extern "C" double svmp_fe_sym_eigenvalue_2x2_v1(const double* A, std::int32_t which);
extern "C" double svmp_fe_sym_eigenvalue_3x3_v1(const double* A, std::int32_t which);

extern "C" double svmp_fe_sym_eigenvalue_dd_2x2_v1(const double* A, const double* dA, std::int32_t which);
extern "C" double svmp_fe_sym_eigenvalue_dd_3x3_v1(const double* A, const double* dA, std::int32_t which);

// Directional derivative of (eig_sym_dd(A, B, which)) w.r.t A in direction dA.
extern "C" double svmp_fe_sym_eigenvalue_ddA_2x2_v1(const double* A, const double* B, const double* dA, std::int32_t which);
extern "C" double svmp_fe_sym_eigenvalue_ddA_3x3_v1(const double* A, const double* B, const double* dA, std::int32_t which);

#endif // SVMP_FE_FORMS_TENSOR_SPECTRAL_EIGEN_H

