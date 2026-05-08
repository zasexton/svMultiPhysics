#ifndef SVMP_FE_FORMS_JIT_JIT_SPECIALIZATION_H
#define SVMP_FE_FORMS_JIT_JIT_SPECIALIZATION_H

/**
 * @file JITSpecialization.h
 * @brief Compile-time size specialization for LLVM JIT kernels
 *
 * This header contains no LLVM dependencies.
 */

#include <cstdint>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

enum class IntegralDomain : std::uint8_t;

namespace jit {

/**
 * @brief Compile-time basis tables for one test/trial side.
 *
 * Tables are stored in the same Q-major layout used by the JIT ABI:
 *   scalar_values_qmajor[(q * n_dofs) + dof]
 *   ref_gradients_qmajor[((q * n_dofs) + dof) * 3 + d]
 *   ref_hessians_qmajor[((q * n_dofs) + dof) * 9 + r * 3 + c]
 */
struct JITBakedBasisSide {
    bool enabled{false};
    bool scalar_basis{false};
    bool ref_gradients_qp_constant{false};

    std::uint32_t n_qpts{0};
    std::uint32_t n_dofs{0};

    std::uint64_t basis_hash{0};
    std::uint64_t quadrature_hash{0};
    std::uint64_t table_hash{0};

    std::vector<double> scalar_values_qmajor{};
    std::vector<double> ref_gradients_qmajor{};
    std::vector<double> ref_hessians_qmajor{};
};

/**
 * @brief Optional JIT-baked basis specialization payload.
 *
 * Scalar basis values are independent of cell geometry and can always be baked
 * for scalar-product bases. Reference gradients and Hessians can be baked for
 * affine cells; the JIT still multiplies by the cell-specific inverse Jacobian.
 */
struct JITBakedBasisSpec {
    bool enabled{false};
    bool geometry_affine{false};
    std::uint64_t hash{0};

    JITBakedBasisSide test{};
    JITBakedBasisSide trial{};
};

/**
 * @brief Optional compile-time specialization for kernel loop trip counts.
 *
 * The specialization applies to the kernel group matching @p domain. For Cell
 * and Boundary kernels, only the "minus" fields are used (single-side
 * integration). For face kernels, both minus and plus may be provided.
 */
struct JITCompileSpecialization {
    IntegralDomain domain{};

    std::optional<std::uint32_t> n_qpts_minus{};
    std::optional<std::uint32_t> n_test_dofs_minus{};
    std::optional<std::uint32_t> n_trial_dofs_minus{};

    std::optional<std::uint32_t> n_qpts_plus{};
    std::optional<std::uint32_t> n_test_dofs_plus{};
    std::optional<std::uint32_t> n_trial_dofs_plus{};

    bool is_affine{false};  ///< True for P1 simplices (Tet4, Tri3) — enables QP-constant term hoisting
    JITBakedBasisSpec baked_basis{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_SPECIALIZATION_H
