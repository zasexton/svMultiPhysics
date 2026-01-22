#ifndef SVMP_FE_ASSEMBLY_JIT_FIELD_SOLUTIONS_H
#define SVMP_FE_ASSEMBLY_JIT_FIELD_SOLUTIONS_H

/**
 * @file FieldSolutions.h
 * @brief Stable POD table entries for multi-field DiscreteField/StateField data in the JIT ABI.
 *
 * This header intentionally contains no LLVM dependencies.
 */

#include "Core/Types.h"
#include "Core/Alignment.h"

#include <cstdint>
#include <type_traits>

namespace svmp {
namespace FE {
namespace assembly {
namespace jit {

/// Preferred alignment for Real-valued JIT ABI tables (cache-line aligned).
inline constexpr std::size_t kJITPointerAlignmentBytes = kFEPreferredAlignmentBytes;

/**
 * @brief One entry describing values/derivatives for a single FE field at quadrature points.
 *
 * Layout notes:
 * - All pointers may be null if the corresponding data was not requested/bound.
 * - Array extents are defined by KernelSideArgs::{n_qpts,value_dim,history_count}.
 *
 * Scalar-valued field (FieldType::Scalar):
 * - values:            [n_qpts]
 * - gradients_xyz:     [n_qpts * 3]
 * - hessians:          [n_qpts * 9]
 * - laplacians:        [n_qpts]
 * - history_values:    [history_count * n_qpts]
 *
 * Vector-valued field (FieldType::Vector):
 * - vector_values_xyz: [n_qpts * 3]  (components beyond value_dim may be zero)
 * - jacobians:         [n_qpts * 9]
 * - component_hessians:[n_qpts * value_dim * 9]
 * - component_laplacians:[n_qpts * value_dim]
 * - history_vector_values_xyz: [history_count * n_qpts * 3]
 */
struct FieldSolutionEntryV1 {
    std::int32_t field_id{-1};
    std::uint32_t field_type{0};   // Core::FieldType
    std::uint32_t value_dim{1};    // 1..3 for vector-valued fields

    // Scalar-valued
    const Real* values{nullptr};
    const Real* gradients_xyz{nullptr};
    const Real* hessians{nullptr};
    const Real* laplacians{nullptr};

    // Vector-valued
    const Real* vector_values_xyz{nullptr};
    const Real* jacobians{nullptr};
    const Real* component_hessians{nullptr};
    const Real* component_laplacians{nullptr};

    // History (k=1 is u^{n-1})
    std::uint32_t history_count{0};
    const Real* history_values{nullptr};
    const Real* history_vector_values_xyz{nullptr};
};

static_assert(std::is_standard_layout_v<FieldSolutionEntryV1>);
static_assert(std::is_trivially_copyable_v<FieldSolutionEntryV1>);

} // namespace jit
} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_JIT_FIELD_SOLUTIONS_H
