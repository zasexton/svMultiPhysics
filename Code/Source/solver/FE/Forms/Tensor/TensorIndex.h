#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_INDEX_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_INDEX_H

/**
 * @file TensorIndex.h
 * @brief Tensor index vocabulary for tensor-calculus-aware symbolic transforms
 *
 * This module is intentionally independent from the existing UFL-like `forms::Index`
 * (which is currently used to build `FormExprType::IndexedAccess` nodes). The
 * tensor-calculus roadmap uses these richer types for variance/symmetry-aware
 * analysis and canonicalization.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

/**
 * @brief Index variance (covariant vs contravariant)
 */
enum class IndexVariance : std::uint8_t {
    Lower,  ///< Covariant (subscript): A_i
    Upper,  ///< Contravariant (superscript): A^i
    None    ///< Scalar/component index (no variance semantics)
};

/**
 * @brief Index classification (relative to an Einstein-summation expression)
 */
enum class IndexRole : std::uint8_t {
    Free,   ///< Appears once
    Dummy,  ///< Appears twice (summed over)
    Fixed   ///< Concrete integer value
};

/**
 * @brief Symbolic tensor index
 */
struct TensorIndex {
    int id{-1};                   ///< Unique identifier (expression-local is sufficient)
    std::string name{};           ///< Display name: "i", "j", ...
    IndexVariance variance{IndexVariance::Lower};
    IndexRole role{IndexRole::Free};
    int dimension{3};             ///< Extent/range: 0..dimension-1
    std::optional<int> fixed_value{}; ///< Present iff role==Fixed

    [[nodiscard]] bool isFree() const noexcept { return role == IndexRole::Free; }
    [[nodiscard]] bool isDummy() const noexcept { return role == IndexRole::Dummy; }
    [[nodiscard]] bool isFixed() const noexcept { return role == IndexRole::Fixed || fixed_value.has_value(); }

    [[nodiscard]] TensorIndex raised() const
    {
        TensorIndex out = *this;
        if (out.variance == IndexVariance::Lower) out.variance = IndexVariance::Upper;
        return out;
    }

    [[nodiscard]] TensorIndex lowered() const
    {
        TensorIndex out = *this;
        if (out.variance == IndexVariance::Upper) out.variance = IndexVariance::Lower;
        return out;
    }
};

/**
 * @brief Multi-index for tensor components
 */
struct MultiIndex {
    std::vector<TensorIndex> indices{};

    [[nodiscard]] int rank() const noexcept { return static_cast<int>(indices.size()); }

    [[nodiscard]] std::vector<int> freeIndices() const;

    /**
     * @brief Return pairs of positions that represent internal trace-style contractions
     *
     * This is limited to repeated ids within this MultiIndex (e.g., A(i,i)).
     * General cross-tensor contractions are analyzed at the expression level.
     */
    [[nodiscard]] std::vector<std::pair<int, int>> contractionPairs() const;

    [[nodiscard]] bool isFullyContracted() const noexcept { return freeIndices().empty(); }
};

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_INDEX_H

