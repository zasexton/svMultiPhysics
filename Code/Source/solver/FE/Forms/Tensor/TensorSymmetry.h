#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_SYMMETRY_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_SYMMETRY_H

/**
 * @file TensorSymmetry.h
 * @brief Symmetry metadata for tensor expressions
 */

#include "Forms/Tensor/TensorIndex.h"

#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

/**
 * @brief Symmetry properties of tensor indices
 */
enum class SymmetryType : std::uint8_t {
    None,
    Symmetric,
    Antisymmetric,
    Hermitian,

    FullySymmetric,
    FullyAntisymmetric,

    MajorSymmetric,
    MinorSymmetric,
    FullElasticity,
};

struct SymmetryPair {
    int index_a{0};
    int index_b{1};
    SymmetryType type{SymmetryType::None};
};

/**
 * @brief Complete symmetry specification for a tensor
 */
struct TensorSymmetry {
    std::vector<SymmetryPair> pairs{};

    [[nodiscard]] bool isSymmetricIn(int i, int j) const noexcept;
    [[nodiscard]] bool isAntisymmetricIn(int i, int j) const noexcept;

    [[nodiscard]] int numIndependentComponents(int dim) const;
    [[nodiscard]] std::vector<MultiIndex> independentComponents(int dim) const;

    static TensorSymmetry symmetric2();
    static TensorSymmetry antisymmetric2();
    static TensorSymmetry elasticity();
};

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_SYMMETRY_H

