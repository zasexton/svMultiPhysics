#ifndef SVMP_FE_FORMS_TENSOR_SPECIAL_TENSORS_H
#define SVMP_FE_FORMS_TENSOR_SPECIAL_TENSORS_H

/**
 * @file SpecialTensors.h
 * @brief Special/intrinsic tensors (Kronecker delta, Levi-Civita, metric)
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <functional>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

/**
 * @brief Classification of special/intrinsic tensors
 */
enum class SpecialTensorKind : std::uint8_t {
    None,

    KroneckerDelta,
    MetricTensor,
    InverseMetric,

    LeviCivita,
    LeviCivitaUpper,

    DeviatoricProjector,
    SymmetricProjector,

    DeformationGradient,
    GreenLagrange,
    CauchyGreen,
};

namespace special {

[[nodiscard]] constexpr int delta(int i, int j) noexcept
{
    return (i == j) ? 1 : 0;
}

[[nodiscard]] constexpr int levicivita(int i, int j, int k) noexcept
{
    if (i == j || j == k || i == k) return 0;
    // Even permutations of (0,1,2) yield +1, odd yield -1.
    // This compact formula matches the sign of the permutation for distinct indices.
    return ((i - j) * (j - k) * (k - i)) / 2;
}

[[nodiscard]] constexpr int levicivita2d(int i, int j) noexcept
{
    if (i == j) return 0;
    return (i < j) ? 1 : -1;
}

using MetricMatrix = std::array<std::array<Real, 3>, 3>;

/**
 * @brief Identity metric tensor g_ij for dim=1..3 (embedded in 3×3 storage)
 */
[[nodiscard]] inline MetricMatrix identityMetric(int dim = 3) noexcept
{
    MetricMatrix g{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.0;
        }
    }
    const int d = (dim < 1) ? 1 : (dim > 3 ? 3 : dim);
    for (int i = 0; i < d; ++i) {
        g[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
    }
    return g;
}

/**
 * @brief Identity inverse metric tensor g^ij for dim=1..3 (embedded in 3×3 storage)
 */
[[nodiscard]] inline MetricMatrix identityInverseMetric(int dim = 3) noexcept
{
    // For the identity metric, g^{-1} == g.
    return identityMetric(dim);
}

/**
 * @brief Metric tensor evaluation hooks for curved/moving meshes
 *
 * The default FE/Forms metric is the identity (Euclidean) metric. For embedded manifolds,
 * ALE/moving meshes, or curvilinear coordinates, callers can provide custom evaluators.
 */
struct MetricTensorHooks {
    int dim{3};
    std::function<MetricMatrix(Real, Real, Real)> metric = [](Real, Real, Real) { return identityMetric(3); };
    std::function<MetricMatrix(Real, Real, Real)> inverse_metric = [](Real, Real, Real) { return identityInverseMetric(3); };
};

[[nodiscard]] inline MetricTensorHooks identityMetricHooks(int dim = 3)
{
    MetricTensorHooks hooks;
    hooks.dim = dim;
    hooks.metric = [dim](Real, Real, Real) { return identityMetric(dim); };
    hooks.inverse_metric = [dim](Real, Real, Real) { return identityInverseMetric(dim); };
    return hooks;
}

} // namespace special

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_SPECIAL_TENSORS_H
