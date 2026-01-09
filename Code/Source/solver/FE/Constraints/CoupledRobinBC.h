#ifndef SVMP_FE_CONSTRAINTS_COUPLED_ROBIN_BC_H
#define SVMP_FE_CONSTRAINTS_COUPLED_ROBIN_BC_H

/**
 * @file CoupledRobinBC.h
 * @brief Robin boundary condition depending on boundary functionals and auxiliary state
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Constraints/CoupledBCContext.h"
#include "Forms/BoundaryFunctional.h"

#include <functional>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

class CoupledRobinBC {
public:
    using CoefficientEvaluator =
        std::function<Real(const CoupledBCContext& ctx, Real x, Real y, Real z)>;

    CoupledRobinBC(int boundary_marker,
                   std::vector<forms::BoundaryFunctional> required_integrals,
                   CoefficientEvaluator alpha,
                   CoefficientEvaluator beta,
                   CoefficientEvaluator g)
        : marker_(boundary_marker)
        , required_integrals_(std::move(required_integrals))
        , alpha_(std::move(alpha))
        , beta_(std::move(beta))
        , g_(std::move(g))
    {
        FE_THROW_IF(marker_ < 0, InvalidArgumentException,
                    "CoupledRobinBC: boundary_marker must be >= 0");
        FE_THROW_IF(!alpha_ || !beta_ || !g_, InvalidArgumentException,
                    "CoupledRobinBC: coefficient evaluators must be set");
    }

    [[nodiscard]] int boundaryMarker() const noexcept { return marker_; }

    [[nodiscard]] const std::vector<forms::BoundaryFunctional>& requiredIntegrals() const noexcept
    {
        return required_integrals_;
    }

    [[nodiscard]] Real alpha(const CoupledBCContext& ctx, Real x, Real y, Real z) const { return alpha_(ctx, x, y, z); }
    [[nodiscard]] Real beta(const CoupledBCContext& ctx, Real x, Real y, Real z) const { return beta_(ctx, x, y, z); }
    [[nodiscard]] Real g(const CoupledBCContext& ctx, Real x, Real y, Real z) const { return g_(ctx, x, y, z); }

private:
    int marker_{-1};
    std::vector<forms::BoundaryFunctional> required_integrals_{};
    CoefficientEvaluator alpha_{};
    CoefficientEvaluator beta_{};
    CoefficientEvaluator g_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_COUPLED_ROBIN_BC_H

