#ifndef SVMP_FE_CONSTRAINTS_COUPLED_NEUMANN_BC_H
#define SVMP_FE_CONSTRAINTS_COUPLED_NEUMANN_BC_H

/**
 * @file CoupledNeumannBC.h
 * @brief Neumann boundary condition depending on boundary functionals and auxiliary state
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Constraints/CoupledBCContext.h"
#include "Forms/BoundaryFunctional.h"

#include <variant>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

class CoupledNeumannBC {
public:
    CoupledNeumannBC(int boundary_marker,
                     std::vector<forms::BoundaryFunctional> required_integrals,
                     CoupledBCEvaluator evaluator)
        : marker_(boundary_marker)
        , required_integrals_(std::move(required_integrals))
        , evaluator_(std::move(evaluator))
    {
        FE_THROW_IF(marker_ < 0, InvalidArgumentException,
                    "CoupledNeumannBC: boundary_marker must be >= 0");
        FE_THROW_IF(!std::holds_alternative<CoupledBCEvaluator>(evaluator_), InvalidArgumentException,
                    "CoupledNeumannBC: evaluator must be scalar-valued");
    }

    CoupledNeumannBC(int boundary_marker,
                     std::vector<forms::BoundaryFunctional> required_integrals,
                     CoupledVectorBCEvaluator evaluator)
        : marker_(boundary_marker)
        , required_integrals_(std::move(required_integrals))
        , evaluator_(std::move(evaluator))
    {
        FE_THROW_IF(marker_ < 0, InvalidArgumentException,
                    "CoupledNeumannBC: boundary_marker must be >= 0");
        FE_THROW_IF(!std::holds_alternative<CoupledVectorBCEvaluator>(evaluator_), InvalidArgumentException,
                    "CoupledNeumannBC: evaluator must be vector-valued");
    }

    [[nodiscard]] int boundaryMarker() const noexcept { return marker_; }

    [[nodiscard]] const std::vector<forms::BoundaryFunctional>& requiredIntegrals() const noexcept
    {
        return required_integrals_;
    }

    [[nodiscard]] Real evaluate(const CoupledBCContext& ctx, Real x, Real y, Real z) const
    {
        if (const auto* f = std::get_if<CoupledBCEvaluator>(&evaluator_)) {
            return (*f)(ctx, x, y, z);
        }
        FE_THROW(NotImplementedException,
                 "CoupledNeumannBC::evaluate: vector-valued evaluator is not directly representable as a scalar coefficient");
    }

    [[nodiscard]] std::array<Real, 3> evaluateVector(const CoupledBCContext& ctx,
                                                     Real x, Real y, Real z,
                                                     const std::array<Real, 3>& normal) const
    {
        if (const auto* f = std::get_if<CoupledVectorBCEvaluator>(&evaluator_)) {
            return (*f)(ctx, x, y, z, normal);
        }
        return {evaluate(ctx, x, y, z), 0.0, 0.0};
    }

private:
    int marker_{-1};
    std::vector<forms::BoundaryFunctional> required_integrals_{};
    std::variant<CoupledBCEvaluator, CoupledVectorBCEvaluator> evaluator_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_COUPLED_NEUMANN_BC_H

