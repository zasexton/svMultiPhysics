#ifndef SVMP_FE_SYSTEMS_AUXILIARY_STATE_BUILDER_H
#define SVMP_FE_SYSTEMS_AUXILIARY_STATE_BUILDER_H

/**
 * @file AuxiliaryStateBuilder.h
 * @brief Convenience builder for AuxiliaryStateRegistration (scalar ODE variables)
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Forms/BoundaryFunctional.h"
#include "Systems/AuxiliaryState.h"
#include "Systems/ODEIntegrator.h"

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class AuxiliaryStateBuilder {
public:
    AuxiliaryStateBuilder(std::string name, Real initial_value)
        : name_(std::move(name))
        , initial_value_(initial_value)
    {
        FE_THROW_IF(name_.empty(), InvalidArgumentException,
                    "AuxiliaryStateBuilder: empty name");
    }

    AuxiliaryStateBuilder& withRHS(forms::FormExpr rhs)
    {
        rhs_ = std::move(rhs);
        return *this;
    }

    AuxiliaryStateBuilder& withJacobian(forms::FormExpr d_rhs_dX)
    {
        d_rhs_dX_ = std::move(d_rhs_dX);
        return *this;
    }

    AuxiliaryStateBuilder& requiresIntegral(forms::BoundaryFunctional functional)
    {
        required_integrals_.push_back(std::move(functional));
        return *this;
    }

    AuxiliaryStateBuilder& withIntegrator(ODEMethod method) noexcept
    {
        integrator_ = method;
        return *this;
    }

    [[nodiscard]] AuxiliaryStateRegistration build() const
    {
        FE_THROW_IF(!rhs_.isValid(), InvalidArgumentException,
                    "AuxiliaryStateBuilder::build: RHS function is not set");

        AuxiliaryStateRegistration reg;
        reg.spec.size = 1;
        reg.spec.name = name_;
        reg.initial_values = {initial_value_};
        reg.required_integrals = required_integrals_;
        reg.rhs = rhs_;
        reg.d_rhs_dX = d_rhs_dX_;
        reg.integrator = integrator_;

        // Best-effort: associate markers from required integrals (if any).
        std::vector<int> markers;
        for (const auto& f : reg.required_integrals) {
            if (f.boundary_marker >= 0) {
                markers.push_back(f.boundary_marker);
            }
        }
        std::sort(markers.begin(), markers.end());
        markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
        reg.spec.associated_markers = std::move(markers);

        return reg;
    }

private:
    std::string name_{};
    Real initial_value_{0.0};
    std::vector<forms::BoundaryFunctional> required_integrals_{};
    forms::FormExpr rhs_{};
    std::optional<forms::FormExpr> d_rhs_dX_{};
    ODEMethod integrator_{ODEMethod::BackwardEuler};
};

[[nodiscard]] inline AuxiliaryStateBuilder auxiliaryODE(std::string name, Real initial_value)
{
    return AuxiliaryStateBuilder(std::move(name), initial_value);
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_STATE_BUILDER_H
