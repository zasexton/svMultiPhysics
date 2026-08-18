#ifndef SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BC_FACTORIES_H

/**
 * @file MeshMotionBCFactories.h
 * @brief Shared mesh-motion boundary-condition translation helpers.
 */

#include "Physics/Formulations/MeshMotion/MeshMotionBoundaryOptions.h"

#include "FE/Analysis/BoundaryConditionDescriptor.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/BoundaryCondition.h"
#include "FE/Forms/JIT/JITValidation.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {
namespace Factories {

[[nodiscard]] inline bool isFinitePositiveScalarLiteral(
    const FE::forms::bc::ScalarValue& value)
{
    const auto literal = [&]() -> std::optional<FE::Real> {
        if (const auto* real = std::get_if<FE::Real>(&value)) {
            return *real;
        }
        if (const auto* expression =
                std::get_if<FE::forms::FormExpr>(&value);
            expression != nullptr && expression->node() != nullptr) {
            return expression->node()->constantValue();
        }
        return std::nullopt;
    }();
    return literal.has_value() && std::isfinite(*literal) &&
           *literal > FE::Real{0.0};
}

class TangentialConstraintBoundaryCondition final : public FE::forms::bc::BoundaryCondition {
public:
    TangentialConstraintBoundaryCondition(int boundary_marker,
                                          FE::forms::FormExpr penalty,
                                          FE::forms::FormExpr target)
        : boundary_marker_(boundary_marker)
        , penalty_(std::move(penalty))
        , target_(std::move(target))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument(
                "TangentialConstraintBoundaryCondition: boundary_marker must be >= 0");
        }
        if (!penalty_.isValid()) {
            throw std::invalid_argument(
                "TangentialConstraintBoundaryCondition: invalid penalty expression");
        }
        if (!target_.isValid()) {
            throw std::invalid_argument(
                "TangentialConstraintBoundaryCondition: invalid target expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FE::forms::FormExpr& residual,
                              const FE::forms::FormExpr& u,
                              const FE::forms::FormExpr& v) const override
    {
        const auto n = FE::forms::FormExpr::normal();
        const auto gap = u - target_;
        const auto tangential_work =
            FE::forms::inner(gap, v) -
            FE::forms::normalTrace(gap, n) * FE::forms::normalTrace(v, n);
        residual = residual + (penalty_ * tangential_work).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<FE::forms::bc::StrongDirichlet>
    getStrongConstraints(FE::FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<FE::analysis::BoundaryConditionDescriptor>
    analysisMetadata(FE::FieldId field_id, const FE::systems::FESystem* /*system*/) const override
    {
        FE::analysis::BoundaryConditionDescriptor d;
        d.primary_variable = FE::analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = FE::analysis::TraceKind::TangentialComponent;
        d.enforcement_kind = FE::analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "TangentialConstraintBoundaryCondition on marker " +
                   std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FE::forms::FormExpr penalty_{};
    FE::forms::FormExpr target_{};
};

template <class NaturalBC>
[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toVectorNaturalBC(const NaturalBC& bc,
                  int dim,
                  std::string_view context,
                  std::string_view value_prefix = "mesh_motion_natural")
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, context);
    auto values = FE::forms::bc::toVectorExpr(
        bc.value,
        dim,
        value_prefix,
        marker,
        FE::forms::bc::ComponentValueNameStyle::Component);
    return std::make_unique<FE::forms::bc::NaturalBC>(
        marker,
        FE::forms::FormExpr::asVector(std::move(values)));
}

template <class RobinBC>
[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toVectorRobinBC(const RobinBC& bc,
                int dim,
                std::string_view context,
                std::string_view alpha_prefix = "mesh_motion_robin_alpha",
                std::string_view target_prefix = "mesh_motion_robin_target")
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, context);
    const auto alpha = FE::forms::bc::toScalarExpr(
        bc.alpha,
        FE::forms::bc::markerValueName(alpha_prefix, marker));
    auto target_components = FE::forms::bc::toVectorExpr(
        bc.target,
        dim,
        target_prefix,
        marker,
        FE::forms::bc::ComponentValueNameStyle::Component);
    const auto target = FE::forms::FormExpr::asVector(std::move(target_components));
    return std::make_unique<FE::forms::bc::RobinBC>(marker, alpha, alpha * target);
}

template <class DirichletBC>
[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toVectorEssentialBC(const DirichletBC& bc,
                    int dim,
                    std::string_view context,
                    std::string_view field_symbol,
                    std::string_view value_prefix = "mesh_displacement")
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, context);
    auto values = FE::forms::bc::toVectorExpr(
        bc.value,
        dim,
        value_prefix,
        marker,
        FE::forms::bc::ComponentValueNameStyle::Component);
    return std::make_unique<FE::forms::bc::EssentialBC>(
        marker,
        std::move(values),
        std::string(field_symbol));
}

[[nodiscard]] inline FE::forms::FormExpr
effectiveNormalConstraintTarget(
    const NormalConstraintBC& bc,
    int dimension,
    std::string_view context,
    std::string_view target_prefix = "mesh_motion_normal_target",
    std::string_view time_scale_prefix = "mesh_motion_normal_time_scale")
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, context);
    auto target = FE::forms::bc::toScalarExpr(
        bc.target,
        FE::forms::bc::markerValueName(target_prefix, marker));

    switch (bc.quantity) {
    case NormalConstraintQuantity::Displacement:
        break;
    case NormalConstraintQuantity::Velocity: {
        if (!isFinitePositiveScalarLiteral(bc.velocity_time_scale)) {
            throw std::invalid_argument(
                std::string(context) +
                ": normal constraint velocity time scale must be a "
                "finite positive literal");
        }
        const auto time_scale = FE::forms::bc::toScalarExpr(
            bc.velocity_time_scale,
            FE::forms::bc::markerValueName(time_scale_prefix, marker));
        target = time_scale * target;
        break;
    }
    default:
        throw std::invalid_argument(
            std::string(context) +
            ": unknown normal constraint quantity");
    }

    if (!FE::forms::jit::hasScalarValueShape(target, dimension)) {
        throw std::invalid_argument(
            std::string(context) +
            ": normal constraint target must be scalar-valued");
    }

    return target;
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toNormalConstraintBC(const NormalConstraintBC& bc,
                     int dimension,
                     std::string_view context,
                     std::string_view penalty_prefix = "mesh_motion_normal_penalty",
                     std::string_view target_prefix = "mesh_motion_normal_target",
                     std::string_view time_scale_prefix = "mesh_motion_normal_time_scale")
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, context);
    if (!isFinitePositiveScalarLiteral(bc.penalty)) {
        throw std::invalid_argument(
            std::string(context) +
            ": normal constraint penalty must be a finite positive "
            "literal");
    }
    const auto penalty = FE::forms::bc::toScalarExpr(
        bc.penalty,
        FE::forms::bc::markerValueName(penalty_prefix, marker));
    const auto target = effectiveNormalConstraintTarget(
        bc, dimension, context, target_prefix, time_scale_prefix);

    return FE::forms::bc::makeTraceRobinBC(
        marker,
        penalty,
        penalty * target,
        FE::forms::bc::ScalarTraceOperator::NormalComponent);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition>
toTangentialConstraintBC(const TangentialPolicyBC& bc,
                         int dim,
                         std::string_view context,
                         std::string_view penalty_prefix = "mesh_motion_tangential_penalty",
                         std::string_view target_prefix = "mesh_motion_tangential_target",
                         std::string_view time_scale_prefix = "mesh_motion_tangential_time_scale")
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, context);
    const auto penalty = FE::forms::bc::toScalarExpr(
        bc.penalty,
        FE::forms::bc::markerValueName(penalty_prefix, marker));
    auto target_components = FE::forms::bc::toVectorExpr(
        bc.target,
        dim,
        target_prefix,
        marker,
        FE::forms::bc::ComponentValueNameStyle::Component);
    auto target = FE::forms::FormExpr::asVector(std::move(target_components));

    switch (bc.quantity) {
    case TangentialConstraintQuantity::Displacement:
        break;
    case TangentialConstraintQuantity::Velocity: {
        const auto time_scale = FE::forms::bc::toScalarExpr(
            bc.velocity_time_scale,
            FE::forms::bc::markerValueName(time_scale_prefix, marker));
        target = time_scale * target;
        break;
    }
    }

    return std::make_unique<TangentialConstraintBoundaryCondition>(
        marker,
        penalty,
        std::move(target));
}

} // namespace Factories
} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BC_FACTORIES_H
