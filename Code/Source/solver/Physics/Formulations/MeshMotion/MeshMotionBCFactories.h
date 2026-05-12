#ifndef SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BC_FACTORIES_H

/**
 * @file MeshMotionBCFactories.h
 * @brief Shared mesh-motion boundary-condition translation helpers.
 */

#include "FE/Forms/StandardBCs.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {
namespace Factories {

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

} // namespace Factories
} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BC_FACTORIES_H
