#ifndef SVMP_PHYSICS_FORMULATIONS_MESHMOTION_HARMONIC_MESH_MOTION_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_MESHMOTION_HARMONIC_MESH_MOTION_MODULE_H

/**
 * @file HarmonicMeshMotionModule.h
 * @brief Baseline coupled-ALE mesh-displacement equation using FE/Forms.
 */

#include "Physics/Core/PhysicsJITPolicy.h"
#include "Physics/Core/PhysicsModule.h"
#include "Physics/Formulations/MeshMotion/MeshMotionBoundaryOptions.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {

struct HarmonicMeshMotionOptions {
    using ScalarValue = FE::forms::bc::ScalarValue;

    struct DirichletBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> value{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    };

    struct NaturalBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> value{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    };

    struct RobinBC {
        int boundary_marker{-1};
        ScalarValue alpha{ScalarValue{1.0}};
        std::array<ScalarValue, 3> target{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    };

    std::string field_name{"mesh_displacement"};
    std::string operator_tag{"equations"};
    ScalarValue kappa{ScalarValue{1.0}};
    std::optional<ScalarValue> stiffness{};

    bool auto_register_field{true};
    bool bind_as_mesh_displacement{true};
    FE::forms::GeometryTangentPath tangent_path{FE::forms::GeometryTangentPath::SymbolicRequired};
    core::PhysicsJITPolicy jit_policy{};

    std::vector<DirichletBC> dirichlet{};
    std::vector<NaturalBC> natural{};
    std::vector<RobinBC> robin{};
    std::vector<NormalConstraintBC> normal_constraint{};
};

class HarmonicMeshMotionModule final : public PhysicsModule {
public:
    explicit HarmonicMeshMotionModule(
        std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space,
        HarmonicMeshMotionOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space_{};
    HarmonicMeshMotionOptions options_{};
};

} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_MESHMOTION_HARMONIC_MESH_MOTION_MODULE_H
