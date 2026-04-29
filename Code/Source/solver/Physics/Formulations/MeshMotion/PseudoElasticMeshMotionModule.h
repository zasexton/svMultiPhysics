#ifndef SVMP_PHYSICS_FORMULATIONS_MESHMOTION_PSEUDOELASTIC_MESH_MOTION_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_MESHMOTION_PSEUDOELASTIC_MESH_MOTION_MODULE_H

/**
 * @file PseudoElasticMeshMotionModule.h
 * @brief Pseudoelastic coupled-ALE mesh-displacement equation using FE/Forms.
 */

#include "Physics/Core/PhysicsModule.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {

struct PseudoElasticMeshMotionOptions {
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
    ScalarValue lambda_mesh{ScalarValue{1.0}};
    ScalarValue mu_mesh{ScalarValue{1.0}};

    bool auto_register_field{true};
    bool bind_as_mesh_displacement{true};
    bool enable_jit{SVMP_FE_ENABLE_LLVM_JIT != 0};
    bool enable_jit_specialization{true};

    std::vector<DirichletBC> dirichlet{};
    std::vector<NaturalBC> natural{};
    std::vector<RobinBC> robin{};
};

class PseudoElasticMeshMotionModule final : public PhysicsModule {
public:
    explicit PseudoElasticMeshMotionModule(
        std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space,
        PseudoElasticMeshMotionOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space_{};
    PseudoElasticMeshMotionOptions options_{};
};

} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_MESHMOTION_PSEUDOELASTIC_MESH_MOTION_MODULE_H
