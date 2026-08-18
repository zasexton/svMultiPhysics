#ifndef SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_MODULE_H

/**
 * @file UstructModule.h
 * @brief Mixed finite-deformation structural mechanics formulation
 */

#include "Physics/Core/PhysicsJITPolicy.h"
#include "Physics/Core/PhysicsModule.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/ConstitutiveModel.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace ustruct {

enum class VolumetricPenaltyModel {
    None,
    Quadratic,
    ST91,
    M94
};

struct UstructOptions {
    using ScalarValue = FE::forms::bc::ScalarValue;

    struct DisplacementDirichletBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> value{
            ScalarValue{0.0},
            ScalarValue{0.0},
            ScalarValue{0.0}};
        std::array<bool, 3> active_components{true, true, true};
    };

    struct VelocityDirichletBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> value{
            ScalarValue{0.0},
            ScalarValue{0.0},
            ScalarValue{0.0}};
        std::array<bool, 3> active_components{true, true, true};
    };

    struct TractionNeumannBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> traction{
            ScalarValue{0.0},
            ScalarValue{0.0},
            ScalarValue{0.0}};
    };

    struct NormalTractionBC {
        int boundary_marker{-1};
        ScalarValue traction{0.0};
    };

    struct FollowerPressureBC {
        struct LinearRamp {
            FE::Real start_time{0.0};
            FE::Real end_time{0.0};
            FE::Real start_value{0.0};
            FE::Real end_value{0.0};
        };

        int boundary_marker{-1};
        ScalarValue pressure{0.0};
        std::optional<LinearRamp> ramp{};
    };

    struct PressureDirichletBC {
        int boundary_marker{-1};
        ScalarValue value{0.0};
    };

    std::string displacement_field_name{"Displacement"};
    std::string velocity_field_name{"Velocity"};
    std::string pressure_field_name{"Pressure"};

    FE::Real density{1.0};
    FE::Real bulk_wave_speed{0.0};
    FE::Real penalty_parameter{0.0};
    FE::Real kinematic_residual_scale{1.0};
    VolumetricPenaltyModel volumetric_model{VolumetricPenaltyModel::ST91};
    std::shared_ptr<const FE::forms::ConstitutiveModel> deviatoric_pk1_model{};

    std::array<FE::Real, 3> body_force{0.0, 0.0, 0.0};

    bool enable_stabilization{true};
    FE::Real ct_m{0.0};
    FE::Real ct_c{0.0};

    core::PhysicsJITPolicy jit_policy{};
    bool use_symbolic_tangent{true};
    bool enable_time_derivative_terms{true};

    bool register_def_grad_output{false};
    bool register_jacobian_output{false};
    bool register_cauchy_stress_output{false};
    bool register_divergence_output{false};
    bool register_strain_output{false};
    bool register_stress_output{false};
    bool register_von_mises_stress_output{false};

    std::vector<DisplacementDirichletBC> displacement_dirichlet{};
    std::vector<VelocityDirichletBC> velocity_dirichlet{};
    std::vector<TractionNeumannBC> traction_neumann{};
    std::vector<NormalTractionBC> normal_traction{};
    std::vector<FollowerPressureBC> follower_pressure{};
    std::vector<PressureDirichletBC> pressure_dirichlet{};
};

class UstructModule final : public PhysicsModule {
public:
    UstructModule(std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space,
                  std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
                  UstructOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> displacement_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space_{};
    UstructOptions options_{};
};

} // namespace ustruct
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_USTRUCT_USTRUCT_MODULE_H
