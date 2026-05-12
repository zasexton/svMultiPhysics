#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H

/**
 * @file LevelSetTransportModule.h
 * @brief Physics module options for level-set free-surface advection.
 */

#include "Physics/Core/PhysicsJITPolicy.h"
#include "Physics/Core/PhysicsModule.h"

#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

using ScalarValue = FE::forms::bc::ScalarValue;

enum class LevelSetFieldSource {
    Unknown,
    PrescribedData
};

enum class LevelSetVelocitySource {
    CoupledField,
    PrescribedData,
    ConstantVector
};

struct LevelSetFieldOptions {
    std::string field_name{"level_set"};
    LevelSetFieldSource source{LevelSetFieldSource::Unknown};
    bool auto_register_field{true};
};

struct LevelSetVelocityOptions {
    std::string field_name{"Velocity"};
    LevelSetVelocitySource source{LevelSetVelocitySource::CoupledField};
    bool auto_register_field{false};
    std::shared_ptr<const FE::spaces::FunctionSpace> space{};
    std::array<FE::Real, 3> constant_value{0.0, 0.0, 0.0};
};

struct LevelSetSUPGOptions {
    bool enabled{false};
    FE::Real tau_scale{0.5};
    FE::Real velocity_epsilon{1.0e-12};
};

enum class LevelSetReinitializationMethod {
    HamiltonJacobiPDE,
    FastMarching,
    Projection
};

struct LevelSetReinitializationOptions {
    bool enabled{false};
    LevelSetReinitializationMethod method{LevelSetReinitializationMethod::HamiltonJacobiPDE};
    int cadence_steps{1};
    int max_iterations{10};
    FE::Real pseudo_time_step_scale{0.3};
    FE::Real interface_band_width{3.0};
    FE::Real signed_distance_tolerance{1.0e-6};
};

struct LevelSetVolumeCorrectionOptions {
    bool enabled{false};
    int cadence_steps{1};
    bool use_initial_negative_volume_as_target{true};
    FE::Real target_negative_volume{0.0};
    FE::Real volume_tolerance{1.0e-10};
    int max_iterations{50};
};

struct LevelSetInflowBoundary {
    int boundary_marker{-1};
    ScalarValue value{0.0};
    FE::Real penalty_scale{1.0};
};

struct LevelSetOutflowBoundary {
    int boundary_marker{-1};
};

struct LevelSetBoundaryOptions {
    std::vector<LevelSetInflowBoundary> inflow{};
    std::vector<LevelSetOutflowBoundary> outflow{};
};

struct LevelSetTransportOptions {
    LevelSetFieldOptions level_set{};
    LevelSetVelocityOptions velocity{};
    LevelSetSUPGOptions supg{};
    LevelSetReinitializationOptions reinitialization{};
    LevelSetVolumeCorrectionOptions volume_correction{};
    LevelSetBoundaryOptions boundaries{};
    core::PhysicsJITPolicy jit_policy{};
};

class LevelSetTransportModule final : public PhysicsModule {
public:
    explicit LevelSetTransportModule(
        std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
        LevelSetTransportOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

    [[nodiscard]] const LevelSetTransportOptions& options() const noexcept { return options_; }

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space_{};
    LevelSetTransportOptions options_{};
};

[[nodiscard]] bool shouldReinitializeLevelSet(
    const LevelSetReinitializationOptions& options,
    int completed_step_index) noexcept;

[[nodiscard]] bool shouldApplyLevelSetVolumeCorrection(
    const LevelSetVolumeCorrectionOptions& options,
    int completed_step_index) noexcept;

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H
