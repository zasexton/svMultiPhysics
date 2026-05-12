#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Public level-set option structs, source enums, and cadence helpers.
 */

#include "Core/Types.h"
#include "Forms/BoundaryConditions.h"
#include "Spaces/FunctionSpace.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

using ScalarValue = forms::bc::ScalarValue;

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
    std::shared_ptr<const spaces::FunctionSpace> space{};
    std::array<Real, 3> constant_value{0.0, 0.0, 0.0};
};

struct LevelSetSUPGOptions {
    bool enabled{false};
    Real tau_scale{0.5};
    Real velocity_epsilon{1.0e-12};
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
    Real pseudo_time_step_scale{0.3};
    Real interface_band_width{3.0};
    Real signed_distance_tolerance{1.0e-6};
};

struct LevelSetVolumeCorrectionOptions {
    bool enabled{false};
    int cadence_steps{1};
    bool use_initial_negative_volume_as_target{true};
    Real target_negative_volume{0.0};
    Real volume_tolerance{1.0e-10};
    int max_iterations{50};
};

struct LevelSetInflowBoundary {
    int boundary_marker{-1};
    ScalarValue value{0.0};
    Real penalty_scale{1.0};
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
};

[[nodiscard]] bool shouldReinitializeLevelSet(
    const LevelSetReinitializationOptions& options,
    int completed_step_index) noexcept;

[[nodiscard]] bool shouldApplyLevelSetVolumeCorrection(
    const LevelSetVolumeCorrectionOptions& options,
    int completed_step_index) noexcept;

} // namespace svmp::FE::level_set
