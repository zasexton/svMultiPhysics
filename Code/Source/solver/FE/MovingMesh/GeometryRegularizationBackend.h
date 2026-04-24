/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_MOVINGMESH_GEOMETRYREGULARIZATIONBACKEND_H
#define SVMP_FE_MOVINGMESH_GEOMETRYREGULARIZATIONBACKEND_H

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Motion/IMotionBackend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace svmp::FE::moving_mesh {

enum class GeometryRegularizationModel : std::uint8_t {
    Harmonic,
    ArtificialPseudoElastic
};

enum class GeometryRegularizationWeightMode : std::uint8_t {
    Uniform,
    ElementSize,
    BoundaryDistance,
    VertexField
};

enum class GeometryConstraintMode : std::uint8_t {
    StrongDirichlet
};

struct GeometryRegularizationScopeContract {
    static constexpr bool depends_on_physics_modules = false;
    static constexpr bool implements_physical_coupling_laws = false;
    static constexpr bool returns_physics_outputs = false;
};

struct GeometryRegularizationOptions {
    GeometryRegularizationModel model{GeometryRegularizationModel::Harmonic};
    GeometryRegularizationWeightMode weight_mode{GeometryRegularizationWeightMode::Uniform};
    GeometryConstraintMode constraint_mode{GeometryConstraintMode::StrongDirichlet};

    int max_linear_iterations{200};
    Real relative_tolerance{1.0e-12};
    Real absolute_tolerance{1.0e-14};
    Real constraint_tolerance{1.0e-12};

    Real artificial_stiffness_scale{1.0};
    Real artificial_pseudo_elastic_blend{0.25};
    Real boundary_distance_floor{1.0e-8};
    Real minimum_weight{1.0e-12};

    std::string vertex_weight_field;
    bool write_velocity{true};
};

struct GeometryRegularizationDiagnostics {
    bool success{false};
    std::string backend_name;
    std::string model;
    std::string weight_mode;
    std::string constraint_mode;
    std::string message;

    std::size_t vertices{0};
    std::size_t constrained_components{0};
    std::size_t free_components{0};
    int linear_iterations{0};
    int nonlinear_iterations{0};
    Real initial_residual_norm{0.0};
    Real final_residual_norm{0.0};
    Real max_constraint_violation{0.0};
    Real accepted_step_scale{1.0};
    Real minimum_quality_jacobian{0.0};
    Real minimum_quality_angle_degrees{0.0};
    Real maximum_quality_skewness{0.0};
};

[[nodiscard]] const char* to_string(GeometryRegularizationModel model) noexcept;
[[nodiscard]] const char* to_string(GeometryRegularizationWeightMode mode) noexcept;
[[nodiscard]] const char* to_string(GeometryConstraintMode mode) noexcept;

[[nodiscard]] std::vector<std::string> geometry_regularization_option_names();
[[nodiscard]] std::vector<std::string> geometry_regularization_diagnostic_field_names();

void validate_geometry_regularization_options(const GeometryRegularizationOptions& options);

class GeometryRegularizationMotionBackend final : public motion::IMotionBackend {
public:
    explicit GeometryRegularizationMotionBackend(GeometryRegularizationOptions options = {});

    [[nodiscard]] const char* name() const noexcept override;
    [[nodiscard]] const GeometryRegularizationOptions& options() const noexcept { return options_; }
    [[nodiscard]] const GeometryRegularizationDiagnostics& last_diagnostics() const noexcept {
        return diagnostics_;
    }

    motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override;

private:
    GeometryRegularizationOptions options_;
    GeometryRegularizationDiagnostics diagnostics_;
};

[[nodiscard]] std::shared_ptr<motion::IMotionBackend>
make_geometry_regularization_motion_backend(GeometryRegularizationOptions options = {});

} // namespace svmp::FE::moving_mesh

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_MOVINGMESH_GEOMETRYREGULARIZATIONBACKEND_H
