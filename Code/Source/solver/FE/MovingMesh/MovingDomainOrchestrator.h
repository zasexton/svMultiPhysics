/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_MOVINGMESH_MOVINGDOMAINORCHESTRATOR_H
#define SVMP_FE_MOVINGMESH_MOVINGDOMAINORCHESTRATOR_H

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "MovingMesh/GeometryRegularizationBackend.h"
#include "Systems/FESystem.h"
#include "TimeStepping/TimeHistory.h"

#include "Mesh/Mesh.h"
#include "Mesh/Motion/IMotionBackend.h"
#include "Mesh/Motion/MeshMotion.h"
#include "Mesh/Motion/MotionConfig.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp::FE::moving_mesh {

enum class MovingMeshMode : std::uint8_t {
    Disabled,
    PrescribedMotion,
    FEBackedSmoothing,
    CoupledMonolithic
};

enum class MotionBackendModel : std::uint8_t {
    None,
    Prescribed,
    FEGeometryRegularization,
    External
};

enum class BoundaryMotionValueMode : std::uint8_t {
    ConstantDisplacement,
    ConstantVelocity,
    Function
};

/**
 * @brief Application-level ordering points for moving-domain orchestration.
 *
 * The first production insertion point is `BeforePhysicsSolve`: after the
 * time-history state is reset for the attempted step and before
 * `TransientSystem::beginTimeStep()`, nonlinear iteration, and assembly. A
 * successful advance notifies FE systems before assembly. Remeshing, restart,
 * checkpoint, and output consume the accepted geometry after the physics solve
 * commits; rejected attempts roll back inside the motion controller.
 */
enum class MovingDomainAdvancePoint : std::uint8_t {
    BeforePhysicsSolve,
    BeforeNonlinearIteration,
    AfterAcceptedNonlinearState,
    BeforeRemesh,
    BeforeCheckpoint,
    BeforeOutput
};

struct BoundaryMotionConfig {
    label_t boundary_label{INVALID_LABEL};
    std::array<bool, 3> component_mask{{true, true, true}};
    BoundaryMotionValueMode value_mode{BoundaryMotionValueMode::ConstantDisplacement};
    std::array<real_t, 3> value{{0.0, 0.0, 0.0}};
    std::function<std::array<real_t, 3>(const std::array<real_t, 3>&,
                                        double,
                                        double)> function{};
};

struct MovingDomainConfig {
    MovingMeshMode mode{MovingMeshMode::Disabled};
    svmp::Configuration fe_coordinate_configuration{svmp::Configuration::Reference};
    MotionBackendModel backend_model{MotionBackendModel::None};

    motion::MotionConfig motion{};
    GeometryRegularizationOptions geometry_regularization{};
    std::vector<BoundaryMotionConfig> boundary_motion{};

    bool notify_fe_systems{true};
    bool exchange_ghost_coordinates{true};
};

struct MovingDomainDiagnostics {
    bool success{true};
    bool advanced_geometry{false};
    bool rolled_back{false};

    std::string mode;
    std::string backend_model;
    std::string backend_name;
    std::string advance_point;
    std::string message;

    double time{0.0};
    double dt{0.0};
    double accepted_step_scale{1.0};

    std::uint64_t geometry_revision_before{0};
    std::uint64_t geometry_revision_after{0};
    real_t minimum_quality_jacobian{0.0};
    real_t minimum_quality_angle_degrees{0.0};
    real_t maximum_quality_skewness{0.0};
    bool has_inverted_cells{false};

    std::size_t boundary_condition_count{0};
    std::size_t notified_fe_systems{0};
    std::size_t synced_prescribed_field_values{0};
};

[[nodiscard]] const char* to_string(MovingMeshMode mode) noexcept;
[[nodiscard]] const char* to_string(MotionBackendModel model) noexcept;
[[nodiscard]] const char* to_string(BoundaryMotionValueMode mode) noexcept;
[[nodiscard]] const char* to_string(MovingDomainAdvancePoint point) noexcept;

[[nodiscard]] MovingMeshMode parse_moving_mesh_mode(std::string_view value);
[[nodiscard]] MotionBackendModel parse_motion_backend_model(std::string_view value);
[[nodiscard]] svmp::Configuration parse_coordinate_configuration(std::string_view value);

/**
 * @brief Build physics-neutral moving-domain orchestration config from flat keys.
 *
 * Supported keys use the `mesh_motion.*` prefix and intentionally describe
 * Mesh/FE orchestration only: motion mode, FE coordinate configuration, backend
 * selection/options, quality-guard options, and boundary displacement/velocity
 * entries by label and component. Physics modules remain responsible for
 * interpreting physical laws and providing any nonconstant motion functions.
 */
[[nodiscard]] MovingDomainConfig moving_domain_config_from_kv(
    const std::unordered_map<std::string, std::string>& kv);

[[nodiscard]] std::vector<motion::MotionDirichletBC>
make_motion_dirichlet_bcs(const std::vector<BoundaryMotionConfig>& configs);

class MovingDomainOrchestrator {
public:
    explicit MovingDomainOrchestrator(std::shared_ptr<Mesh> mesh,
                                      MovingDomainConfig config = {});

    [[nodiscard]] const MovingDomainConfig& config() const noexcept { return config_; }
    void configure(MovingDomainConfig config);

    void setExternalBackend(std::shared_ptr<motion::IMotionBackend> backend);
    [[nodiscard]] const MovingDomainDiagnostics& lastDiagnostics() const noexcept { return diagnostics_; }

    [[nodiscard]] MovingDomainDiagnostics advance(
        MovingDomainAdvancePoint point,
        double time,
        double dt,
        std::span<systems::FESystem* const> systems = {});

    [[nodiscard]] std::function<bool(timestepping::TimeHistory&, double, double)>
    makeBeforePhysicsSolveCallback(std::vector<systems::FESystem*> systems = {});

private:
    [[nodiscard]] std::shared_ptr<motion::IMotionBackend> buildBackend();
    [[nodiscard]] std::vector<motion::MotionDirichletBC> buildBoundaryConditions() const;
    void notifySystems(std::span<systems::FESystem* const> systems);
    void validateSystemCoordinateConfiguration(std::span<systems::FESystem* const> systems) const;

    std::shared_ptr<Mesh> mesh_;
    MovingDomainConfig config_{};
    std::shared_ptr<motion::IMotionBackend> external_backend_{};
    MovingDomainDiagnostics diagnostics_{};
};

[[nodiscard]] std::unique_ptr<systems::FESystem>
make_moving_domain_fe_system(std::shared_ptr<Mesh> mesh,
                             const MovingDomainConfig& config);

} // namespace svmp::FE::moving_mesh

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_MOVINGMESH_MOVINGDOMAINORCHESTRATOR_H
