/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_MOVING_MESH_TIME_INTEGRATION_H
#define SVMP_FE_TIMESTEPPING_MOVING_MESH_TIME_INTEGRATION_H

#include "Assembly/TimeIntegrationContext.h"
#include "Core/Types.h"

#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {
namespace moving_mesh {

/**
 * @brief Physics-neutral time history for moving-domain coordinates and mesh displacement.
 *
 * The data are stored as flat entity-major vectors with `dimension()` components
 * per entity. The class intentionally has no Mesh or physics-module dependency;
 * callers own the mapping from mesh vertices, control-volume centers, or
 * geometry DOFs into these arrays.
 */
class MovingMeshTimeHistory {
public:
    MovingMeshTimeHistory() = default;
    MovingMeshTimeHistory(int dimension, std::size_t entity_count, int history_depth = 2);

    void configure(int dimension, std::size_t entity_count, int history_depth = 2);

    [[nodiscard]] int dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t entityCount() const noexcept { return entity_count_; }
    [[nodiscard]] std::size_t valueSize() const noexcept { return current_coordinates_.size(); }
    [[nodiscard]] int historyDepth() const noexcept { return static_cast<int>(coordinate_history_.size()); }

    void setCurrentCoordinates(std::span<const Real> values);
    void setCurrentMeshDisplacements(std::span<const Real> values);
    void setPreviousCoordinates(int steps_back, std::span<const Real> values);
    void setPreviousMeshDisplacements(int steps_back, std::span<const Real> values);

    [[nodiscard]] std::span<Real> currentCoordinates() noexcept { return {current_coordinates_.data(), current_coordinates_.size()}; }
    [[nodiscard]] std::span<const Real> currentCoordinates() const noexcept { return {current_coordinates_.data(), current_coordinates_.size()}; }
    [[nodiscard]] std::span<Real> currentMeshDisplacements() noexcept { return {current_displacements_.data(), current_displacements_.size()}; }
    [[nodiscard]] std::span<const Real> currentMeshDisplacements() const noexcept { return {current_displacements_.data(), current_displacements_.size()}; }
    [[nodiscard]] std::span<const Real> previousCoordinates(int steps_back = 1) const;
    [[nodiscard]] std::span<const Real> previousMeshDisplacements(int steps_back = 1) const;

    [[nodiscard]] std::vector<std::span<const Real>> coordinateHistorySpans() const;
    [[nodiscard]] std::vector<std::span<const Real>> meshDisplacementHistorySpans() const;

    void setTime(double time) noexcept { time_ = time; }
    [[nodiscard]] double time() const noexcept { return time_; }

    void setDt(double dt) noexcept { dt_ = dt; }
    [[nodiscard]] double dt() const noexcept { return dt_; }

    void setPrevDt(double dt_prev) noexcept;
    [[nodiscard]] double dtPrev() const noexcept { return dt_prev_; }

    void setStepIndex(int step) noexcept { step_index_ = step; }
    [[nodiscard]] int stepIndex() const noexcept { return step_index_; }

    [[nodiscard]] std::span<const double> dtHistory() const noexcept { return dt_history_; }
    void setDtHistory(std::span<const double> dt_history);
    void primeDtHistory(double dt_default) noexcept;

    /**
     * @brief Shift accepted current data into history.
     *
     * After the call, previous state 1 is the accepted current state and older
     * history entries move back by one slot.
     */
    void acceptStep(double accepted_dt);

private:
    [[nodiscard]] std::vector<Real>& previousVector(std::vector<std::vector<Real>>& history,
                                                    int steps_back,
                                                    const char* label);
    [[nodiscard]] const std::vector<Real>& previousVector(const std::vector<std::vector<Real>>& history,
                                                          int steps_back,
                                                          const char* label) const;
    void requireConfigured(const char* label) const;
    void requireValueSize(std::span<const Real> values, const char* label) const;

    int dimension_{0};
    std::size_t entity_count_{0};
    std::vector<Real> current_coordinates_{};
    std::vector<Real> current_displacements_{};
    std::vector<std::vector<Real>> coordinate_history_{};
    std::vector<std::vector<Real>> displacement_history_{};

    double time_{0.0};
    double dt_{0.0};
    double dt_prev_{0.0};
    std::vector<double> dt_history_{};
    int step_index_{0};
};

void applyTimeDerivativeStencil(std::span<const Real> current,
                                std::span<const std::span<const Real>> history,
                                const assembly::TimeDerivativeStencil& stencil,
                                std::span<Real> out);

[[nodiscard]] std::vector<Real>
computeTimeDerivative(std::span<const Real> current,
                      std::span<const std::span<const Real>> history,
                      const assembly::TimeDerivativeStencil& stencil);

[[nodiscard]] std::vector<Real>
computeMeshVelocityFromCoordinates(const MovingMeshTimeHistory& history,
                                   const assembly::TimeDerivativeStencil& stencil);

[[nodiscard]] std::vector<Real>
computeMeshVelocityFromDisplacements(const MovingMeshTimeHistory& history,
                                     const assembly::TimeDerivativeStencil& stencil);

[[nodiscard]] std::vector<Real>
computeMeshAccelerationFromCoordinates(const MovingMeshTimeHistory& history,
                                       const assembly::TimeDerivativeStencil& stencil);

[[nodiscard]] std::vector<Real>
computeMeshAccelerationFromDisplacements(const MovingMeshTimeHistory& history,
                                         const assembly::TimeDerivativeStencil& stencil);

void blendMeshVelocity(std::span<const Real> previous_velocity,
                       std::span<const Real> current_velocity,
                       Real current_weight,
                       std::span<Real> out);

[[nodiscard]] std::vector<Real>
blendMeshVelocity(std::span<const Real> previous_velocity,
                  std::span<const Real> current_velocity,
                  Real current_weight);

struct GCLDiagnostic {
    Real discrete_measure_rate{0.0};
    Real supplied_mesh_measure_rate{0.0};
    Real residual{0.0};
    Real relative_residual{0.0};
};

/**
 * @brief Evaluate the discrete geometric conservation law residual.
 *
 * The supplied rate can be a moving-boundary flux, an integrated mesh-velocity
 * divergence, or any other formulation-specific estimate. This helper only
 * compares that rate to the discrete control-volume measure change.
 */
[[nodiscard]] GCLDiagnostic evaluateControlVolumeGCL(Real current_measure,
                                                     Real previous_measure,
                                                     Real dt,
                                                     Real supplied_mesh_measure_rate);

[[nodiscard]] std::vector<GCLDiagnostic>
evaluateControlVolumeGCL(std::span<const Real> current_measures,
                         std::span<const Real> previous_measures,
                         Real dt,
                         std::span<const Real> supplied_mesh_measure_rates);

} // namespace moving_mesh
} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_MOVING_MESH_TIME_INTEGRATION_H
