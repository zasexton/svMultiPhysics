/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/MovingMeshTimeIntegration.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace svmp {
namespace FE {
namespace timestepping {
namespace moving_mesh {

namespace {

void copyChecked(std::span<Real> dst, std::span<const Real> src, const char* label)
{
    FE_THROW_IF(dst.size() != src.size(), InvalidArgumentException,
                std::string(label) + ": size mismatch");
    std::copy(src.begin(), src.end(), dst.begin());
}

void validateStencil(const assembly::TimeDerivativeStencil& stencil,
                     std::size_t history_size,
                     const char* label)
{
    FE_THROW_IF(stencil.order <= 0, InvalidArgumentException,
                std::string(label) + ": stencil order must be positive");
    FE_THROW_IF(stencil.a.empty(), InvalidArgumentException,
                std::string(label) + ": stencil coefficient array is empty");
    FE_THROW_IF(stencil.requiredHistoryStates() > static_cast<int>(history_size),
                InvalidArgumentException,
                std::string(label) + ": stencil requires " +
                    std::to_string(stencil.requiredHistoryStates()) +
                    " history states but only " + std::to_string(history_size) +
                    " are available");
    for (const Real a : stencil.a) {
        FE_THROW_IF(!std::isfinite(static_cast<double>(a)), InvalidArgumentException,
                    std::string(label) + ": stencil coefficients must be finite");
    }
}

std::span<const std::span<const Real>> asSpan(const std::vector<std::span<const Real>>& spans)
{
    return std::span<const std::span<const Real>>(spans.data(), spans.size());
}

} // namespace

MovingMeshTimeHistory::MovingMeshTimeHistory(int dimension,
                                             std::size_t entity_count,
                                             int history_depth)
{
    configure(dimension, entity_count, history_depth);
}

void MovingMeshTimeHistory::configure(int dimension,
                                      std::size_t entity_count,
                                      int history_depth)
{
    FE_THROW_IF(dimension <= 0 || dimension > 3, InvalidArgumentException,
                "MovingMeshTimeHistory::configure: dimension must be in [1,3]");
    FE_THROW_IF(history_depth <= 0, InvalidArgumentException,
                "MovingMeshTimeHistory::configure: history_depth must be positive");

    dimension_ = dimension;
    entity_count_ = entity_count;
    const auto size = entity_count_ * static_cast<std::size_t>(dimension_);
    current_coordinates_.assign(size, Real(0));
    current_displacements_.assign(size, Real(0));
    coordinate_history_.assign(static_cast<std::size_t>(history_depth),
                               std::vector<Real>(size, Real(0)));
    displacement_history_.assign(static_cast<std::size_t>(history_depth),
                                 std::vector<Real>(size, Real(0)));
    dt_history_.assign(static_cast<std::size_t>(history_depth), 0.0);
}

void MovingMeshTimeHistory::requireConfigured(const char* label) const
{
    FE_THROW_IF(dimension_ <= 0, InvalidArgumentException,
                std::string(label) + ": history is not configured");
}

void MovingMeshTimeHistory::requireValueSize(std::span<const Real> values, const char* label) const
{
    requireConfigured(label);
    FE_THROW_IF(values.size() != current_coordinates_.size(), InvalidArgumentException,
                std::string(label) + ": value size mismatch");
}

std::vector<Real>& MovingMeshTimeHistory::previousVector(std::vector<std::vector<Real>>& history,
                                                         int steps_back,
                                                         const char* label)
{
    FE_THROW_IF(steps_back <= 0, InvalidArgumentException,
                std::string(label) + ": steps_back must be positive");
    FE_THROW_IF(static_cast<std::size_t>(steps_back) > history.size(), InvalidArgumentException,
                std::string(label) + ": requested history state is not allocated");
    return history[static_cast<std::size_t>(steps_back - 1)];
}

const std::vector<Real>& MovingMeshTimeHistory::previousVector(const std::vector<std::vector<Real>>& history,
                                                               int steps_back,
                                                               const char* label) const
{
    FE_THROW_IF(steps_back <= 0, InvalidArgumentException,
                std::string(label) + ": steps_back must be positive");
    FE_THROW_IF(static_cast<std::size_t>(steps_back) > history.size(), InvalidArgumentException,
                std::string(label) + ": requested history state is not allocated");
    return history[static_cast<std::size_t>(steps_back - 1)];
}

void MovingMeshTimeHistory::setCurrentCoordinates(std::span<const Real> values)
{
    requireValueSize(values, "MovingMeshTimeHistory::setCurrentCoordinates");
    copyChecked(std::span<Real>(current_coordinates_.data(), current_coordinates_.size()),
                values,
                "MovingMeshTimeHistory::setCurrentCoordinates");
}

void MovingMeshTimeHistory::setCurrentMeshDisplacements(std::span<const Real> values)
{
    requireValueSize(values, "MovingMeshTimeHistory::setCurrentMeshDisplacements");
    copyChecked(std::span<Real>(current_displacements_.data(), current_displacements_.size()),
                values,
                "MovingMeshTimeHistory::setCurrentMeshDisplacements");
}

void MovingMeshTimeHistory::setPreviousCoordinates(int steps_back, std::span<const Real> values)
{
    requireValueSize(values, "MovingMeshTimeHistory::setPreviousCoordinates");
    auto& dst = previousVector(coordinate_history_, steps_back,
                               "MovingMeshTimeHistory::setPreviousCoordinates");
    copyChecked(std::span<Real>(dst.data(), dst.size()), values,
                "MovingMeshTimeHistory::setPreviousCoordinates");
}

void MovingMeshTimeHistory::setPreviousMeshDisplacements(int steps_back, std::span<const Real> values)
{
    requireValueSize(values, "MovingMeshTimeHistory::setPreviousMeshDisplacements");
    auto& dst = previousVector(displacement_history_, steps_back,
                               "MovingMeshTimeHistory::setPreviousMeshDisplacements");
    copyChecked(std::span<Real>(dst.data(), dst.size()), values,
                "MovingMeshTimeHistory::setPreviousMeshDisplacements");
}

std::span<const Real> MovingMeshTimeHistory::previousCoordinates(int steps_back) const
{
    const auto& v = previousVector(coordinate_history_, steps_back,
                                   "MovingMeshTimeHistory::previousCoordinates");
    return std::span<const Real>(v.data(), v.size());
}

std::span<const Real> MovingMeshTimeHistory::previousMeshDisplacements(int steps_back) const
{
    const auto& v = previousVector(displacement_history_, steps_back,
                                   "MovingMeshTimeHistory::previousMeshDisplacements");
    return std::span<const Real>(v.data(), v.size());
}

std::vector<std::span<const Real>> MovingMeshTimeHistory::coordinateHistorySpans() const
{
    std::vector<std::span<const Real>> spans;
    spans.reserve(coordinate_history_.size());
    for (const auto& v : coordinate_history_) {
        spans.emplace_back(v.data(), v.size());
    }
    return spans;
}

std::vector<std::span<const Real>> MovingMeshTimeHistory::meshDisplacementHistorySpans() const
{
    std::vector<std::span<const Real>> spans;
    spans.reserve(displacement_history_.size());
    for (const auto& v : displacement_history_) {
        spans.emplace_back(v.data(), v.size());
    }
    return spans;
}

void MovingMeshTimeHistory::setPrevDt(double dt_prev) noexcept
{
    dt_prev_ = dt_prev;
    if (!dt_history_.empty()) {
        dt_history_[0] = dt_prev;
    }
}

void MovingMeshTimeHistory::setDtHistory(std::span<const double> dt_history)
{
    FE_THROW_IF(dt_history.size() != dt_history_.size(), InvalidArgumentException,
                "MovingMeshTimeHistory::setDtHistory: size must match history depth");
    std::copy(dt_history.begin(), dt_history.end(), dt_history_.begin());
    if (!dt_history_.empty()) {
        dt_prev_ = dt_history_[0];
    }
}

void MovingMeshTimeHistory::primeDtHistory(double dt_default) noexcept
{
    for (double& v : dt_history_) {
        if (!(v > 0.0) || !std::isfinite(v)) {
            v = dt_default;
        }
    }
    if (!(dt_prev_ > 0.0) || !std::isfinite(dt_prev_)) {
        dt_prev_ = dt_default;
    }
}

void MovingMeshTimeHistory::acceptStep(double accepted_dt)
{
    requireConfigured("MovingMeshTimeHistory::acceptStep");
    FE_THROW_IF(!(accepted_dt > 0.0) || !std::isfinite(accepted_dt),
                InvalidArgumentException,
                "MovingMeshTimeHistory::acceptStep: accepted_dt must be finite and positive");

    auto shift_history = [](std::vector<std::vector<Real>>& history,
                            const std::vector<Real>& current) {
        for (std::size_t i = history.size(); i > 1; --i) {
            history[i - 1] = history[i - 2];
        }
        if (!history.empty()) {
            history[0] = current;
        }
    };
    shift_history(coordinate_history_, current_coordinates_);
    shift_history(displacement_history_, current_displacements_);

    for (std::size_t i = dt_history_.size(); i > 1; --i) {
        dt_history_[i - 1] = dt_history_[i - 2];
    }
    if (!dt_history_.empty()) {
        dt_history_[0] = accepted_dt;
    }
    dt_prev_ = accepted_dt;
    dt_ = accepted_dt;
    ++step_index_;
}

void applyTimeDerivativeStencil(std::span<const Real> current,
                                std::span<const std::span<const Real>> history,
                                const assembly::TimeDerivativeStencil& stencil,
                                std::span<Real> out)
{
    validateStencil(stencil, history.size(), "applyTimeDerivativeStencil");
    FE_THROW_IF(out.size() != current.size(), InvalidArgumentException,
                "applyTimeDerivativeStencil: output size mismatch");
    for (std::size_t k = 0; k < history.size(); ++k) {
        if (stencil.coeff(static_cast<int>(k + 1)) != Real(0)) {
            FE_THROW_IF(history[k].size() != current.size(), InvalidArgumentException,
                        "applyTimeDerivativeStencil: history size mismatch");
        }
    }

    std::fill(out.begin(), out.end(), Real(0));
    const Real a0 = stencil.coeff(0);
    for (std::size_t i = 0; i < current.size(); ++i) {
        out[i] = a0 * current[i];
    }
    for (std::size_t k = 0; k < history.size(); ++k) {
        const Real a = stencil.coeff(static_cast<int>(k + 1));
        if (a == Real(0)) {
            continue;
        }
        const auto hk = history[k];
        for (std::size_t i = 0; i < current.size(); ++i) {
            out[i] += a * hk[i];
        }
    }
}

std::vector<Real>
computeTimeDerivative(std::span<const Real> current,
                      std::span<const std::span<const Real>> history,
                      const assembly::TimeDerivativeStencil& stencil)
{
    std::vector<Real> out(current.size(), Real(0));
    applyTimeDerivativeStencil(current, history, stencil, out);
    return out;
}

std::vector<Real>
computeMeshVelocityFromCoordinates(const MovingMeshTimeHistory& history,
                                   const assembly::TimeDerivativeStencil& stencil)
{
    FE_THROW_IF(stencil.order != 1, InvalidArgumentException,
                "computeMeshVelocityFromCoordinates: stencil order must be 1");
    const auto spans = history.coordinateHistorySpans();
    return computeTimeDerivative(history.currentCoordinates(), asSpan(spans), stencil);
}

std::vector<Real>
computeMeshVelocityFromDisplacements(const MovingMeshTimeHistory& history,
                                     const assembly::TimeDerivativeStencil& stencil)
{
    FE_THROW_IF(stencil.order != 1, InvalidArgumentException,
                "computeMeshVelocityFromDisplacements: stencil order must be 1");
    const auto spans = history.meshDisplacementHistorySpans();
    return computeTimeDerivative(history.currentMeshDisplacements(), asSpan(spans), stencil);
}

std::vector<Real>
computeMeshAccelerationFromCoordinates(const MovingMeshTimeHistory& history,
                                       const assembly::TimeDerivativeStencil& stencil)
{
    FE_THROW_IF(stencil.order != 2, InvalidArgumentException,
                "computeMeshAccelerationFromCoordinates: stencil order must be 2");
    const auto spans = history.coordinateHistorySpans();
    return computeTimeDerivative(history.currentCoordinates(), asSpan(spans), stencil);
}

std::vector<Real>
computeMeshAccelerationFromDisplacements(const MovingMeshTimeHistory& history,
                                         const assembly::TimeDerivativeStencil& stencil)
{
    FE_THROW_IF(stencil.order != 2, InvalidArgumentException,
                "computeMeshAccelerationFromDisplacements: stencil order must be 2");
    const auto spans = history.meshDisplacementHistorySpans();
    return computeTimeDerivative(history.currentMeshDisplacements(), asSpan(spans), stencil);
}

void blendMeshVelocity(std::span<const Real> previous_velocity,
                       std::span<const Real> current_velocity,
                       Real current_weight,
                       std::span<Real> out)
{
    FE_THROW_IF(previous_velocity.size() != current_velocity.size() ||
                    out.size() != current_velocity.size(),
                InvalidArgumentException,
                "blendMeshVelocity: size mismatch");
    FE_THROW_IF(!std::isfinite(static_cast<double>(current_weight)),
                InvalidArgumentException,
                "blendMeshVelocity: current_weight must be finite");
    const Real previous_weight = Real(1) - current_weight;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = previous_weight * previous_velocity[i] + current_weight * current_velocity[i];
    }
}

std::vector<Real>
blendMeshVelocity(std::span<const Real> previous_velocity,
                  std::span<const Real> current_velocity,
                  Real current_weight)
{
    std::vector<Real> out(current_velocity.size(), Real(0));
    blendMeshVelocity(previous_velocity, current_velocity, current_weight, out);
    return out;
}

GCLDiagnostic evaluateControlVolumeGCL(Real current_measure,
                                       Real previous_measure,
                                       Real dt,
                                       Real supplied_mesh_measure_rate)
{
    FE_THROW_IF(!(dt > Real(0)) || !std::isfinite(static_cast<double>(dt)),
                InvalidArgumentException,
                "evaluateControlVolumeGCL: dt must be finite and positive");
    FE_THROW_IF(!std::isfinite(static_cast<double>(current_measure)) ||
                    !std::isfinite(static_cast<double>(previous_measure)) ||
                    !std::isfinite(static_cast<double>(supplied_mesh_measure_rate)),
                InvalidArgumentException,
                "evaluateControlVolumeGCL: measures and supplied rate must be finite");

    GCLDiagnostic out;
    out.discrete_measure_rate = (current_measure - previous_measure) / dt;
    out.supplied_mesh_measure_rate = supplied_mesh_measure_rate;
    out.residual = out.discrete_measure_rate - supplied_mesh_measure_rate;
    const Real denom = std::max<Real>(std::abs(out.discrete_measure_rate),
                                      std::abs(supplied_mesh_measure_rate));
    out.relative_residual = denom > Real(0) ? std::abs(out.residual) / denom
                                            : std::abs(out.residual);
    return out;
}

std::vector<GCLDiagnostic>
evaluateControlVolumeGCL(std::span<const Real> current_measures,
                         std::span<const Real> previous_measures,
                         Real dt,
                         std::span<const Real> supplied_mesh_measure_rates)
{
    FE_THROW_IF(current_measures.size() != previous_measures.size() ||
                    current_measures.size() != supplied_mesh_measure_rates.size(),
                InvalidArgumentException,
                "evaluateControlVolumeGCL: array size mismatch");
    std::vector<GCLDiagnostic> out;
    out.reserve(current_measures.size());
    for (std::size_t i = 0; i < current_measures.size(); ++i) {
        out.push_back(evaluateControlVolumeGCL(current_measures[i],
                                               previous_measures[i],
                                               dt,
                                               supplied_mesh_measure_rates[i]));
    }
    return out;
}

} // namespace moving_mesh
} // namespace timestepping
} // namespace FE
} // namespace svmp
