/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/InterfaceOperators.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {
namespace {

std::size_t target_size(const svmp::search::InterfaceMap& map) {
    std::size_t size = 0;
    for (const auto& pair : map.pairs) {
        if (pair.target_face >= 0) {
            size = std::max(size, static_cast<std::size_t>(pair.target_face) + 1u);
        }
    }
    return size;
}

Real source_value(const std::vector<Real>& source_face_values, svmp::index_t face) {
    if (face < 0 || static_cast<std::size_t>(face) >= source_face_values.size()) {
        throw std::out_of_range("interface source face value is missing");
    }
    return source_face_values[static_cast<std::size_t>(face)];
}

Real positive_measure(Real measure) noexcept {
    return measure > Real{0.0} ? measure : Real{1.0};
}

bool has_revision_anchors(const svmp::search::InterfaceMap& map) noexcept {
    return map.source.valid() && map.target.valid();
}

bool stale_for_revision_anchors(const svmp::search::InterfaceMap& map) noexcept {
    return has_revision_anchors(map) && !map.valid_for_current_revisions();
}

std::array<Real, 3> matvec(const std::array<std::array<Real, 3>, 3>& a,
                           const std::array<Real, 3>& x) noexcept {
    return std::array<Real, 3>{
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2]};
}

std::array<std::array<Real, 3>, 3> transpose(
    const std::array<std::array<Real, 3>, 3>& a) noexcept {
    return std::array<std::array<Real, 3>, 3>{{
        {{a[0][0], a[1][0], a[2][0]}},
        {{a[0][1], a[1][1], a[2][1]}},
        {{a[0][2], a[1][2], a[2][2]}}}};
}

std::array<std::array<Real, 3>, 3> matmul(
    const std::array<std::array<Real, 3>, 3>& a,
    const std::array<std::array<Real, 3>, 3>& b) noexcept {
    std::array<std::array<Real, 3>, 3> out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            for (std::size_t k = 0; k < 3u; ++k) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return out;
}

std::uint32_t effective_components(const InterfaceTransferOptions& options) {
    if (options.component_count != 0u) {
        return options.component_count;
    }
    switch (options.field_kind) {
        case InterfaceFieldKind::Scalar: return 1u;
        case InterfaceFieldKind::Vector: return 3u;
        case InterfaceFieldKind::Rank2Tensor: return 9u;
        case InterfaceFieldKind::MixedBlock: return 1u;
    }
    return 1u;
}

std::vector<Real> transform_source_values(
    const std::vector<Real>& values,
    const InterfaceTransferOptions& options,
    std::uint32_t components) {
    if (options.frame_policy == InterfaceFrameTransformPolicy::None) {
        return values;
    }
    if (components == 0u || values.size() % components != 0u) {
        throw std::invalid_argument("interface transfer source value size is not divisible by component count");
    }

    std::vector<Real> out(values.size(), Real{0.0});
    const std::size_t faces = values.size() / components;
    for (std::size_t f = 0; f < faces; ++f) {
        const std::size_t base = f * components;
        if (options.frame_policy == InterfaceFrameTransformPolicy::SourceToTargetVector) {
            if (components < 3u) {
                throw std::invalid_argument("vector interface frame transform requires at least 3 components");
            }
            const auto y = matvec(options.source_to_target_rotation,
                                  std::array<Real, 3>{values[base], values[base + 1u], values[base + 2u]});
            out[base] = y[0];
            out[base + 1u] = y[1];
            out[base + 2u] = y[2];
            for (std::uint32_t c = 3u; c < components; ++c) {
                out[base + c] = values[base + c];
            }
        } else if (options.frame_policy == InterfaceFrameTransformPolicy::SourceToTargetRank2Tensor) {
            if (components < 9u) {
                throw std::invalid_argument("rank-2 interface frame transform requires at least 9 components");
            }
            std::array<std::array<Real, 3>, 3> tensor{};
            for (std::size_t i = 0; i < 3u; ++i) {
                for (std::size_t j = 0; j < 3u; ++j) {
                    tensor[i][j] = values[base + i * 3u + j];
                }
            }
            const auto rt = transpose(options.source_to_target_rotation);
            const auto transformed = matmul(options.source_to_target_rotation, matmul(tensor, rt));
            for (std::size_t i = 0; i < 3u; ++i) {
                for (std::size_t j = 0; j < 3u; ++j) {
                    out[base + i * 3u + j] = transformed[i][j];
                }
            }
            for (std::uint32_t c = 9u; c < components; ++c) {
                out[base + c] = values[base + c];
            }
        }
    }
    return out;
}

InterfaceTransferResult conservative_apply(
    InterfaceOperatorKind kind,
    const svmp::search::InterfaceMap& interface_map,
    const std::vector<Real>& source_face_values) {
    InterfaceTransferResult result;
    result.kind = kind;
    result.target_values.assign(target_size(interface_map), Real{0.0});
    result.target_weights.assign(result.target_values.size(), Real{0.0});

    for (const auto& pair : interface_map.pairs) {
        if (pair.target_face < 0) {
            continue;
        }
        const auto target = static_cast<std::size_t>(pair.target_face);
        const Real value = source_value(source_face_values, pair.source_face);
        const Real source_measure = positive_measure(pair.source_measure);
        const Real target_measure = positive_measure(pair.target_measure);
        result.target_values[target] += value * source_measure;
        result.target_weights[target] = std::max(result.target_weights[target], target_measure);
        result.source_integral += value * source_measure;
    }

    for (std::size_t i = 0; i < result.target_values.size(); ++i) {
        if (result.target_weights[i] > Real{0.0}) {
            result.target_values[i] /= result.target_weights[i];
            result.target_integral += result.target_values[i] * result.target_weights[i];
        }
    }
    return result;
}

} // namespace

bool SlidingInterfaceMap::valid_for_current_revisions() const noexcept {
    return !stale_for_revision_anchors(interface_map) &&
           accepted_revision_key == interface_map.revision_key() &&
           (state == InterfaceOperatorState::AcceptedNonlinearState ||
            state == InterfaceOperatorState::AcceptedTimeStep);
}

bool SlidingInterfaceMap::valid_for_time_level(
    Real query_time,
    std::uint64_t query_epoch,
    Real time_tolerance) const noexcept {
    if (!valid_for_current_revisions()) {
        return false;
    }
    if (!std::isfinite(time) || !std::isfinite(query_time) || time_tolerance < 0.0) {
        return false;
    }
    return time_level_epoch == query_epoch &&
           std::abs(time - query_time) <= time_tolerance;
}

void SlidingInterfaceMap::set_trial_map(
    svmp::search::InterfaceMap map,
    Real map_time,
    std::uint64_t epoch) {
    name = map.name;
    interface_map = std::move(map);
    interface_map.state = svmp::search::InterfaceMapState::Trial;
    trial_revision_key = interface_map.revision_key();
    time = map_time;
    time_level_epoch = epoch;
    state = InterfaceOperatorState::Trial;
}

void SlidingInterfaceMap::accept_trial(InterfaceOperatorState accepted_state) {
    interface_map.accept_trial();
    accepted_revision_key = interface_map.revision_key();
    trial_revision_key = 0;
    state = accepted_state;
}

void SlidingInterfaceMap::rollback_trial() {
    interface_map.rollback_trial();
    trial_revision_key = 0;
    state = InterfaceOperatorState::RolledBack;
}

InterfaceTransferResult PointwiseInterfaceInterpolation::apply(
    const svmp::search::InterfaceMap& interface_map,
    const std::vector<Real>& source_face_values) const {
    InterfaceTransferResult result;
    result.kind = kind();
    result.target_values.assign(target_size(interface_map), Real{0.0});
    result.target_weights.assign(result.target_values.size(), Real{0.0});

    for (const auto& pair : interface_map.pairs) {
        if (pair.target_face < 0) {
            continue;
        }
        const auto target = static_cast<std::size_t>(pair.target_face);
        result.target_values[target] += source_value(source_face_values, pair.source_face);
        result.target_weights[target] += Real{1.0};
        result.source_integral += source_value(source_face_values, pair.source_face) *
                                  positive_measure(pair.source_measure);
    }

    for (std::size_t i = 0; i < result.target_values.size(); ++i) {
        if (result.target_weights[i] > Real{0.0}) {
            result.target_values[i] /= result.target_weights[i];
            result.target_integral += result.target_values[i] * result.target_weights[i];
        }
    }
    return result;
}

InterfaceTransferResult ConservativeInterfaceProjection::apply(
    const svmp::search::InterfaceMap& interface_map,
    const std::vector<Real>& source_face_values) const {
    return conservative_apply(kind(), interface_map, source_face_values);
}

InterfaceTransferResult MortarInterfaceProjection::apply(
    const svmp::search::InterfaceMap& interface_map,
    const std::vector<Real>& source_face_values) const {
    return conservative_apply(kind(), interface_map, source_face_values);
}

std::unique_ptr<InterfaceTransferOperator> makeInterfaceTransferOperator(
    InterfaceOperatorKind kind) {
    switch (kind) {
        case InterfaceOperatorKind::PointwiseInterpolation:
            return std::make_unique<PointwiseInterfaceInterpolation>();
        case InterfaceOperatorKind::ConservativeProjection:
            return std::make_unique<ConservativeInterfaceProjection>();
        case InterfaceOperatorKind::Mortar:
            return std::make_unique<MortarInterfaceProjection>();
    }
    throw std::invalid_argument("unknown interface operator kind");
}

InterfaceTransferResult applyInterfaceTransfer(
    const InterfaceTransferOperator& op,
    const svmp::search::InterfaceMap& interface_map,
    const std::vector<Real>& source_face_values,
    const InterfaceTransferOptions& options) {
    const std::uint32_t components = effective_components(options);
    if (components == 1u && options.frame_policy == InterfaceFrameTransformPolicy::None) {
        auto result = op.apply(interface_map, source_face_values);
        result.field_kind = options.field_kind;
        result.frame_policy = options.frame_policy;
        result.component_count = components;
        return result;
    }
    if (components == 0u || source_face_values.size() % components != 0u) {
        throw std::invalid_argument("interface transfer source value size is not divisible by component count");
    }

    const auto transformed = transform_source_values(source_face_values, options, components);
    const std::size_t source_faces = transformed.size() / components;

    InterfaceTransferResult result;
    result.kind = op.kind();
    result.field_kind = options.field_kind;
    result.frame_policy = options.frame_policy;
    result.component_count = components;
    result.target_values.assign(target_size(interface_map) * components, Real{0.0});

    for (std::uint32_t c = 0; c < components; ++c) {
        std::vector<Real> component_values(source_faces, Real{0.0});
        for (std::size_t f = 0; f < source_faces; ++f) {
            component_values[f] = transformed[f * components + c];
        }
        const auto component_result = op.apply(interface_map, component_values);
        if (c == 0u) {
            result.target_weights = component_result.target_weights;
            result.source_integral = component_result.source_integral;
            result.target_integral = component_result.target_integral;
        }
        for (std::size_t target = 0; target < component_result.target_values.size(); ++target) {
            result.target_values[target * components + c] = component_result.target_values[target];
        }
    }
    return result;
}

InterfaceTransferResult applySlidingInterfaceTransfer(
    const InterfaceTransferOperator& op,
    const SlidingInterfaceMap& sliding_map,
    const std::vector<Real>& source_face_values,
    const InterfaceTransferOptions& options) {
    if (!sliding_map.valid_for_current_revisions()) {
        throw std::invalid_argument("sliding interface map is not accepted for current revisions");
    }
    auto result = applyInterfaceTransfer(op,
                                         sliding_map.interface_map,
                                         source_face_values,
                                         options);
    result.interface_name = sliding_map.name;
    result.sliding_map_kind = sliding_map.map_kind;
    result.interface_state = sliding_map.state;
    result.interface_time = sliding_map.time;
    result.interface_time_level_epoch = sliding_map.time_level_epoch;
    result.interface_revision_key = sliding_map.accepted_revision_key;
    return result;
}

InterfaceTransferDiagnostics diagnoseInterfaceTransfer(
    const svmp::search::InterfaceMap& interface_map,
    const InterfaceTransferResult& result,
    Real tolerance) {
    InterfaceTransferDiagnostics diagnostics;
    diagnostics.source_pair_count = interface_map.pairs.size();
    diagnostics.target_value_count = result.target_values.size();
    diagnostics.source_integral = result.source_integral;
    diagnostics.target_integral = result.target_integral;
    diagnostics.conservation_error = std::abs(result.source_integral - result.target_integral);
    if ((result.kind == InterfaceOperatorKind::ConservativeProjection ||
         result.kind == InterfaceOperatorKind::Mortar) &&
        diagnostics.conservation_error > tolerance) {
        diagnostics.ok = false;
        diagnostics.messages.push_back("interface projection conservation error exceeds tolerance");
    }
    if (stale_for_revision_anchors(interface_map)) {
        diagnostics.ok = false;
        diagnostics.messages.push_back("interface map is stale for current mesh revisions");
    }
    if (result.target_values.empty() && !interface_map.pairs.empty()) {
        diagnostics.ok = false;
        diagnostics.messages.push_back("interface transfer produced no target values");
    }
    return diagnostics;
}

InterfaceOperatorInvalidationPolicy interfaceOperatorInvalidation(
    const SlidingInterfaceMap& sliding_map,
    std::uint64_t fe_dof_layout_revision,
    std::uint64_t cached_fe_dof_layout_revision) {
    InterfaceOperatorInvalidationPolicy policy;
    const bool mesh_changed =
        stale_for_revision_anchors(sliding_map.interface_map) ||
        sliding_map.accepted_revision_key != sliding_map.interface_map.revision_key();
    const bool dof_changed = fe_dof_layout_revision != cached_fe_dof_layout_revision;
    if (mesh_changed) {
        policy.rebuild_search = true;
        policy.rebuild_interface_operator = true;
        policy.refresh_matrix_free_geometry = true;
        policy.rebuild_matrix = true;
        policy.refresh_preconditioner = true;
        policy.refresh_restart_metadata = true;
    }
    if (dof_changed) {
        policy.rebuild_interface_operator = true;
        policy.rebuild_matrix = true;
        policy.refresh_preconditioner = true;
        policy.refresh_restart_metadata = true;
    }
    return policy;
}

InterfaceOperatorInvalidationPolicy interfaceOperatorInvalidationForTime(
    const SlidingInterfaceMap& sliding_map,
    Real current_time,
    std::uint64_t current_time_level_epoch,
    std::uint64_t fe_dof_layout_revision,
    std::uint64_t cached_fe_dof_layout_revision,
    Real time_tolerance) {
    auto policy = interfaceOperatorInvalidation(sliding_map,
                                                fe_dof_layout_revision,
                                                cached_fe_dof_layout_revision);
    if (!sliding_map.valid_for_time_level(current_time,
                                          current_time_level_epoch,
                                          time_tolerance)) {
        policy.rebuild_search = true;
        policy.rebuild_interface_operator = true;
        policy.refresh_matrix_free_geometry = true;
        policy.rebuild_matrix = true;
        policy.refresh_preconditioner = true;
        policy.refresh_restart_metadata = true;
    }
    return policy;
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
