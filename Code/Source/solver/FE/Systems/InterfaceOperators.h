/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_INTERFACEOPERATORS_H
#define SVMP_FE_SYSTEMS_INTERFACEOPERATORS_H

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Search/MultiMeshInterface.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

enum class InterfaceOperatorKind {
    PointwiseInterpolation,
    ConservativeProjection,
    Mortar
};

enum class InterfaceFieldKind : std::uint8_t {
    Scalar,
    Vector,
    Rank2Tensor,
    MixedBlock
};

enum class InterfaceFrameTransformPolicy : std::uint8_t {
    None,
    SourceToTargetVector,
    SourceToTargetRank2Tensor
};

enum class SlidingInterfaceMapKind : std::uint8_t {
    FittedNonmatching,
    Sliding,
    RotatingSliding,
    CyclicAngularPeriodic
};

enum class InterfaceOperatorState : std::uint8_t {
    Empty,
    Trial,
    AcceptedNonlinearState,
    AcceptedTimeStep,
    RolledBack
};

struct InterfaceTransferOptions {
    InterfaceFieldKind field_kind{InterfaceFieldKind::Scalar};
    InterfaceFrameTransformPolicy frame_policy{InterfaceFrameTransformPolicy::None};
    std::uint32_t component_count{1};
    std::array<std::array<Real, 3>, 3> source_to_target_rotation{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}}};
    Real conservation_tolerance{1.0e-10};
};

struct InterfaceTransferResult {
    InterfaceOperatorKind kind = InterfaceOperatorKind::PointwiseInterpolation;
    InterfaceFieldKind field_kind = InterfaceFieldKind::Scalar;
    InterfaceFrameTransformPolicy frame_policy = InterfaceFrameTransformPolicy::None;
    std::uint32_t component_count = 1;
    std::string interface_name{};
    SlidingInterfaceMapKind sliding_map_kind{SlidingInterfaceMapKind::Sliding};
    InterfaceOperatorState interface_state{InterfaceOperatorState::Empty};
    Real interface_time{0.0};
    std::uint64_t interface_time_level_epoch{0};
    std::uint64_t interface_revision_key{0};
    std::vector<Real> target_values;
    std::vector<Real> target_weights;
    Real source_integral = 0.0;
    Real target_integral = 0.0;
};

struct InterfaceTransferDiagnostics {
    bool ok{true};
    Real conservation_error{0.0};
    Real source_integral{0.0};
    Real target_integral{0.0};
    std::size_t source_pair_count{0};
    std::size_t target_value_count{0};
    std::vector<std::string> messages{};
};

struct SlidingInterfaceMap {
    std::string name{};
    SlidingInterfaceMapKind map_kind{SlidingInterfaceMapKind::Sliding};
    svmp::search::InterfaceMap interface_map{};
    InterfaceOperatorState state{InterfaceOperatorState::Empty};
    std::uint64_t accepted_revision_key{0};
    std::uint64_t trial_revision_key{0};
    Real time{0.0};
    std::uint64_t time_level_epoch{0};

    [[nodiscard]] bool valid_for_current_revisions() const noexcept;
    [[nodiscard]] bool valid_for_time_level(
        Real query_time,
        std::uint64_t query_epoch,
        Real time_tolerance = 1.0e-12) const noexcept;
    void set_trial_map(svmp::search::InterfaceMap map, Real map_time, std::uint64_t epoch);
    void accept_trial(InterfaceOperatorState accepted_state = InterfaceOperatorState::AcceptedTimeStep);
    void rollback_trial();
};

struct InterfaceOperatorInvalidationPolicy {
    bool rebuild_search{false};
    bool rebuild_interface_operator{false};
    bool refresh_matrix_free_geometry{false};
    bool rebuild_matrix{false};
    bool refresh_preconditioner{false};
    bool refresh_restart_metadata{false};

    [[nodiscard]] bool any() const noexcept {
        return rebuild_search || rebuild_interface_operator ||
               refresh_matrix_free_geometry || rebuild_matrix ||
               refresh_preconditioner || refresh_restart_metadata;
    }
};

class InterfaceTransferOperator {
public:
    virtual ~InterfaceTransferOperator() = default;
    [[nodiscard]] virtual InterfaceOperatorKind kind() const noexcept = 0;

    [[nodiscard]] virtual InterfaceTransferResult apply(
        const svmp::search::InterfaceMap& interface_map,
        const std::vector<Real>& source_face_values) const = 0;
};

class PointwiseInterfaceInterpolation final : public InterfaceTransferOperator {
public:
    [[nodiscard]] InterfaceOperatorKind kind() const noexcept override {
        return InterfaceOperatorKind::PointwiseInterpolation;
    }

    [[nodiscard]] InterfaceTransferResult apply(
        const svmp::search::InterfaceMap& interface_map,
        const std::vector<Real>& source_face_values) const override;
};

class ConservativeInterfaceProjection final : public InterfaceTransferOperator {
public:
    [[nodiscard]] InterfaceOperatorKind kind() const noexcept override {
        return InterfaceOperatorKind::ConservativeProjection;
    }

    [[nodiscard]] InterfaceTransferResult apply(
        const svmp::search::InterfaceMap& interface_map,
        const std::vector<Real>& source_face_values) const override;
};

class MortarInterfaceProjection final : public InterfaceTransferOperator {
public:
    [[nodiscard]] InterfaceOperatorKind kind() const noexcept override {
        return InterfaceOperatorKind::Mortar;
    }

    [[nodiscard]] InterfaceTransferResult apply(
        const svmp::search::InterfaceMap& interface_map,
        const std::vector<Real>& source_face_values) const override;
};

[[nodiscard]] std::unique_ptr<InterfaceTransferOperator> makeInterfaceTransferOperator(
    InterfaceOperatorKind kind);

[[nodiscard]] InterfaceTransferResult applyInterfaceTransfer(
    const InterfaceTransferOperator& op,
    const svmp::search::InterfaceMap& interface_map,
    const std::vector<Real>& source_face_values,
    const InterfaceTransferOptions& options = {});

[[nodiscard]] InterfaceTransferResult applySlidingInterfaceTransfer(
    const InterfaceTransferOperator& op,
    const SlidingInterfaceMap& sliding_map,
    const std::vector<Real>& source_face_values,
    const InterfaceTransferOptions& options = {});

[[nodiscard]] InterfaceTransferDiagnostics diagnoseInterfaceTransfer(
    const svmp::search::InterfaceMap& interface_map,
    const InterfaceTransferResult& result,
    Real tolerance = 1.0e-10);

[[nodiscard]] InterfaceOperatorInvalidationPolicy interfaceOperatorInvalidation(
    const SlidingInterfaceMap& sliding_map,
    std::uint64_t fe_dof_layout_revision,
    std::uint64_t cached_fe_dof_layout_revision);

[[nodiscard]] InterfaceOperatorInvalidationPolicy interfaceOperatorInvalidationForTime(
    const SlidingInterfaceMap& sliding_map,
    Real current_time,
    std::uint64_t current_time_level_epoch,
    std::uint64_t fe_dof_layout_revision,
    std::uint64_t cached_fe_dof_layout_revision,
    Real time_tolerance = 1.0e-12);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_SYSTEMS_INTERFACEOPERATORS_H
