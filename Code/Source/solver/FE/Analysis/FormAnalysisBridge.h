/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_FORMANALYSISBRIDGE_H
#define SVMP_FE_ANALYSIS_FORMANALYSISBRIDGE_H

/**
 * @file FormAnalysisBridge.h
 * @brief Public Forms/Systems metadata bridge for setup-time diagnostics.
 *
 * The bridge normalizes public Forms and Analysis records into explicit
 * terminal, dependency, and installed-block evidence. It is intentionally owned
 * by the lower FE layers so higher-level setup code can consume metadata
 * without depending on private Forms implementation details.
 */

#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

enum class FormBridgeFeature : std::uint8_t {
    ContributionIdentity,
    OwningSystem,
    FieldTerminals,
    NonFieldTerminals,
    TemporalTerminals,
    GeometryTerminals,
    GeometrySensitivity,
    InstalledDependencies,
    InstalledBlocks,
};

enum class FormBridgeFeatureStatus : std::uint8_t {
    Available,
    Partial,
    Unavailable,
};

struct FormBridgeFeatureGate {
    FormBridgeFeature feature{FormBridgeFeature::FieldTerminals};
    FormBridgeFeatureStatus status{FormBridgeFeatureStatus::Unavailable};
    std::string reason;
};

enum class FormTerminalKind : std::uint8_t {
    TestField,
    TrialField,
    StateField,
    DiscreteField,
    ParameterSymbol,
    ParameterRef,
    Coefficient,
    BoundaryFunctionalSymbol,
    BoundaryIntegralSymbol,
    BoundaryIntegralRef,
    AuxiliaryStateSymbol,
    AuxiliaryStateRef,
    AuxiliaryInputSymbol,
    AuxiliaryInputRef,
    AuxiliaryOutputSymbol,
    AuxiliaryOutputRef,
    MaterialStateOldRef,
    MaterialStateWorkRef,
    Time,
    TimeStep,
    EffectiveTimeStep,
    TimeDerivative,
    PreviousSolutionRef,
    Coordinate,
    ReferenceCoordinate,
    CurrentCoordinate,
    PreviousCoordinate,
    ReferencePhysicalCoordinate,
    MeshDisplacement,
    MeshVelocity,
    MeshAcceleration,
    PreviousMeshVelocity,
    PredictedMeshVelocity,
    CurrentJacobian,
    ReferenceJacobian,
    CurrentJacobianDeterminant,
    ReferenceJacobianDeterminant,
    CurrentNormal,
    ReferenceNormal,
    CurrentMeasure,
    ReferenceMeasure,
    SurfaceJacobian,
    GeometryTrialVectorVariation,
    GeometryTrialJacobianVariation,
    MeshVelocityVariation,
    CurrentMeasureVariation,
    CurrentNormalVariation,
    SurfaceJacobianVariation,
    CellDiameter,
    CellVolume,
    FacetArea,
    CellDomainId,
    FrameTransform,
};

struct FormTerminalMetadata {
    FormTerminalKind kind{FormTerminalKind::StateField};
    std::size_t terminal_sequence{0u};
    std::string provider{"Forms"};
    FieldType value_type{FieldType::Scalar};
    int value_dimension{1};
    std::string owner_system_name;
    std::string owner_participant_name;
    std::string contribution_name;
    std::string origin;
    std::string operator_tag;
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
    FieldId field_id{INVALID_FIELD_ID};
    std::optional<VariableKey> graph_variable;
    std::string symbol_name;
    std::optional<std::uint32_t> slot;
    std::optional<std::uint32_t> state_offset_bytes;
    std::optional<int> history_index;
    int derivative_order{0};
    std::optional<forms::GeometryConfiguration> frame_from;
    std::optional<forms::GeometryConfiguration> frame_to;
};

struct FormInstalledDependencyMetadata {
    VariableKey residual_row;
    VariableKey dependency;
    DomainKind domain{DomainKind::Cell};
    bool contributes_matrix_block{false};
    bool contributes_vector{true};
    std::string provider;
};

struct FormInstalledBlockMetadata {
    VariableKey residual_row;
    VariableKey dependency;
    std::vector<DomainKind> domains;
    bool has_matrix{false};
    bool has_vector{false};
    std::string provider;
};

struct FormAnalysisBridgeOptions {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    std::string owner_participant_name;
    forms::GeometrySensitivityOptions geometry_sensitivity{};
};

struct FormContributionAnalysisMetadata {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    std::string operator_tag;
    bool contribution_name_explicit{false};
    std::vector<FieldId> installed_fields;
    forms::GeometrySensitivityOptions geometry_sensitivity{};
    std::vector<FormTerminalMetadata> terminals;
    std::vector<FormInstalledDependencyMetadata> installed_dependencies;
    std::vector<FormInstalledBlockMetadata> installed_blocks;
    std::vector<FormBridgeFeatureGate> feature_gates;
};

[[nodiscard]] const char* toString(FormBridgeFeature feature) noexcept;
[[nodiscard]] const char* toString(FormBridgeFeatureStatus status) noexcept;
[[nodiscard]] const char* toString(FormTerminalKind kind) noexcept;

[[nodiscard]] std::vector<FormTerminalMetadata> collectFormTerminalMetadata(
    const forms::FormExprNode& root,
    const FormAnalysisBridgeOptions& options = {},
    std::string operator_tag = {});

[[nodiscard]] FormContributionAnalysisMetadata buildFormAnalysisMetadata(
    const FormulationRecord& formulation,
    const std::vector<ContributionDescriptor>& contributions,
    const FormAnalysisBridgeOptions& options = {});

[[nodiscard]] bool bridgeFeatureAvailable(
    const FormContributionAnalysisMetadata& metadata,
    FormBridgeFeature feature) noexcept;

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_FORMANALYSISBRIDGE_H
