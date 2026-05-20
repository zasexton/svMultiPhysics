#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Backend interface for generated implicit level-set cut quadrature.
 */

#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"

#include <array>
#include <cstddef>
#include <string>

namespace svmp::FE::level_set {

enum class ImplicitCutQuadratureDiagnosticStatus {
    ExactNoCut,
    Cut,
    Tangent,
    Degenerate,
    Fallback,
    Unsupported,
    Failed,
};

enum class ImplicitCutQuadratureBackendQualification {
    Unavailable,
    Experimental,
    ProductionQualified,
};

[[nodiscard]] const char* implicitCutQuadratureDiagnosticStatusName(
    ImplicitCutQuadratureDiagnosticStatus status) noexcept;

[[nodiscard]] const char* implicitCutQuadratureBackendQualificationName(
    ImplicitCutQuadratureBackendQualification qualification) noexcept;

struct ImplicitCutQuadratureBackendCellResult {
    interfaces::LevelSetCellCutResult cut{};
    int requested_interface_quadrature_order{0};
    int requested_volume_quadrature_order{0};
    int possible_interface_quadrature_order{0};
    int possible_volume_quadrature_order{0};
    int achieved_interface_quadrature_order{0};
    int achieved_volume_quadrature_order{0};
    int verified_interface_quadrature_order{0};
    int verified_volume_quadrature_order{0};
    bool fallback_used{false};
    bool requested_high_order_downgrade{false};
    int root_branch_count{0};
    int root_finder_iteration_count{0};
    int curved_fragment_count{0};
    int linearized_leaf_count{0};
    int subdivision_count{0};
    int max_subdivision_depth_reached{0};
    int full_negative_region_count{0};
    int full_positive_region_count{0};
    int interface_fragment_count{0};
    std::size_t volume_quadrature_point_count{0};
    std::size_t interface_quadrature_point_count{0};
    double backend_elapsed_seconds{0.0};
    std::string fallback_reason{};
    ImplicitCutQuadratureBackend selected_backend{
        ImplicitCutQuadratureBackend::LinearCorner};
    ImplicitCutQuadratureDiagnosticStatus diagnostic_status{
        ImplicitCutQuadratureDiagnosticStatus::Failed};
};

struct ImplicitCutQuadratureBackendValidation {
    bool ok{true};
    ImplicitCutQuadratureDiagnosticStatus status{
        ImplicitCutQuadratureDiagnosticStatus::Cut};
    std::string diagnostic{};
};

struct ImplicitCutQuadratureBackendCellInput {
    interfaces::LevelSetCellCutInput linearized_input{};
    const LevelSetCellEvaluator* evaluator{nullptr};
    Real isovalue{0.0};
    std::array<Real, 3> reference_min{{0.0, 0.0, 0.0}};
    std::array<Real, 3> reference_max{{1.0, 1.0, 1.0}};
};

struct ImplicitCutQuadratureBackendCapability {
    ImplicitCutQuadratureBackend backend{ImplicitCutQuadratureBackend::LinearCorner};
    int mesh_dimension{0};
    ElementType element_type{ElementType::Unknown};
    bool implemented{false};
    bool supports_element_type{false};
    bool supports_high_order_geometry{false};
    ImplicitCutQuadratureBackendQualification qualification{
        ImplicitCutQuadratureBackendQualification::Unavailable};
    std::string qualification_diagnostic{};
    bool requires_scalar_h1_c0_level_set{true};
    int minimum_level_set_order{1};
    int validation_level_set_order{3};
    bool returns_reference_frame_rules{true};
    bool requires_positive_volume_weights{true};
    bool requires_deterministic_rule_order{true};
    bool prunes_tiny_slivers_in_context{true};
    bool near_tangent_requires_diagnostic{true};
    Real tiny_sliver_volume_fraction{1.0e-8};
    int maximum_reported_interface_order{-1};
    int maximum_reported_volume_order{-1};
};

class ImplicitCutQuadratureBackendDriver {
public:
    virtual ~ImplicitCutQuadratureBackendDriver() = default;

    [[nodiscard]] virtual ImplicitCutQuadratureBackend kind() const noexcept = 0;

    [[nodiscard]] virtual const char* name() const noexcept = 0;

    [[nodiscard]] virtual bool supports(int mesh_dimension,
                                        ElementType element_type) const noexcept = 0;

    [[nodiscard]] virtual int achievedInterfaceQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept = 0;

    [[nodiscard]] virtual int achievedVolumeQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept = 0;

    [[nodiscard]] virtual ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const = 0;
};

[[nodiscard]] const ImplicitCutQuadratureBackendDriver&
implicitCutQuadratureBackendDriver(ImplicitCutQuadratureBackend backend);

[[nodiscard]] ImplicitCutQuadratureBackendCapability
implicitCutQuadratureBackendCapability(ImplicitCutQuadratureBackend backend,
                                       int mesh_dimension,
                                       ElementType element_type) noexcept;

[[nodiscard]] ImplicitCutQuadratureBackendValidation
validateImplicitCutQuadratureBackendCellResult(
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const ImplicitCutQuadratureBackendCellResult& result);

} // namespace svmp::FE::level_set
