#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Generated interface domain lifecycle for level-set fields.
 */

#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace svmp::FE::level_set {

enum class GeneratedInterfaceGeometryMode {
    LinearCorner,
    HighOrderImplicit,
};

enum class ImplicitCutQuadratureBackend {
    LinearCorner,
    SayeHyperrectangle,
    HighOrderSubcell,
    MomentFit,
};

enum class ImplicitCutFallbackPolicy {
    Fail,
    LinearCorner,
};

enum class GeometryTangentPolicy {
    RefreshedFrozenQuadrature,
    DifferentiatedQuadrature,
};

struct GeometryQuadratureSensitivitySupport {
    GeometryTangentPolicy policy{
        GeometryTangentPolicy::RefreshedFrozenQuadrature};
    bool point_location_sensitivity_available{false};
    bool quadrature_weight_sensitivity_available{false};
    bool measure_sensitivity_available{false};
    bool normal_sensitivity_available{false};
    bool topology_transition_sensitivity_available{false};
    std::string diagnostic{};

    [[nodiscard]] bool complete() const noexcept
    {
        return point_location_sensitivity_available &&
               quadrature_weight_sensitivity_available &&
               measure_sensitivity_available &&
               normal_sensitivity_available &&
               topology_transition_sensitivity_available;
    }
};

[[nodiscard]] const char* generatedInterfaceGeometryModeName(
    GeneratedInterfaceGeometryMode mode) noexcept;

[[nodiscard]] const char* implicitCutQuadratureBackendName(
    ImplicitCutQuadratureBackend backend) noexcept;

[[nodiscard]] const char* implicitCutFallbackPolicyName(
    ImplicitCutFallbackPolicy policy) noexcept;

[[nodiscard]] const char* geometryTangentPolicyName(
    GeometryTangentPolicy policy) noexcept;

[[nodiscard]] GeometryQuadratureSensitivitySupport
geometryQuadratureSensitivitySupport(GeometryTangentPolicy policy);

[[nodiscard]] std::string levelSetImplicitCutBackendCellDiagnostic(
    ImplicitCutQuadratureBackend backend,
    GlobalIndex cell_id,
    ElementType type,
    const std::string& diagnostic);

struct LevelSetGeneratedInterfaceOptions {
    std::string level_set_field_name{"level_set"};
    std::string domain_id{"free_surface"};
    int requested_interface_marker{-1};
    Real isovalue{0.0};
    Real tolerance{1.0e-12};
    int quadrature_order{interfaces::defaultLevelSetCutVolumeQuadratureOrder(
        /*geometry_order=*/1,
        /*field_order=*/1,
        /*form_order=*/1)};
    int interface_quadrature_order{-1};
    int volume_quadrature_order{interfaces::defaultLevelSetCutVolumeQuadratureOrder(
        /*geometry_order=*/1,
        /*field_order=*/1,
        /*form_order=*/2)};
    GeneratedInterfaceGeometryMode geometry_mode{
        GeneratedInterfaceGeometryMode::LinearCorner};
    ImplicitCutQuadratureBackend implicit_cut_quadrature_backend{
        ImplicitCutQuadratureBackend::LinearCorner};
    ImplicitCutFallbackPolicy implicit_cut_fallback_policy{
        ImplicitCutFallbackPolicy::Fail};
    Real implicit_cut_root_tolerance{1.0e-10};
    int implicit_cut_max_subdivision_depth{16};
    bool keep_degenerate_fragments{false};
    bool allow_corner_linearized_geometry{false};
    GeometryTangentPolicy geometry_tangent_policy{
        GeometryTangentPolicy::RefreshedFrozenQuadrature};
};

struct LevelSetGeneratedInterfaceResult {
    bool success{false};
    int interface_marker{-1};
    std::uint64_t value_revision{0};
    interfaces::LevelSetInterfaceDomain domain{};
    interfaces::CutInterfaceDomainSummary summary{};
    std::string diagnostic{};
    std::size_t cell_count{0};
    std::size_t corner_linearized_cell_count{0};
    std::size_t max_cell_node_count{0};
    std::size_t max_corner_node_count{0};
    ImplicitCutQuadratureBackend implicit_cut_quadrature_backend{
        ImplicitCutQuadratureBackend::LinearCorner};
    int achieved_interface_quadrature_order{0};
    int achieved_volume_quadrature_order{0};
    std::size_t implicit_cut_fallback_cell_count{0};
    GeometryTangentPolicy geometry_tangent_policy{
        GeometryTangentPolicy::RefreshedFrozenQuadrature};
};

class LevelSetGeneratedInterfaceLifecycle {
public:
    explicit LevelSetGeneratedInterfaceLifecycle(int marker_base = 1000000,
                                                 int marker_range = 1000000);

    [[nodiscard]] LevelSetGeneratedInterfaceResult build(
        const systems::FESystem& system,
        const LevelSetGeneratedInterfaceOptions& options,
        std::span<const Real> solution);

    [[nodiscard]] std::uint64_t valueRevision() const noexcept { return value_revision_; }
    void restoreValueRevision(std::uint64_t value_revision) noexcept;

private:
    interfaces::GeneratedInterfaceMarkerRegistry marker_registry_;
    std::uint64_t value_revision_{0};
};

} // namespace svmp::FE::level_set
