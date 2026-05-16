#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Backend interface for generated implicit level-set cut quadrature.
 */

#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"

namespace svmp::FE::level_set {

struct ImplicitCutQuadratureBackendCellResult {
    interfaces::LevelSetCellCutResult cut{};
    int achieved_interface_quadrature_order{0};
    int achieved_volume_quadrature_order{0};
    bool fallback_used{false};
};

struct ImplicitCutQuadratureBackendCapability {
    ImplicitCutQuadratureBackend backend{ImplicitCutQuadratureBackend::LinearCorner};
    int mesh_dimension{0};
    ElementType element_type{ElementType::Unknown};
    bool implemented{false};
    bool supports_element_type{false};
    bool supports_high_order_geometry{false};
    bool requires_scalar_h1_c0_level_set{true};
    int minimum_level_set_order{1};
    int validation_level_set_order{3};
    bool returns_reference_frame_rules{true};
    bool requires_positive_volume_weights{true};
    bool requires_deterministic_rule_order{true};
    bool prunes_tiny_slivers_in_context{true};
    bool near_tangent_requires_diagnostic{true};
    Real tiny_sliver_volume_fraction{1.0e-10};
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
        const interfaces::CutInterfaceDomainRequest& request) const noexcept = 0;

    [[nodiscard]] virtual int achievedVolumeQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept = 0;

    [[nodiscard]] virtual ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const interfaces::LevelSetCellCutInput& input) const = 0;
};

[[nodiscard]] const ImplicitCutQuadratureBackendDriver&
implicitCutQuadratureBackendDriver(ImplicitCutQuadratureBackend backend);

[[nodiscard]] ImplicitCutQuadratureBackendCapability
implicitCutQuadratureBackendCapability(ImplicitCutQuadratureBackend backend,
                                       int mesh_dimension,
                                       ElementType element_type) noexcept;

} // namespace svmp::FE::level_set
