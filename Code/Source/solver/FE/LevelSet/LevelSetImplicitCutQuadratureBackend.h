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

} // namespace svmp::FE::level_set
