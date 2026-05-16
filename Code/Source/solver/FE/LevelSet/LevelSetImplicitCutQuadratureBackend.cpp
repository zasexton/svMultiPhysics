#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace svmp::FE::level_set {
namespace {

class LinearCornerImplicitCutBackend final
    : public ImplicitCutQuadratureBackendDriver {
public:
    [[nodiscard]] ImplicitCutQuadratureBackend kind() const noexcept override {
        return ImplicitCutQuadratureBackend::LinearCorner;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return implicitCutQuadratureBackendName(kind());
    }

    [[nodiscard]] bool supports(int mesh_dimension,
                                ElementType element_type) const noexcept override
    {
        if (mesh_dimension == 2) {
            return interfaces::supportsLinearLevelSetCellCut2D(element_type);
        }
        if (mesh_dimension == 3) {
            return interfaces::supportsLinearLevelSetCellCut3D(element_type);
        }
        return false;
    }

    [[nodiscard]] int achievedInterfaceQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return std::min(request.resolvedInterfaceQuadratureOrder(), 1);
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return interfaces::implementedLevelSetCutVolumeExactOrder(
            request.resolvedVolumeQuadratureOrder());
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const interfaces::LevelSetCellCutInput& input) const override
    {
        ImplicitCutQuadratureBackendCellResult result{};
        result.achieved_interface_quadrature_order =
            achievedInterfaceQuadratureOrder(request);
        result.achieved_volume_quadrature_order =
            achievedVolumeQuadratureOrder(request);

        if (!supports(mesh_dimension, input.element_type)) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "LinearCorner implicit cut quadrature backend does not support "
                "element type " +
                std::to_string(static_cast<unsigned>(input.element_type)) +
                " in mesh dimension " + std::to_string(mesh_dimension);
            return result;
        }

        if (mesh_dimension == 2) {
            result.cut =
                interfaces::cutLinearLevelSetCell2D(request, input);
        } else if (mesh_dimension == 3) {
            result.cut =
                interfaces::cutLinearLevelSetCell3D(request, input);
        }
        return result;
    }
};

} // namespace

const ImplicitCutQuadratureBackendDriver&
implicitCutQuadratureBackendDriver(ImplicitCutQuadratureBackend backend)
{
    static const LinearCornerImplicitCutBackend linear_corner_backend;

    switch (backend) {
    case ImplicitCutQuadratureBackend::LinearCorner:
        return linear_corner_backend;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
    case ImplicitCutQuadratureBackend::MomentFit:
        throw std::invalid_argument(
            std::string(implicitCutQuadratureBackendName(backend)) +
            " implicit cut quadrature backend is not implemented yet");
    }
    throw std::invalid_argument("unknown implicit cut quadrature backend");
}

} // namespace svmp::FE::level_set
