#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] bool finitePoint(const geometry::CutQuadraturePoint& point) noexcept
{
    return std::isfinite(point.point[0]) &&
           std::isfinite(point.point[1]) &&
           std::isfinite(point.point[2]) &&
           std::isfinite(point.normal[0]) &&
           std::isfinite(point.normal[1]) &&
           std::isfinite(point.normal[2]) &&
           std::isfinite(point.weight);
}

[[nodiscard]] bool finiteArray(const std::array<Real, 3>& values) noexcept
{
    return std::isfinite(values[0]) &&
           std::isfinite(values[1]) &&
           std::isfinite(values[2]);
}

[[nodiscard]] ImplicitCutQuadratureDiagnosticStatus
classifyCutStatus(const interfaces::LevelSetCellCutResult& cut,
                  bool fallback_used) noexcept
{
    if (!cut.supported) {
        return ImplicitCutQuadratureDiagnosticStatus::Unsupported;
    }
    if (fallback_used) {
        return ImplicitCutQuadratureDiagnosticStatus::Fallback;
    }
    switch (cut.degeneracy) {
    case interfaces::CutInterfaceDegeneracy::None:
        return cut.hasActiveFragments()
                   ? ImplicitCutQuadratureDiagnosticStatus::Cut
                   : ImplicitCutQuadratureDiagnosticStatus::ExactNoCut;
    case interfaces::CutInterfaceDegeneracy::NoCut:
        return ImplicitCutQuadratureDiagnosticStatus::ExactNoCut;
    case interfaces::CutInterfaceDegeneracy::NearlyTangent:
        return ImplicitCutQuadratureDiagnosticStatus::Tangent;
    case interfaces::CutInterfaceDegeneracy::FullZeroCell:
    case interfaces::CutInterfaceDegeneracy::VertexTouch:
    case interfaces::CutInterfaceDegeneracy::EdgeTouch:
    case interfaces::CutInterfaceDegeneracy::SmallFragment:
        return ImplicitCutQuadratureDiagnosticStatus::Degenerate;
    }
    return ImplicitCutQuadratureDiagnosticStatus::Failed;
}

[[nodiscard]] ImplicitCutQuadratureBackendValidation failedValidation(
    ImplicitCutQuadratureDiagnosticStatus status,
    std::string diagnostic)
{
    return ImplicitCutQuadratureBackendValidation{
        .ok = false,
        .status = status,
        .diagnostic = std::move(diagnostic)};
}

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
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            return result;
        }

        if (mesh_dimension == 2) {
            result.cut =
                interfaces::cutLinearLevelSetCell2D(request, input);
        } else if (mesh_dimension == 3) {
            result.cut =
                interfaces::cutLinearLevelSetCell3D(request, input);
        }
        result.diagnostic_status =
            classifyCutStatus(result.cut, result.fallback_used);
        return result;
    }
};

[[nodiscard]] bool supportsSayeHyperrectangleMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept
{
    if (mesh_dimension != 2) {
        return false;
    }
    switch (element_type) {
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return true;
    default:
        return false;
    }
}

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

ImplicitCutQuadratureBackendCapability
implicitCutQuadratureBackendCapability(ImplicitCutQuadratureBackend backend,
                                       int mesh_dimension,
                                       ElementType element_type) noexcept
{
    ImplicitCutQuadratureBackendCapability capability{};
    capability.backend = backend;
    capability.mesh_dimension = mesh_dimension;
    capability.element_type = element_type;

    switch (backend) {
    case ImplicitCutQuadratureBackend::LinearCorner:
        capability.implemented = true;
        capability.supports_element_type =
            (mesh_dimension == 2 &&
             interfaces::supportsLinearLevelSetCellCut2D(element_type)) ||
            (mesh_dimension == 3 &&
             interfaces::supportsLinearLevelSetCellCut3D(element_type));
        capability.supports_high_order_geometry = false;
        capability.validation_level_set_order = 1;
        capability.maximum_reported_interface_order = 1;
        capability.maximum_reported_volume_order = 2;
        return capability;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        capability.implemented = false;
        capability.supports_element_type =
            supportsSayeHyperrectangleMilestone(mesh_dimension, element_type);
        capability.supports_high_order_geometry = true;
        capability.maximum_reported_interface_order = -1;
        capability.maximum_reported_volume_order = -1;
        return capability;
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
    case ImplicitCutQuadratureBackend::MomentFit:
        capability.implemented = false;
        capability.supports_element_type = false;
        capability.supports_high_order_geometry = true;
        capability.maximum_reported_interface_order = -1;
        capability.maximum_reported_volume_order = -1;
        return capability;
    }
    return capability;
}

const char* implicitCutQuadratureDiagnosticStatusName(
    ImplicitCutQuadratureDiagnosticStatus status) noexcept
{
    switch (status) {
    case ImplicitCutQuadratureDiagnosticStatus::ExactNoCut:
        return "ExactNoCut";
    case ImplicitCutQuadratureDiagnosticStatus::Cut:
        return "Cut";
    case ImplicitCutQuadratureDiagnosticStatus::Tangent:
        return "Tangent";
    case ImplicitCutQuadratureDiagnosticStatus::Degenerate:
        return "Degenerate";
    case ImplicitCutQuadratureDiagnosticStatus::Fallback:
        return "Fallback";
    case ImplicitCutQuadratureDiagnosticStatus::Unsupported:
        return "Unsupported";
    case ImplicitCutQuadratureDiagnosticStatus::Failed:
        return "Failed";
    }
    return "Failed";
}

ImplicitCutQuadratureBackendValidation
validateImplicitCutQuadratureBackendCellResult(
    const interfaces::CutInterfaceDomainRequest& request,
    const interfaces::LevelSetCellCutInput& input,
    const ImplicitCutQuadratureBackendCellResult& result)
{
    const auto status = result.diagnostic_status ==
                                ImplicitCutQuadratureDiagnosticStatus::Failed
                            ? classifyCutStatus(result.cut, result.fallback_used)
                            : result.diagnostic_status;
    if (!result.cut.supported) {
        return ImplicitCutQuadratureBackendValidation{
            .ok = true,
            .status = status,
            .diagnostic = result.cut.diagnostic};
    }
    if (result.achieved_interface_quadrature_order < 0 ||
        result.achieved_volume_quadrature_order < 0) {
        return failedValidation(
            ImplicitCutQuadratureDiagnosticStatus::Failed,
            "implicit cut backend reported a negative achieved quadrature order");
    }

    for (const auto& fragment : result.cut.fragments) {
        if (fragment.parent_cell != input.parent_cell) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned an interface fragment for the wrong parent cell");
        }
        if (fragment.interface_marker >= 0 &&
            fragment.interface_marker != request.interface_marker) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned an interface fragment with the wrong marker");
        }
        if (!std::isfinite(fragment.measure) ||
            fragment.measure < Real{0.0} ||
            !finiteArray(fragment.normal)) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned invalid interface fragment measure or normal");
        }
        for (const auto& point : fragment.quadrature_points) {
            geometry::CutQuadraturePoint qp;
            qp.point = point.point;
            qp.normal = point.normal;
            qp.weight = point.weight;
            if (!finitePoint(qp) || qp.weight <= Real{0.0}) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned an invalid interface quadrature point");
            }
        }
    }

    Real parent_measure = Real{0.0};
    Real negative_measure = Real{0.0};
    Real positive_measure = Real{0.0};
    for (const auto& region : result.cut.volume_regions) {
        if (region.parent_cell != input.parent_cell) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned a volume region for the wrong parent cell");
        }
        if (region.interface_marker >= 0 &&
            region.interface_marker != request.interface_marker) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned a volume region with the wrong marker");
        }
        if (region.side == geometry::CutIntegrationSide::Interface) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned an interface side for a volume region");
        }
        if (!std::isfinite(region.measure) ||
            !std::isfinite(region.parent_measure) ||
            !std::isfinite(region.volume_fraction) ||
            region.measure < Real{0.0} ||
            region.parent_measure < Real{0.0} ||
            !finiteArray(region.centroid) ||
            !finiteArray(region.normal)) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned invalid volume region metadata");
        }
        for (const auto& point : region.quadrature_points) {
            if (!finitePoint(point) || point.weight <= Real{0.0}) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned an invalid volume quadrature point");
            }
        }
        parent_measure = std::max(parent_measure, region.parent_measure);
        if (region.side == geometry::CutIntegrationSide::Negative) {
            negative_measure += region.measure;
        } else if (region.side == geometry::CutIntegrationSide::Positive) {
            positive_measure += region.measure;
        }
    }

    if (parent_measure > Real{0.0}) {
        const Real total = negative_measure + positive_measure;
        const Real tolerance =
            std::max(request.tolerance, request.tolerance * parent_measure);
        if (std::abs(total - parent_measure) > tolerance) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend volume measures do not sum to the parent measure");
        }
    }

    return ImplicitCutQuadratureBackendValidation{
        .ok = true,
        .status = status,
        .diagnostic = result.cut.diagnostic};
}

} // namespace svmp::FE::level_set
