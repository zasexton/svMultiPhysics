#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

[[nodiscard]] bool supportsSayeHyperrectangleMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept;

[[nodiscard]] bool supportsHighOrderSubcellTriangleMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept;

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
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        ImplicitCutQuadratureBackendCellResult result{};
        result.achieved_interface_quadrature_order =
            achievedInterfaceQuadratureOrder(request);
        result.achieved_volume_quadrature_order =
            achievedVolumeQuadratureOrder(request);

        if (!supports(mesh_dimension, input.linearized_input.element_type)) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "LinearCorner implicit cut quadrature backend does not support "
                "element type " +
                std::to_string(static_cast<unsigned>(
                    input.linearized_input.element_type)) +
                " in mesh dimension " + std::to_string(mesh_dimension);
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            return result;
        }

        if (mesh_dimension == 2) {
            result.cut =
                interfaces::cutLinearLevelSetCell2D(
                    request, input.linearized_input);
        } else if (mesh_dimension == 3) {
            result.cut =
                interfaces::cutLinearLevelSetCell3D(
                    request, input.linearized_input);
        }
        result.diagnostic_status =
            classifyCutStatus(result.cut, result.fallback_used);
        return result;
    }
};

struct Rectangle2D {
    Real xmin{0.0};
    Real xmax{0.0};
    Real ymin{0.0};
    Real ymax{0.0};
};

struct Triangle2D {
    std::array<Real, 3> a{{0.0, 0.0, 0.0}};
    std::array<Real, 3> b{{0.0, 0.0, 0.0}};
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
};

struct SayeHyperrectangleDiagnostics {
    int max_depth_reached{0};
    int subdivision_count{0};
    int full_negative_region_count{0};
    int full_positive_region_count{0};
    int linearized_leaf_count{0};
    int interface_fragment_count{0};
};

[[nodiscard]] Real rectangleMeasure(const Rectangle2D& rect) noexcept
{
    return std::max(Real{0.0}, rect.xmax - rect.xmin) *
           std::max(Real{0.0}, rect.ymax - rect.ymin);
}

[[nodiscard]] Real triangleMeasure(const Triangle2D& tri) noexcept
{
    const Real x0 = tri.b[0] - tri.a[0];
    const Real y0 = tri.b[1] - tri.a[1];
    const Real x1 = tri.c[0] - tri.a[0];
    const Real y1 = tri.c[1] - tri.a[1];
    return Real{0.5} * std::abs(x0 * y1 - y0 * x1);
}

[[nodiscard]] std::array<Real, 3> rectangleCentroid(
    const Rectangle2D& rect) noexcept
{
    return {{Real{0.5} * (rect.xmin + rect.xmax),
             Real{0.5} * (rect.ymin + rect.ymax),
             0.0}};
}

[[nodiscard]] std::array<Real, 3> triangleCentroid(
    const Triangle2D& tri) noexcept
{
    return {{(tri.a[0] + tri.b[0] + tri.c[0]) / Real{3.0},
             (tri.a[1] + tri.b[1] + tri.c[1]) / Real{3.0},
             (tri.a[2] + tri.b[2] + tri.c[2]) / Real{3.0}}};
}

[[nodiscard]] std::array<Real, 3> midpoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {{Real{0.5} * (a[0] + b[0]),
             Real{0.5} * (a[1] + b[1]),
             Real{0.5} * (a[2] + b[2])}};
}

[[nodiscard]] std::array<Real, 3> normalizedOrDefault(
    const std::array<Real, 3>& value) noexcept
{
    const Real norm =
        std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (norm <= Real{1.0e-30}) {
        return {{1.0, 0.0, 0.0}};
    }
    return {{value[0] / norm, value[1] / norm, value[2] / norm}};
}

[[nodiscard]] Real signedLevelSetValue(
    const ImplicitCutQuadratureBackendCellInput& input,
    const std::array<Real, 3>& point)
{
    return input.evaluator
               ->evaluate(input.linearized_input.parent_cell, point)
               .value -
           input.isovalue;
}

[[nodiscard]] std::array<Real, 3> interfaceNormalAt(
    const ImplicitCutQuadratureBackendCellInput& input,
    const std::array<Real, 3>& point) noexcept
{
    try {
        return normalizedOrDefault(
            input.evaluator
                ->evaluate(input.linearized_input.parent_cell, point)
                .reference_gradient);
    } catch (...) {
        return {{1.0, 0.0, 0.0}};
    }
}

[[nodiscard]] std::vector<std::array<Real, 3>> rectangleSamplePoints(
    const Rectangle2D& rect)
{
    const Real xm = Real{0.5} * (rect.xmin + rect.xmax);
    const Real ym = Real{0.5} * (rect.ymin + rect.ymax);
    return {
        {{rect.xmin, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymax, 0.0}},
        {{rect.xmin, rect.ymax, 0.0}},
        {{xm, rect.ymin, 0.0}},
        {{rect.xmax, ym, 0.0}},
        {{xm, rect.ymax, 0.0}},
        {{rect.xmin, ym, 0.0}},
        {{xm, ym, 0.0}},
    };
}

[[nodiscard]] std::vector<std::array<Real, 3>> triangleSamplePoints(
    const Triangle2D& tri)
{
    return {
        tri.a,
        tri.b,
        tri.c,
        midpoint(tri.a, tri.b),
        midpoint(tri.b, tri.c),
        midpoint(tri.c, tri.a),
        triangleCentroid(tri),
    };
}

void appendFullRectangleRegion(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    geometry::CutIntegrationSide side,
    Real parent_measure,
    Real min_signed_value,
    Real max_signed_value,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    if (side == geometry::CutIntegrationSide::Negative) {
        ++diagnostics.full_negative_region_count;
    } else if (side == geometry::CutIntegrationSide::Positive) {
        ++diagnostics.full_positive_region_count;
    }

    const auto centroid = rectangleCentroid(rect);
    auto normal = interfaceNormalAt(input, centroid);
    if (side == geometry::CutIntegrationSide::Positive) {
        normal = {{-normal[0], -normal[1], -normal[2]}};
    }

    interfaces::CutInterfaceVolumeRegion region;
    region.interface_marker = request.interface_marker;
    region.parent_cell = input.linearized_input.parent_cell;
    region.side = side;
    region.centroid = centroid;
    region.normal = normal;
    region.parent_measure = parent_measure;
    region.measure = rectangleMeasure(rect);
    region.volume_fraction =
        parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
    region.min_level_set_value = min_signed_value;
    region.max_level_set_value = max_signed_value;
    region.full_cell_equivalent = std::abs(region.measure - parent_measure) <=
                                  std::max(request.tolerance,
                                           request.tolerance * parent_measure);
    if (region.measure > Real{0.0}) {
        geometry::CutQuadraturePoint qp;
        qp.point = region.centroid;
        qp.normal = region.normal;
        qp.weight = region.measure;
        region.quadrature_points.push_back(qp);
        cut.volume_regions.push_back(std::move(region));
    }
}

void appendFullTriangleRegion(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Triangle2D& tri,
    geometry::CutIntegrationSide side,
    Real parent_measure,
    Real min_signed_value,
    Real max_signed_value,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    if (side == geometry::CutIntegrationSide::Negative) {
        ++diagnostics.full_negative_region_count;
    } else if (side == geometry::CutIntegrationSide::Positive) {
        ++diagnostics.full_positive_region_count;
    }

    const auto centroid = triangleCentroid(tri);
    auto normal = interfaceNormalAt(input, centroid);
    if (side == geometry::CutIntegrationSide::Positive) {
        normal = {{-normal[0], -normal[1], -normal[2]}};
    }

    interfaces::CutInterfaceVolumeRegion region;
    region.interface_marker = request.interface_marker;
    region.parent_cell = input.linearized_input.parent_cell;
    region.side = side;
    region.centroid = centroid;
    region.normal = normal;
    region.parent_measure = parent_measure;
    region.measure = triangleMeasure(tri);
    region.volume_fraction =
        parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
    region.min_level_set_value = min_signed_value;
    region.max_level_set_value = max_signed_value;
    region.full_cell_equivalent = std::abs(region.measure - parent_measure) <=
                                  std::max(request.tolerance,
                                           request.tolerance * parent_measure);
    if (region.measure > Real{0.0}) {
        geometry::CutQuadraturePoint qp;
        qp.point = region.centroid;
        qp.normal = region.normal;
        qp.weight = region.measure;
        region.quadrature_points.push_back(qp);
        cut.volume_regions.push_back(std::move(region));
    }
}

void appendLinearizedRectangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    Real parent_measure,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    ++diagnostics.linearized_leaf_count;

    interfaces::LevelSetCellCutInput leaf;
    leaf.parent_cell = input.linearized_input.parent_cell;
    leaf.element_type = ElementType::Quad4;
    leaf.node_coordinates = {
        std::array<Real, 3>{rect.xmin, rect.ymin, 0.0},
        std::array<Real, 3>{rect.xmax, rect.ymin, 0.0},
        std::array<Real, 3>{rect.xmax, rect.ymax, 0.0},
        std::array<Real, 3>{rect.xmin, rect.ymax, 0.0},
    };
    leaf.level_set_values.reserve(leaf.node_coordinates.size());
    for (const auto& point : leaf.node_coordinates) {
        leaf.level_set_values.push_back(
            input.evaluator
                ->evaluate(input.linearized_input.parent_cell, point)
                .value);
    }

    auto leaf_cut = interfaces::cutLinearLevelSetCell2D(request, leaf);
    diagnostics.interface_fragment_count +=
        static_cast<int>(leaf_cut.fragments.size());
    for (auto& fragment : leaf_cut.fragments) {
        fragment.parent_cell = input.linearized_input.parent_cell;
        fragment.interface_marker = request.interface_marker;
        cut.fragments.push_back(std::move(fragment));
    }
    for (auto& region : leaf_cut.volume_regions) {
        region.parent_cell = input.linearized_input.parent_cell;
        region.interface_marker = request.interface_marker;
        region.parent_measure = parent_measure;
        region.volume_fraction =
            parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
        cut.volume_regions.push_back(std::move(region));
    }
    if (cut.degeneracy == interfaces::CutInterfaceDegeneracy::None) {
        cut.degeneracy = leaf_cut.degeneracy;
    }
}

void appendLinearizedTriangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Triangle2D& tri,
    Real parent_measure,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    ++diagnostics.linearized_leaf_count;

    interfaces::LevelSetCellCutInput leaf;
    leaf.parent_cell = input.linearized_input.parent_cell;
    leaf.element_type = ElementType::Triangle3;
    leaf.node_coordinates = {tri.a, tri.b, tri.c};
    leaf.level_set_values.reserve(leaf.node_coordinates.size());
    for (const auto& point : leaf.node_coordinates) {
        leaf.level_set_values.push_back(
            input.evaluator
                ->evaluate(input.linearized_input.parent_cell, point)
                .value);
    }

    auto leaf_cut = interfaces::cutLinearLevelSetCell2D(request, leaf);
    diagnostics.interface_fragment_count +=
        static_cast<int>(leaf_cut.fragments.size());
    for (auto& fragment : leaf_cut.fragments) {
        fragment.parent_cell = input.linearized_input.parent_cell;
        fragment.interface_marker = request.interface_marker;
        cut.fragments.push_back(std::move(fragment));
    }
    for (auto& region : leaf_cut.volume_regions) {
        region.parent_cell = input.linearized_input.parent_cell;
        region.interface_marker = request.interface_marker;
        region.parent_measure = parent_measure;
        region.volume_fraction =
            parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
        cut.volume_regions.push_back(std::move(region));
    }
    if (leaf_cut.hasActiveFragments() &&
        cut.degeneracy == interfaces::CutInterfaceDegeneracy::None) {
        cut.degeneracy = leaf_cut.degeneracy;
    }
}

void appendAdaptiveRectangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    Real parent_measure,
    int depth,
    int max_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    diagnostics.max_depth_reached =
        std::max(diagnostics.max_depth_reached, depth);
    const auto samples = rectangleSamplePoints(rect);
    bool has_negative = false;
    bool has_positive = false;
    Real min_signed = std::numeric_limits<Real>::infinity();
    Real max_signed = -std::numeric_limits<Real>::infinity();
    for (const auto& point : samples) {
        const Real value = signedLevelSetValue(input, point);
        min_signed = std::min(min_signed, value);
        max_signed = std::max(max_signed, value);
        has_negative = has_negative || value <= request.implicit_cut_root_tolerance;
        has_positive = has_positive || value >= -request.implicit_cut_root_tolerance;
    }

    if (!has_negative || !has_positive) {
        appendFullRectangleRegion(
            cut,
            request,
            input,
            rect,
            has_negative ? geometry::CutIntegrationSide::Negative
                         : geometry::CutIntegrationSide::Positive,
            parent_measure,
            min_signed,
            max_signed,
            diagnostics);
        return;
    }

    if (depth >= max_depth) {
        appendLinearizedRectangleCut(
            cut, request, input, rect, parent_measure, diagnostics);
        return;
    }

    ++diagnostics.subdivision_count;
    const Real xm = Real{0.5} * (rect.xmin + rect.xmax);
    const Real ym = Real{0.5} * (rect.ymin + rect.ymax);
    const std::array<Rectangle2D, 4> children{{
        Rectangle2D{rect.xmin, xm, rect.ymin, ym},
        Rectangle2D{xm, rect.xmax, rect.ymin, ym},
        Rectangle2D{xm, rect.xmax, ym, rect.ymax},
        Rectangle2D{rect.xmin, xm, ym, rect.ymax},
    }};
    for (const auto& child : children) {
        appendAdaptiveRectangleCut(
            cut,
            request,
            input,
            child,
            parent_measure,
            depth + 1,
            max_depth,
            diagnostics);
    }
}

void appendAdaptiveTriangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Triangle2D& tri,
    Real parent_measure,
    int depth,
    int max_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    diagnostics.max_depth_reached =
        std::max(diagnostics.max_depth_reached, depth);
    const auto samples = triangleSamplePoints(tri);
    bool has_negative = false;
    bool has_positive = false;
    Real min_signed = std::numeric_limits<Real>::infinity();
    Real max_signed = -std::numeric_limits<Real>::infinity();
    for (const auto& point : samples) {
        const Real value = signedLevelSetValue(input, point);
        min_signed = std::min(min_signed, value);
        max_signed = std::max(max_signed, value);
        has_negative = has_negative || value <= request.implicit_cut_root_tolerance;
        has_positive = has_positive || value >= -request.implicit_cut_root_tolerance;
    }

    if (!has_negative || !has_positive) {
        appendFullTriangleRegion(
            cut,
            request,
            input,
            tri,
            has_negative ? geometry::CutIntegrationSide::Negative
                         : geometry::CutIntegrationSide::Positive,
            parent_measure,
            min_signed,
            max_signed,
            diagnostics);
        return;
    }

    if (depth >= max_depth) {
        appendLinearizedTriangleCut(
            cut, request, input, tri, parent_measure, diagnostics);
        return;
    }

    ++diagnostics.subdivision_count;
    const auto ab = midpoint(tri.a, tri.b);
    const auto bc = midpoint(tri.b, tri.c);
    const auto ca = midpoint(tri.c, tri.a);
    const std::array<Triangle2D, 4> children{{
        Triangle2D{tri.a, ab, ca},
        Triangle2D{ab, tri.b, bc},
        Triangle2D{ca, bc, tri.c},
        Triangle2D{ab, bc, ca},
    }};
    for (const auto& child : children) {
        appendAdaptiveTriangleCut(
            cut,
            request,
            input,
            child,
            parent_measure,
            depth + 1,
            max_depth,
            diagnostics);
    }
}

[[nodiscard]] std::string formatSayeHyperrectangleDiagnostics(
    const SayeHyperrectangleDiagnostics& diagnostics,
    int max_depth_limit)
{
    return "SayeHyperrectangle recursive 2D hyperrectangle quadrature"
           "; max_depth_limit=" + std::to_string(max_depth_limit) +
           "; max_depth_reached=" +
           std::to_string(diagnostics.max_depth_reached) +
           "; subdivisions=" + std::to_string(diagnostics.subdivision_count) +
           "; linearized_leaves=" +
           std::to_string(diagnostics.linearized_leaf_count) +
           "; full_negative_regions=" +
           std::to_string(diagnostics.full_negative_region_count) +
           "; full_positive_regions=" +
           std::to_string(diagnostics.full_positive_region_count) +
           "; interface_fragments=" +
           std::to_string(diagnostics.interface_fragment_count);
}

[[nodiscard]] std::string formatHighOrderSubcellDiagnostics(
    const SayeHyperrectangleDiagnostics& diagnostics,
    int max_depth_limit)
{
    return "HighOrderSubcell recursive 2D triangle quadrature"
           "; max_depth_limit=" + std::to_string(max_depth_limit) +
           "; max_depth_reached=" +
           std::to_string(diagnostics.max_depth_reached) +
           "; subdivisions=" + std::to_string(diagnostics.subdivision_count) +
           "; linearized_leaves=" +
           std::to_string(diagnostics.linearized_leaf_count) +
           "; full_negative_regions=" +
           std::to_string(diagnostics.full_negative_region_count) +
           "; full_positive_regions=" +
           std::to_string(diagnostics.full_positive_region_count) +
           "; interface_fragments=" +
           std::to_string(diagnostics.interface_fragment_count);
}

class SayeHyperrectangleImplicitCutBackend final
    : public ImplicitCutQuadratureBackendDriver {
public:
    [[nodiscard]] ImplicitCutQuadratureBackend kind() const noexcept override {
        return ImplicitCutQuadratureBackend::SayeHyperrectangle;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return implicitCutQuadratureBackendName(kind());
    }

    [[nodiscard]] bool supports(int mesh_dimension,
                                ElementType element_type) const noexcept override
    {
        return supportsSayeHyperrectangleMilestone(mesh_dimension, element_type);
    }

    [[nodiscard]] int achievedInterfaceQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return std::min(1, std::max(0, request.resolvedInterfaceQuadratureOrder()));
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return interfaces::implementedLevelSetCutVolumeExactOrder(
            std::max(0, request.resolvedVolumeQuadratureOrder()));
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        ImplicitCutQuadratureBackendCellResult result;
        result.achieved_interface_quadrature_order =
            achievedInterfaceQuadratureOrder(request);
        result.achieved_volume_quadrature_order =
            achievedVolumeQuadratureOrder(request);

        if (!supports(mesh_dimension, input.linearized_input.element_type)) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "SayeHyperrectangle implicit cut quadrature backend supports only quadrilateral cells in two dimensions";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            return result;
        }
        if (input.evaluator == nullptr) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "SayeHyperrectangle implicit cut quadrature backend requires a level-set evaluator";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            return result;
        }
        if (input.evaluator->interpolationOrder(
                input.linearized_input.parent_cell) <= 1) {
            result.cut =
                interfaces::cutLinearLevelSetCell2D(
                    request, input.linearized_input);
            result.diagnostic_status =
                classifyCutStatus(result.cut, result.fallback_used);
            return result;
        }

        const Rectangle2D root{
            input.reference_min[0],
            input.reference_max[0],
            input.reference_min[1],
            input.reference_max[1]};
        const Real parent_measure = rectangleMeasure(root);
        const int max_depth =
            std::max(0, std::min(request.implicit_cut_max_subdivision_depth, 8));
        SayeHyperrectangleDiagnostics diagnostics;
        appendAdaptiveRectangleCut(
            result.cut,
            request,
            input,
            root,
            parent_measure,
            0,
            max_depth,
            diagnostics);
        result.cut.supported = true;
        result.cut.diagnostic =
            formatSayeHyperrectangleDiagnostics(diagnostics, max_depth);
        result.diagnostic_status =
            classifyCutStatus(result.cut, result.fallback_used);
        return result;
    }
};

class HighOrderSubcellImplicitCutBackend final
    : public ImplicitCutQuadratureBackendDriver {
public:
    [[nodiscard]] ImplicitCutQuadratureBackend kind() const noexcept override {
        return ImplicitCutQuadratureBackend::HighOrderSubcell;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return implicitCutQuadratureBackendName(kind());
    }

    [[nodiscard]] bool supports(int mesh_dimension,
                                ElementType element_type) const noexcept override
    {
        return supportsHighOrderSubcellTriangleMilestone(mesh_dimension, element_type);
    }

    [[nodiscard]] int achievedInterfaceQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return std::min(1, std::max(0, request.resolvedInterfaceQuadratureOrder()));
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return interfaces::implementedLevelSetCutVolumeExactOrder(
            std::max(0, request.resolvedVolumeQuadratureOrder()));
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        ImplicitCutQuadratureBackendCellResult result;
        result.achieved_interface_quadrature_order =
            achievedInterfaceQuadratureOrder(request);
        result.achieved_volume_quadrature_order =
            achievedVolumeQuadratureOrder(request);

        if (!supports(mesh_dimension, input.linearized_input.element_type)) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "HighOrderSubcell implicit cut quadrature backend supports only triangular cells in two dimensions";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            return result;
        }
        if (input.evaluator == nullptr) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "HighOrderSubcell implicit cut quadrature backend requires a level-set evaluator";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            return result;
        }
        if (input.evaluator->interpolationOrder(
                input.linearized_input.parent_cell) <= 1) {
            result.cut =
                interfaces::cutLinearLevelSetCell2D(
                    request, input.linearized_input);
            result.diagnostic_status =
                classifyCutStatus(result.cut, result.fallback_used);
            return result;
        }
        if (input.linearized_input.node_coordinates.size() < 3u) {
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "HighOrderSubcell implicit cut quadrature backend requires triangle corner coordinates";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            return result;
        }

        const Triangle2D root{
            input.linearized_input.node_coordinates[0],
            input.linearized_input.node_coordinates[1],
            input.linearized_input.node_coordinates[2]};
        const Real parent_measure = triangleMeasure(root);
        const int max_depth =
            std::max(0, std::min(request.implicit_cut_max_subdivision_depth, 8));
        SayeHyperrectangleDiagnostics diagnostics;
        appendAdaptiveTriangleCut(
            result.cut,
            request,
            input,
            root,
            parent_measure,
            0,
            max_depth,
            diagnostics);
        result.cut.supported = true;
        result.cut.diagnostic =
            formatHighOrderSubcellDiagnostics(diagnostics, max_depth);
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

[[nodiscard]] bool supportsHighOrderSubcellTriangleMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept
{
    if (mesh_dimension != 2) {
        return false;
    }
    switch (element_type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
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
    static const SayeHyperrectangleImplicitCutBackend saye_hyperrectangle_backend;
    static const HighOrderSubcellImplicitCutBackend high_order_subcell_backend;

    switch (backend) {
    case ImplicitCutQuadratureBackend::LinearCorner:
        return linear_corner_backend;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        return saye_hyperrectangle_backend;
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
        return high_order_subcell_backend;
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
        capability.implemented = true;
        capability.supports_element_type =
            supportsSayeHyperrectangleMilestone(mesh_dimension, element_type);
        capability.supports_high_order_geometry = true;
        capability.maximum_reported_interface_order = 1;
        capability.maximum_reported_volume_order = 2;
        return capability;
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
        capability.implemented = true;
        capability.supports_element_type =
            supportsHighOrderSubcellTriangleMilestone(mesh_dimension, element_type);
        capability.supports_high_order_geometry = true;
        capability.maximum_reported_interface_order = 1;
        capability.maximum_reported_volume_order = 2;
        return capability;
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
    const ImplicitCutQuadratureBackendCellInput& input,
    const ImplicitCutQuadratureBackendCellResult& result)
{
    const auto& linearized_input = input.linearized_input;
    if (input.evaluator == nullptr) {
        return failedValidation(
            ImplicitCutQuadratureDiagnosticStatus::Failed,
            "implicit cut backend input is missing a level-set evaluator");
    }
    if (!finiteArray(input.reference_min) ||
        !finiteArray(input.reference_max) ||
        input.reference_min[0] > input.reference_max[0] ||
        input.reference_min[1] > input.reference_max[1] ||
        input.reference_min[2] > input.reference_max[2]) {
        return failedValidation(
            ImplicitCutQuadratureDiagnosticStatus::Failed,
            "implicit cut backend input has invalid reference bounds");
    }
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
        if (fragment.parent_cell != linearized_input.parent_cell) {
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
        if (region.parent_cell != linearized_input.parent_cell) {
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
