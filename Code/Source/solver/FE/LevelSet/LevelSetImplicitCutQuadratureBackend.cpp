#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
           std::isfinite(point.weight) &&
           std::isfinite(point.parent_coordinate[0]) &&
           std::isfinite(point.parent_coordinate[1]) &&
           std::isfinite(point.parent_coordinate[2]) &&
           std::isfinite(point.reference_measure_factor) &&
           std::isfinite(point.level_set_residual) &&
           std::isfinite(point.gradient_norm);
}

[[nodiscard]] bool finiteArray(const std::array<Real, 3>& values) noexcept
{
    return std::isfinite(values[0]) &&
           std::isfinite(values[1]) &&
           std::isfinite(values[2]);
}

[[nodiscard]] Real norm3(const std::array<Real, 3>& values) noexcept
{
    return std::sqrt(values[0] * values[0] +
                     values[1] * values[1] +
                     values[2] * values[2]);
}

[[nodiscard]] Real dot3(const std::array<Real, 3>& a,
                        const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real measureTolerance(Real request_tolerance,
                                    Real measure) noexcept
{
    const Real scale = std::max(Real{1.0}, std::abs(measure));
    const Real roundoff =
        Real{64.0} * std::numeric_limits<Real>::epsilon() * scale;
    return std::max(request_tolerance * scale, roundoff);
}

[[nodiscard]] std::string formatReal(Real value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

[[nodiscard]] Real rootResidualTolerance(
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    return std::max(request.implicit_cut_root_tolerance, request.tolerance);
}

[[nodiscard]] Real rootCoordinateTolerance(
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    return std::max(request.implicit_cut_root_coordinate_tolerance,
                    Real{8.0} * std::numeric_limits<Real>::epsilon());
}

[[nodiscard]] Real rootParametricCoordinateTolerance(
    const interfaces::CutInterfaceDomainRequest& request,
    Real span) noexcept
{
    return std::max(Real{16.0} * std::numeric_limits<Real>::epsilon(),
                    rootCoordinateTolerance(request) /
                        std::max(span, Real{1.0}));
}

[[nodiscard]] Real rootUniquenessTolerance(
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    return Real{10.0} *
           std::max({request.tolerance,
                     request.implicit_cut_root_tolerance,
                     request.implicit_cut_root_coordinate_tolerance});
}

[[nodiscard]] int rootMaxIterations(
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    return std::max(1, request.implicit_cut_root_max_iterations);
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

[[nodiscard]] bool supportsHighOrderSubcellMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept;

[[nodiscard]] bool selectAutoImplicitCutBackend(
    int mesh_dimension,
    ElementType element_type,
    ImplicitCutQuadratureBackend& selected) noexcept;

void appendDetailedBackendDiagnostics(
    ImplicitCutQuadratureBackendCellResult& result,
    const interfaces::CutInterfaceDomainRequest& request);

void setOrderMetadata(ImplicitCutQuadratureBackendCellResult& result,
                      const interfaces::CutInterfaceDomainRequest& request,
                      int possible_interface_order,
                      int possible_volume_order,
                      int achieved_interface_order,
                      int achieved_volume_order,
                      int verified_interface_order,
                      int verified_volume_order) noexcept
{
    result.requested_interface_quadrature_order =
        request.resolvedInterfaceQuadratureOrder();
    result.requested_volume_quadrature_order =
        request.resolvedVolumeQuadratureOrder();
    result.possible_interface_quadrature_order = possible_interface_order;
    result.possible_volume_quadrature_order = possible_volume_order;
    result.achieved_interface_quadrature_order = achieved_interface_order;
    result.achieved_volume_quadrature_order = achieved_volume_order;
    result.verified_interface_quadrature_order = verified_interface_order;
    result.verified_volume_quadrature_order = verified_volume_order;
}

void setUnavailableOrderMetadata(
    ImplicitCutQuadratureBackendCellResult& result,
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    setOrderMetadata(result, request,
                     /*possible_interface_order=*/-1,
                     /*possible_volume_order=*/-1,
                     /*achieved_interface_order=*/-1,
                     /*achieved_volume_order=*/-1,
                     /*verified_interface_order=*/-1,
                     /*verified_volume_order=*/-1);
}

[[nodiscard]] int capabilityLimitedOrder(
    ImplicitCutQuadratureBackend backend,
    int mesh_dimension,
    ElementType element_type,
    int requested_order,
    bool interface_order) noexcept
{
    if (requested_order < 0) {
        return -1;
    }
    const auto capability =
        implicitCutQuadratureBackendCapability(
            backend, mesh_dimension, element_type);
    if (!capability.implemented || !capability.supports_element_type) {
        return -1;
    }
    const int maximum_order = interface_order
                                  ? capability.maximum_reported_interface_order
                                  : capability.maximum_reported_volume_order;
    if (maximum_order < 0) {
        return -1;
    }
    return std::min(requested_order, maximum_order);
}

[[nodiscard]] std::size_t activeVolumeQuadraturePointCount(
    const interfaces::LevelSetCellCutResult& cut) noexcept
{
    std::size_t count = 0u;
    for (const auto& region : cut.volume_regions) {
        if (!region.active()) {
            continue;
        }
        count += region.quadrature_points.empty()
                     ? 1u
                     : region.quadrature_points.size();
    }
    return count;
}

[[nodiscard]] std::size_t activeInterfaceQuadraturePointCount(
    const interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    std::size_t count = 0u;
    for (const auto& fragment : cut.fragments) {
        if (!fragment.active()) {
            continue;
        }
        count += fragment.quadraturePointCount(request);
    }
    return count;
}

void appendBackendRuntimeDiagnostics(
    ImplicitCutQuadratureBackendCellResult& result)
{
    if (result.cut.diagnostic.empty() ||
        result.cut.diagnostic.find("volume_quadrature_points=") !=
            std::string::npos) {
        return;
    }
    result.cut.diagnostic +=
        "; volume_quadrature_points=" +
        std::to_string(result.volume_quadrature_point_count) +
        "; interface_quadrature_points=" +
        std::to_string(result.interface_quadrature_point_count) +
        "; backend_elapsed_seconds=" +
        std::to_string(result.backend_elapsed_seconds);
}

void appendRootPolishDiagnostics(std::string& diagnostic,
                                 int root_finder_iteration_count)
{
    if (root_finder_iteration_count > 0) {
        diagnostic +=
            "; root_finder_iterations=" +
            std::to_string(root_finder_iteration_count);
        return;
    }
    diagnostic += "; root_polishing=not_performed";
}

[[nodiscard]] ImplicitCutQuadratureBackendCellResult finalizeBackendResult(
    ImplicitCutQuadratureBackendCellResult result,
    const interfaces::CutInterfaceDomainRequest& request,
    std::chrono::steady_clock::time_point backend_start)
{
    auto counting_request = request;
    counting_request.achieved_interface_quadrature_order =
        result.achieved_interface_quadrature_order;
    counting_request.achieved_volume_quadrature_order =
        result.achieved_volume_quadrature_order;
    result.volume_quadrature_point_count =
        activeVolumeQuadraturePointCount(result.cut);
    result.interface_quadrature_point_count =
        activeInterfaceQuadraturePointCount(result.cut, counting_request);
    const auto elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - backend_start)
            .count();
    result.backend_elapsed_seconds = std::max(0.0, elapsed);
    appendBackendRuntimeDiagnostics(result);
    return result;
}

void reduceOrderMetadataToGeneratedRules(
    ImplicitCutQuadratureBackendCellResult& result,
    const interfaces::CutInterfaceDomainRequest& request)
{
    if (!result.cut.supported ||
        result.achieved_interface_quadrature_order < 0 ||
        result.achieved_volume_quadrature_order < 0) {
        return;
    }

    auto rule_request = request;
    rule_request.achieved_interface_quadrature_order =
        result.achieved_interface_quadrature_order;
    rule_request.achieved_volume_quadrature_order =
        result.achieved_volume_quadrature_order;

    int generated_interface_order =
        result.achieved_interface_quadrature_order;
    bool saw_active_interface_rule = false;
    for (const auto& fragment : result.cut.fragments) {
        if (!fragment.active()) {
            continue;
        }
        const auto rule = fragment.toCutQuadratureRule(rule_request);
        generated_interface_order =
            std::min(generated_interface_order,
                     rule.provenance.achieved_quadrature_order);
        saw_active_interface_rule = true;
    }
    if (saw_active_interface_rule) {
        result.achieved_interface_quadrature_order =
            generated_interface_order;
        result.verified_interface_quadrature_order =
            std::min(result.verified_interface_quadrature_order,
                     generated_interface_order);
        rule_request.achieved_interface_quadrature_order =
            generated_interface_order;
    }

    int generated_volume_order =
        result.achieved_volume_quadrature_order;
    bool saw_active_volume_rule = false;
    for (const auto& region : result.cut.volume_regions) {
        if (!region.active()) {
            continue;
        }
        const auto rule = region.toCutQuadratureRule(rule_request);
        generated_volume_order =
            std::min(generated_volume_order,
                     rule.provenance.achieved_quadrature_order);
        saw_active_volume_rule = true;
    }
    if (saw_active_volume_rule) {
        result.achieved_volume_quadrature_order = generated_volume_order;
        result.verified_volume_quadrature_order =
            std::min(result.verified_volume_quadrature_order,
                     generated_volume_order);
    }
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
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedInterfaceQuadratureOrder()),
                                      /*interface_order=*/true);
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedVolumeQuadratureOrder()),
                                      /*interface_order=*/false);
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        const auto backend_start = std::chrono::steady_clock::now();
        ImplicitCutQuadratureBackendCellResult result{};
        result.selected_backend = kind();
        const int possible_interface_order =
            achievedInterfaceQuadratureOrder(
                mesh_dimension,
                input.linearized_input.element_type,
                request);
        const int possible_volume_order =
            achievedVolumeQuadratureOrder(
                mesh_dimension,
                input.linearized_input.element_type,
                request);
        setOrderMetadata(result, request,
                         possible_interface_order,
                         possible_volume_order,
                         possible_interface_order,
                         possible_volume_order,
                         possible_interface_order,
                         possible_volume_order);

        if (!supports(mesh_dimension, input.linearized_input.element_type)) {
            setUnavailableOrderMetadata(result, request);
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
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
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
        reduceOrderMetadataToGeneratedRules(result, request);
        result.diagnostic_status =
            classifyCutStatus(result.cut, result.fallback_used);
        return finalizeBackendResult(std::move(result), request, backend_start);
    }
};

struct Rectangle2D {
    Real xmin{0.0};
    Real xmax{0.0};
    Real ymin{0.0};
    Real ymax{0.0};
};

struct Box3D {
    Real xmin{0.0};
    Real xmax{0.0};
    Real ymin{0.0};
    Real ymax{0.0};
    Real zmin{0.0};
    Real zmax{0.0};
};

struct Triangle2D {
    std::array<Real, 3> a{{0.0, 0.0, 0.0}};
    std::array<Real, 3> b{{0.0, 0.0, 0.0}};
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
};

struct Tetrahedron3D {
    std::array<Real, 3> a{{0.0, 0.0, 0.0}};
    std::array<Real, 3> b{{0.0, 0.0, 0.0}};
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    std::array<Real, 3> d{{0.0, 0.0, 0.0}};
};

struct SayeHyperrectangleDiagnostics {
    int max_depth_reached{0};
    int subdivision_count{0};
    int root_branch_count{0};
    int root_finder_iteration_count{0};
    int curved_fragment_count{0};
    int full_negative_region_count{0};
    int full_positive_region_count{0};
    int linearized_leaf_count{0};
    int interface_fragment_count{0};
    int curved_fragment_failure_count{0};
    int curved_fragment_vertex_count_failure{0};
    int curved_fragment_seed_failure{0};
    int curved_fragment_search_segment_failure{0};
    int curved_fragment_root_solve_failure{0};
    int curved_fragment_gradient_failure{0};
    int curved_fragment_weight_failure{0};
    int curved_fragment_edge_root_mismatch_count{0};
    int curved_fragment_root_solve_edge_root_mismatch{0};
    int curved_fragment_boundary_degenerate_count{0};
};

[[nodiscard]] Real rectangleMeasure(const Rectangle2D& rect) noexcept
{
    return std::max(Real{0.0}, rect.xmax - rect.xmin) *
           std::max(Real{0.0}, rect.ymax - rect.ymin);
}

[[nodiscard]] Real boxMeasure(const Box3D& box) noexcept
{
    return std::max(Real{0.0}, box.xmax - box.xmin) *
           std::max(Real{0.0}, box.ymax - box.ymin) *
           std::max(Real{0.0}, box.zmax - box.zmin);
}

[[nodiscard]] Real triangleMeasure(const Triangle2D& tri) noexcept
{
    const Real x0 = tri.b[0] - tri.a[0];
    const Real y0 = tri.b[1] - tri.a[1];
    const Real x1 = tri.c[0] - tri.a[0];
    const Real y1 = tri.c[1] - tri.a[1];
    return Real{0.5} * std::abs(x0 * y1 - y0 * x1);
}

[[nodiscard]] std::array<Real, 3> subtract(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<Real, 3> cross(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real tetrahedronMeasure(const Tetrahedron3D& tet) noexcept
{
    return std::abs(dot(subtract(tet.b, tet.a),
                        cross(subtract(tet.c, tet.a),
                              subtract(tet.d, tet.a)))) /
           Real{6.0};
}

[[nodiscard]] std::array<Real, 3> rectangleCentroid(
    const Rectangle2D& rect) noexcept
{
    return {{Real{0.5} * (rect.xmin + rect.xmax),
             Real{0.5} * (rect.ymin + rect.ymax),
             0.0}};
}

[[nodiscard]] std::array<Real, 3> boxCentroid(const Box3D& box) noexcept
{
    return {{Real{0.5} * (box.xmin + box.xmax),
             Real{0.5} * (box.ymin + box.ymax),
             Real{0.5} * (box.zmin + box.zmax)}};
}

[[nodiscard]] std::array<Real, 3> triangleCentroid(
    const Triangle2D& tri) noexcept
{
    return {{(tri.a[0] + tri.b[0] + tri.c[0]) / Real{3.0},
             (tri.a[1] + tri.b[1] + tri.c[1]) / Real{3.0},
             (tri.a[2] + tri.b[2] + tri.c[2]) / Real{3.0}}};
}

[[nodiscard]] std::array<Real, 3> tetrahedronCentroid(
    const Tetrahedron3D& tet) noexcept
{
    return {{(tet.a[0] + tet.b[0] + tet.c[0] + tet.d[0]) / Real{4.0},
             (tet.a[1] + tet.b[1] + tet.c[1] + tet.d[1]) / Real{4.0},
             (tet.a[2] + tet.b[2] + tet.c[2] + tet.d[2]) / Real{4.0}}};
}

[[nodiscard]] std::array<Real, 3> midpoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b) noexcept
{
    return {{Real{0.5} * (a[0] + b[0]),
             Real{0.5} * (a[1] + b[1]),
             Real{0.5} * (a[2] + b[2])}};
}

[[nodiscard]] Real distance2D(const std::array<Real, 3>& a,
                              const std::array<Real, 3>& b) noexcept
{
    const Real dx = a[0] - b[0];
    const Real dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy);
}

[[nodiscard]] Real distance3D(const std::array<Real, 3>& a,
                              const std::array<Real, 3>& b) noexcept
{
    return norm3(subtract(a, b));
}

[[nodiscard]] Real cross2D(const std::array<Real, 3>& a,
                           const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[1] - a[1] * b[0];
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

[[nodiscard]] bool polishRootOnSegment(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    std::array<Real, 3>& root,
    int& iterations)
{
    const Real tolerance = rootResidualTolerance(request);
    const Real coordinate_tolerance = rootCoordinateTolerance(request);
    Real fa = signedLevelSetValue(input, a);
    Real fb = signedLevelSetValue(input, b);
    if (!std::isfinite(fa) || !std::isfinite(fb)) {
        return false;
    }
    if (std::abs(fa) <= tolerance) {
        root = a;
        return true;
    }
    if (std::abs(fb) <= tolerance) {
        root = b;
        return true;
    }
    if ((fa < Real{0.0}) == (fb < Real{0.0})) {
        return false;
    }

    std::array<Real, 3> lo = a;
    std::array<Real, 3> hi = b;
    Real flo = fa;
    Real fhi = fb;
    root = midpoint(lo, hi);
    const int max_iterations = rootMaxIterations(request);
    for (int iter = 0; iter < max_iterations; ++iter) {
        ++iterations;
        root = midpoint(lo, hi);
        const Real fm = signedLevelSetValue(input, root);
        if (!std::isfinite(fm)) {
            return false;
        }
        if (std::abs(fm) <= tolerance ||
            distance3D(lo, hi) <= coordinate_tolerance) {
            return true;
        }
        if ((flo < Real{0.0}) == (fm < Real{0.0})) {
            lo = root;
            flo = fm;
        } else {
            hi = root;
            fhi = fm;
        }
        (void)fhi;
    }
    return std::abs(signedLevelSetValue(input, root)) <=
           Real{10.0} * tolerance;
}

void addUniqueRoot(std::vector<std::array<Real, 3>>& roots,
                   const std::array<Real, 3>& root,
                   Real tolerance)
{
    const auto existing =
        std::find_if(roots.begin(), roots.end(), [&](const auto& point) {
            return distance3D(point, root) <= tolerance;
        });
    if (existing == roots.end()) {
        roots.push_back(root);
    }
}

[[nodiscard]] std::vector<std::array<Real, 3>> rectangleEdgeRoots(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const Rectangle2D& rect,
    int& iterations)
{
    const std::array<std::array<Real, 3>, 4> corners{{
        {{rect.xmin, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymax, 0.0}},
        {{rect.xmin, rect.ymax, 0.0}},
    }};
    const std::array<std::array<std::size_t, 2>, 4> edges{{
        {{0u, 1u}},
        {{1u, 2u}},
        {{2u, 3u}},
        {{3u, 0u}},
    }};
    const Real uniqueness_tolerance = rootUniquenessTolerance(request);
    std::vector<std::array<Real, 3>> roots;
    roots.reserve(4u);
    for (const auto& edge : edges) {
        std::array<Real, 3> root;
        if (polishRootOnSegment(input,
                                request,
                                corners[edge[0]],
                                corners[edge[1]],
                                root,
                                iterations)) {
            addUniqueRoot(roots, root, uniqueness_tolerance);
        }
    }
    return roots;
}

[[nodiscard]] std::vector<std::array<Real, 3>> tetrahedronEdgeRoots(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const Tetrahedron3D& tet,
    int& iterations)
{
    const std::array<std::array<Real, 3>, 4> corners{{tet.a, tet.b, tet.c, tet.d}};
    const std::array<std::array<std::size_t, 2>, 6> edges{{
        {{0u, 1u}},
        {{0u, 2u}},
        {{0u, 3u}},
        {{1u, 2u}},
        {{1u, 3u}},
        {{2u, 3u}},
    }};
    const Real uniqueness_tolerance = rootUniquenessTolerance(request);
    std::vector<std::array<Real, 3>> roots;
    roots.reserve(4u);
    for (const auto& edge : edges) {
        std::array<Real, 3> root;
        if (polishRootOnSegment(input,
                                request,
                                corners[edge[0]],
                                corners[edge[1]],
                                root,
                                iterations)) {
            addUniqueRoot(roots, root, uniqueness_tolerance);
        }
    }
    return roots;
}

[[nodiscard]] bool solve3x3(std::array<std::array<Real, 3>, 3> matrix,
                            std::array<Real, 3> rhs,
                            std::array<Real, 3>& solution) noexcept
{
    for (std::size_t pivot = 0u; pivot < 3u; ++pivot) {
        std::size_t best = pivot;
        Real best_abs = std::abs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1u; row < 3u; ++row) {
            const Real value = std::abs(matrix[row][pivot]);
            if (value > best_abs) {
                best = row;
                best_abs = value;
            }
        }
        if (best_abs <= Real{1.0e-30}) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[best], matrix[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }
        const Real inv_pivot = Real{1.0} / matrix[pivot][pivot];
        for (std::size_t col = pivot; col < 3u; ++col) {
            matrix[pivot][col] *= inv_pivot;
        }
        rhs[pivot] *= inv_pivot;
        for (std::size_t row = 0u; row < 3u; ++row) {
            if (row == pivot) {
                continue;
            }
            const Real factor = matrix[row][pivot];
            for (std::size_t col = pivot; col < 3u; ++col) {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}

[[nodiscard]] bool tetrahedronBarycentricCoordinates(
    const Tetrahedron3D& tet,
    const std::array<Real, 3>& point,
    std::array<Real, 4>& lambda) noexcept
{
    const auto ba = subtract(tet.b, tet.a);
    const auto ca = subtract(tet.c, tet.a);
    const auto da = subtract(tet.d, tet.a);
    std::array<std::array<Real, 3>, 3> matrix{{
        {{ba[0], ca[0], da[0]}},
        {{ba[1], ca[1], da[1]}},
        {{ba[2], ca[2], da[2]}},
    }};
    std::array<Real, 3> local{{0.0, 0.0, 0.0}};
    if (!solve3x3(matrix, subtract(point, tet.a), local)) {
        return false;
    }
    lambda[1] = local[0];
    lambda[2] = local[1];
    lambda[3] = local[2];
    lambda[0] = Real{1.0} - lambda[1] - lambda[2] - lambda[3];
    return finiteArray({{lambda[0], lambda[1], lambda[2]}}) &&
           std::isfinite(lambda[3]);
}

[[nodiscard]] bool tetrahedronBarycentricDirection(
    const Tetrahedron3D& tet,
    const std::array<Real, 3>& direction,
    std::array<Real, 4>& lambda_direction) noexcept
{
    const auto ba = subtract(tet.b, tet.a);
    const auto ca = subtract(tet.c, tet.a);
    const auto da = subtract(tet.d, tet.a);
    std::array<std::array<Real, 3>, 3> matrix{{
        {{ba[0], ca[0], da[0]}},
        {{ba[1], ca[1], da[1]}},
        {{ba[2], ca[2], da[2]}},
    }};
    std::array<Real, 3> local{{0.0, 0.0, 0.0}};
    if (!solve3x3(matrix, direction, local)) {
        return false;
    }
    lambda_direction[1] = local[0];
    lambda_direction[2] = local[1];
    lambda_direction[3] = local[2];
    lambda_direction[0] =
        -lambda_direction[1] - lambda_direction[2] - lambda_direction[3];
    return finiteArray({{lambda_direction[0],
                         lambda_direction[1],
                         lambda_direction[2]}}) &&
           std::isfinite(lambda_direction[3]);
}

[[nodiscard]] bool lineTetrahedronSearchSegment(
    const Tetrahedron3D& tet,
    const std::array<Real, 3>& origin,
    const std::array<Real, 3>& direction,
    std::array<Real, 3>& start,
    std::array<Real, 3>& end,
    Real& guess_fraction)
{
    std::array<Real, 4> lambda0{{0.0, 0.0, 0.0, 0.0}};
    std::array<Real, 4> lambda_direction{{0.0, 0.0, 0.0, 0.0}};
    if (!tetrahedronBarycentricCoordinates(tet, origin, lambda0) ||
        !tetrahedronBarycentricDirection(tet, direction, lambda_direction)) {
        return false;
    }

    constexpr Real tolerance = Real{1.0e-12};
    Real t_min = -std::numeric_limits<Real>::infinity();
    Real t_max = std::numeric_limits<Real>::infinity();
    for (std::size_t i = 0u; i < 4u; ++i) {
        const Real slope = lambda_direction[i];
        if (std::abs(slope) <= tolerance) {
            if (lambda0[i] < -tolerance) {
                return false;
            }
            continue;
        }
        const Real bound = -lambda0[i] / slope;
        if (slope > Real{0.0}) {
            t_min = std::max(t_min, bound);
        } else {
            t_max = std::min(t_max, bound);
        }
    }
    if (!std::isfinite(t_min) || !std::isfinite(t_max) ||
        !(t_max - t_min > tolerance)) {
        return false;
    }
    start = {{origin[0] + t_min * direction[0],
              origin[1] + t_min * direction[1],
              origin[2] + t_min * direction[2]}};
    end = {{origin[0] + t_max * direction[0],
            origin[1] + t_max * direction[1],
            origin[2] + t_max * direction[2]}};
    guess_fraction =
        std::clamp(-t_min / (t_max - t_min), Real{0.0}, Real{1.0});
    return true;
}

[[nodiscard]] std::array<Real, 3> polygonCentroid(
    const std::vector<std::array<Real, 3>>& points) noexcept
{
    std::array<Real, 3> centroid{{0.0, 0.0, 0.0}};
    if (points.empty()) {
        return centroid;
    }
    for (const auto& point : points) {
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    const Real inv_count = Real{1.0} / static_cast<Real>(points.size());
    centroid[0] *= inv_count;
    centroid[1] *= inv_count;
    centroid[2] *= inv_count;
    return centroid;
}

[[nodiscard]] std::array<Real, 3> polygonNormalOrDefault(
    const std::vector<std::array<Real, 3>>& points,
    const std::array<Real, 3>& fallback) noexcept
{
    if (points.size() < 3u) {
        return normalizedOrDefault(fallback);
    }
    const auto centroid = polygonCentroid(points);
    std::array<Real, 3> accumulated{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0u; i < points.size(); ++i) {
        const auto a = subtract(points[i], centroid);
        const auto b = subtract(points[(i + 1u) % points.size()], centroid);
        const auto area_normal = cross(a, b);
        accumulated[0] += area_normal[0];
        accumulated[1] += area_normal[1];
        accumulated[2] += area_normal[2];
    }
    auto normal = normalizedOrDefault(accumulated);
    if (dot3(normal, fallback) < Real{0.0}) {
        normal = {{-normal[0], -normal[1], -normal[2]}};
    }
    return normal;
}

void appendTriangleSurfaceQuadratureSeeds(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& c,
    std::vector<std::pair<std::array<Real, 3>, Real>>& seeds)
{
    const Real area = Real{0.5} * norm3(cross(subtract(b, a), subtract(c, a)));
    if (!(area > Real{0.0}) || !std::isfinite(area)) {
        return;
    }
    constexpr Real high = Real{2.0} / Real{3.0};
    constexpr Real low = Real{1.0} / Real{6.0};
    const Real weight = area / Real{3.0};
    const auto point = [](const std::array<Real, 3>& v0,
                          const std::array<Real, 3>& v1,
                          const std::array<Real, 3>& v2,
                          Real w0,
                          Real w1,
                          Real w2) {
        return std::array<Real, 3>{{
            w0 * v0[0] + w1 * v1[0] + w2 * v2[0],
            w0 * v0[1] + w1 * v1[1] + w2 * v2[1],
            w0 * v0[2] + w1 * v1[2] + w2 * v2[2],
        }};
    };
    seeds.push_back({point(a, b, c, high, low, low), weight});
    seeds.push_back({point(a, b, c, low, high, low), weight});
    seeds.push_back({point(a, b, c, low, low, high), weight});
}

[[nodiscard]] bool matchPolishedRootsToBaseVertices(
    const std::vector<std::array<Real, 3>>& roots,
    const std::vector<std::array<Real, 3>>& base_vertices,
    std::vector<std::array<Real, 3>>& matched_roots)
{
    if (roots.size() != base_vertices.size()) {
        return false;
    }
    matched_roots.clear();
    matched_roots.reserve(base_vertices.size());
    std::vector<bool> used(roots.size(), false);
    for (const auto& base_vertex : base_vertices) {
        std::size_t best = roots.size();
        Real best_distance = std::numeric_limits<Real>::infinity();
        for (std::size_t i = 0u; i < roots.size(); ++i) {
            if (used[i]) {
                continue;
            }
            const Real candidate_distance = distance3D(base_vertex, roots[i]);
            if (candidate_distance < best_distance) {
                best = i;
                best_distance = candidate_distance;
            }
        }
        if (best == roots.size()) {
            return false;
        }
        used[best] = true;
        matched_roots.push_back(roots[best]);
    }
    return true;
}

[[nodiscard]] std::vector<std::array<Real, 3>> triangleEdgeRoots(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const Triangle2D& tri,
    int& iterations)
{
    const std::array<std::array<Real, 3>, 3> corners{{tri.a, tri.b, tri.c}};
    const std::array<std::array<std::size_t, 2>, 3> edges{{
        {{0u, 1u}},
        {{1u, 2u}},
        {{2u, 0u}},
    }};
    const Real uniqueness_tolerance = rootUniquenessTolerance(request);
    std::vector<std::array<Real, 3>> roots;
    roots.reserve(3u);
    for (const auto& edge : edges) {
        std::array<Real, 3> root;
        if (polishRootOnSegment(input,
                                request,
                                corners[edge[0]],
                                corners[edge[1]],
                                root,
                                iterations)) {
            addUniqueRoot(roots, root, uniqueness_tolerance);
        }
    }
    return roots;
}

void addUniqueParameter(std::vector<Real>& parameters,
                        Real parameter,
                        Real tolerance)
{
    const auto existing =
        std::find_if(parameters.begin(), parameters.end(), [&](Real value) {
            return std::abs(value - parameter) <= tolerance;
        });
    if (existing == parameters.end()) {
        parameters.push_back(parameter);
    }
}

[[nodiscard]] bool lineTriangleSearchSegment(
    const Triangle2D& tri,
    const std::array<Real, 3>& origin,
    const std::array<Real, 3>& direction,
    std::array<Real, 3>& start,
    std::array<Real, 3>& end,
    Real& guess_fraction)
{
    const std::array<std::array<Real, 3>, 3> vertices{{tri.a, tri.b, tri.c}};
    const std::array<std::array<std::size_t, 2>, 3> edges{{
        {{0u, 1u}},
        {{1u, 2u}},
        {{2u, 0u}},
    }};
    constexpr Real tolerance = Real{1.0e-12};
    std::vector<Real> parameters;
    parameters.reserve(3u);
    for (const auto& edge : edges) {
        const auto& a = vertices[edge[0]];
        const auto& b = vertices[edge[1]];
        const auto edge_vector = subtract(b, a);
        const Real denominator = cross2D(direction, edge_vector);
        if (std::abs(denominator) <= tolerance) {
            continue;
        }
        const auto a_minus_origin = subtract(a, origin);
        const Real line_parameter =
            cross2D(a_minus_origin, edge_vector) / denominator;
        const Real edge_parameter =
            cross2D(a_minus_origin, direction) / denominator;
        if (edge_parameter >= -tolerance &&
            edge_parameter <= Real{1.0} + tolerance) {
            addUniqueParameter(parameters, line_parameter, tolerance);
        }
    }
    if (parameters.size() < 2u) {
        return false;
    }
    std::sort(parameters.begin(), parameters.end());
    const Real min_parameter = parameters.front();
    const Real max_parameter = parameters.back();
    const Real span = max_parameter - min_parameter;
    if (!(span > tolerance) || !std::isfinite(span)) {
        return false;
    }
    start = {{origin[0] + min_parameter * direction[0],
              origin[1] + min_parameter * direction[1],
              origin[2] + min_parameter * direction[2]}};
    end = {{origin[0] + max_parameter * direction[0],
            origin[1] + max_parameter * direction[1],
            origin[2] + max_parameter * direction[2]}};
    guess_fraction = std::clamp(-min_parameter / span, Real{0.0}, Real{1.0});
    return true;
}

[[nodiscard]] std::array<Real, 3> pointOnSegment(
    const std::array<Real, 3>& start,
    const std::array<Real, 3>& end,
    Real t) noexcept
{
    return {{(Real{1.0} - t) * start[0] + t * end[0],
             (Real{1.0} - t) * start[1] + t * end[1],
             (Real{1.0} - t) * start[2] + t * end[2]}};
}

[[nodiscard]] bool newtonPolishRootAlongSegmentNearGuess(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const std::array<Real, 3>& start,
    const std::array<Real, 3>& end,
    Real guess_fraction,
    std::array<Real, 3>& root,
    int& iterations)
{
    const auto direction = subtract(end, start);
    const Real span = norm3(direction);
    if (!(span > Real{1.0e-30}) || !std::isfinite(span)) {
        return false;
    }

    const Real tolerance = rootResidualTolerance(request);
    const Real step_tolerance =
        rootParametricCoordinateTolerance(request, span);
    Real t = std::clamp(guess_fraction, Real{0.0}, Real{1.0});

    const int max_iterations = rootMaxIterations(request);
    for (int iter = 0; iter < max_iterations; ++iter) {
        ++iterations;
        const auto point = pointOnSegment(start, end, t);
        const auto evaluation =
            input.evaluator->evaluate(input.linearized_input.parent_cell, point);
        const Real value = evaluation.value - input.isovalue;
        if (!std::isfinite(value) || !finiteArray(evaluation.reference_gradient)) {
            return false;
        }
        if (std::abs(value) <= tolerance) {
            root = point;
            return true;
        }
        const Real derivative = dot3(evaluation.reference_gradient, direction);
        const Real derivative_scale =
            std::max(Real{1.0}, norm3(evaluation.reference_gradient) * span);
        if (std::abs(derivative) <=
            Real{64.0} * std::numeric_limits<Real>::epsilon() *
                derivative_scale) {
            return false;
        }
        const Real candidate_t = t - value / derivative;
        if (!std::isfinite(candidate_t)) {
            return false;
        }
        const Real next_t = std::clamp(candidate_t, Real{0.0}, Real{1.0});
        if (std::abs(next_t - t) <= step_tolerance) {
            const auto candidate_point = pointOnSegment(start, end, next_t);
            const Real candidate_value =
                signedLevelSetValue(input, candidate_point);
            if (std::isfinite(candidate_value) &&
                std::abs(candidate_value) <= Real{10.0} * tolerance) {
                root = candidate_point;
                return true;
            }
            return false;
        }
        t = next_t;
    }

    const auto candidate_point = pointOnSegment(start, end, t);
    const Real value = signedLevelSetValue(input, candidate_point);
    if (std::isfinite(value) && std::abs(value) <= Real{10.0} * tolerance) {
        root = candidate_point;
        return true;
    }
    return false;
}

[[nodiscard]] bool pointInsideTetrahedron(
    const Tetrahedron3D& tet,
    const std::array<Real, 3>& point,
    Real tolerance) noexcept
{
    std::array<Real, 4> lambda{{0.0, 0.0, 0.0, 0.0}};
    if (!tetrahedronBarycentricCoordinates(tet, point, lambda)) {
        return false;
    }
    for (Real value : lambda) {
        if (value < -tolerance || value > Real{1.0} + tolerance) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool newtonProjectRootInsideTetrahedron(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const Tetrahedron3D& tet,
    const std::array<Real, 3>& seed,
    std::array<Real, 3>& root,
    int& iterations)
{
    const Real tolerance = rootResidualTolerance(request);
    const Real containment_tolerance =
        std::max(rootCoordinateTolerance(request),
                 Real{100.0} * request.tolerance);
    std::array<Real, 3> point = seed;
    if (!pointInsideTetrahedron(tet, point, containment_tolerance)) {
        return false;
    }

    const int max_iterations = rootMaxIterations(request);
    for (int iter = 0; iter < max_iterations; ++iter) {
        ++iterations;
        const auto evaluation =
            input.evaluator->evaluate(input.linearized_input.parent_cell, point);
        const Real value = evaluation.value - input.isovalue;
        if (!std::isfinite(value) || !finiteArray(evaluation.reference_gradient)) {
            return false;
        }
        if (std::abs(value) <= tolerance) {
            root = point;
            return true;
        }

        const Real gradient_norm_sq =
            dot3(evaluation.reference_gradient, evaluation.reference_gradient);
        if (!(gradient_norm_sq > Real{1.0e-28}) ||
            !std::isfinite(gradient_norm_sq)) {
            return false;
        }
        const std::array<Real, 3> full_step{{
            -value * evaluation.reference_gradient[0] / gradient_norm_sq,
            -value * evaluation.reference_gradient[1] / gradient_norm_sq,
            -value * evaluation.reference_gradient[2] / gradient_norm_sq,
        }};

        bool advanced = false;
        Real damping = Real{1.0};
        for (int backtrack = 0; backtrack < 8; ++backtrack) {
            const std::array<Real, 3> candidate{{
                point[0] + damping * full_step[0],
                point[1] + damping * full_step[1],
                point[2] + damping * full_step[2],
            }};
            if (!pointInsideTetrahedron(
                    tet, candidate, containment_tolerance)) {
                damping *= Real{0.5};
                continue;
            }
            const Real candidate_value =
                signedLevelSetValue(input, candidate);
            if (!std::isfinite(candidate_value)) {
                damping *= Real{0.5};
                continue;
            }
            if (std::abs(candidate_value) < std::abs(value) ||
                std::abs(candidate_value) <= Real{10.0} * tolerance) {
                point = candidate;
                advanced = true;
                break;
            }
            damping *= Real{0.5};
        }
        if (!advanced) {
            return false;
        }
    }

    const Real value = signedLevelSetValue(input, point);
    if (std::isfinite(value) && std::abs(value) <= Real{10.0} * tolerance) {
        root = point;
        return true;
    }
    return false;
}

[[nodiscard]] bool solveRootAlongSegmentNearGuess(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const std::array<Real, 3>& start,
    const std::array<Real, 3>& end,
    Real guess_fraction,
    std::array<Real, 3>& root,
    int& iterations)
{
    if (newtonPolishRootAlongSegmentNearGuess(input,
                                              request,
                                              start,
                                              end,
                                              guess_fraction,
                                              root,
                                              iterations)) {
        return true;
    }

    constexpr int sample_count = 16;
    const Real tolerance = rootResidualTolerance(request);
    bool have_bracket = false;
    Real best_distance = std::numeric_limits<Real>::infinity();
    std::array<Real, 3> best_a = start;
    std::array<Real, 3> best_b = end;
    std::array<Real, 3> previous = start;
    Real f_previous = signedLevelSetValue(input, previous);
    if (!std::isfinite(f_previous)) {
        return false;
    }
    if (std::abs(f_previous) <= tolerance) {
        root = previous;
        return true;
    }
    for (int i = 1; i <= sample_count; ++i) {
        const Real t =
            static_cast<Real>(i) / static_cast<Real>(sample_count);
        const auto current = pointOnSegment(start, end, t);
        const Real f_current = signedLevelSetValue(input, current);
        if (!std::isfinite(f_current)) {
            return false;
        }
        if (std::abs(f_current) <= tolerance) {
            root = current;
            return true;
        }
        if ((f_previous < Real{0.0}) != (f_current < Real{0.0})) {
            const Real midpoint_t =
                (static_cast<Real>(i) - Real{0.5}) /
                static_cast<Real>(sample_count);
            const Real distance_to_guess = std::abs(midpoint_t - guess_fraction);
            if (distance_to_guess < best_distance) {
                best_distance = distance_to_guess;
                best_a = previous;
                best_b = current;
                have_bracket = true;
            }
        }
        previous = current;
        f_previous = f_current;
    }
    if (!have_bracket) {
        return false;
    }
    return polishRootOnSegment(input, request, best_a, best_b, root, iterations);
}

[[nodiscard]] std::vector<std::pair<Real, Real>> gaussLegendreUnitRule(
    int requested_order)
{
    if (requested_order <= 1) {
        return {{Real{0.5}, Real{1.0}}};
    }
    if (requested_order <= 3) {
        constexpr Real offset = Real{0.28867513459481288225};
        return {{Real{0.5} - offset, Real{0.5}},
                {Real{0.5} + offset, Real{0.5}}};
    }
    constexpr Real offset = Real{0.38729833462074168852};
    return {{Real{0.5} - offset, Real{5.0} / Real{18.0}},
            {Real{0.5}, Real{4.0} / Real{9.0}},
            {Real{0.5} + offset, Real{5.0} / Real{18.0}}};
}

[[nodiscard]] int planarVolumeQuadratureOrder(
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    return interfaces::implementedPlanarLevelSetCutVolumeExactOrder(
        request.resolvedVolumeQuadratureOrder());
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint>
triangleVolumeQuadraturePoints(const std::array<Real, 3>& a,
                               const std::array<Real, 3>& b,
                               const std::array<Real, 3>& c,
                               int requested_order)
{
    const Real area = Real{0.5} * norm3(cross(subtract(b, a), subtract(c, a)));
    if (!(area > Real{0.0}) || !std::isfinite(area)) {
        return {};
    }
    const int achieved_order =
        interfaces::implementedPlanarLevelSetCutVolumeExactOrder(requested_order);
    const auto make_point = [&](Real l0, Real l1, Real l2, Real weight) {
        const auto point = std::array<Real, 3>{{
            l0 * a[0] + l1 * b[0] + l2 * c[0],
            l0 * a[1] + l1 * b[1] + l2 * c[1],
            l0 * a[2] + l1 * b[2] + l2 * c[2],
        }};
        return geometry::CutQuadraturePoint{
            .point = point,
            .weight = area * weight,
            .parent_coordinate = point,
            .reference_measure_factor = area};
    };

    std::vector<geometry::CutQuadraturePoint> points;
    if (achieved_order <= 2) {
        constexpr Real high = Real{2.0} / Real{3.0};
        constexpr Real low = Real{1.0} / Real{6.0};
        constexpr Real weight = Real{1.0} / Real{3.0};
        points.reserve(3u);
        points.push_back(make_point(high, low, low, weight));
        points.push_back(make_point(low, high, low, weight));
        points.push_back(make_point(low, low, high, weight));
        return points;
    }

    constexpr Real center_weight = Real{0.225};
    constexpr Real a1 = Real{0.0597158717897698};
    constexpr Real b1 = Real{0.470142064105115};
    constexpr Real w1 = Real{0.132394152788506};
    constexpr Real a2 = Real{0.797426985353087};
    constexpr Real b2 = Real{0.101286507323456};
    constexpr Real w2 = Real{0.125939180544827};
    points.reserve(7u);
    points.push_back(make_point(Real{1.0} / Real{3.0},
                                Real{1.0} / Real{3.0},
                                Real{1.0} / Real{3.0},
                                center_weight));
    points.push_back(make_point(a1, b1, b1, w1));
    points.push_back(make_point(b1, a1, b1, w1));
    points.push_back(make_point(b1, b1, a1, w1));
    points.push_back(make_point(a2, b2, b2, w2));
    points.push_back(make_point(b2, a2, b2, w2));
    points.push_back(make_point(b2, b2, a2, w2));
    return points;
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint>
rectangleVolumeQuadraturePoints(const Rectangle2D& rect, int requested_order)
{
    const Real dx = rect.xmax - rect.xmin;
    const Real dy = rect.ymax - rect.ymin;
    const Real area = dx * dy;
    if (!(area > Real{0.0}) || !std::isfinite(area)) {
        return {};
    }
    const auto rule = gaussLegendreUnitRule(
        interfaces::implementedPlanarLevelSetCutVolumeExactOrder(requested_order));
    std::vector<geometry::CutQuadraturePoint> points;
    points.reserve(rule.size() * rule.size());
    for (const auto& [tx, wx] : rule) {
        for (const auto& [ty, wy] : rule) {
            const std::array<Real, 3> point{{
                rect.xmin + tx * dx,
                rect.ymin + ty * dy,
                0.0,
            }};
            points.push_back(geometry::CutQuadraturePoint{
                .point = point,
                .weight = area * wx * wy,
                .parent_coordinate = point,
                .reference_measure_factor = area});
        }
    }
    return points;
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint>
boxVolumeQuadraturePoints(const Box3D& box, int requested_order)
{
    const Real dx = box.xmax - box.xmin;
    const Real dy = box.ymax - box.ymin;
    const Real dz = box.zmax - box.zmin;
    const Real volume = dx * dy * dz;
    if (!(volume > Real{0.0}) || !std::isfinite(volume)) {
        return {};
    }
    const auto rule = gaussLegendreUnitRule(
        interfaces::implementedLevelSetCutVolumeExactOrder(requested_order));
    std::vector<geometry::CutQuadraturePoint> points;
    points.reserve(rule.size() * rule.size() * rule.size());
    for (const auto& [tx, wx] : rule) {
        for (const auto& [ty, wy] : rule) {
            for (const auto& [tz, wz] : rule) {
                const std::array<Real, 3> point{{
                    box.xmin + tx * dx,
                    box.ymin + ty * dy,
                    box.zmin + tz * dz,
                }};
                points.push_back(geometry::CutQuadraturePoint{
                    .point = point,
                    .weight = volume * wx * wy * wz,
                    .parent_coordinate = point,
                    .reference_measure_factor = volume});
            }
        }
    }
    return points;
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint>
tetrahedronVolumeQuadraturePoints(const Tetrahedron3D& tet,
                                  int requested_order)
{
    const Real volume = tetrahedronMeasure(tet);
    if (!(volume > Real{0.0}) || !std::isfinite(volume)) {
        return {};
    }

    const auto make_point = [&](Real l0,
                                Real l1,
                                Real l2,
                                Real l3,
                                Real weight) {
        const auto point = std::array<Real, 3>{{
            l0 * tet.a[0] + l1 * tet.b[0] + l2 * tet.c[0] + l3 * tet.d[0],
            l0 * tet.a[1] + l1 * tet.b[1] + l2 * tet.c[1] + l3 * tet.d[1],
            l0 * tet.a[2] + l1 * tet.b[2] + l2 * tet.c[2] + l3 * tet.d[2],
        }};
        return geometry::CutQuadraturePoint{
            .point = point,
            .weight = volume * weight,
            .parent_coordinate = point,
            .reference_measure_factor = volume};
    };

    const int achieved_order =
        interfaces::implementedLevelSetCutVolumeExactOrder(requested_order);
    std::vector<geometry::CutQuadraturePoint> points;
    if (achieved_order <= 1) {
        points.push_back(make_point(Real{0.25},
                                    Real{0.25},
                                    Real{0.25},
                                    Real{0.25},
                                    Real{1.0}));
        return points;
    }

    constexpr Real high = Real{0.5854101966249685};
    constexpr Real low = Real{0.1381966011250105};
    constexpr Real weight = Real{0.25};
    points.reserve(4u);
    points.push_back(make_point(high, low, low, low, weight));
    points.push_back(make_point(low, high, low, low, weight));
    points.push_back(make_point(low, low, high, low, weight));
    points.push_back(make_point(low, low, low, high, weight));
    return points;
}

[[nodiscard]] bool solveRootAtFixedX(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    Real x,
    Real y_min,
    Real y_max,
    Real guess_y,
    std::array<Real, 3>& root,
    int& iterations)
{
    constexpr int sample_count = 16;
    const Real tolerance = rootResidualTolerance(request);
    bool have_bracket = false;
    Real best_distance = std::numeric_limits<Real>::infinity();
    std::array<Real, 3> best_a{{x, y_min, 0.0}};
    std::array<Real, 3> best_b{{x, y_max, 0.0}};
    std::array<Real, 3> previous = best_a;
    Real f_previous = signedLevelSetValue(input, previous);
    if (!std::isfinite(f_previous)) {
        return false;
    }
    if (std::abs(f_previous) <= tolerance) {
        root = previous;
        return true;
    }
    for (int i = 1; i <= sample_count; ++i) {
        const Real y =
            y_min + (y_max - y_min) *
                        static_cast<Real>(i) / static_cast<Real>(sample_count);
        const std::array<Real, 3> current{{x, y, 0.0}};
        const Real f_current = signedLevelSetValue(input, current);
        if (!std::isfinite(f_current)) {
            return false;
        }
        if (std::abs(f_current) <= tolerance) {
            root = current;
            return true;
        }
        if ((f_previous < Real{0.0}) != (f_current < Real{0.0})) {
            const Real midpoint_y = Real{0.5} * (previous[1] + current[1]);
            const Real distance_to_guess = std::abs(midpoint_y - guess_y);
            if (distance_to_guess < best_distance) {
                best_distance = distance_to_guess;
                best_a = previous;
                best_b = current;
                have_bracket = true;
            }
        }
        previous = current;
        f_previous = f_current;
    }
    if (!have_bracket) {
        return false;
    }
    return polishRootOnSegment(input, request, best_a, best_b, root, iterations);
}

[[nodiscard]] bool solveRootAtFixedY(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    Real y,
    Real x_min,
    Real x_max,
    Real guess_x,
    std::array<Real, 3>& root,
    int& iterations)
{
    constexpr int sample_count = 16;
    const Real tolerance = rootResidualTolerance(request);
    bool have_bracket = false;
    Real best_distance = std::numeric_limits<Real>::infinity();
    std::array<Real, 3> best_a{{x_min, y, 0.0}};
    std::array<Real, 3> best_b{{x_max, y, 0.0}};
    std::array<Real, 3> previous = best_a;
    Real f_previous = signedLevelSetValue(input, previous);
    if (!std::isfinite(f_previous)) {
        return false;
    }
    if (std::abs(f_previous) <= tolerance) {
        root = previous;
        return true;
    }
    for (int i = 1; i <= sample_count; ++i) {
        const Real x =
            x_min + (x_max - x_min) *
                        static_cast<Real>(i) / static_cast<Real>(sample_count);
        const std::array<Real, 3> current{{x, y, 0.0}};
        const Real f_current = signedLevelSetValue(input, current);
        if (!std::isfinite(f_current)) {
            return false;
        }
        if (std::abs(f_current) <= tolerance) {
            root = current;
            return true;
        }
        if ((f_previous < Real{0.0}) != (f_current < Real{0.0})) {
            const Real midpoint_x = Real{0.5} * (previous[0] + current[0]);
            const Real distance_to_guess = std::abs(midpoint_x - guess_x);
            if (distance_to_guess < best_distance) {
                best_distance = distance_to_guess;
                best_a = previous;
                best_b = current;
                have_bracket = true;
            }
        }
        previous = current;
        f_previous = f_current;
    }
    if (!have_bracket) {
        return false;
    }
    return polishRootOnSegment(input, request, best_a, best_b, root, iterations);
}

[[nodiscard]] bool replaceWithRootPolishedRectangleFragment(
    interfaces::CutInterfaceFragment& fragment,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    if (!fragment.active() || request.resolvedInterfaceQuadratureOrder() <= 1) {
        return false;
    }

    int local_iterations = 0;
    auto roots = rectangleEdgeRoots(input, request, rect, local_iterations);
    if (roots.size() != 2u) {
        return false;
    }
    const auto a = roots[0];
    const auto b = roots[1];
    const Real dx = b[0] - a[0];
    const Real dy = b[1] - a[1];
    if (std::abs(dx) <= Real{1.0e-14} &&
        std::abs(dy) <= Real{1.0e-14}) {
        return false;
    }
    const bool solve_y_as_function_of_x = std::abs(dx) >= std::abs(dy);
    const auto rule =
        gaussLegendreUnitRule(request.resolvedInterfaceQuadratureOrder());

    std::vector<interfaces::CutInterfaceQuadraturePoint> quadrature_points;
    quadrature_points.reserve(rule.size());
    std::array<Real, 3> accumulated_normal{{0.0, 0.0, 0.0}};
    Real measure = 0.0;
    Real max_root_residual = 0.0;
    Real min_gradient_norm = std::numeric_limits<Real>::infinity();
    for (const auto& [t, unit_weight] : rule) {
        const Real x_guess = (Real{1.0} - t) * a[0] + t * b[0];
        const Real y_guess = (Real{1.0} - t) * a[1] + t * b[1];
        std::array<Real, 3> point;
        if (solve_y_as_function_of_x) {
            if (!solveRootAtFixedX(input,
                                   request,
                                   x_guess,
                                   rect.ymin,
                                   rect.ymax,
                                   y_guess,
                                   point,
                                   local_iterations)) {
                return false;
            }
        } else {
            if (!solveRootAtFixedY(input,
                                   request,
                                   y_guess,
                                   rect.xmin,
                                   rect.xmax,
                                   x_guess,
                                   point,
                                   local_iterations)) {
                return false;
            }
        }
        const auto evaluation =
            input.evaluator->evaluate(input.linearized_input.parent_cell, point);
        const Real root_residual =
            std::abs(evaluation.value - input.isovalue);
        const Real gradient_norm = norm3(evaluation.reference_gradient);
        if (!std::isfinite(root_residual) ||
            !std::isfinite(gradient_norm) ||
            gradient_norm <= Real{1.0e-14}) {
            return false;
        }
        max_root_residual = std::max(max_root_residual, root_residual);
        min_gradient_norm = std::min(min_gradient_norm, gradient_norm);
        const auto normal = normalizedOrDefault(evaluation.reference_gradient);
        const Real denominator =
            solve_y_as_function_of_x ? evaluation.reference_gradient[1]
                                     : evaluation.reference_gradient[0];
        if (std::abs(denominator) <= Real{1.0e-14}) {
            return false;
        }
        const Real slope =
            solve_y_as_function_of_x
                ? -evaluation.reference_gradient[0] / denominator
                : -evaluation.reference_gradient[1] / denominator;
        const Real coordinate_span =
            solve_y_as_function_of_x ? std::abs(dx) : std::abs(dy);
        const Real reference_measure_factor =
            coordinate_span * std::sqrt(Real{1.0} + slope * slope);
        const Real weight = unit_weight * reference_measure_factor;
        if (!std::isfinite(weight) || weight <= Real{0.0}) {
            return false;
        }
        quadrature_points.push_back(
            interfaces::CutInterfaceQuadraturePoint{
                .point = point,
                .parent_coordinate = point,
                .normal = normal,
                .weight = weight,
                .reference_measure_factor = reference_measure_factor,
                .level_set_residual = root_residual,
                .gradient_norm = gradient_norm});
        accumulated_normal[0] += normal[0] * weight;
        accumulated_normal[1] += normal[1] * weight;
        accumulated_normal[2] += normal[2] * weight;
        measure += weight;
    }
    if (!std::isfinite(measure) || measure <= request.tolerance) {
        return false;
    }

    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = a,
            .parent_coordinate = a,
            .level_set_value = 0.0,
            .stable_id = interfaces::cutInterfaceStableId(
                request.interface_marker,
                input.linearized_input.parent_cell,
                0u,
                request.source.value_revision)},
        interfaces::CutInterfaceVertex{
            .point = b,
            .parent_coordinate = b,
            .level_set_value = 0.0,
            .stable_id = interfaces::cutInterfaceStableId(
                request.interface_marker,
                input.linearized_input.parent_cell,
                1u,
                request.source.value_revision)}};
    fragment.measure = measure;
    fragment.normal = normalizedOrDefault(accumulated_normal);
    fragment.kind = interfaces::CutInterfaceFragmentKind::CurvedPatch;
    fragment.quadrature_points = std::move(quadrature_points);
    fragment.topology_id =
        "cell-" + std::to_string(input.linearized_input.parent_cell) +
        "-root-polished-rectangle-branch-" +
        std::to_string(diagnostics.curved_fragment_count);
    fragment.branch_id = fragment.topology_id;
    fragment.conditioning_diagnostic = "ok";
    fragment.root_finder_iterations = local_iterations;
    fragment.max_root_residual = max_root_residual;
    fragment.min_gradient_norm =
        std::isfinite(min_gradient_norm) ? min_gradient_norm : 0.0;
    fragment.root_polished = true;

    ++diagnostics.root_branch_count;
    diagnostics.root_finder_iteration_count += local_iterations;
    ++diagnostics.curved_fragment_count;
    return true;
}

[[nodiscard]] bool replaceWithRootPolishedTriangleFragment(
    interfaces::CutInterfaceFragment& fragment,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Triangle2D& tri,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    if (!fragment.active() || request.resolvedInterfaceQuadratureOrder() <= 1) {
        return false;
    }

    int local_iterations = 0;
    auto roots = triangleEdgeRoots(input, request, tri, local_iterations);
    if (roots.size() != 2u) {
        return false;
    }
    const auto a = roots[0];
    const auto b = roots[1];
    const Real chord_length = distance2D(a, b);
    if (!(chord_length > Real{1.0e-14})) {
        return false;
    }
    const std::array<Real, 3> tangent{{
        (b[0] - a[0]) / chord_length,
        (b[1] - a[1]) / chord_length,
        0.0,
    }};
    const std::array<Real, 3> transverse{{-tangent[1], tangent[0], 0.0}};
    const auto rule =
        gaussLegendreUnitRule(request.resolvedInterfaceQuadratureOrder());

    std::vector<interfaces::CutInterfaceQuadraturePoint> quadrature_points;
    quadrature_points.reserve(rule.size());
    std::array<Real, 3> accumulated_normal{{0.0, 0.0, 0.0}};
    Real measure = 0.0;
    Real max_root_residual = 0.0;
    Real min_gradient_norm = std::numeric_limits<Real>::infinity();
    for (const auto& [t, unit_weight] : rule) {
        const std::array<Real, 3> origin{{
            (Real{1.0} - t) * a[0] + t * b[0],
            (Real{1.0} - t) * a[1] + t * b[1],
            0.0,
        }};
        std::array<Real, 3> search_start;
        std::array<Real, 3> search_end;
        Real guess_fraction = 0.5;
        if (!lineTriangleSearchSegment(
                tri, origin, transverse, search_start, search_end, guess_fraction)) {
            return false;
        }
        std::array<Real, 3> point;
        if (!solveRootAlongSegmentNearGuess(input,
                                            request,
                                            search_start,
                                            search_end,
                                            guess_fraction,
                                            point,
                                            local_iterations)) {
            return false;
        }
        const auto evaluation =
            input.evaluator->evaluate(input.linearized_input.parent_cell, point);
        const Real root_residual =
            std::abs(evaluation.value - input.isovalue);
        const Real gradient_norm = norm3(evaluation.reference_gradient);
        if (!std::isfinite(root_residual) ||
            !std::isfinite(gradient_norm) ||
            gradient_norm <= Real{1.0e-14}) {
            return false;
        }
        const Real transverse_derivative =
            dot3(evaluation.reference_gradient, transverse);
        if (std::abs(transverse_derivative) <= Real{1.0e-14}) {
            return false;
        }
        const Real tangent_derivative =
            dot3(evaluation.reference_gradient, tangent);
        const Real height_slope = -tangent_derivative / transverse_derivative;
        const Real reference_measure_factor =
            chord_length * std::sqrt(Real{1.0} + height_slope * height_slope);
        const Real weight = unit_weight * reference_measure_factor;
        if (!std::isfinite(weight) || weight <= Real{0.0}) {
            return false;
        }

        max_root_residual = std::max(max_root_residual, root_residual);
        min_gradient_norm = std::min(min_gradient_norm, gradient_norm);
        const auto normal = normalizedOrDefault(evaluation.reference_gradient);
        quadrature_points.push_back(
            interfaces::CutInterfaceQuadraturePoint{
                .point = point,
                .parent_coordinate = point,
                .normal = normal,
                .weight = weight,
                .reference_measure_factor = reference_measure_factor,
                .level_set_residual = root_residual,
                .gradient_norm = gradient_norm});
        accumulated_normal[0] += normal[0] * weight;
        accumulated_normal[1] += normal[1] * weight;
        accumulated_normal[2] += normal[2] * weight;
        measure += weight;
    }
    if (!std::isfinite(measure) || measure <= request.tolerance) {
        return false;
    }

    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = a,
            .parent_coordinate = a,
            .level_set_value = 0.0,
            .stable_id = interfaces::cutInterfaceStableId(
                request.interface_marker,
                input.linearized_input.parent_cell,
                0u,
                request.source.value_revision)},
        interfaces::CutInterfaceVertex{
            .point = b,
            .parent_coordinate = b,
            .level_set_value = 0.0,
            .stable_id = interfaces::cutInterfaceStableId(
                request.interface_marker,
                input.linearized_input.parent_cell,
                1u,
                request.source.value_revision)}};
    fragment.measure = measure;
    fragment.normal = normalizedOrDefault(accumulated_normal);
    fragment.kind = interfaces::CutInterfaceFragmentKind::CurvedPatch;
    fragment.quadrature_points = std::move(quadrature_points);
    fragment.topology_id =
        "cell-" + std::to_string(input.linearized_input.parent_cell) +
        "-root-polished-triangle-branch-" +
        std::to_string(diagnostics.curved_fragment_count);
    fragment.branch_id = fragment.topology_id;
    fragment.conditioning_diagnostic = "ok";
    fragment.root_finder_iterations = local_iterations;
    fragment.max_root_residual = max_root_residual;
    fragment.min_gradient_norm =
        std::isfinite(min_gradient_norm) ? min_gradient_norm : 0.0;
    fragment.root_polished = true;

    ++diagnostics.root_branch_count;
    diagnostics.root_finder_iteration_count += local_iterations;
    ++diagnostics.curved_fragment_count;
    return true;
}

[[nodiscard]] bool replaceWithRootPolishedTetrahedronFragment(
    interfaces::CutInterfaceFragment& fragment,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    if (!fragment.active() ||
        fragment.kind != interfaces::CutInterfaceFragmentKind::Polygon ||
        request.resolvedInterfaceQuadratureOrder() <= 1 ||
        fragment.vertices.size() < 3u) {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_vertex_count_failure;
        return false;
    }

    const auto fail_seed = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_seed_failure;
        return false;
    };
    const auto fail_search_segment = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_search_segment_failure;
        return false;
    };
    const auto fail_root_solve = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_root_solve_failure;
        return false;
    };
    const auto fail_gradient = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_gradient_failure;
        return false;
    };
    const auto fail_weight = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_weight_failure;
        return false;
    };

    std::vector<std::array<Real, 3>> base_vertices;
    base_vertices.reserve(fragment.vertices.size());
    for (const auto& vertex : fragment.vertices) {
        base_vertices.push_back(vertex.point);
    }

    int local_iterations = 0;
    const auto edge_roots =
        tetrahedronEdgeRoots(input, request, tet, local_iterations);
    std::vector<std::array<Real, 3>> matched_roots;
    const bool matched_edge_roots =
        matchPolishedRootsToBaseVertices(edge_roots, base_vertices, matched_roots);
    if (matched_edge_roots) {
        base_vertices = matched_roots;
    } else {
        ++diagnostics.curved_fragment_edge_root_mismatch_count;
    }

    const auto base_centroid = polygonCentroid(base_vertices);
    const auto gradient_normal = interfaceNormalAt(input, base_centroid);
    const auto base_normal =
        polygonNormalOrDefault(base_vertices, gradient_normal);
    if (norm3(base_normal) <= Real{1.0e-14}) {
        return fail_seed();
    }

    const auto mark_boundary_degenerate = [&]() {
        const Real root_tolerance =
            Real{100.0} * rootResidualTolerance(request);
        Real max_root_residual = 0.0;
        Real min_gradient_norm = std::numeric_limits<Real>::infinity();
        for (const auto& vertex : base_vertices) {
            const auto evaluation =
                input.evaluator->evaluate(input.linearized_input.parent_cell, vertex);
            const Real root_residual =
                std::abs(evaluation.value - input.isovalue);
            if (!std::isfinite(root_residual) ||
                root_residual > root_tolerance) {
                return false;
            }
            max_root_residual = std::max(max_root_residual, root_residual);
            const Real gradient_norm = norm3(evaluation.reference_gradient);
            if (std::isfinite(gradient_norm)) {
                min_gradient_norm = std::min(min_gradient_norm, gradient_norm);
            }
        }

        fragment.vertices.clear();
        fragment.vertices.reserve(base_vertices.size());
        for (std::size_t i = 0u; i < base_vertices.size(); ++i) {
            fragment.vertices.push_back(
                interfaces::CutInterfaceVertex{
                    .point = base_vertices[i],
                    .parent_coordinate = base_vertices[i],
                    .level_set_value = 0.0,
                    .stable_id = interfaces::cutInterfaceStableId(
                        request.interface_marker,
                        input.linearized_input.parent_cell,
                        static_cast<LocalIndex>(i + 1u),
                        request.source.value_revision)});
        }
        fragment.measure = 0.0;
        fragment.normal = base_normal;
        fragment.kind = interfaces::CutInterfaceFragmentKind::CurvedPatch;
        fragment.degeneracy = interfaces::CutInterfaceDegeneracy::SmallFragment;
        fragment.quadrature_points.clear();
        fragment.topology_id =
            "cell-" + std::to_string(input.linearized_input.parent_cell) +
            "-root-polished-tetrahedron-boundary-degenerate-" +
            std::to_string(diagnostics.curved_fragment_boundary_degenerate_count);
        fragment.branch_id = fragment.topology_id;
        fragment.conditioning_diagnostic = "boundary-only-curved-degenerate";
        fragment.root_finder_iterations = local_iterations;
        fragment.max_root_residual = max_root_residual;
        fragment.min_gradient_norm =
            std::isfinite(min_gradient_norm) ? min_gradient_norm : 0.0;
        fragment.root_polished = true;

        ++diagnostics.root_branch_count;
        diagnostics.root_finder_iteration_count += local_iterations;
        ++diagnostics.curved_fragment_boundary_degenerate_count;
        return true;
    };

    std::vector<std::pair<std::array<Real, 3>, Real>> surface_seeds;
    surface_seeds.reserve((base_vertices.size() - 2u) * 3u);
    for (std::size_t i = 1u; i + 1u < base_vertices.size(); ++i) {
        appendTriangleSurfaceQuadratureSeeds(
            base_vertices[0], base_vertices[i], base_vertices[i + 1u],
            surface_seeds);
    }
    if (surface_seeds.empty()) {
        return fail_seed();
    }

    std::vector<interfaces::CutInterfaceQuadraturePoint> quadrature_points;
    quadrature_points.reserve(surface_seeds.size());
    std::array<Real, 3> accumulated_normal{{0.0, 0.0, 0.0}};
    Real measure = 0.0;
    Real max_root_residual = 0.0;
    Real min_gradient_norm = std::numeric_limits<Real>::infinity();
    for (const auto& [seed, planar_weight] : surface_seeds) {
        const auto seed_normal = interfaceNormalAt(input, seed);
        const std::array<std::array<Real, 3>, 3> projection_directions{{
            base_normal,
            gradient_normal,
            seed_normal,
        }};

        bool saw_search_segment = false;
        bool saw_root = false;
        bool saw_gradient = false;
        bool accepted = false;
        interfaces::CutInterfaceQuadraturePoint accepted_point{};
        for (const auto& projection_direction : projection_directions) {
            const Real plane_projection =
                std::abs(dot3(base_normal, projection_direction));
            if (plane_projection <= Real{1.0e-14}) {
                continue;
            }

            std::array<Real, 3> search_start;
            std::array<Real, 3> search_end;
            Real guess_fraction = 0.5;
            if (!lineTetrahedronSearchSegment(tet,
                                               seed,
                                               projection_direction,
                                               search_start,
                                               search_end,
                                               guess_fraction)) {
                continue;
            }
            saw_search_segment = true;

            std::array<Real, 3> point;
            if (!solveRootAlongSegmentNearGuess(input,
                                                request,
                                                search_start,
                                                search_end,
                                                guess_fraction,
                                                point,
                                                local_iterations)) {
                continue;
            }
            saw_root = true;

            const auto evaluation =
                input.evaluator->evaluate(input.linearized_input.parent_cell, point);
            const Real root_residual =
                std::abs(evaluation.value - input.isovalue);
            const Real gradient_norm = norm3(evaluation.reference_gradient);
            if (!std::isfinite(root_residual) ||
                !std::isfinite(gradient_norm) ||
                gradient_norm <= Real{1.0e-14}) {
                continue;
            }
            const Real directional_derivative =
                std::abs(dot3(evaluation.reference_gradient,
                              projection_direction));
            if (directional_derivative <= Real{1.0e-14}) {
                continue;
            }
            saw_gradient = true;

            const Real reference_measure_factor =
                plane_projection * gradient_norm / directional_derivative;
            const Real weight = planar_weight * reference_measure_factor;
            if (!std::isfinite(reference_measure_factor) ||
                !std::isfinite(weight) ||
                weight <= Real{0.0}) {
                continue;
            }

            const auto normal =
                normalizedOrDefault(evaluation.reference_gradient);
            accepted_point = interfaces::CutInterfaceQuadraturePoint{
                .point = point,
                .parent_coordinate = point,
                .normal = normal,
                .weight = weight,
                .reference_measure_factor = reference_measure_factor,
                .level_set_residual = root_residual,
                .gradient_norm = gradient_norm};
            accepted = true;
            break;
        }

        if (!accepted) {
            std::array<Real, 3> point;
            if (newtonProjectRootInsideTetrahedron(input,
                                                   request,
                                                   tet,
                                                   seed,
                                                   point,
                                                   local_iterations)) {
                saw_root = true;
                const auto evaluation =
                    input.evaluator->evaluate(input.linearized_input.parent_cell,
                                              point);
                const Real root_residual =
                    std::abs(evaluation.value - input.isovalue);
                const Real gradient_norm = norm3(evaluation.reference_gradient);
                if (std::isfinite(root_residual) &&
                    std::isfinite(gradient_norm) &&
                    gradient_norm > Real{1.0e-14}) {
                    const Real normal_projection =
                        std::abs(dot3(evaluation.reference_gradient,
                                      base_normal));
                    if (normal_projection > Real{1.0e-14}) {
                        saw_gradient = true;
                        const Real reference_measure_factor =
                            gradient_norm / normal_projection;
                        const Real weight =
                            planar_weight * reference_measure_factor;
                        if (std::isfinite(reference_measure_factor) &&
                            std::isfinite(weight) &&
                            weight > Real{0.0}) {
                            const auto normal = normalizedOrDefault(
                                evaluation.reference_gradient);
                            accepted_point =
                                interfaces::CutInterfaceQuadraturePoint{
                                    .point = point,
                                    .parent_coordinate = point,
                                    .normal = normal,
                                    .weight = weight,
                                    .reference_measure_factor =
                                        reference_measure_factor,
                                    .level_set_residual = root_residual,
                                    .gradient_norm = gradient_norm};
                            accepted = true;
                        }
                    }
                }
            }
        }

        if (!accepted) {
            if (matched_edge_roots && mark_boundary_degenerate()) {
                return true;
            }
            if (!saw_search_segment) {
                return fail_search_segment();
            }
            if (!saw_root) {
                if (!matched_edge_roots) {
                    ++diagnostics.curved_fragment_root_solve_edge_root_mismatch;
                }
                return fail_root_solve();
            }
            if (!saw_gradient) {
                return fail_gradient();
            }
            return fail_weight();
        }

        max_root_residual = std::max(max_root_residual,
                                     accepted_point.level_set_residual);
        min_gradient_norm = std::min(min_gradient_norm,
                                     accepted_point.gradient_norm);
        quadrature_points.push_back(accepted_point);
        accumulated_normal[0] += accepted_point.normal[0] * accepted_point.weight;
        accumulated_normal[1] += accepted_point.normal[1] * accepted_point.weight;
        accumulated_normal[2] += accepted_point.normal[2] * accepted_point.weight;
        measure += accepted_point.weight;
    }
    if (!std::isfinite(measure) || measure <= request.tolerance) {
        return fail_weight();
    }

    fragment.vertices.clear();
    const auto& output_vertices =
        matched_edge_roots ? matched_roots : base_vertices;
    fragment.vertices.reserve(output_vertices.size());
    for (std::size_t i = 0u; i < output_vertices.size(); ++i) {
        fragment.vertices.push_back(
            interfaces::CutInterfaceVertex{
                .point = output_vertices[i],
                .parent_coordinate = output_vertices[i],
                .level_set_value = 0.0,
                .stable_id = interfaces::cutInterfaceStableId(
                    request.interface_marker,
                    input.linearized_input.parent_cell,
                    static_cast<LocalIndex>(i + 1u),
                    request.source.value_revision)});
    }
    fragment.measure = measure;
    fragment.normal = normalizedOrDefault(accumulated_normal);
    fragment.kind = interfaces::CutInterfaceFragmentKind::CurvedPatch;
    fragment.quadrature_points = std::move(quadrature_points);
    fragment.topology_id =
        "cell-" + std::to_string(input.linearized_input.parent_cell) +
        "-root-polished-tetrahedron-branch-" +
        std::to_string(diagnostics.curved_fragment_count);
    fragment.branch_id = fragment.topology_id;
    fragment.conditioning_diagnostic =
        matched_edge_roots ? "ok" : "edge-root-vertex-mismatch";
    fragment.root_finder_iterations = local_iterations;
    fragment.max_root_residual = max_root_residual;
    fragment.min_gradient_norm =
        std::isfinite(min_gradient_norm) ? min_gradient_norm : 0.0;
    fragment.root_polished = true;

    ++diagnostics.root_branch_count;
    diagnostics.root_finder_iteration_count += local_iterations;
    ++diagnostics.curved_fragment_count;
    return true;
}

void alignLeafCutNormalsWithEvaluator(
    interfaces::LevelSetCellCutResult& leaf_cut,
    const ImplicitCutQuadratureBackendCellInput& input)
{
    for (auto& fragment : leaf_cut.fragments) {
        std::array<Real, 3> accumulated{{0.0, 0.0, 0.0}};
        for (auto& qp : fragment.quadrature_points) {
            qp.normal = interfaceNormalAt(input, qp.point);
            accumulated[0] += qp.normal[0];
            accumulated[1] += qp.normal[1];
            accumulated[2] += qp.normal[2];
        }
        if (!fragment.quadrature_points.empty()) {
            fragment.normal = normalizedOrDefault(accumulated);
            continue;
        }
        if (!fragment.vertices.empty()) {
            std::array<Real, 3> centroid{{0.0, 0.0, 0.0}};
            for (const auto& vertex : fragment.vertices) {
                centroid[0] += vertex.point[0];
                centroid[1] += vertex.point[1];
                centroid[2] += vertex.point[2];
            }
            const Real inv_count =
                Real{1.0} / static_cast<Real>(fragment.vertices.size());
            centroid[0] *= inv_count;
            centroid[1] *= inv_count;
            centroid[2] *= inv_count;
            fragment.normal = interfaceNormalAt(input, centroid);
        }
    }

    for (auto& region : leaf_cut.volume_regions) {
        auto normal = interfaceNormalAt(input, region.centroid);
        if (region.side == geometry::CutIntegrationSide::Positive) {
            normal = {{-normal[0], -normal[1], -normal[2]}};
        }
        region.normal = normal;
        for (auto& qp : region.quadrature_points) {
            qp.normal = region.normal;
        }
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

[[nodiscard]] std::vector<std::array<Real, 3>> boxSamplePoints(
    const Box3D& box)
{
    const Real xm = Real{0.5} * (box.xmin + box.xmax);
    const Real ym = Real{0.5} * (box.ymin + box.ymax);
    const Real zm = Real{0.5} * (box.zmin + box.zmax);
    const std::array<Real, 3> xs{{box.xmin, xm, box.xmax}};
    const std::array<Real, 3> ys{{box.ymin, ym, box.ymax}};
    const std::array<Real, 3> zs{{box.zmin, zm, box.zmax}};
    std::vector<std::array<Real, 3>> points;
    points.reserve(27u);
    for (const Real z : zs) {
        for (const Real y : ys) {
            for (const Real x : xs) {
                points.push_back({{x, y, z}});
            }
        }
    }
    return points;
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

[[nodiscard]] std::array<Real, 3> faceCentroid(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& c) noexcept
{
    return {{(a[0] + b[0] + c[0]) / Real{3.0},
             (a[1] + b[1] + c[1]) / Real{3.0},
             (a[2] + b[2] + c[2]) / Real{3.0}}};
}

[[nodiscard]] std::vector<std::array<Real, 3>> tetrahedronSamplePoints(
    const Tetrahedron3D& tet)
{
    return {
        tet.a,
        tet.b,
        tet.c,
        tet.d,
        midpoint(tet.a, tet.b),
        midpoint(tet.a, tet.c),
        midpoint(tet.a, tet.d),
        midpoint(tet.b, tet.c),
        midpoint(tet.b, tet.d),
        midpoint(tet.c, tet.d),
        faceCentroid(tet.a, tet.b, tet.c),
        faceCentroid(tet.a, tet.b, tet.d),
        faceCentroid(tet.a, tet.c, tet.d),
        faceCentroid(tet.b, tet.c, tet.d),
        tetrahedronCentroid(tet),
    };
}

[[nodiscard]] const char* sideTopologyToken(
    geometry::CutIntegrationSide side) noexcept
{
    switch (side) {
    case geometry::CutIntegrationSide::Negative:
        return "negative";
    case geometry::CutIntegrationSide::Positive:
        return "positive";
    case geometry::CutIntegrationSide::Interface:
        return "interface";
    }
    return "unknown";
}

void stampGeneratedVolumeRegionMetadata(
    interfaces::CutInterfaceVolumeRegion& region,
    const interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const char* construction_token)
{
    if (region.local_region_index == INVALID_LOCAL_INDEX) {
        region.local_region_index =
            static_cast<LocalIndex>(cut.volume_regions.size());
    }
    if (region.topology_id.empty()) {
        region.topology_id =
            "cell-" + std::to_string(input.linearized_input.parent_cell) +
            "-volume-" + construction_token + "-" +
            sideTopologyToken(region.side) + "-" +
            std::to_string(region.local_region_index);
    }
    if (region.stable_id == 0u) {
        region.stable_id =
            interfaces::cutVolumeStableId(request.interface_marker,
                                          input.linearized_input.parent_cell,
                                          region.local_region_index,
                                          region.side,
                                          request.source.value_revision);
    }
    if (region.achieved_quadrature_order < 0) {
        region.achieved_quadrature_order =
            interfaces::implementedLevelSetCutVolumeExactOrder(
                request.resolvedVolumeQuadratureOrder());
    }
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
        region.achieved_quadrature_order = planarVolumeQuadratureOrder(request);
        if (!region.full_cell_equivalent) {
            region.quadrature_points =
                rectangleVolumeQuadraturePoints(
                    rect, request.resolvedVolumeQuadratureOrder());
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "rectangle");
        cut.volume_regions.push_back(std::move(region));
    }
}

void appendFullBoxRegion(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Box3D& box,
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

    const auto centroid = boxCentroid(box);
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
    region.measure = boxMeasure(box);
    region.volume_fraction =
        parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
    region.min_level_set_value = min_signed_value;
    region.max_level_set_value = max_signed_value;
    region.full_cell_equivalent = std::abs(region.measure - parent_measure) <=
                                  std::max(request.tolerance,
                                           request.tolerance * parent_measure);
    if (region.measure > Real{0.0}) {
        region.achieved_quadrature_order =
            interfaces::implementedLevelSetCutVolumeExactOrder(
                request.resolvedVolumeQuadratureOrder());
        if (!region.full_cell_equivalent) {
            region.quadrature_points =
                boxVolumeQuadraturePoints(
                    box, request.resolvedVolumeQuadratureOrder());
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "box");
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
        region.achieved_quadrature_order = planarVolumeQuadratureOrder(request);
        if (!region.full_cell_equivalent) {
            region.quadrature_points =
                triangleVolumeQuadraturePoints(
                    tri.a, tri.b, tri.c, request.resolvedVolumeQuadratureOrder());
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "triangle");
        cut.volume_regions.push_back(std::move(region));
    }
}

void appendFullTetrahedronRegion(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
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

    const auto centroid = tetrahedronCentroid(tet);
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
    region.measure = tetrahedronMeasure(tet);
    region.volume_fraction =
        parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
    region.min_level_set_value = min_signed_value;
    region.max_level_set_value = max_signed_value;
    region.full_cell_equivalent = std::abs(region.measure - parent_measure) <=
                                  std::max(request.tolerance,
                                           request.tolerance * parent_measure);
    if (region.measure > Real{0.0}) {
        region.achieved_quadrature_order =
            interfaces::implementedLevelSetCutVolumeExactOrder(
                request.resolvedVolumeQuadratureOrder());
        if (!region.full_cell_equivalent) {
            region.quadrature_points =
                tetrahedronVolumeQuadraturePoints(
                    tet, request.resolvedVolumeQuadratureOrder());
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "tetrahedron");
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
    alignLeafCutNormalsWithEvaluator(leaf_cut, input);
    diagnostics.interface_fragment_count +=
        static_cast<int>(leaf_cut.fragments.size());
    for (auto& fragment : leaf_cut.fragments) {
        (void)replaceWithRootPolishedRectangleFragment(
            fragment, request, input, rect, diagnostics);
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
    alignLeafCutNormalsWithEvaluator(leaf_cut, input);
    diagnostics.interface_fragment_count +=
        static_cast<int>(leaf_cut.fragments.size());
    for (auto& fragment : leaf_cut.fragments) {
        (void)replaceWithRootPolishedTriangleFragment(
            fragment, request, input, tri, diagnostics);
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
    if (!leaf_cut.hasActiveFragments() &&
        cut.degeneracy == interfaces::CutInterfaceDegeneracy::None &&
        leaf_cut.degeneracy != interfaces::CutInterfaceDegeneracy::None &&
        leaf_cut.degeneracy != interfaces::CutInterfaceDegeneracy::NoCut) {
        cut.degeneracy = leaf_cut.degeneracy;
    }
}

void appendLinearizedTetrahedronCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    Real parent_measure,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    ++diagnostics.linearized_leaf_count;

    interfaces::LevelSetCellCutInput leaf;
    leaf.parent_cell = input.linearized_input.parent_cell;
    leaf.element_type = ElementType::Tetra4;
    leaf.node_coordinates = {tet.a, tet.b, tet.c, tet.d};
    leaf.level_set_values.reserve(leaf.node_coordinates.size());
    for (const auto& point : leaf.node_coordinates) {
        leaf.level_set_values.push_back(
            input.evaluator
                ->evaluate(input.linearized_input.parent_cell, point)
                .value);
    }

    auto leaf_cut = interfaces::cutLinearLevelSetCell3D(request, leaf);
    alignLeafCutNormalsWithEvaluator(leaf_cut, input);
    diagnostics.interface_fragment_count +=
        static_cast<int>(leaf_cut.fragments.size());
    for (auto& fragment : leaf_cut.fragments) {
        (void)replaceWithRootPolishedTetrahedronFragment(
            fragment, request, input, tet, diagnostics);
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
    if (!leaf_cut.hasActiveFragments() &&
        cut.degeneracy == interfaces::CutInterfaceDegeneracy::None &&
        leaf_cut.degeneracy != interfaces::CutInterfaceDegeneracy::None &&
        leaf_cut.degeneracy != interfaces::CutInterfaceDegeneracy::NoCut) {
        cut.degeneracy = leaf_cut.degeneracy;
    }
}

[[nodiscard]] std::array<std::array<Real, 3>, 8> boxVertices(
    const Box3D& box) noexcept
{
    return {{
        {{box.xmin, box.ymin, box.zmin}},
        {{box.xmax, box.ymin, box.zmin}},
        {{box.xmax, box.ymax, box.zmin}},
        {{box.xmin, box.ymax, box.zmin}},
        {{box.xmin, box.ymin, box.zmax}},
        {{box.xmax, box.ymin, box.zmax}},
        {{box.xmax, box.ymax, box.zmax}},
        {{box.xmin, box.ymax, box.zmax}},
    }};
}

void appendLinearizedBoxCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Box3D& box,
    Real parent_measure,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    const auto v = boxVertices(box);
    const std::array<Tetrahedron3D, 6> tetrahedra{{
        Tetrahedron3D{v[0], v[1], v[2], v[6]},
        Tetrahedron3D{v[0], v[2], v[3], v[6]},
        Tetrahedron3D{v[0], v[3], v[7], v[6]},
        Tetrahedron3D{v[0], v[7], v[4], v[6]},
        Tetrahedron3D{v[0], v[4], v[5], v[6]},
        Tetrahedron3D{v[0], v[5], v[1], v[6]},
    }};
    for (const auto& tet : tetrahedra) {
        appendLinearizedTetrahedronCut(
            cut, request, input, tet, parent_measure, diagnostics);
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

void appendAdaptiveBoxCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Box3D& box,
    Real parent_measure,
    int depth,
    int max_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    diagnostics.max_depth_reached =
        std::max(diagnostics.max_depth_reached, depth);
    const auto samples = boxSamplePoints(box);
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
        appendFullBoxRegion(
            cut,
            request,
            input,
            box,
            has_negative ? geometry::CutIntegrationSide::Negative
                         : geometry::CutIntegrationSide::Positive,
            parent_measure,
            min_signed,
            max_signed,
            diagnostics);
        return;
    }

    if (depth >= max_depth) {
        appendLinearizedBoxCut(
            cut, request, input, box, parent_measure, diagnostics);
        return;
    }

    ++diagnostics.subdivision_count;
    const Real xm = Real{0.5} * (box.xmin + box.xmax);
    const Real ym = Real{0.5} * (box.ymin + box.ymax);
    const Real zm = Real{0.5} * (box.zmin + box.zmax);
    const std::array<Real, 3> xs{{box.xmin, xm, box.xmax}};
    const std::array<Real, 3> ys{{box.ymin, ym, box.ymax}};
    const std::array<Real, 3> zs{{box.zmin, zm, box.zmax}};
    for (std::size_t iz = 0u; iz < 2u; ++iz) {
        for (std::size_t iy = 0u; iy < 2u; ++iy) {
            for (std::size_t ix = 0u; ix < 2u; ++ix) {
                appendAdaptiveBoxCut(
                    cut,
                    request,
                    input,
                    Box3D{xs[ix], xs[ix + 1u],
                          ys[iy], ys[iy + 1u],
                          zs[iz], zs[iz + 1u]},
                    parent_measure,
                    depth + 1,
                    max_depth,
                    diagnostics);
            }
        }
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

void appendAdaptiveTetrahedronCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    Real parent_measure,
    int depth,
    int max_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    diagnostics.max_depth_reached =
        std::max(diagnostics.max_depth_reached, depth);
    const auto samples = tetrahedronSamplePoints(tet);
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
        appendFullTetrahedronRegion(
            cut,
            request,
            input,
            tet,
            has_negative ? geometry::CutIntegrationSide::Negative
                         : geometry::CutIntegrationSide::Positive,
            parent_measure,
            min_signed,
            max_signed,
            diagnostics);
        return;
    }

    if (depth >= max_depth) {
        appendLinearizedTetrahedronCut(
            cut, request, input, tet, parent_measure, diagnostics);
        return;
    }

    ++diagnostics.subdivision_count;
    const auto ab = midpoint(tet.a, tet.b);
    const auto ac = midpoint(tet.a, tet.c);
    const auto ad = midpoint(tet.a, tet.d);
    const auto bc = midpoint(tet.b, tet.c);
    const auto bd = midpoint(tet.b, tet.d);
    const auto cd = midpoint(tet.c, tet.d);
    const std::array<Tetrahedron3D, 8> children{{
        Tetrahedron3D{tet.a, ab, ac, ad},
        Tetrahedron3D{ab, tet.b, bc, bd},
        Tetrahedron3D{ac, bc, tet.c, cd},
        Tetrahedron3D{ad, bd, cd, tet.d},
        Tetrahedron3D{ab, ac, ad, cd},
        Tetrahedron3D{ab, ac, bc, cd},
        Tetrahedron3D{ab, ad, bd, cd},
        Tetrahedron3D{ab, bc, bd, cd},
    }};
    for (const auto& child : children) {
        appendAdaptiveTetrahedronCut(
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
    std::string diagnostic =
        "SayeHyperrectangle recursive 2D hyperrectangle quadrature"
           "; max_depth_limit=" + std::to_string(max_depth_limit) +
           "; max_depth_reached=" +
           std::to_string(diagnostics.max_depth_reached) +
           "; subdivisions=" + std::to_string(diagnostics.subdivision_count) +
           "; root_branches=" +
           std::to_string(diagnostics.root_branch_count) +
           "; curved_fragments=" +
           std::to_string(diagnostics.curved_fragment_count) +
           "; linearized_leaves=" +
           std::to_string(diagnostics.linearized_leaf_count) +
           "; full_negative_regions=" +
           std::to_string(diagnostics.full_negative_region_count) +
           "; full_positive_regions=" +
           std::to_string(diagnostics.full_positive_region_count) +
           "; interface_fragments=" +
           std::to_string(diagnostics.interface_fragment_count) +
           "; curved_fragment_failures=" +
           std::to_string(diagnostics.curved_fragment_failure_count) +
           "; curved_failure_vertex_count=" +
           std::to_string(diagnostics.curved_fragment_vertex_count_failure) +
           "; curved_failure_seed=" +
           std::to_string(diagnostics.curved_fragment_seed_failure) +
           "; curved_failure_search_segment=" +
           std::to_string(diagnostics.curved_fragment_search_segment_failure) +
           "; curved_failure_root_solve=" +
           std::to_string(diagnostics.curved_fragment_root_solve_failure) +
           "; curved_failure_gradient=" +
           std::to_string(diagnostics.curved_fragment_gradient_failure) +
           "; curved_failure_weight=" +
           std::to_string(diagnostics.curved_fragment_weight_failure) +
           "; curved_edge_root_mismatches=" +
           std::to_string(diagnostics.curved_fragment_edge_root_mismatch_count) +
           "; curved_root_solve_edge_root_mismatches=" +
           std::to_string(
               diagnostics.curved_fragment_root_solve_edge_root_mismatch) +
           "; curved_boundary_degenerate=" +
           std::to_string(
               diagnostics.curved_fragment_boundary_degenerate_count);
    appendRootPolishDiagnostics(diagnostic,
                                diagnostics.root_finder_iteration_count);
    return diagnostic;
}

[[nodiscard]] std::string formatSayeHyperrectangle3DDiagnostics(
    const SayeHyperrectangleDiagnostics& diagnostics,
    int max_depth_limit)
{
    std::string diagnostic =
        "SayeHyperrectangle recursive 3D hyperrectangle quadrature"
           "; max_depth_limit=" + std::to_string(max_depth_limit) +
           "; max_depth_reached=" +
           std::to_string(diagnostics.max_depth_reached) +
           "; subdivisions=" + std::to_string(diagnostics.subdivision_count) +
           "; root_branches=" +
           std::to_string(diagnostics.root_branch_count) +
           "; curved_fragments=" +
           std::to_string(diagnostics.curved_fragment_count) +
           "; linearized_leaves=" +
           std::to_string(diagnostics.linearized_leaf_count) +
           "; full_negative_regions=" +
           std::to_string(diagnostics.full_negative_region_count) +
           "; full_positive_regions=" +
           std::to_string(diagnostics.full_positive_region_count) +
           "; interface_fragments=" +
           std::to_string(diagnostics.interface_fragment_count) +
           "; curved_fragment_failures=" +
           std::to_string(diagnostics.curved_fragment_failure_count) +
           "; curved_failure_vertex_count=" +
           std::to_string(diagnostics.curved_fragment_vertex_count_failure) +
           "; curved_failure_seed=" +
           std::to_string(diagnostics.curved_fragment_seed_failure) +
           "; curved_failure_search_segment=" +
           std::to_string(diagnostics.curved_fragment_search_segment_failure) +
           "; curved_failure_root_solve=" +
           std::to_string(diagnostics.curved_fragment_root_solve_failure) +
           "; curved_failure_gradient=" +
           std::to_string(diagnostics.curved_fragment_gradient_failure) +
           "; curved_failure_weight=" +
           std::to_string(diagnostics.curved_fragment_weight_failure) +
           "; curved_edge_root_mismatches=" +
           std::to_string(diagnostics.curved_fragment_edge_root_mismatch_count) +
           "; curved_root_solve_edge_root_mismatches=" +
           std::to_string(
               diagnostics.curved_fragment_root_solve_edge_root_mismatch) +
           "; curved_boundary_degenerate=" +
           std::to_string(
               diagnostics.curved_fragment_boundary_degenerate_count);
    appendRootPolishDiagnostics(diagnostic,
                                diagnostics.root_finder_iteration_count);
    return diagnostic;
}

[[nodiscard]] std::string formatHighOrderSubcellDiagnostics(
    const SayeHyperrectangleDiagnostics& diagnostics,
    int max_depth_limit)
{
    std::string diagnostic =
        "HighOrderSubcell recursive 2D triangle quadrature"
           "; max_depth_limit=" + std::to_string(max_depth_limit) +
           "; max_depth_reached=" +
           std::to_string(diagnostics.max_depth_reached) +
           "; subdivisions=" + std::to_string(diagnostics.subdivision_count) +
           "; root_branches=" +
           std::to_string(diagnostics.root_branch_count) +
           "; curved_fragments=" +
           std::to_string(diagnostics.curved_fragment_count) +
           "; linearized_leaves=" +
           std::to_string(diagnostics.linearized_leaf_count) +
           "; full_negative_regions=" +
           std::to_string(diagnostics.full_negative_region_count) +
           "; full_positive_regions=" +
           std::to_string(diagnostics.full_positive_region_count) +
           "; interface_fragments=" +
           std::to_string(diagnostics.interface_fragment_count) +
           "; curved_fragment_failures=" +
           std::to_string(diagnostics.curved_fragment_failure_count) +
           "; curved_failure_vertex_count=" +
           std::to_string(diagnostics.curved_fragment_vertex_count_failure) +
           "; curved_failure_seed=" +
           std::to_string(diagnostics.curved_fragment_seed_failure) +
           "; curved_failure_search_segment=" +
           std::to_string(diagnostics.curved_fragment_search_segment_failure) +
           "; curved_failure_root_solve=" +
           std::to_string(diagnostics.curved_fragment_root_solve_failure) +
           "; curved_failure_gradient=" +
           std::to_string(diagnostics.curved_fragment_gradient_failure) +
           "; curved_failure_weight=" +
           std::to_string(diagnostics.curved_fragment_weight_failure) +
           "; curved_edge_root_mismatches=" +
           std::to_string(diagnostics.curved_fragment_edge_root_mismatch_count) +
           "; curved_root_solve_edge_root_mismatches=" +
           std::to_string(
               diagnostics.curved_fragment_root_solve_edge_root_mismatch) +
           "; curved_boundary_degenerate=" +
           std::to_string(
               diagnostics.curved_fragment_boundary_degenerate_count);
    appendRootPolishDiagnostics(diagnostic,
                                diagnostics.root_finder_iteration_count);
    return diagnostic;
}

[[nodiscard]] std::string formatHighOrderSubcellTetrahedronDiagnostics(
    const SayeHyperrectangleDiagnostics& diagnostics,
    int max_depth_limit)
{
    std::string diagnostic =
        "HighOrderSubcell recursive 3D tetrahedron quadrature"
           "; max_depth_limit=" + std::to_string(max_depth_limit) +
           "; max_depth_reached=" +
           std::to_string(diagnostics.max_depth_reached) +
           "; subdivisions=" + std::to_string(diagnostics.subdivision_count) +
           "; root_branches=" +
           std::to_string(diagnostics.root_branch_count) +
           "; curved_fragments=" +
           std::to_string(diagnostics.curved_fragment_count) +
           "; linearized_leaves=" +
           std::to_string(diagnostics.linearized_leaf_count) +
           "; full_negative_regions=" +
           std::to_string(diagnostics.full_negative_region_count) +
           "; full_positive_regions=" +
           std::to_string(diagnostics.full_positive_region_count) +
           "; interface_fragments=" +
           std::to_string(diagnostics.interface_fragment_count) +
           "; curved_fragment_failures=" +
           std::to_string(diagnostics.curved_fragment_failure_count) +
           "; curved_failure_vertex_count=" +
           std::to_string(diagnostics.curved_fragment_vertex_count_failure) +
           "; curved_failure_seed=" +
           std::to_string(diagnostics.curved_fragment_seed_failure) +
           "; curved_failure_search_segment=" +
           std::to_string(diagnostics.curved_fragment_search_segment_failure) +
           "; curved_failure_root_solve=" +
           std::to_string(diagnostics.curved_fragment_root_solve_failure) +
           "; curved_failure_gradient=" +
           std::to_string(diagnostics.curved_fragment_gradient_failure) +
           "; curved_failure_weight=" +
           std::to_string(diagnostics.curved_fragment_weight_failure) +
           "; curved_edge_root_mismatches=" +
           std::to_string(diagnostics.curved_fragment_edge_root_mismatch_count) +
           "; curved_root_solve_edge_root_mismatches=" +
           std::to_string(
               diagnostics.curved_fragment_root_solve_edge_root_mismatch) +
           "; curved_boundary_degenerate=" +
           std::to_string(
               diagnostics.curved_fragment_boundary_degenerate_count);
    appendRootPolishDiagnostics(diagnostic,
                                diagnostics.root_finder_iteration_count);
    return diagnostic;
}

void recordRecursiveBackendDiagnostics(
    ImplicitCutQuadratureBackendCellResult& result,
    const SayeHyperrectangleDiagnostics& diagnostics,
    const interfaces::CutInterfaceDomainRequest& request)
{
    result.max_subdivision_depth_reached = diagnostics.max_depth_reached;
    result.subdivision_count = diagnostics.subdivision_count;
    result.full_negative_region_count = diagnostics.full_negative_region_count;
    result.full_positive_region_count = diagnostics.full_positive_region_count;
    result.linearized_leaf_count = diagnostics.linearized_leaf_count;
    result.interface_fragment_count = diagnostics.interface_fragment_count;
    result.curved_fragment_count = diagnostics.curved_fragment_count;
    result.root_branch_count = diagnostics.root_branch_count;
    result.root_finder_iteration_count =
        diagnostics.root_finder_iteration_count;

    const bool requested_unachieved_high_order =
        request.resolvedInterfaceQuadratureOrder() >
            result.achieved_interface_quadrature_order ||
        request.resolvedVolumeQuadratureOrder() >
            result.achieved_volume_quadrature_order;
    if (result.linearized_leaf_count > 0 && requested_unachieved_high_order) {
        result.fallback_used = true;
        result.requested_high_order_downgrade = true;
        result.fallback_reason =
            "requested high-order rule downgraded to terminal linearized leaves";
    }
}

[[nodiscard]] std::string enforceConservativeVolumeClosure(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    Real parent_measure)
{
    if (!(parent_measure > Real{0.0}) || cut.volume_regions.empty()) {
        return {};
    }

    Real total = Real{0.0};
    for (const auto& region : cut.volume_regions) {
        if (region.side == geometry::CutIntegrationSide::Negative ||
            region.side == geometry::CutIntegrationSide::Positive) {
            total += region.measure;
        }
    }
    if (!(total > Real{0.0}) || !std::isfinite(total)) {
        return {};
    }

    const Real validation_tolerance =
        std::max(request.tolerance, request.tolerance * parent_measure);
    const Real abs_error = std::abs(total - parent_measure);
    if (abs_error <= validation_tolerance) {
        return {};
    }

    const Real closure_tolerance =
        std::max(validation_tolerance, Real{1.0e-4} * parent_measure);
    if (abs_error > closure_tolerance) {
        return {};
    }

    const Real scale = parent_measure / total;
    if (!(scale > Real{0.0}) || !std::isfinite(scale)) {
        return {};
    }

    for (auto& region : cut.volume_regions) {
        if (region.side != geometry::CutIntegrationSide::Negative &&
            region.side != geometry::CutIntegrationSide::Positive) {
            continue;
        }
        region.measure *= scale;
        region.volume_fraction =
            std::clamp(region.measure / parent_measure, Real{0.0}, Real{1.0});
        for (auto& point : region.quadrature_points) {
            point.weight *= scale;
            point.reference_measure_factor *= scale;
        }
    }

    return "; conservative_volume_closure_scale=" + formatReal(scale) +
           "; conservative_volume_closure_abs_error=" + formatReal(abs_error) +
           "; conservative_volume_closure_tolerance=" +
           formatReal(closure_tolerance);
}

void appendDetailedBackendDiagnostics(
    ImplicitCutQuadratureBackendCellResult& result,
    const interfaces::CutInterfaceDomainRequest& request)
{
    if (result.cut.diagnostic.empty()) {
        return;
    }
    result.cut.diagnostic +=
        "; root_branches=" +
        std::to_string(result.root_branch_count);
    result.cut.diagnostic +=
        "; root_coordinate_tolerance=" +
        formatReal(request.implicit_cut_root_coordinate_tolerance) +
        "; root_max_iterations=" +
        std::to_string(request.implicit_cut_root_max_iterations);
    appendRootPolishDiagnostics(result.cut.diagnostic,
                                result.root_finder_iteration_count);
    result.cut.diagnostic +=
        "; curved_fragments=" +
        std::to_string(result.curved_fragment_count) +
        "; fallback_used=" +
        std::string(result.fallback_used ? "true" : "false") +
        "; high_order_downgrade=" +
        std::string(result.requested_high_order_downgrade ? "true" : "false") +
        "; selected_backend=" +
        implicitCutQuadratureBackendName(result.selected_backend) +
        "; requested_interface_order=" +
        std::to_string(result.requested_interface_quadrature_order) +
        "; requested_volume_order=" +
        std::to_string(result.requested_volume_quadrature_order) +
        "; possible_interface_order=" +
        std::to_string(result.possible_interface_quadrature_order) +
        "; possible_volume_order=" +
        std::to_string(result.possible_volume_quadrature_order) +
        "; achieved_interface_order=" +
        std::to_string(result.achieved_interface_quadrature_order) +
        "; achieved_volume_order=" +
        std::to_string(result.achieved_volume_quadrature_order) +
        "; verified_interface_order=" +
        std::to_string(result.verified_interface_quadrature_order) +
        "; verified_volume_order=" +
        std::to_string(result.verified_volume_quadrature_order) +
        "; status=" +
        implicitCutQuadratureDiagnosticStatusName(result.diagnostic_status);
    if (!result.fallback_reason.empty()) {
        result.cut.diagnostic +=
            "; fallback_reason=" + result.fallback_reason;
    }
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
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedInterfaceQuadratureOrder()),
                                      /*interface_order=*/true);
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedVolumeQuadratureOrder()),
                                      /*interface_order=*/false);
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        const auto backend_start = std::chrono::steady_clock::now();
        ImplicitCutQuadratureBackendCellResult result;
        result.selected_backend = kind();
        const int possible_interface_order =
            achievedInterfaceQuadratureOrder(
                mesh_dimension,
                input.linearized_input.element_type,
                request);
        const int possible_volume_order =
            achievedVolumeQuadratureOrder(
                mesh_dimension,
                input.linearized_input.element_type,
                request);
        setOrderMetadata(result, request,
                         possible_interface_order,
                         possible_volume_order,
                         possible_interface_order,
                         possible_volume_order,
                         possible_interface_order,
                         possible_volume_order);

        if (!supports(mesh_dimension, input.linearized_input.element_type)) {
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "SayeHyperrectangle implicit cut quadrature backend supports only quadrilateral cells in two dimensions and hexahedron cells in three dimensions";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }
        if (input.evaluator == nullptr) {
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "SayeHyperrectangle implicit cut quadrature backend requires a level-set evaluator";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }
        if (input.evaluator->interpolationOrder(
                input.linearized_input.parent_cell) <= 1) {
            if (mesh_dimension == 2) {
                result.cut =
                    interfaces::cutLinearLevelSetCell2D(
                        request, input.linearized_input);
                reduceOrderMetadataToGeneratedRules(result, request);
            } else {
                const Box3D root{
                    input.reference_min[0],
                    input.reference_max[0],
                    input.reference_min[1],
                    input.reference_max[1],
                    input.reference_min[2],
                    input.reference_max[2]};
                SayeHyperrectangleDiagnostics diagnostics;
                appendLinearizedBoxCut(
                    result.cut,
                    request,
                    input,
                    root,
                    boxMeasure(root),
                    diagnostics);
                reduceOrderMetadataToGeneratedRules(result, request);
                recordRecursiveBackendDiagnostics(result, diagnostics, request);
                result.cut.diagnostic =
                    formatSayeHyperrectangle3DDiagnostics(diagnostics, 0);
            }
            result.diagnostic_status =
                classifyCutStatus(result.cut, result.fallback_used);
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }

        const int max_depth =
            std::max(0, std::min(request.implicit_cut_max_subdivision_depth, 8));
        SayeHyperrectangleDiagnostics diagnostics;
        if (mesh_dimension == 2) {
            const Rectangle2D root{
                input.reference_min[0],
                input.reference_max[0],
                input.reference_min[1],
                input.reference_max[1]};
            appendAdaptiveRectangleCut(
                result.cut,
                request,
                input,
                root,
                rectangleMeasure(root),
                0,
                max_depth,
                diagnostics);
            if (request.resolvedInterfaceQuadratureOrder() > 1 &&
                diagnostics.interface_fragment_count > 0 &&
                diagnostics.curved_fragment_count +
                        diagnostics.curved_fragment_boundary_degenerate_count !=
                    diagnostics.interface_fragment_count) {
                result.achieved_interface_quadrature_order = 1;
                result.verified_interface_quadrature_order = 1;
            }
            reduceOrderMetadataToGeneratedRules(result, request);
            recordRecursiveBackendDiagnostics(result, diagnostics, request);
            result.cut.diagnostic =
                formatSayeHyperrectangleDiagnostics(diagnostics, max_depth);
        } else {
            const Box3D root{
                input.reference_min[0],
                input.reference_max[0],
                input.reference_min[1],
                input.reference_max[1],
                input.reference_min[2],
                input.reference_max[2]};
            appendAdaptiveBoxCut(
                result.cut,
                request,
                input,
                root,
                boxMeasure(root),
                0,
                max_depth,
                diagnostics);
            reduceOrderMetadataToGeneratedRules(result, request);
            recordRecursiveBackendDiagnostics(result, diagnostics, request);
            result.cut.diagnostic =
                formatSayeHyperrectangle3DDiagnostics(diagnostics, max_depth);
        }
        result.cut.supported = true;
        result.diagnostic_status =
            classifyCutStatus(result.cut, result.fallback_used);
        appendDetailedBackendDiagnostics(result, request);
        return finalizeBackendResult(std::move(result), request, backend_start);
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
        return supportsHighOrderSubcellMilestone(mesh_dimension, element_type);
    }

    [[nodiscard]] int achievedInterfaceQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedInterfaceQuadratureOrder()),
                                      /*interface_order=*/true);
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedVolumeQuadratureOrder()),
                                      /*interface_order=*/false);
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        const auto backend_start = std::chrono::steady_clock::now();
        ImplicitCutQuadratureBackendCellResult result;
        result.selected_backend = kind();
        const int possible_interface_order =
            achievedInterfaceQuadratureOrder(
                mesh_dimension,
                input.linearized_input.element_type,
                request);
        const int possible_volume_order =
            achievedVolumeQuadratureOrder(
                mesh_dimension,
                input.linearized_input.element_type,
                request);
        setOrderMetadata(result, request,
                         possible_interface_order,
                         possible_volume_order,
                         possible_interface_order,
                         possible_volume_order,
                         possible_interface_order,
                         possible_volume_order);

        if (!supports(mesh_dimension, input.linearized_input.element_type)) {
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "HighOrderSubcell implicit cut quadrature backend supports only triangular cells in two dimensions and tetrahedron cells in three dimensions";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }
        if (input.evaluator == nullptr) {
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "HighOrderSubcell implicit cut quadrature backend requires a level-set evaluator";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }
        if (input.evaluator->interpolationOrder(
                input.linearized_input.parent_cell) <= 1) {
            if (mesh_dimension == 2) {
                result.cut =
                    interfaces::cutLinearLevelSetCell2D(
                        request, input.linearized_input);
            } else {
                result.cut =
                    interfaces::cutLinearLevelSetCell3D(
                        request, input.linearized_input);
                result.achieved_interface_quadrature_order =
                    std::min(result.achieved_interface_quadrature_order, 1);
                result.verified_interface_quadrature_order =
                    std::min(result.verified_interface_quadrature_order, 1);
            }
            reduceOrderMetadataToGeneratedRules(result, request);
            result.diagnostic_status =
                classifyCutStatus(result.cut, result.fallback_used);
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }
        const std::size_t required_node_count = mesh_dimension == 2 ? 3u : 4u;
        if (input.linearized_input.node_coordinates.size() < required_node_count) {
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "HighOrderSubcell implicit cut quadrature backend requires simplex corner coordinates";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }

        const int max_depth =
            std::max(0, std::min(request.implicit_cut_max_subdivision_depth, 8));
        SayeHyperrectangleDiagnostics diagnostics;
        if (mesh_dimension == 2) {
            const Triangle2D root{
                input.linearized_input.node_coordinates[0],
                input.linearized_input.node_coordinates[1],
                input.linearized_input.node_coordinates[2]};
            appendAdaptiveTriangleCut(
                result.cut,
                request,
                input,
                root,
                triangleMeasure(root),
                0,
                max_depth,
                diagnostics);
            if (request.resolvedInterfaceQuadratureOrder() > 1 &&
                diagnostics.interface_fragment_count > 0 &&
                diagnostics.curved_fragment_count +
                        diagnostics.curved_fragment_boundary_degenerate_count !=
                    diagnostics.interface_fragment_count) {
                result.achieved_interface_quadrature_order = 1;
                result.verified_interface_quadrature_order = 1;
            }
            const auto volume_closure_diagnostic =
                enforceConservativeVolumeClosure(
                    result.cut, request, triangleMeasure(root));
            reduceOrderMetadataToGeneratedRules(result, request);
            recordRecursiveBackendDiagnostics(result, diagnostics, request);
            result.cut.diagnostic =
                formatHighOrderSubcellDiagnostics(diagnostics, max_depth) +
                volume_closure_diagnostic;
        } else {
            const Tetrahedron3D root{
                input.linearized_input.node_coordinates[0],
                input.linearized_input.node_coordinates[1],
                input.linearized_input.node_coordinates[2],
                input.linearized_input.node_coordinates[3]};
            appendAdaptiveTetrahedronCut(
                result.cut,
                request,
                input,
                root,
                tetrahedronMeasure(root),
                0,
                max_depth,
                diagnostics);
            if (request.resolvedInterfaceQuadratureOrder() > 1 &&
                diagnostics.interface_fragment_count > 0 &&
                diagnostics.curved_fragment_count +
                        diagnostics.curved_fragment_boundary_degenerate_count !=
                    diagnostics.interface_fragment_count) {
                result.achieved_interface_quadrature_order = 1;
                result.verified_interface_quadrature_order = 1;
            }
            const auto volume_closure_diagnostic =
                enforceConservativeVolumeClosure(
                    result.cut, request, tetrahedronMeasure(root));
            reduceOrderMetadataToGeneratedRules(result, request);
            recordRecursiveBackendDiagnostics(result, diagnostics, request);
            result.cut.diagnostic =
                formatHighOrderSubcellTetrahedronDiagnostics(diagnostics, max_depth) +
                volume_closure_diagnostic;
        }
        result.cut.supported = true;
        result.diagnostic_status =
            classifyCutStatus(result.cut, result.fallback_used);
        appendDetailedBackendDiagnostics(result, request);
        return finalizeBackendResult(std::move(result), request, backend_start);
    }
};

class AutoImplicitCutBackend final
    : public ImplicitCutQuadratureBackendDriver {
public:
    [[nodiscard]] ImplicitCutQuadratureBackend kind() const noexcept override {
        return ImplicitCutQuadratureBackend::Auto;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return implicitCutQuadratureBackendName(kind());
    }

    [[nodiscard]] bool supports(int mesh_dimension,
                                ElementType element_type) const noexcept override
    {
        ImplicitCutQuadratureBackend selected =
            ImplicitCutQuadratureBackend::LinearCorner;
        return selectAutoImplicitCutBackend(
            mesh_dimension, element_type, selected);
    }

    [[nodiscard]] int achievedInterfaceQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedInterfaceQuadratureOrder()),
                                      /*interface_order=*/true);
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        int mesh_dimension,
        ElementType element_type,
        const interfaces::CutInterfaceDomainRequest& request) const noexcept override
    {
        return capabilityLimitedOrder(kind(),
                                      mesh_dimension,
                                      element_type,
                                      std::max(0, request.resolvedVolumeQuadratureOrder()),
                                      /*interface_order=*/false);
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        const auto backend_start = std::chrono::steady_clock::now();
        ImplicitCutQuadratureBackend selected =
            ImplicitCutQuadratureBackend::LinearCorner;
        if (!selectAutoImplicitCutBackend(
                mesh_dimension, input.linearized_input.element_type, selected)) {
            ImplicitCutQuadratureBackendCellResult result{};
            result.selected_backend = kind();
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "Auto implicit cut quadrature backend cannot select a production "
                "backend for element type " +
                std::to_string(static_cast<unsigned>(
                    input.linearized_input.element_type)) +
                " in mesh dimension " + std::to_string(mesh_dimension) +
                "; supported production dispatch is quads/hexes to "
                "SayeHyperrectangle and triangles/tetrahedra to HighOrderSubcell";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Unsupported;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(std::move(result), request, backend_start);
        }

        auto result =
            implicitCutQuadratureBackendDriver(selected)
                .cut(mesh_dimension, request, input);
        result.selected_backend = selected;
        if (!result.cut.diagnostic.empty()) {
            result.cut.diagnostic =
                std::string("Auto selected_backend=") +
                implicitCutQuadratureBackendName(selected) + "; " +
                result.cut.diagnostic;
        }
        return result;
    }
};

class MomentFitImplicitCutBackend final
    : public ImplicitCutQuadratureBackendDriver {
public:
    [[nodiscard]] ImplicitCutQuadratureBackend kind() const noexcept override {
        return ImplicitCutQuadratureBackend::MomentFit;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return implicitCutQuadratureBackendName(kind());
    }

    [[nodiscard]] bool supports(int /*mesh_dimension*/,
                                ElementType /*element_type*/) const noexcept override
    {
        return false;
    }

    [[nodiscard]] int achievedInterfaceQuadratureOrder(
        int /*mesh_dimension*/,
        ElementType /*element_type*/,
        const interfaces::CutInterfaceDomainRequest& /*request*/) const noexcept override
    {
        return -1;
    }

    [[nodiscard]] int achievedVolumeQuadratureOrder(
        int /*mesh_dimension*/,
        ElementType /*element_type*/,
        const interfaces::CutInterfaceDomainRequest& /*request*/) const noexcept override
    {
        return -1;
    }

    [[nodiscard]] ImplicitCutQuadratureBackendCellResult cut(
        int mesh_dimension,
        const interfaces::CutInterfaceDomainRequest& request,
        const ImplicitCutQuadratureBackendCellInput& input) const override
    {
        const auto backend_start = std::chrono::steady_clock::now();
        ImplicitCutQuadratureBackendCellResult result{};
        result.selected_backend = kind();
        setUnavailableOrderMetadata(result, request);
        result.fallback_reason = "MomentFit backend unavailable";
        result.cut.supported = false;
        result.cut.degeneracy = interfaces::CutInterfaceDegeneracy::NoCut;
        result.cut.diagnostic =
            "MomentFit implicit cut quadrature backend capability_state=" +
            std::string(implicitCutQuadratureBackendQualificationName(
                ImplicitCutQuadratureBackendQualification::Unavailable)) +
            "; experimental=false; production_qualified=false"
            "; reason=moment system construction, conditioning diagnostics, "
            "positive-weight policy, and exactness tests are not implemented"
            "; mesh_dimension=" +
            std::to_string(mesh_dimension) +
            "; element_type=" +
            std::to_string(static_cast<unsigned>(
                input.linearized_input.element_type));
        result.diagnostic_status =
            ImplicitCutQuadratureDiagnosticStatus::Unsupported;
        appendDetailedBackendDiagnostics(result, request);
        return finalizeBackendResult(std::move(result), request, backend_start);
    }
};

[[nodiscard]] bool supportsSayeHyperrectangleMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept
{
    if (mesh_dimension == 2) {
        switch (element_type) {
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            return true;
        default:
            return false;
        }
    }
    if (mesh_dimension == 3) {
        switch (element_type) {
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return true;
        default:
            return false;
        }
    }
    return false;
}

[[nodiscard]] bool supportsHighOrderSubcellMilestone(
    int mesh_dimension,
    ElementType element_type) noexcept
{
    if (mesh_dimension == 2) {
        switch (element_type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return true;
        default:
            return false;
        }
    }
    if (mesh_dimension == 3) {
        switch (element_type) {
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return true;
        default:
            return false;
        }
    }
    return false;
}

[[nodiscard]] bool selectAutoImplicitCutBackend(
    int mesh_dimension,
    ElementType element_type,
    ImplicitCutQuadratureBackend& selected) noexcept
{
    if (supportsSayeHyperrectangleMilestone(mesh_dimension, element_type)) {
        selected = ImplicitCutQuadratureBackend::SayeHyperrectangle;
        return true;
    }
    if (supportsHighOrderSubcellMilestone(mesh_dimension, element_type)) {
        selected = ImplicitCutQuadratureBackend::HighOrderSubcell;
        return true;
    }
    return false;
}

} // namespace

const ImplicitCutQuadratureBackendDriver&
implicitCutQuadratureBackendDriver(ImplicitCutQuadratureBackend backend)
{
    static const LinearCornerImplicitCutBackend linear_corner_backend;
    static const SayeHyperrectangleImplicitCutBackend saye_hyperrectangle_backend;
    static const HighOrderSubcellImplicitCutBackend high_order_subcell_backend;
    static const AutoImplicitCutBackend auto_backend;
    static const MomentFitImplicitCutBackend moment_fit_backend;

    switch (backend) {
    case ImplicitCutQuadratureBackend::LinearCorner:
        return linear_corner_backend;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        return saye_hyperrectangle_backend;
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
        return high_order_subcell_backend;
    case ImplicitCutQuadratureBackend::MomentFit:
        return moment_fit_backend;
    case ImplicitCutQuadratureBackend::Auto:
        return auto_backend;
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
        capability.qualification =
            ImplicitCutQuadratureBackendQualification::ProductionQualified;
        capability.qualification_diagnostic =
            "LinearCorner is production-qualified for linear corner cuts";
        capability.supports_element_type =
            (mesh_dimension == 2 &&
             interfaces::supportsLinearLevelSetCellCut2D(element_type)) ||
            (mesh_dimension == 3 &&
             interfaces::supportsLinearLevelSetCellCut3D(element_type));
        capability.supports_high_order_geometry = false;
        capability.validation_level_set_order = 1;
        if (!capability.supports_element_type) {
            capability.maximum_reported_interface_order = -1;
            capability.maximum_reported_volume_order = -1;
            return capability;
        }
        capability.maximum_reported_interface_order =
            mesh_dimension == 2 ? 5 : 1;
        capability.maximum_reported_volume_order =
            mesh_dimension == 2 ? 5 : 2;
        return capability;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        capability.implemented = true;
        capability.supports_element_type =
            supportsSayeHyperrectangleMilestone(mesh_dimension, element_type);
        capability.supports_high_order_geometry = true;
        if (capability.supports_element_type && mesh_dimension == 2) {
            capability.qualification =
                ImplicitCutQuadratureBackendQualification::ProductionQualified;
            capability.qualification_diagnostic =
                "SayeHyperrectangle is production-qualified for 2D quadrilateral "
                "high-order generated-interface cut quadrature";
        } else {
            capability.qualification =
                ImplicitCutQuadratureBackendQualification::Experimental;
            capability.qualification_diagnostic =
                "SayeHyperrectangle remains experimental outside the qualified "
                "2D quadrilateral path; current production qualification is "
                "blocked by terminal linearized volume leaves and missing "
                "qualified 3D hyperrectangle height-function rules";
        }
        if (!capability.supports_element_type) {
            capability.maximum_reported_interface_order = -1;
            capability.maximum_reported_volume_order = -1;
            return capability;
        }
        capability.maximum_reported_interface_order =
            mesh_dimension == 2 ? 5 : 1;
        capability.maximum_reported_volume_order =
            mesh_dimension == 2 ? 5 : 2;
        return capability;
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
        capability.implemented = true;
        capability.qualification =
            ImplicitCutQuadratureBackendQualification::Experimental;
        capability.qualification_diagnostic =
            "HighOrderSubcell is an experimental high-order milestone backend; "
            "current production qualification is blocked by terminal linearized "
            "volume subcells and incomplete 3D curved simplex qualification";
        capability.supports_element_type =
            supportsHighOrderSubcellMilestone(mesh_dimension, element_type);
        capability.supports_high_order_geometry = true;
        if (!capability.supports_element_type) {
            capability.maximum_reported_interface_order = -1;
            capability.maximum_reported_volume_order = -1;
            return capability;
        }
        capability.maximum_reported_interface_order =
            mesh_dimension == 2 ? 5 : (mesh_dimension == 3 ? 2 : 1);
        capability.maximum_reported_volume_order =
            mesh_dimension == 2 ? 5 : 2;
        return capability;
    case ImplicitCutQuadratureBackend::MomentFit:
        capability.implemented = false;
        capability.qualification =
            ImplicitCutQuadratureBackendQualification::Unavailable;
        capability.qualification_diagnostic =
            "MomentFit is unavailable: moment system construction, conditioning "
            "diagnostics, positive-weight policy, and exactness tests are not "
            "implemented or production-qualified";
        capability.supports_element_type = false;
        capability.supports_high_order_geometry = true;
        capability.maximum_reported_interface_order = -1;
        capability.maximum_reported_volume_order = -1;
        return capability;
    case ImplicitCutQuadratureBackend::Auto: {
        capability.implemented = true;
        capability.qualification =
            ImplicitCutQuadratureBackendQualification::Experimental;
        capability.qualification_diagnostic =
            "Auto delegates to the selected supported backend and inherits its "
            "qualification state";
        capability.supports_high_order_geometry = false;
        ImplicitCutQuadratureBackend selected =
            ImplicitCutQuadratureBackend::LinearCorner;
        if (!selectAutoImplicitCutBackend(mesh_dimension, element_type, selected)) {
            capability.supports_element_type = false;
            capability.maximum_reported_interface_order = -1;
            capability.maximum_reported_volume_order = -1;
            return capability;
        }
        const auto selected_capability =
            implicitCutQuadratureBackendCapability(
                selected, mesh_dimension, element_type);
        capability.supports_element_type =
            selected_capability.supports_element_type;
        capability.supports_high_order_geometry =
            selected_capability.supports_high_order_geometry;
        capability.requires_scalar_h1_c0_level_set =
            selected_capability.requires_scalar_h1_c0_level_set;
        capability.minimum_level_set_order =
            selected_capability.minimum_level_set_order;
        capability.validation_level_set_order =
            selected_capability.validation_level_set_order;
        capability.returns_reference_frame_rules =
            selected_capability.returns_reference_frame_rules;
        capability.requires_positive_volume_weights =
            selected_capability.requires_positive_volume_weights;
        capability.requires_deterministic_rule_order =
            selected_capability.requires_deterministic_rule_order;
        capability.prunes_tiny_slivers_in_context =
            selected_capability.prunes_tiny_slivers_in_context;
        capability.near_tangent_requires_diagnostic =
            selected_capability.near_tangent_requires_diagnostic;
        capability.tiny_sliver_volume_fraction =
            selected_capability.tiny_sliver_volume_fraction;
        capability.maximum_reported_interface_order =
            selected_capability.maximum_reported_interface_order;
        capability.maximum_reported_volume_order =
            selected_capability.maximum_reported_volume_order;
        capability.qualification = selected_capability.qualification;
        capability.qualification_diagnostic =
            std::string("Auto selects ") +
            implicitCutQuadratureBackendName(selected) + ": " +
            selected_capability.qualification_diagnostic;
        return capability;
    }
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

const char* implicitCutQuadratureBackendQualificationName(
    ImplicitCutQuadratureBackendQualification qualification) noexcept
{
    switch (qualification) {
    case ImplicitCutQuadratureBackendQualification::Unavailable:
        return "Unavailable";
    case ImplicitCutQuadratureBackendQualification::Experimental:
        return "Experimental";
    case ImplicitCutQuadratureBackendQualification::ProductionQualified:
        return "ProductionQualified";
    }
    return "Unavailable";
}

ImplicitCutQuadratureBackendValidation
validateImplicitCutQuadratureBackendCellResult(
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const ImplicitCutQuadratureBackendCellResult& result)
{
    const auto& linearized_input = input.linearized_input;
    if (!(request.implicit_cut_root_tolerance > Real{0.0}) ||
        !(request.implicit_cut_root_coordinate_tolerance > Real{0.0}) ||
        request.implicit_cut_root_max_iterations <= 0) {
        return failedValidation(
            ImplicitCutQuadratureDiagnosticStatus::Failed,
            "implicit cut backend request has invalid root-polishing controls");
    }
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
        Real interface_weight_sum = Real{0.0};
        for (const auto& point : fragment.quadrature_points) {
            geometry::CutQuadraturePoint qp;
            qp.point = point.point;
            qp.normal = point.normal;
            qp.weight = point.weight;
            qp.parent_coordinate = point.parent_coordinate;
            qp.reference_measure_factor = point.reference_measure_factor;
            qp.level_set_residual = point.level_set_residual;
            qp.gradient_norm = point.gradient_norm;
            if (!finitePoint(qp) || qp.weight <= Real{0.0}) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned an invalid interface quadrature point");
            }
            interface_weight_sum += point.weight;
            if (fragment.root_polished) {
                const Real residual_tolerance =
                    Real{10.0} * rootResidualTolerance(request);
                if (point.level_set_residual > residual_tolerance ||
                    point.gradient_norm <= Real{0.0}) {
                    return failedValidation(
                        ImplicitCutQuadratureDiagnosticStatus::Failed,
                        "implicit cut backend returned an invalid root-polished interface quadrature point");
                }
            }
            const auto normal_norm = norm3(point.normal);
            if (normal_norm <= Real{1.0e-30}) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned a zero interface quadrature normal");
            }
            const auto evaluation =
                input.evaluator->evaluate(fragment.parent_cell, point.point);
            const auto gradient_norm = norm3(evaluation.reference_gradient);
            if (finiteArray(evaluation.reference_gradient) &&
                gradient_norm > Real{1.0e-30}) {
                const Real alignment =
                    dot3(point.normal, evaluation.reference_gradient) /
                    (normal_norm * gradient_norm);
                if (!std::isfinite(alignment) || alignment < Real{0.0}) {
                    return failedValidation(
                        ImplicitCutQuadratureDiagnosticStatus::Failed,
                        "implicit cut backend returned an inconsistent interface quadrature normal");
                }
            }
        }
        if (!fragment.quadrature_points.empty() &&
            fragment.measure > Real{0.0} &&
            std::abs(interface_weight_sum - fragment.measure) >
                measureTolerance(request.tolerance, fragment.measure)) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend interface quadrature weights do not sum to the fragment measure");
        }
        if (fragment.root_polished && fragment.active()) {
            const Real residual_tolerance =
                Real{10.0} * rootResidualTolerance(request);
            if (!std::isfinite(fragment.max_root_residual) ||
                fragment.max_root_residual > residual_tolerance ||
                !std::isfinite(fragment.min_gradient_norm) ||
                fragment.min_gradient_norm <= Real{0.0}) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned invalid root-polished interface fragment metadata");
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
        if (region.parent_measure > Real{0.0} &&
            std::abs(region.measure -
                     region.parent_measure * region.volume_fraction) >
                measureTolerance(request.tolerance, region.parent_measure)) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend volume region measure is inconsistent with its volume fraction");
        }
        Real volume_weight_sum = Real{0.0};
        for (const auto& point : region.quadrature_points) {
            if (!finitePoint(point) || point.weight <= Real{0.0}) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned an invalid volume quadrature point");
            }
            volume_weight_sum += point.weight;
        }
        if (!region.quadrature_points.empty() &&
            region.measure > Real{0.0} &&
            std::abs(volume_weight_sum - region.measure) >
                measureTolerance(request.tolerance, region.measure)) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend volume quadrature weights do not sum to the region measure");
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
                "implicit cut backend volume measures do not sum to the parent measure"
                "; total=" + formatReal(total) +
                "; parent=" + formatReal(parent_measure) +
                "; negative=" + formatReal(negative_measure) +
                "; positive=" + formatReal(positive_measure) +
                "; abs_error=" + formatReal(std::abs(total - parent_measure)) +
                "; tolerance=" + formatReal(tolerance));
        }
    }

    return ImplicitCutQuadratureBackendValidation{
        .ok = true,
        .status = status,
        .diagnostic = result.cut.diagnostic};
}

} // namespace svmp::FE::level_set
