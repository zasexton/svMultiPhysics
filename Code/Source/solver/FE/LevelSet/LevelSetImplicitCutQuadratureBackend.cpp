#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include "Basis/BasisTraits.h"
#include "Basis/NodeOrderingConventions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <optional>
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

[[nodiscard]] std::string formatPoint(const std::array<Real, 3>& point)
{
    return "(" + formatReal(point[0]) + "," + formatReal(point[1]) + "," +
           formatReal(point[2]) + ")";
}

[[nodiscard]] std::string formatPointList(
    const std::vector<std::array<Real, 3>>& points)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0u; i < points.size(); ++i) {
        if (i > 0u) {
            out << ",";
        }
        out << formatPoint(points[i]);
    }
    out << "]";
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

void alignLeafCutNormalsWithEvaluator(
    interfaces::LevelSetCellCutResult& leaf_cut,
    const ImplicitCutQuadratureBackendCellInput& input);

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
        if (result.cut.supported && input.evaluator != nullptr &&
            !input.high_order_sample_points.empty()) {
            alignLeafCutNormalsWithEvaluator(result.cut, input);
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

struct OutwardInterval {
    long double lower{0.0L};
    long double upper{0.0L};
};

[[nodiscard]] long double roundDown(long double value) noexcept
{
    return std::nextafter(value,
                          -std::numeric_limits<long double>::infinity());
}

[[nodiscard]] long double roundUp(long double value) noexcept
{
    return std::nextafter(value,
                          std::numeric_limits<long double>::infinity());
}

[[nodiscard]] OutwardInterval pointInterval(Real value) noexcept
{
    const auto exact = static_cast<long double>(value);
    return OutwardInterval{exact, exact};
}

[[nodiscard]] OutwardInterval addIntervals(const OutwardInterval& lhs,
                                            const OutwardInterval& rhs) noexcept
{
    return OutwardInterval{roundDown(lhs.lower + rhs.lower),
                           roundUp(lhs.upper + rhs.upper)};
}

[[nodiscard]] OutwardInterval subtractIntervals(
    const OutwardInterval& lhs,
    const OutwardInterval& rhs) noexcept
{
    return OutwardInterval{roundDown(lhs.lower - rhs.upper),
                           roundUp(lhs.upper - rhs.lower)};
}

[[nodiscard]] OutwardInterval multiplyIntervals(
    const OutwardInterval& lhs,
    const OutwardInterval& rhs) noexcept
{
    const std::array<long double, 4> products{{
        lhs.lower * rhs.lower,
        lhs.lower * rhs.upper,
        lhs.upper * rhs.lower,
        lhs.upper * rhs.upper,
    }};
    const auto [minimum, maximum] =
        std::minmax_element(products.begin(), products.end());
    return OutwardInterval{roundDown(*minimum), roundUp(*maximum)};
}

[[nodiscard]] OutwardInterval divideIntervals(
    const OutwardInterval& numerator,
    const OutwardInterval& denominator)
{
    if (denominator.lower <= 0.0L && denominator.upper >= 0.0L) {
        throw std::invalid_argument(
            "certified tensor polynomial interval division crossed zero");
    }
    const std::array<long double, 4> quotients{{
        numerator.lower / denominator.lower,
        numerator.lower / denominator.upper,
        numerator.upper / denominator.lower,
        numerator.upper / denominator.upper,
    }};
    const auto [minimum, maximum] =
        std::minmax_element(quotients.begin(), quotients.end());
    return OutwardInterval{roundDown(*minimum), roundUp(*maximum)};
}

[[nodiscard]] OutwardInterval rationalInterval(long double numerator,
                                                long double denominator)
{
    return divideIntervals(OutwardInterval{numerator, numerator},
                           OutwardInterval{denominator, denominator});
}

[[nodiscard]] OutwardInterval scaleInterval(
    const OutwardInterval& value,
    long double numerator,
    long double denominator = 1.0L)
{
    return multiplyIntervals(
        value, rationalInterval(numerator, denominator));
}

[[nodiscard]] std::vector<OutwardInterval> nodalLineToBernstein(
    const std::vector<OutwardInterval>& nodal,
    int order)
{
    if (order == 2 && nodal.size() == 3u) {
        return {
            nodal[0],
            addIntervals(
                scaleInterval(nodal[1], 2.0L),
                addIntervals(scaleInterval(nodal[0], -1.0L, 2.0L),
                             scaleInterval(nodal[2], -1.0L, 2.0L))),
            nodal[2],
        };
    }
    if (order == 3 && nodal.size() == 4u) {
        return {
            nodal[0],
            addIntervals(
                addIntervals(scaleInterval(nodal[0], -5.0L, 6.0L),
                             scaleInterval(nodal[1], 3.0L)),
                addIntervals(scaleInterval(nodal[2], -3.0L, 2.0L),
                             scaleInterval(nodal[3], 1.0L, 3.0L))),
            addIntervals(
                addIntervals(scaleInterval(nodal[0], 1.0L, 3.0L),
                             scaleInterval(nodal[1], -3.0L, 2.0L)),
                addIntervals(scaleInterval(nodal[2], 3.0L),
                             scaleInterval(nodal[3], -5.0L, 6.0L))),
            nodal[3],
        };
    }
    throw std::invalid_argument(
        "certified tensor polynomial range supports only Q2 and Q3 lines");
}

struct TensorBernsteinCertificate {
    int dimension{0};
    int order{0};
    std::size_t axis_size{0u};
    std::array<Real, 3> reference_min{{0.0, 0.0, 0.0}};
    std::array<Real, 3> reference_max{{0.0, 0.0, 0.0}};
    std::vector<OutwardInterval> root_control_values{};
};

[[nodiscard]] std::size_t integerPower(std::size_t base, int exponent) noexcept
{
    std::size_t value = 1u;
    for (int i = 0; i < exponent; ++i) {
        value *= base;
    }
    return value;
}

void transformTensorAxisToBernstein(
    std::vector<OutwardInterval>& values,
    int dimension,
    int order,
    int axis)
{
    const auto n = static_cast<std::size_t>(order + 1);
    const auto stride = integerPower(n, axis);
    const auto block = stride * n;
    const auto total = integerPower(n, dimension);
    auto transformed = values;
    for (std::size_t base = 0u; base < total; base += block) {
        for (std::size_t inner = 0u; inner < stride; ++inner) {
            std::vector<OutwardInterval> line;
            line.reserve(n);
            for (std::size_t i = 0u; i < n; ++i) {
                line.push_back(values[base + inner + i * stride]);
            }
            const auto bernstein = nodalLineToBernstein(line, order);
            for (std::size_t i = 0u; i < n; ++i) {
                transformed[base + inner + i * stride] = bernstein[i];
            }
        }
    }
    values = std::move(transformed);
}

[[nodiscard]] std::optional<TensorBernsteinCertificate>
makeTensorBernsteinCertificate(
    int mesh_dimension,
    const ImplicitCutQuadratureBackendCellInput& input)
{
    if (input.evaluator == nullptr ||
        (mesh_dimension != 2 && mesh_dimension != 3)) {
        return std::nullopt;
    }
    const auto cell_id = input.linearized_input.parent_cell;
    const int order = input.evaluator->interpolationOrder(cell_id);
    if (order != 2 && order != 3) {
        return std::nullopt;
    }

    const auto element_type = input.linearized_input.element_type;
    const ElementType canonical_type =
        mesh_dimension == 2 && basis::is_quadrilateral(element_type)
            ? ElementType::Quad4
            : mesh_dimension == 3 && basis::is_hexahedron(element_type)
                  ? ElementType::Hex8
                  : ElementType::Unknown;
    if (canonical_type == ElementType::Unknown) {
        return std::nullopt;
    }
    const auto nodes = basis::ReferenceNodeLayout::get_lagrange_node_coords(
        canonical_type, order);
    const auto coefficients =
        input.evaluator->gatherCellCoefficients(cell_id);
    const bool coefficients_are_complete_tensor_nodes =
        input.evaluator->usesCompleteTensorLagrangeBasis(cell_id);
    const bool coefficients_use_tensor_serendipity =
        input.evaluator->usesTensorSerendipityBasis(cell_id);
    if (!coefficients_are_complete_tensor_nodes &&
        !coefficients_use_tensor_serendipity) {
        return std::nullopt;
    }
    const auto n = static_cast<std::size_t>(order + 1);
    const auto expected = integerPower(n, mesh_dimension);
    if (nodes.size() != expected ||
        (coefficients_are_complete_tensor_nodes &&
         coefficients.size() != expected)) {
        throw std::invalid_argument(
            "certified tensor polynomial range found an inconsistent complete Lagrange control net");
    }

    long double coefficient_scale = 1.0L;
    for (const Real coefficient : coefficients) {
        coefficient_scale += std::abs(static_cast<long double>(coefficient));
    }
    const long double evaluated_node_uncertainty = roundUp(
        4096.0L * std::numeric_limits<Real>::epsilon() *
        coefficient_scale);

    std::vector<OutwardInterval> tensor_values(expected);
    std::vector<unsigned char> assigned(expected, 0u);
    constexpr Real coordinate_tolerance = Real{256.0} *
                                          std::numeric_limits<Real>::epsilon();
    for (std::size_t local = 0u; local < nodes.size(); ++local) {
        std::array<std::size_t, 3> index{{0u, 0u, 0u}};
        for (int axis = 0; axis < mesh_dimension; ++axis) {
            const Real scaled =
                (nodes[local][static_cast<std::size_t>(axis)] + Real{1.0}) *
                static_cast<Real>(order) / Real{2.0};
            const auto nearest = static_cast<long long>(std::llround(scaled));
            if (nearest < 0 || nearest > order ||
                std::abs(scaled - static_cast<Real>(nearest)) >
                    coordinate_tolerance) {
                throw std::invalid_argument(
                    "certified tensor polynomial range found a non-equispaced Lagrange node");
            }
            index[static_cast<std::size_t>(axis)] =
                static_cast<std::size_t>(nearest);
        }
        const auto flat = index[0] + n * (index[1] + n * index[2]);
        if (flat >= expected || assigned[flat] != 0u) {
            throw std::invalid_argument(
                "certified tensor polynomial range found duplicate tensor nodes");
        }
        if (coefficients_are_complete_tensor_nodes) {
            tensor_values[flat] = pointInterval(coefficients[local]);
        } else {
            const auto value = input.evaluator
                                   ->evaluate(
                                       cell_id,
                                       {{nodes[local][0],
                                         nodes[local][1],
                                         nodes[local][2]}})
                                   .value;
            const auto extended = static_cast<long double>(value);
            tensor_values[flat] = OutwardInterval{
                roundDown(extended - evaluated_node_uncertainty),
                roundUp(extended + evaluated_node_uncertainty)};
        }
        assigned[flat] = 1u;
    }
    if (std::find(assigned.begin(), assigned.end(), 0u) != assigned.end()) {
        throw std::invalid_argument(
            "certified tensor polynomial range found an incomplete tensor node set");
    }

    for (int axis = 0; axis < mesh_dimension; ++axis) {
        transformTensorAxisToBernstein(
            tensor_values, mesh_dimension, order, axis);
    }
    const auto isovalue = pointInterval(input.isovalue);
    for (auto& value : tensor_values) {
        value = subtractIntervals(value, isovalue);
    }
    return TensorBernsteinCertificate{
        .dimension = mesh_dimension,
        .order = order,
        .axis_size = n,
        .reference_min = input.reference_min,
        .reference_max = input.reference_max,
        .root_control_values = std::move(tensor_values),
    };
}

struct BernsteinSplitLine {
    std::vector<OutwardInterval> left{};
    std::vector<OutwardInterval> right{};
};

[[nodiscard]] BernsteinSplitLine splitBernsteinLine(
    const std::vector<OutwardInterval>& control,
    const OutwardInterval& parameter)
{
    const auto n = control.size();
    if (n == 0u) {
        return {};
    }
    auto work = control;
    BernsteinSplitLine split;
    split.left.resize(n);
    split.right.resize(n);
    split.left[0] = work[0];
    split.right[n - 1u] = work[n - 1u];
    const auto one_minus_parameter =
        subtractIntervals(OutwardInterval{1.0L, 1.0L}, parameter);
    for (std::size_t level = 1u; level < n; ++level) {
        for (std::size_t i = 0u; i + level < n; ++i) {
            work[i] = addIntervals(
                multiplyIntervals(one_minus_parameter, work[i]),
                multiplyIntervals(parameter, work[i + 1u]));
        }
        split.left[level] = work[0];
        split.right[n - level - 1u] = work[n - level - 1u];
    }
    return split;
}

void restrictTensorControlAxis(
    std::vector<OutwardInterval>& values,
    const TensorBernsteinCertificate& certificate,
    int axis,
    Real lower,
    Real upper)
{
    const auto n = certificate.axis_size;
    const auto stride = integerPower(n, axis);
    const auto block = stride * n;
    const auto total = integerPower(n, certificate.dimension);
    const Real root_lower =
        certificate.reference_min[static_cast<std::size_t>(axis)];
    const Real root_upper =
        certificate.reference_max[static_cast<std::size_t>(axis)];
    const auto root_span = subtractIntervals(pointInterval(root_upper),
                                             pointInterval(root_lower));
    const auto upper_parameter = divideIntervals(
        subtractIntervals(pointInterval(upper), pointInterval(root_lower)),
        root_span);
    const auto lower_parameter_on_left = divideIntervals(
        subtractIntervals(pointInterval(lower), pointInterval(root_lower)),
        subtractIntervals(pointInterval(upper), pointInterval(root_lower)));

    auto restricted = values;
    for (std::size_t base = 0u; base < total; base += block) {
        for (std::size_t inner = 0u; inner < stride; ++inner) {
            std::vector<OutwardInterval> line;
            line.reserve(n);
            for (std::size_t i = 0u; i < n; ++i) {
                line.push_back(values[base + inner + i * stride]);
            }
            if (upper != root_upper) {
                line = splitBernsteinLine(line, upper_parameter).left;
            }
            if (lower != root_lower) {
                line = splitBernsteinLine(
                           line, lower_parameter_on_left)
                           .right;
            }
            for (std::size_t i = 0u; i < n; ++i) {
                restricted[base + inner + i * stride] = line[i];
            }
        }
    }
    values = std::move(restricted);
}

[[nodiscard]] OutwardInterval certifiedRectangleRange(
    const TensorBernsteinCertificate& certificate,
    const Rectangle2D& rect)
{
    auto control = certificate.root_control_values;
    restrictTensorControlAxis(
        control, certificate, 0, rect.xmin, rect.xmax);
    restrictTensorControlAxis(
        control, certificate, 1, rect.ymin, rect.ymax);
    OutwardInterval range{std::numeric_limits<long double>::infinity(),
                          -std::numeric_limits<long double>::infinity()};
    for (const auto& value : control) {
        range.lower = std::min(range.lower, value.lower);
        range.upper = std::max(range.upper, value.upper);
    }
    return range;
}

[[nodiscard]] OutwardInterval certifiedBoxRange(
    const TensorBernsteinCertificate& certificate,
    const Box3D& box)
{
    auto control = certificate.root_control_values;
    restrictTensorControlAxis(
        control, certificate, 0, box.xmin, box.xmax);
    restrictTensorControlAxis(
        control, certificate, 1, box.ymin, box.ymax);
    restrictTensorControlAxis(
        control, certificate, 2, box.zmin, box.zmax);
    OutwardInterval range{std::numeric_limits<long double>::infinity(),
                          -std::numeric_limits<long double>::infinity()};
    for (const auto& value : control) {
        range.lower = std::min(range.lower, value.lower);
        range.upper = std::max(range.upper, value.upper);
    }
    return range;
}

struct CertifiedRegularGraph {
    int axis{-1};
    bool lower_face_zero{false};
    bool upper_face_zero{false};
    bool coordinatewise_monotone{false};
};

[[nodiscard]] long double certifiedControlRoundoffZeroBound(
    const std::vector<OutwardInterval>& control) noexcept
{
    // This factor exceeds the scalar-operation count of the supported Q3
    // transform and two-axis restriction while remaining far below Real root
    // tolerances at unit scale.
    constexpr long double operation_roundoff_factor = 16384.0L;
    long double control_scale = 1.0L;
    for (const auto& value : control) {
        control_scale = std::max(
            control_scale,
            std::max(std::abs(value.lower), std::abs(value.upper)));
    }
    return roundUp(
        operation_roundoff_factor *
        std::numeric_limits<long double>::epsilon() *
        control_scale);
}

[[nodiscard]] bool isCertifiedRoundoffZero(
    const OutwardInterval& value,
    long double zero_bound) noexcept
{
    return value.lower >= -zero_bound && value.upper <= zero_bound;
}

[[nodiscard]] bool certifiedFaceEndpointTouch(
    const std::vector<OutwardInterval>& control,
    const TensorBernsteinCertificate& certificate,
    int graph_axis,
    bool lower_face,
    bool nonnegative_face,
    Real root_tolerance,
    long double roundoff_zero_bound)
{
    if (certificate.dimension != 2) {
        return false;
    }
    const auto n = certificate.axis_size;
    const int tangent_axis = 1 - graph_axis;
    std::vector<OutwardInterval> face;
    face.reserve(n);
    for (std::size_t tangent = 0u; tangent < n; ++tangent) {
        std::array<std::size_t, 2> index{{0u, 0u}};
        index[static_cast<std::size_t>(graph_axis)] =
            lower_face ? 0u : n - 1u;
        index[static_cast<std::size_t>(tangent_axis)] = tangent;
        const auto flat = index[0] + n * index[1];
        const auto& value = control[flat];
        face.push_back(nonnegative_face
                           ? value
                           : OutwardInterval{-value.upper, -value.lower});
    }

    for (const auto& value : face) {
        if (value.lower < 0.0L &&
            !isCertifiedRoundoffZero(value, roundoff_zero_bound)) {
            return false;
        }
    }

    const auto tolerance = static_cast<long double>(root_tolerance);
    for (const std::size_t zero_endpoint :
         {std::size_t{0u}, n - 1u}) {
        if (!isCertifiedRoundoffZero(
                face[zero_endpoint], roundoff_zero_bound)) {
            continue;
        }
        const std::size_t opposite_endpoint =
            zero_endpoint == 0u ? n - 1u : 0u;
        if (face[opposite_endpoint].lower <= tolerance) {
            continue;
        }
        const bool zero_endpoint_is_upper = zero_endpoint == n - 1u;
        bool strict_step = false;
        bool monotone_toward_zero = true;
        for (std::size_t i = 0u; i + 1u < n; ++i) {
            const auto difference =
                subtractIntervals(face[i + 1u], face[i]);
            if (zero_endpoint_is_upper) {
                monotone_toward_zero =
                    difference.upper <= 0.0L ||
                    isCertifiedRoundoffZero(
                        difference, roundoff_zero_bound);
                strict_step = strict_step ||
                    difference.upper < -roundoff_zero_bound;
            } else {
                monotone_toward_zero =
                    difference.lower >= 0.0L ||
                    isCertifiedRoundoffZero(
                        difference, roundoff_zero_bound);
                strict_step = strict_step ||
                    difference.lower > roundoff_zero_bound;
            }
            if (!monotone_toward_zero) {
                break;
            }
        }
        if (monotone_toward_zero && strict_step) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::optional<CertifiedRegularGraph> certifiedRegularGraph(
    const TensorBernsteinCertificate& certificate,
    const std::array<Real, 3>& lower,
    const std::array<Real, 3>& upper,
    Real root_tolerance)
{
    auto control = certificate.root_control_values;
    for (int axis = 0; axis < certificate.dimension; ++axis) {
        restrictTensorControlAxis(
            control,
            certificate,
            axis,
            lower[static_cast<std::size_t>(axis)],
            upper[static_cast<std::size_t>(axis)]);
    }

    const auto n = certificate.axis_size;
    const auto total = integerPower(n, certificate.dimension);
    const long double roundoff_zero_bound =
        certifiedControlRoundoffZeroBound(control);
    std::optional<CertifiedRegularGraph> face_bracketed_graph;
    int strictly_monotone_axis_count = 0;
    int first_strictly_monotone_axis = -1;
    for (int axis = 0; axis < certificate.dimension; ++axis) {
        const auto stride = integerPower(n, axis);
        const auto block = stride * n;
        bool strictly_increasing = true;
        bool strictly_decreasing = true;
        bool lower_face_negative = true;
        bool lower_face_positive = true;
        bool upper_face_negative = true;
        bool upper_face_positive = true;
        bool lower_face_zero = true;
        bool upper_face_zero = true;
        const auto tolerance =
            static_cast<long double>(root_tolerance);
        for (std::size_t base = 0u; base < total; base += block) {
            for (std::size_t inner = 0u; inner < stride; ++inner) {
                const auto& lower_face_value = control[base + inner];
                const auto& upper_face_value =
                    control[base + inner + (n - 1u) * stride];
                lower_face_negative =
                    lower_face_negative &&
                    lower_face_value.upper < -tolerance;
                lower_face_positive =
                    lower_face_positive &&
                    lower_face_value.lower > tolerance;
                upper_face_negative =
                    upper_face_negative &&
                    upper_face_value.upper < -tolerance;
                upper_face_positive =
                    upper_face_positive &&
                    upper_face_value.lower > tolerance;
                lower_face_zero =
                    lower_face_zero &&
                    lower_face_value.lower >= -tolerance &&
                    lower_face_value.upper <= tolerance;
                upper_face_zero =
                    upper_face_zero &&
                    upper_face_value.lower >= -tolerance &&
                    upper_face_value.upper <= tolerance;
                for (std::size_t i = 0u; i + 1u < n; ++i) {
                    const auto difference = subtractIntervals(
                        control[base + inner + (i + 1u) * stride],
                        control[base + inner + i * stride]);
                    strictly_increasing =
                        strictly_increasing && difference.lower > 0.0L;
                    strictly_decreasing =
                        strictly_decreasing && difference.upper < 0.0L;
                }
            }
        }
        const bool increasing_graph_bracket = strictly_increasing &&
            ((lower_face_negative && upper_face_positive) ||
             (lower_face_zero && upper_face_positive) ||
             (lower_face_negative && upper_face_zero) ||
             (certifiedFaceEndpointTouch(
                  control,
                  certificate,
                  axis,
                  /*lower_face=*/true,
                  /*nonnegative_face=*/false,
                  root_tolerance,
                  roundoff_zero_bound) &&
              upper_face_positive) ||
             (lower_face_negative &&
              certifiedFaceEndpointTouch(
                  control,
                  certificate,
                  axis,
                  /*lower_face=*/false,
                  /*nonnegative_face=*/true,
                  root_tolerance,
                  roundoff_zero_bound)));
        const bool decreasing_graph_bracket = strictly_decreasing &&
            ((lower_face_positive && upper_face_negative) ||
             (lower_face_zero && upper_face_negative) ||
             (lower_face_positive && upper_face_zero) ||
             (certifiedFaceEndpointTouch(
                  control,
                  certificate,
                  axis,
                  /*lower_face=*/true,
                  /*nonnegative_face=*/true,
                  root_tolerance,
                  roundoff_zero_bound) &&
              upper_face_negative) ||
             (lower_face_positive &&
              certifiedFaceEndpointTouch(
                  control,
                  certificate,
                  axis,
                  /*lower_face=*/false,
                  /*nonnegative_face=*/false,
                  root_tolerance,
                  roundoff_zero_bound)));
        if (strictly_increasing || strictly_decreasing) {
            ++strictly_monotone_axis_count;
            if (first_strictly_monotone_axis < 0) {
                first_strictly_monotone_axis = axis;
            }
        }
        if (!face_bracketed_graph.has_value() &&
            (increasing_graph_bracket || decreasing_graph_bracket)) {
            face_bracketed_graph = CertifiedRegularGraph{
                .axis = axis,
                .lower_face_zero = lower_face_zero,
                .upper_face_zero = upper_face_zero,
                .coordinatewise_monotone = false,
            };
        }
    }
    if (face_bracketed_graph.has_value()) {
        return face_bracketed_graph;
    }
    if (certificate.dimension == 2 &&
        strictly_monotone_axis_count == certificate.dimension) {
        bool certified_negative_corner = false;
        bool certified_positive_corner = false;
        const auto tolerance = static_cast<long double>(root_tolerance);
        for (std::size_t flat = 0u; flat < total; ++flat) {
            std::size_t remaining = flat;
            bool is_corner = true;
            for (int axis = 0; axis < certificate.dimension; ++axis) {
                const auto coordinate = remaining % n;
                remaining /= n;
                is_corner = is_corner &&
                    (coordinate == 0u || coordinate + 1u == n);
            }
            if (!is_corner) {
                continue;
            }
            certified_negative_corner = certified_negative_corner ||
                control[flat].upper < -tolerance;
            certified_positive_corner = certified_positive_corner ||
                control[flat].lower > tolerance;
        }
        if (certified_negative_corner && certified_positive_corner) {
            return CertifiedRegularGraph{
                .axis = first_strictly_monotone_axis,
                .lower_face_zero = false,
                .upper_face_zero = false,
                .coordinatewise_monotone = true,
            };
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<CertifiedRegularGraph> certifiedRegularGraph(
    const TensorBernsteinCertificate& certificate,
    const Rectangle2D& rect,
    Real root_tolerance)
{
    return certifiedRegularGraph(
        certificate,
        {{rect.xmin, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymax, 0.0}},
        root_tolerance);
}

[[nodiscard]] std::optional<geometry::CutIntegrationSide>
certifiedOneSidedBoundaryTouch(
    const TensorBernsteinCertificate& certificate,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    const OutwardInterval& certified_range,
    Real root_tolerance)
{
    if (certificate.dimension != 2 || input.evaluator == nullptr) {
        return std::nullopt;
    }

    auto control = certificate.root_control_values;
    restrictTensorControlAxis(
        control, certificate, 0, rect.xmin, rect.xmax);
    restrictTensorControlAxis(
        control, certificate, 1, rect.ymin, rect.ymax);

    const long double roundoff_zero_bound =
        certifiedControlRoundoffZeroBound(control);
    const auto is_roundoff_zero = [&](const OutwardInterval& value) {
        return isCertifiedRoundoffZero(value, roundoff_zero_bound);
    };

    const auto n = certificate.axis_size;
    const auto total = integerPower(n, certificate.dimension);
    const auto corner_flat_index = [n](std::size_t x, std::size_t y) {
        return x + n * y;
    };
    for (const auto side : {geometry::CutIntegrationSide::Positive,
                            geometry::CutIntegrationSide::Negative}) {
        const bool positive_side =
            side == geometry::CutIntegrationSide::Positive;
        const bool range_is_one_sided =
            positive_side
                ? certified_range.lower >= -roundoff_zero_bound &&
                      certified_range.upper > roundoff_zero_bound
                : certified_range.upper <= roundoff_zero_bound &&
                      certified_range.lower < -roundoff_zero_bound;
        if (!range_is_one_sided) {
            continue;
        }

        for (const std::size_t corner_y : {std::size_t{0u}, n - 1u}) {
            for (const std::size_t corner_x : {std::size_t{0u}, n - 1u}) {
                const auto corner_index =
                    corner_flat_index(corner_x, corner_y);
                if (!is_roundoff_zero(control[corner_index])) {
                    continue;
                }
                const std::array<Real, 3> corner{{
                    corner_x == 0u ? rect.xmin : rect.xmax,
                    corner_y == 0u ? rect.ymin : rect.ymax,
                    0.0,
                }};
                const Real corner_value =
                    input.evaluator
                        ->evaluate(input.linearized_input.parent_cell, corner)
                        .value -
                    input.isovalue;
                if ((positive_side &&
                     (corner_value < Real{0.0} ||
                      corner_value > root_tolerance)) ||
                    (!positive_side &&
                     (corner_value > Real{0.0} ||
                      corner_value < -root_tolerance))) {
                    continue;
                }

                bool all_axes_monotone = true;
                for (int axis = 0; axis < 2; ++axis) {
                    const auto stride = integerPower(n, axis);
                    const auto block = stride * n;
                    const bool corner_is_lower =
                        axis == 0 ? corner_x == 0u : corner_y == 0u;
                    const bool increasing_with_axis =
                        positive_side == corner_is_lower;
                    bool strict_step = false;
                    for (std::size_t base = 0u;
                         base < total && all_axes_monotone;
                         base += block) {
                        for (std::size_t inner = 0u;
                             inner < stride && all_axes_monotone;
                             ++inner) {
                            for (std::size_t i = 0u; i + 1u < n; ++i) {
                                const auto difference = subtractIntervals(
                                    control[base + inner + (i + 1u) * stride],
                                    control[base + inner + i * stride]);
                                if (increasing_with_axis) {
                                    all_axes_monotone =
                                        difference.lower >= 0.0L ||
                                        is_roundoff_zero(difference);
                                    strict_step = strict_step ||
                                        difference.lower >
                                            roundoff_zero_bound;
                                } else {
                                    all_axes_monotone =
                                        difference.upper <= 0.0L ||
                                        is_roundoff_zero(difference);
                                    strict_step = strict_step ||
                                        difference.upper <
                                            -roundoff_zero_bound;
                                }
                            }
                        }
                    }
                    all_axes_monotone = all_axes_monotone && strict_step;
                    if (!all_axes_monotone) {
                        break;
                    }
                }
                if (!all_axes_monotone) {
                    continue;
                }

                bool other_corners_have_strict_side_sign = true;
                for (const std::size_t other_y :
                     {std::size_t{0u}, n - 1u}) {
                    for (const std::size_t other_x :
                         {std::size_t{0u}, n - 1u}) {
                        if (other_x == corner_x && other_y == corner_y) {
                            continue;
                        }
                        const auto& value = control[
                            corner_flat_index(other_x, other_y)];
                        other_corners_have_strict_side_sign =
                            other_corners_have_strict_side_sign &&
                            (positive_side
                                 ? value.lower >
                                       static_cast<long double>(root_tolerance)
                                 : value.upper <
                                       -static_cast<long double>(root_tolerance));
                    }
                }
                if (other_corners_have_strict_side_sign) {
                    return side;
                }
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<CertifiedRegularGraph> certifiedRegularGraph(
    const TensorBernsteinCertificate& certificate,
    const Box3D& box,
    Real root_tolerance)
{
    return certifiedRegularGraph(
        certificate,
        {{box.xmin, box.ymin, box.zmin}},
        {{box.xmax, box.ymax, box.zmax}},
        root_tolerance);
}

[[nodiscard]] bool certifiedGraphLeafOwnsInterface(
    const TensorBernsteinCertificate& certificate,
    const CertifiedRegularGraph& graph,
    const std::array<Real, 3>& lower) noexcept
{
    if (!graph.lower_face_zero || graph.upper_face_zero) {
        return true;
    }
    const auto axis = static_cast<std::size_t>(graph.axis);
    return lower[axis] <= certificate.reference_min[axis];
}

[[nodiscard]] Real intervalLowerAsReal(long double value) noexcept
{
    Real converted = static_cast<Real>(value);
    if (static_cast<long double>(converted) > value) {
        converted = std::nextafter(
            converted, -std::numeric_limits<Real>::infinity());
    }
    return std::nextafter(
        converted, -std::numeric_limits<Real>::infinity());
}

[[nodiscard]] Real intervalUpperAsReal(long double value) noexcept
{
    Real converted = static_cast<Real>(value);
    if (static_cast<long double>(converted) < value) {
        converted = std::nextafter(
            converted, std::numeric_limits<Real>::infinity());
    }
    return std::nextafter(
        converted, std::numeric_limits<Real>::infinity());
}

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
    int terminal_topology_refinement_count{0};
    int max_terminal_topology_extra_depth{0};
    int root_branch_count{0};
    int root_finder_iteration_count{0};
    int curved_fragment_count{0};
    int full_negative_region_count{0};
    int full_positive_region_count{0};
    int certified_tensor_range_available{0};
    int certified_range_query_count{0};
    int certified_full_sign_region_count{0};
    int certified_same_sign_refinement_count{0};
    int certified_range_fail_closed_count{0};
    int certified_topology_query_count{0};
    int certified_regular_graph_leaf_count{0};
    int certified_topology_refinement_count{0};
    int certified_topology_fail_closed_count{0};
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
    std::string first_curved_fragment_failure_detail;
};

constexpr int kTerminalTopologyExtraSubdivisionDepth = 2;
constexpr int kCertifiedRangeExtraSubdivisionDepth = 6;

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

[[nodiscard]] interfaces::CutInterfaceReferenceSimplex referenceTriangle(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& c) noexcept
{
    interfaces::CutInterfaceReferenceSimplex simplex;
    simplex.vertex_count = 3u;
    simplex.vertices[0] = a;
    simplex.vertices[1] = b;
    simplex.vertices[2] = c;
    return simplex;
}

[[nodiscard]] interfaces::CutInterfaceReferenceSimplex referenceTetrahedron(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& c,
    const std::array<Real, 3>& d) noexcept
{
    interfaces::CutInterfaceReferenceSimplex simplex;
    simplex.vertex_count = 4u;
    simplex.vertices[0] = a;
    simplex.vertices[1] = b;
    simplex.vertices[2] = c;
    simplex.vertices[3] = d;
    return simplex;
}

[[nodiscard]] Real referenceSimplexMeasure(
    const interfaces::CutInterfaceReferenceSimplex& simplex,
    int dimension) noexcept
{
    if (simplex.vertex_count != static_cast<std::uint8_t>(dimension + 1)) {
        return Real{-1.0};
    }
    for (std::size_t i = 0u; i < simplex.vertex_count; ++i) {
        if (!finiteArray(simplex.vertices[i])) {
            return Real{-1.0};
        }
        if (simplex.has_represented_signed_values &&
            !std::isfinite(simplex.represented_signed_values[i])) {
            return Real{-1.0};
        }
    }
    if (!std::isfinite(simplex.measure_scale) ||
        !(simplex.measure_scale > Real{0.0})) {
        return Real{-1.0};
    }
    const auto ab = subtract(simplex.vertices[1], simplex.vertices[0]);
    Real measure = Real{-1.0};
    if (dimension == 2) {
        const auto ac = subtract(simplex.vertices[2], simplex.vertices[0]);
        measure = Real{0.5} * norm3(cross(ab, ac));
    } else if (dimension == 3) {
        const auto ac = subtract(simplex.vertices[2], simplex.vertices[0]);
        const auto ad = subtract(simplex.vertices[3], simplex.vertices[0]);
        measure = std::abs(dot(ab, cross(ac, ad))) / Real{6.0};
    }
    if (!std::isfinite(measure) || !(measure > Real{0.0})) {
        return Real{-1.0};
    }
    return measure * simplex.measure_scale;
}

[[nodiscard]] std::vector<interfaces::CutInterfaceReferenceSimplex>
rectangleReferenceSubcells(const Rectangle2D& rect)
{
    const std::array<Real, 3> v0{{rect.xmin, rect.ymin, 0.0}};
    const std::array<Real, 3> v1{{rect.xmax, rect.ymin, 0.0}};
    const std::array<Real, 3> v2{{rect.xmax, rect.ymax, 0.0}};
    const std::array<Real, 3> v3{{rect.xmin, rect.ymax, 0.0}};
    return {referenceTriangle(v0, v1, v2),
            referenceTriangle(v0, v2, v3)};
}

[[nodiscard]] std::vector<interfaces::CutInterfaceReferenceSimplex>
boxReferenceSubcells(const Box3D& box)
{
    const std::array<Real, 3> v0{{box.xmin, box.ymin, box.zmin}};
    const std::array<Real, 3> v1{{box.xmax, box.ymin, box.zmin}};
    const std::array<Real, 3> v2{{box.xmax, box.ymax, box.zmin}};
    const std::array<Real, 3> v3{{box.xmin, box.ymax, box.zmin}};
    const std::array<Real, 3> v4{{box.xmin, box.ymin, box.zmax}};
    const std::array<Real, 3> v5{{box.xmax, box.ymin, box.zmax}};
    const std::array<Real, 3> v6{{box.xmax, box.ymax, box.zmax}};
    const std::array<Real, 3> v7{{box.xmin, box.ymax, box.zmax}};
    return {
        referenceTetrahedron(v0, v1, v2, v6),
        referenceTetrahedron(v0, v2, v3, v6),
        referenceTetrahedron(v0, v3, v7, v6),
        referenceTetrahedron(v0, v7, v4, v6),
        referenceTetrahedron(v0, v4, v5, v6),
        referenceTetrahedron(v0, v5, v1, v6),
    };
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

void appendSampledSegmentRoots(
    const ImplicitCutQuadratureBackendCellInput& input,
    const interfaces::CutInterfaceDomainRequest& request,
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    Real uniqueness_tolerance,
    int& iterations,
    std::vector<std::array<Real, 3>>& roots)
{
    constexpr int sample_count = 32;
    const Real residual_tolerance = rootResidualTolerance(request);
    const auto point_at = [&](int i) {
        const Real t =
            static_cast<Real>(i) / static_cast<Real>(sample_count);
        return std::array<Real, 3>{{
            (Real{1.0} - t) * a[0] + t * b[0],
            (Real{1.0} - t) * a[1] + t * b[1],
            (Real{1.0} - t) * a[2] + t * b[2],
        }};
    };

    std::array<Real, 3> previous = a;
    Real f_previous = signedLevelSetValue(input, previous);
    if (!std::isfinite(f_previous)) {
        return;
    }
    if (std::abs(f_previous) <= residual_tolerance) {
        addUniqueRoot(roots, previous, uniqueness_tolerance);
    }

    for (int i = 1; i <= sample_count; ++i) {
        const auto current = point_at(i);
        const Real f_current = signedLevelSetValue(input, current);
        if (!std::isfinite(f_current)) {
            return;
        }
        if (std::abs(f_current) <= residual_tolerance) {
            addUniqueRoot(roots, current, uniqueness_tolerance);
        } else if (std::abs(f_previous) > residual_tolerance &&
                   (f_previous < Real{0.0}) != (f_current < Real{0.0})) {
            std::array<Real, 3> root;
            if (polishRootOnSegment(
                    input, request, previous, current, root, iterations)) {
                addUniqueRoot(roots, root, uniqueness_tolerance);
            }
        }
        previous = current;
        f_previous = f_current;
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
        appendSampledSegmentRoots(input,
                                  request,
                                  corners[edge[0]],
                                  corners[edge[1]],
                                  uniqueness_tolerance,
                                  iterations,
                                  roots);
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
    roots.reserve(8u);
    for (const auto& edge : edges) {
        appendSampledSegmentRoots(input,
                                  request,
                                  corners[edge[0]],
                                  corners[edge[1]],
                                  uniqueness_tolerance,
                                  iterations,
                                  roots);
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
    const std::vector<std::array<Real, 4>>& unit_rule,
    std::vector<std::pair<std::array<Real, 3>, Real>>& seeds)
{
    const Real area =
        Real{0.5} * norm3(cross(subtract(b, a), subtract(c, a)));
    if (!(area > Real{0.0}) || !std::isfinite(area) ||
        unit_rule.empty()) {
        return;
    }
    for (const auto& sample : unit_rule) {
        const std::array<Real, 3> point{{
            sample[0] * a[0] + sample[1] * b[0] + sample[2] * c[0],
            sample[0] * a[1] + sample[1] * b[1] + sample[2] * c[1],
            sample[0] * a[2] + sample[1] * b[2] + sample[2] * c[2],
        }};
        const Real weight = sample[3] * area;
        if (std::isfinite(weight) && weight > Real{0.0}) {
            seeds.push_back({point, weight});
        }
    }
}

[[nodiscard]] std::vector<std::array<Real, 4>>
positiveTriangleSurfaceUnitRule(int point_count)
{
    std::vector<std::array<Real, 4>> rule;
    const auto append_three = [&](Real repeated,
                                  Real distinct,
                                  Real weight) {
        rule.push_back({{distinct, repeated, repeated, weight}});
        rule.push_back({{repeated, distinct, repeated, weight}});
        rule.push_back({{repeated, repeated, distinct, weight}});
    };
    const auto append_six = [&](Real first,
                                Real second,
                                Real third,
                                Real weight) {
        rule.push_back({{first, second, third, weight}});
        rule.push_back({{first, third, second, weight}});
        rule.push_back({{second, first, third, weight}});
        rule.push_back({{second, third, first, weight}});
        rule.push_back({{third, first, second, weight}});
        rule.push_back({{third, second, first, weight}});
    };
    if (point_count == 3) {
        append_three(Real{1.0} / Real{6.0},
                     Real{2.0} / Real{3.0},
                     Real{1.0} / Real{3.0});
        return rule;
    }
    if (point_count == 7) {
        rule.push_back({{Real{1.0} / Real{3.0},
                         Real{1.0} / Real{3.0},
                         Real{1.0} / Real{3.0},
                         Real{0.225}}});
        append_three(Real{0.470142064105115},
                     Real{0.059715871789770},
                     Real{0.132394152788506});
        append_three(Real{0.101286507323456},
                     Real{0.797426985353087},
                     Real{0.125939180544827});
        return rule;
    }
    if (point_count == 12) {
        append_three(Real{0.063089014491502},
                     Real{0.873821971016996},
                     Real{0.050844906370207});
        append_three(Real{0.249286745170910},
                     Real{0.501426509658179},
                     Real{0.116786275726379});
        append_six(Real{0.636502499121399},
                   Real{0.310352451033785},
                   Real{0.053145049844816},
                   Real{0.082851075618374});
        return rule;
    }
    if (point_count == 16) {
        rule.push_back({{Real{1.0} / Real{3.0},
                         Real{1.0} / Real{3.0},
                         Real{1.0} / Real{3.0},
                         Real{0.144315607677787}}});
        append_three(Real{0.459292588292723},
                     Real{0.081414823414554},
                     Real{0.095091634267285});
        append_three(Real{0.170569307751760},
                     Real{0.658861384496480},
                     Real{0.103217370534718});
        append_three(Real{0.050547228317031},
                     Real{0.898905543365938},
                     Real{0.032458497623198});
        append_six(Real{0.008394777409958},
                   Real{0.263112829634638},
                   Real{0.728492392955404},
                   Real{0.027230314174435});
        return rule;
    }
    return {};
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
        appendSampledSegmentRoots(input,
                                  request,
                                  corners[edge[0]],
                                  corners[edge[1]],
                                  uniqueness_tolerance,
                                  iterations,
                                  roots);
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

[[nodiscard]] bool lineRectangleSearchSegment(
    const Rectangle2D& rect,
    const std::array<Real, 3>& origin,
    const std::array<Real, 3>& direction,
    std::array<Real, 3>& start,
    std::array<Real, 3>& end,
    Real& guess_fraction)
{
    constexpr Real tolerance = Real{1.0e-12};
    if (std::abs(direction[0]) <= tolerance &&
        std::abs(direction[1]) <= tolerance) {
        return false;
    }

    std::vector<Real> parameters;
    parameters.reserve(4u);
    const auto add_if_inside = [&](Real parameter) {
        const std::array<Real, 3> point{{
            origin[0] + parameter * direction[0],
            origin[1] + parameter * direction[1],
            origin[2] + parameter * direction[2],
        }};
        if (point[0] >= rect.xmin - tolerance &&
            point[0] <= rect.xmax + tolerance &&
            point[1] >= rect.ymin - tolerance &&
            point[1] <= rect.ymax + tolerance) {
            addUniqueParameter(parameters, parameter, tolerance);
        }
    };

    if (std::abs(direction[0]) > tolerance) {
        add_if_inside((rect.xmin - origin[0]) / direction[0]);
        add_if_inside((rect.xmax - origin[0]) / direction[0]);
    }
    if (std::abs(direction[1]) > tolerance) {
        add_if_inside((rect.ymin - origin[1]) / direction[1]);
        add_if_inside((rect.ymax - origin[1]) / direction[1]);
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

[[nodiscard]] std::vector<std::pair<Real, Real>>
gaussLegendreUnitRulePointCount(int point_count)
{
    if (point_count <= 0) {
        return {};
    }
    std::vector<std::pair<Real, Real>> rule(
        static_cast<std::size_t>(point_count));
    const Real pi = std::acos(Real{-1.0});
    const int root_count = (point_count + 1) / 2;
    for (int i = 0; i < root_count; ++i) {
        Real root = std::cos(
            pi * (static_cast<Real>(i) + Real{0.75}) /
            (static_cast<Real>(point_count) + Real{0.5}));
        Real derivative = Real{0.0};
        for (int iteration = 0; iteration < 64; ++iteration) {
            Real previous = Real{1.0};
            Real current = root;
            for (int degree = 2; degree <= point_count; ++degree) {
                const Real next =
                    ((Real{2.0} * static_cast<Real>(degree) - Real{1.0}) *
                         root * current -
                     (static_cast<Real>(degree) - Real{1.0}) * previous) /
                    static_cast<Real>(degree);
                previous = current;
                current = next;
            }
            derivative = static_cast<Real>(point_count) *
                         (root * current - previous) /
                         (root * root - Real{1.0});
            const Real update = current / derivative;
            root -= update;
            if (std::abs(update) <=
                Real{8.0} * std::numeric_limits<Real>::epsilon()) {
                break;
            }
        }
        Real previous = Real{1.0};
        Real current = root;
        for (int degree = 2; degree <= point_count; ++degree) {
            const Real next =
                ((Real{2.0} * static_cast<Real>(degree) - Real{1.0}) *
                     root * current -
                 (static_cast<Real>(degree) - Real{1.0}) * previous) /
                static_cast<Real>(degree);
            previous = current;
            current = next;
        }
        derivative = static_cast<Real>(point_count) *
                     (root * current - previous) /
                     (root * root - Real{1.0});
        const Real weight =
            Real{1.0} /
            ((Real{1.0} - root * root) * derivative * derivative);
        const auto left = static_cast<std::size_t>(i);
        const auto right =
            static_cast<std::size_t>(point_count - 1 - i);
        rule[left] = {Real{0.5} * (Real{1.0} - root), weight};
        rule[right] = {Real{0.5} * (Real{1.0} + root), weight};
    }
    return rule;
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

enum class PlanarLineRuleFailure {
    None,
    SearchSegment,
    RootSolve,
    Gradient,
    Weight,
    MomentConvergence,
};

struct PlanarLineRuleResult {
    bool ok{false};
    PlanarLineRuleFailure failure{PlanarLineRuleFailure::None};
    Real failed_parameter{0.0};
    std::vector<interfaces::CutInterfaceQuadraturePoint> points{};
    std::array<Real, 3> accumulated_normal{{0.0, 0.0, 0.0}};
    Real measure{0.0};
    Real max_root_residual{0.0};
    Real min_gradient_norm{std::numeric_limits<Real>::infinity()};
};

template <typename SampleBuilder>
[[nodiscard]] PlanarLineRuleResult buildPlanarLineRule(
    const std::vector<std::pair<Real, Real>>& rule,
    SampleBuilder&& build_sample)
{
    PlanarLineRuleResult result;
    result.points.reserve(rule.size());
    for (const auto& [parameter, unit_weight] : rule) {
        interfaces::CutInterfaceQuadraturePoint point;
        PlanarLineRuleFailure failure = PlanarLineRuleFailure::RootSolve;
        if (!build_sample(parameter, point, failure)) {
            result.failure = failure;
            result.failed_parameter = parameter;
            return result;
        }
        point.weight = unit_weight * point.reference_measure_factor;
        if (!std::isfinite(point.weight) || !(point.weight > Real{0.0})) {
            result.failure = PlanarLineRuleFailure::Weight;
            result.failed_parameter = parameter;
            return result;
        }
        result.max_root_residual = std::max(
            result.max_root_residual, point.level_set_residual);
        result.min_gradient_norm = std::min(
            result.min_gradient_norm, point.gradient_norm);
        for (std::size_t component = 0u; component < 3u; ++component) {
            result.accumulated_normal[component] +=
                point.normal[component] * point.weight;
        }
        result.measure += point.weight;
        result.points.push_back(point);
    }
    if (!std::isfinite(result.measure) || !(result.measure > Real{0.0})) {
        result.failure = PlanarLineRuleFailure::Weight;
        return result;
    }
    result.ok = true;
    return result;
}

[[nodiscard]] Real integerPowerNonnegative(Real value, int exponent) noexcept
{
    Real result{1.0};
    for (int i = 0; i < exponent; ++i) {
        result *= value;
    }
    return result;
}

[[nodiscard]] bool planarLineMomentRulesAgree(
    const PlanarLineRuleResult& production,
    const PlanarLineRuleResult& reference,
    int polynomial_order,
    Real tolerance) noexcept
{
    const auto moments_agree =
        [&](int x_degree, int y_degree, int z_degree) {
            Real production_moment{0.0};
            Real reference_moment{0.0};
            Real absolute_sum{0.0};
            for (const auto& point : production.points) {
                const Real contribution =
                    point.weight *
                    integerPowerNonnegative(point.point[0], x_degree) *
                    integerPowerNonnegative(point.point[1], y_degree) *
                    integerPowerNonnegative(point.point[2], z_degree);
                production_moment += contribution;
                absolute_sum += std::abs(contribution);
            }
            for (const auto& point : reference.points) {
                reference_moment +=
                    point.weight *
                    integerPowerNonnegative(point.point[0], x_degree) *
                    integerPowerNonnegative(point.point[1], y_degree) *
                    integerPowerNonnegative(point.point[2], z_degree);
            }
            const Real scale = std::max(
                {Real{1.0}, std::abs(reference_moment), absolute_sum});
            const Real allowed =
                tolerance + Real{4096.0} *
                                std::numeric_limits<Real>::epsilon() * scale;
            return std::abs(production_moment - reference_moment) <= allowed;
        };
    for (int total_degree = 0; total_degree <= polynomial_order;
         ++total_degree) {
        for (int x_degree = 0; x_degree <= total_degree; ++x_degree) {
            const int y_degree = total_degree - x_degree;
            if (!moments_agree(x_degree, y_degree, 0)) {
                return false;
            }
        }
    }
    return true;
}

struct SurfacePatchRuleResult {
    bool ok{false};
    PlanarLineRuleFailure failure{PlanarLineRuleFailure::None};
    std::array<Real, 3> failed_seed{{0.0, 0.0, 0.0}};
    bool saw_search_segment{false};
    bool saw_root{false};
    bool saw_gradient{false};
    std::vector<interfaces::CutInterfaceQuadraturePoint> points{};
    std::array<Real, 3> accumulated_normal{{0.0, 0.0, 0.0}};
    Real measure{0.0};
    Real max_root_residual{0.0};
    Real min_gradient_norm{std::numeric_limits<Real>::infinity()};
    Real maximum_moment_error{0.0};
    Real maximum_moment_scaled_error{0.0};
};

[[nodiscard]] bool surfacePatchMomentRulesAgree(
    const SurfacePatchRuleResult& production,
    const SurfacePatchRuleResult& reference,
    int polynomial_order,
    Real tolerance,
    Real& maximum_error,
    Real& maximum_scaled_error) noexcept
{
    const auto moments_agree =
        [&](int x_degree, int y_degree, int z_degree) {
            Real production_moment{0.0};
            Real reference_moment{0.0};
            Real absolute_sum{0.0};
            for (const auto& point : production.points) {
                const Real contribution =
                    point.weight *
                    integerPowerNonnegative(point.point[0], x_degree) *
                    integerPowerNonnegative(point.point[1], y_degree) *
                    integerPowerNonnegative(point.point[2], z_degree);
                production_moment += contribution;
                absolute_sum += std::abs(contribution);
            }
            for (const auto& point : reference.points) {
                reference_moment +=
                    point.weight *
                    integerPowerNonnegative(point.point[0], x_degree) *
                    integerPowerNonnegative(point.point[1], y_degree) *
                    integerPowerNonnegative(point.point[2], z_degree);
            }
            const Real scale = std::max(
                {Real{1.0}, std::abs(reference_moment), absolute_sum});
            const Real allowed =
                std::max(tolerance * scale,
                         Real{64.0} *
                             std::numeric_limits<Real>::epsilon() * scale);
            const Real error =
                std::abs(production_moment - reference_moment);
            maximum_error = std::max(maximum_error, error);
            maximum_scaled_error = std::max(
                maximum_scaled_error, error / allowed);
            return error <= allowed;
        };
    for (int total_degree = 0; total_degree <= polynomial_order;
         ++total_degree) {
        for (int x_degree = 0; x_degree <= total_degree; ++x_degree) {
            for (int y_degree = 0;
                 y_degree <= total_degree - x_degree;
                 ++y_degree) {
                const int z_degree = total_degree - x_degree - y_degree;
                if (!moments_agree(x_degree, y_degree, z_degree)) {
                    return false;
                }
            }
        }
    }
    return true;
}

void mergeSurfacePatchRule(SurfacePatchRuleResult& destination,
                           SurfacePatchRuleResult source)
{
    destination.points.insert(destination.points.end(),
                              std::make_move_iterator(source.points.begin()),
                              std::make_move_iterator(source.points.end()));
    for (std::size_t component = 0u; component < 3u; ++component) {
        destination.accumulated_normal[component] +=
            source.accumulated_normal[component];
    }
    destination.measure += source.measure;
    destination.max_root_residual = std::max(
        destination.max_root_residual, source.max_root_residual);
    destination.min_gradient_norm = std::min(
        destination.min_gradient_norm, source.min_gradient_norm);
}

void copySurfacePatchFailure(SurfacePatchRuleResult& destination,
                             const SurfacePatchRuleResult& source)
{
    destination.failure = source.failure;
    destination.failed_seed = source.failed_seed;
    destination.saw_search_segment = source.saw_search_segment;
    destination.saw_root = source.saw_root;
    destination.saw_gradient = source.saw_gradient;
}

[[nodiscard]] std::array<Tetrahedron3D, 8> subdivideTetrahedron(
    const Tetrahedron3D& tet) noexcept
{
    const auto ab = midpoint(tet.a, tet.b);
    const auto ac = midpoint(tet.a, tet.c);
    const auto ad = midpoint(tet.a, tet.d);
    const auto bc = midpoint(tet.b, tet.c);
    const auto bd = midpoint(tet.b, tet.d);
    const auto cd = midpoint(tet.c, tet.d);
    return {{
        Tetrahedron3D{tet.a, ab, ac, ad},
        Tetrahedron3D{ab, tet.b, bc, bd},
        Tetrahedron3D{ac, bc, tet.c, cd},
        Tetrahedron3D{ad, bd, cd, tet.d},
        Tetrahedron3D{ab, ac, ad, cd},
        Tetrahedron3D{ab, ac, bc, cd},
        Tetrahedron3D{ab, ad, bd, cd},
        Tetrahedron3D{ab, bc, bd, cd},
    }};
}

[[nodiscard]] bool projectTetrahedronSurfaceFragmentRule(
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    const interfaces::CutInterfaceFragment& linear_fragment,
    int point_count,
    int& local_iterations,
    SurfacePatchRuleResult& result)
{
    std::vector<std::array<Real, 3>> base_vertices;
    base_vertices.reserve(linear_fragment.vertices.size());
    for (const auto& vertex : linear_fragment.vertices) {
        base_vertices.push_back(vertex.point);
    }
    if (base_vertices.size() < 3u) {
        result.failure = PlanarLineRuleFailure::Weight;
        return false;
    }

    const auto edge_roots =
        tetrahedronEdgeRoots(input, request, tet, local_iterations);
    std::vector<std::array<Real, 3>> matched_roots;
    if (matchPolishedRootsToBaseVertices(
            edge_roots, base_vertices, matched_roots)) {
        base_vertices = std::move(matched_roots);
    }

    const auto base_centroid = polygonCentroid(base_vertices);
    const auto gradient_normal = interfaceNormalAt(input, base_centroid);
    const auto base_normal =
        polygonNormalOrDefault(base_vertices, gradient_normal);
    if (!(norm3(base_normal) > Real{1.0e-14})) {
        result.failure = PlanarLineRuleFailure::Gradient;
        return false;
    }

    const auto unit_rule = positiveTriangleSurfaceUnitRule(point_count);
    std::vector<std::pair<std::array<Real, 3>, Real>> surface_seeds;
    surface_seeds.reserve(
        (base_vertices.size() - 2u) * static_cast<std::size_t>(point_count));
    for (std::size_t i = 1u; i + 1u < base_vertices.size(); ++i) {
        appendTriangleSurfaceQuadratureSeeds(
            base_vertices[0],
            base_vertices[i],
            base_vertices[i + 1u],
            unit_rule,
            surface_seeds);
    }
    if (surface_seeds.empty()) {
        result.failure = PlanarLineRuleFailure::Weight;
        return false;
    }

    result.points.reserve(surface_seeds.size());
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
            if (!(plane_projection > Real{1.0e-14})) {
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

            const auto evaluation = input.evaluator->evaluate(
                input.linearized_input.parent_cell, point);
            const Real root_residual =
                std::abs(evaluation.value - input.isovalue);
            const Real gradient_norm = norm3(evaluation.reference_gradient);
            if (!std::isfinite(root_residual) ||
                !std::isfinite(gradient_norm) ||
                !(gradient_norm > Real{1.0e-14})) {
                continue;
            }
            const Real directional_derivative =
                std::abs(dot3(evaluation.reference_gradient,
                              projection_direction));
            if (!(directional_derivative > Real{1.0e-14})) {
                continue;
            }
            saw_gradient = true;

            const Real reference_measure_factor =
                plane_projection * gradient_norm / directional_derivative;
            const Real weight = planar_weight * reference_measure_factor;
            if (!std::isfinite(reference_measure_factor) ||
                !std::isfinite(weight) || !(weight > Real{0.0})) {
                continue;
            }

            accepted_point = interfaces::CutInterfaceQuadraturePoint{
                .point = point,
                .parent_coordinate = point,
                .normal = normalizedOrDefault(evaluation.reference_gradient),
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
                const auto evaluation = input.evaluator->evaluate(
                    input.linearized_input.parent_cell, point);
                const Real root_residual =
                    std::abs(evaluation.value - input.isovalue);
                const Real gradient_norm =
                    norm3(evaluation.reference_gradient);
                const Real normal_projection =
                    std::abs(dot3(evaluation.reference_gradient, base_normal));
                if (std::isfinite(root_residual) &&
                    std::isfinite(gradient_norm) &&
                    gradient_norm > Real{1.0e-14} &&
                    normal_projection > Real{1.0e-14}) {
                    saw_gradient = true;
                    const Real reference_measure_factor =
                        gradient_norm / normal_projection;
                    const Real weight =
                        planar_weight * reference_measure_factor;
                    if (std::isfinite(reference_measure_factor) &&
                        std::isfinite(weight) && weight > Real{0.0}) {
                        accepted_point =
                            interfaces::CutInterfaceQuadraturePoint{
                                .point = point,
                                .parent_coordinate = point,
                                .normal = normalizedOrDefault(
                                    evaluation.reference_gradient),
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

        if (!accepted) {
            result.failed_seed = seed;
            result.saw_search_segment = saw_search_segment;
            result.saw_root = saw_root;
            result.saw_gradient = saw_gradient;
            result.failure =
                !saw_search_segment
                    ? PlanarLineRuleFailure::SearchSegment
                    : (!saw_root ? PlanarLineRuleFailure::RootSolve
                                : (!saw_gradient
                                       ? PlanarLineRuleFailure::Gradient
                                       : PlanarLineRuleFailure::Weight));
            return false;
        }

        result.max_root_residual = std::max(
            result.max_root_residual, accepted_point.level_set_residual);
        result.min_gradient_norm = std::min(
            result.min_gradient_norm, accepted_point.gradient_norm);
        result.points.push_back(accepted_point);
        for (std::size_t component = 0u; component < 3u; ++component) {
            result.accumulated_normal[component] +=
                accepted_point.normal[component] * accepted_point.weight;
        }
        result.measure += accepted_point.weight;
    }
    return true;
}

[[nodiscard]] bool appendPairedRefinedTetrahedronSurfaceRules(
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    int forced_refinement_depth,
    int recovery_depth,
    int maximum_recovery_depth,
    int production_point_count,
    int reference_point_count,
    int& local_iterations,
    SurfacePatchRuleResult& production,
    SurfacePatchRuleResult& reference)
{
    const auto average_three = [](const std::array<Real, 3>& a,
                                  const std::array<Real, 3>& b,
                                  const std::array<Real, 3>& c) {
        return std::array<Real, 3>{{
            (a[0] + b[0] + c[0]) / Real{3.0},
            (a[1] + b[1] + c[1]) / Real{3.0},
            (a[2] + b[2] + c[2]) / Real{3.0},
        }};
    };
    const std::array<std::array<Real, 3>, 15> samples{{
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
        average_three(tet.a, tet.b, tet.c),
        average_three(tet.a, tet.b, tet.d),
        average_three(tet.a, tet.c, tet.d),
        average_three(tet.b, tet.c, tet.d),
        tetrahedronCentroid(tet),
    }};
    Real minimum = std::numeric_limits<Real>::infinity();
    Real maximum = -std::numeric_limits<Real>::infinity();
    for (const auto& sample : samples) {
        const Real value = signedLevelSetValue(input, sample);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const Real strict_tolerance = rootResidualTolerance(request);
    if (!(minimum < -strict_tolerance && maximum > strict_tolerance)) {
        return true;
    }

    const auto append_children = [&](int next_forced_depth,
                                     int next_recovery_depth) {
        SurfacePatchRuleResult child_production;
        SurfacePatchRuleResult child_reference;
        for (const auto& child : subdivideTetrahedron(tet)) {
            if (!appendPairedRefinedTetrahedronSurfaceRules(
                    request,
                    input,
                    child,
                    next_forced_depth,
                    next_recovery_depth,
                    maximum_recovery_depth,
                    production_point_count,
                    reference_point_count,
                    local_iterations,
                    child_production,
                    child_reference)) {
                copySurfacePatchFailure(production, child_production);
                copySurfacePatchFailure(reference, child_reference);
                return false;
            }
        }
        mergeSurfacePatchRule(production, std::move(child_production));
        mergeSurfacePatchRule(reference, std::move(child_reference));
        return true;
    };

    if (forced_refinement_depth > 0) {
        return append_children(
            forced_refinement_depth - 1, recovery_depth);
    }

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
    SurfacePatchRuleResult local_production;
    SurfacePatchRuleResult local_reference;
    bool found_active_fragment = false;
    for (const auto& child_fragment : leaf_cut.fragments) {
        if (!child_fragment.active()) {
            continue;
        }
        found_active_fragment = true;
        SurfacePatchRuleResult fragment_production;
        SurfacePatchRuleResult fragment_reference;
        const bool production_ok = projectTetrahedronSurfaceFragmentRule(
            request,
            input,
            tet,
            child_fragment,
            production_point_count,
            local_iterations,
            fragment_production);
        const bool reference_ok = projectTetrahedronSurfaceFragmentRule(
            request,
            input,
            tet,
            child_fragment,
            reference_point_count,
            local_iterations,
            fragment_reference);
        if (!production_ok || !reference_ok) {
            if (recovery_depth < maximum_recovery_depth) {
                return append_children(
                    /*next_forced_depth=*/0,
                    recovery_depth + 1);
            }
            copySurfacePatchFailure(
                production,
                production_ok ? fragment_reference : fragment_production);
            copySurfacePatchFailure(reference, fragment_reference);
            return false;
        }
        mergeSurfacePatchRule(
            local_production, std::move(fragment_production));
        mergeSurfacePatchRule(
            local_reference, std::move(fragment_reference));
    }
    if (!found_active_fragment) {
        if (recovery_depth < maximum_recovery_depth) {
            return append_children(
                /*next_forced_depth=*/0,
                recovery_depth + 1);
        }
        production.failure = PlanarLineRuleFailure::RootSolve;
        reference.failure = PlanarLineRuleFailure::RootSolve;
        return false;
    }
    mergeSurfacePatchRule(production, std::move(local_production));
    mergeSurfacePatchRule(reference, std::move(local_reference));
    return true;
}

[[nodiscard]] bool buildPairedRefinedTetrahedronSurfaceRules(
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    int refinement_depth,
    int production_point_count,
    int reference_point_count,
    int& local_iterations,
    SurfacePatchRuleResult& production,
    SurfacePatchRuleResult& reference)
{
    if (!appendPairedRefinedTetrahedronSurfaceRules(
            request,
            input,
            tet,
            refinement_depth,
            /*recovery_depth=*/0,
            /*maximum_recovery_depth=*/2,
            production_point_count,
            reference_point_count,
            local_iterations,
            production,
            reference)) {
        return false;
    }
    if (!std::isfinite(production.measure) ||
        !std::isfinite(reference.measure) || production.measure < Real{0.0} ||
        reference.measure < Real{0.0}) {
        production.failure = PlanarLineRuleFailure::Weight;
        reference.failure = PlanarLineRuleFailure::Weight;
        return false;
    }
    production.ok = true;
    reference.ok = true;
    return true;
}

template <typename SampleBuilder>
[[nodiscard]] bool buildCertifiedPlanarLineRules(
    int polynomial_order,
    Real tolerance,
    SampleBuilder&& build_sample,
    PlanarLineRuleResult& production,
    PlanarLineRuleResult& reference)
{
    constexpr std::array<std::pair<int, int>, 4> point_counts{{
        {8, 12},
        {12, 16},
        {16, 24},
        {24, 32},
    }};
    for (const auto& [production_count, reference_count] : point_counts) {
        auto candidate = buildPlanarLineRule(
            gaussLegendreUnitRulePointCount(production_count), build_sample);
        if (!candidate.ok) {
            production = std::move(candidate);
            return false;
        }
        auto certificate = buildPlanarLineRule(
            gaussLegendreUnitRulePointCount(reference_count), build_sample);
        if (!certificate.ok) {
            production = std::move(certificate);
            return false;
        }
        if (planarLineMomentRulesAgree(
                candidate, certificate, polynomial_order, tolerance)) {
            production = std::move(candidate);
            reference = std::move(certificate);
            return true;
        }
    }
    production = PlanarLineRuleResult{
        .failure = PlanarLineRuleFailure::MomentConvergence};
    return false;
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

    const auto fail_edge_root_mismatch = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_edge_root_mismatch_count;
        return false;
    };
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

    int local_iterations = 0;
    auto roots = rectangleEdgeRoots(input, request, rect, local_iterations);
    if (roots.size() != 2u) {
        if (diagnostics.first_curved_fragment_failure_detail.empty()) {
            const std::array<std::array<Real, 3>, 4> corners{{
                {{rect.xmin, rect.ymin, 0.0}},
                {{rect.xmax, rect.ymin, 0.0}},
                {{rect.xmax, rect.ymax, 0.0}},
                {{rect.xmin, rect.ymax, 0.0}},
            }};
            std::ostringstream detail;
            detail << "shape=rectangle"
                   << "; rect=(" << formatReal(rect.xmin) << ","
                   << formatReal(rect.xmax) << "," << formatReal(rect.ymin)
                   << "," << formatReal(rect.ymax) << ")"
                   << "; edge_root_count=" << roots.size()
                   << "; edge_roots=" << formatPointList(roots)
                   << "; corner_values=[";
            for (std::size_t i = 0u; i < corners.size(); ++i) {
                if (i > 0u) {
                    detail << ",";
                }
                detail << formatReal(signedLevelSetValue(input, corners[i]));
            }
            detail << "]";
            diagnostics.first_curved_fragment_failure_detail = detail.str();
        }
        return fail_edge_root_mismatch();
    }
    const auto a = roots[0];
    const auto b = roots[1];
    const Real dx = b[0] - a[0];
    const Real dy = b[1] - a[1];
    if (std::abs(dx) <= Real{1.0e-14} &&
        std::abs(dy) <= Real{1.0e-14}) {
        return fail_seed();
    }
    const Real chord_length = std::sqrt(dx * dx + dy * dy);
    if (!(chord_length > Real{1.0e-14}) || !std::isfinite(chord_length)) {
        return fail_seed();
    }
    const std::array<Real, 3> tangent{{
        dx / chord_length,
        dy / chord_length,
        0.0,
    }};
    const std::array<Real, 3> transverse{{-tangent[1], tangent[0], 0.0}};
    const bool solve_y_as_function_of_x = std::abs(dx) >= std::abs(dy);
    auto polishing_request = request;
    polishing_request.tolerance =
        std::min(polishing_request.tolerance, Real{1.0e-14});
    polishing_request.implicit_cut_root_tolerance =
        std::min(polishing_request.implicit_cut_root_tolerance,
                 Real{1.0e-14});
    polishing_request.implicit_cut_root_coordinate_tolerance =
        std::min(polishing_request.implicit_cut_root_coordinate_tolerance,
                 Real{1.0e-14});
    const auto build_sample =
        [&](Real t,
            interfaces::CutInterfaceQuadraturePoint& output,
            PlanarLineRuleFailure& failure) {
        const std::array<Real, 3> origin{{
            (Real{1.0} - t) * a[0] + t * b[0],
            (Real{1.0} - t) * a[1] + t * b[1],
            0.0,
        }};
        std::array<Real, 3> point{};
        std::array<Real, 3> normal{};
        Real root_residual = 0.0;
        Real gradient_norm = 0.0;
        Real reference_measure_factor = 0.0;
        bool accepted = false;

        const auto try_fixed_axis = [&]() {
            if (solve_y_as_function_of_x) {
                if (!solveRootAtFixedX(input,
                                       polishing_request,
                                       origin[0],
                                       rect.ymin,
                                       rect.ymax,
                                       origin[1],
                                       point,
                                       local_iterations)) {
                    failure = PlanarLineRuleFailure::RootSolve;
                    return false;
                }
            } else if (!solveRootAtFixedY(input,
                                          polishing_request,
                                          origin[1],
                                          rect.xmin,
                                          rect.xmax,
                                          origin[0],
                                          point,
                                          local_iterations)) {
                failure = PlanarLineRuleFailure::RootSolve;
                return false;
            }

            const auto evaluation =
                input.evaluator->evaluate(input.linearized_input.parent_cell, point);
            root_residual = std::abs(evaluation.value - input.isovalue);
            gradient_norm = norm3(evaluation.reference_gradient);
            if (!std::isfinite(root_residual) ||
                !std::isfinite(gradient_norm) ||
                gradient_norm <= Real{1.0e-14}) {
                failure = PlanarLineRuleFailure::Gradient;
                return false;
            }
            const Real denominator =
                solve_y_as_function_of_x ? evaluation.reference_gradient[1]
                                         : evaluation.reference_gradient[0];
            if (std::abs(denominator) <= Real{1.0e-14}) {
                failure = PlanarLineRuleFailure::Gradient;
                return false;
            }
            const Real slope =
                solve_y_as_function_of_x
                    ? -evaluation.reference_gradient[0] / denominator
                    : -evaluation.reference_gradient[1] / denominator;
            const Real coordinate_span =
                solve_y_as_function_of_x ? std::abs(dx) : std::abs(dy);
            reference_measure_factor =
                coordinate_span * std::sqrt(Real{1.0} + slope * slope);
            if (!std::isfinite(reference_measure_factor) ||
                reference_measure_factor <= Real{0.0}) {
                failure = PlanarLineRuleFailure::Weight;
                return false;
            }
            normal = normalizedOrDefault(evaluation.reference_gradient);
            return true;
        };

        const auto try_transverse_segment = [&]() {
            std::array<Real, 3> search_start;
            std::array<Real, 3> search_end;
            Real guess_fraction = 0.5;
            if (!lineRectangleSearchSegment(
                    rect, origin, transverse, search_start, search_end,
                    guess_fraction)) {
                failure = PlanarLineRuleFailure::SearchSegment;
                return false;
            }
            if (!solveRootAlongSegmentNearGuess(input,
                                                polishing_request,
                                                search_start,
                                                search_end,
                                                guess_fraction,
                                                point,
                                                local_iterations)) {
                failure = PlanarLineRuleFailure::RootSolve;
                return false;
            }
            const auto evaluation =
                input.evaluator->evaluate(input.linearized_input.parent_cell, point);
            root_residual = std::abs(evaluation.value - input.isovalue);
            gradient_norm = norm3(evaluation.reference_gradient);
            if (!std::isfinite(root_residual) ||
                !std::isfinite(gradient_norm) ||
                gradient_norm <= Real{1.0e-14}) {
                failure = PlanarLineRuleFailure::Gradient;
                return false;
            }
            const Real transverse_derivative =
                dot3(evaluation.reference_gradient, transverse);
            if (std::abs(transverse_derivative) <= Real{1.0e-14}) {
                failure = PlanarLineRuleFailure::Gradient;
                return false;
            }
            const Real tangent_derivative =
                dot3(evaluation.reference_gradient, tangent);
            const Real height_slope =
                -tangent_derivative / transverse_derivative;
            reference_measure_factor =
                chord_length * std::sqrt(Real{1.0} +
                                         height_slope * height_slope);
            if (!std::isfinite(reference_measure_factor) ||
                reference_measure_factor <= Real{0.0}) {
                failure = PlanarLineRuleFailure::Weight;
                return false;
            }
            normal = normalizedOrDefault(evaluation.reference_gradient);
            return true;
        };

        accepted = try_transverse_segment() || try_fixed_axis();
        if (!accepted) {
            return false;
        }
        output = interfaces::CutInterfaceQuadraturePoint{
            .point = point,
            .parent_coordinate = point,
            .normal = normal,
            .weight = 0.0,
            .reference_measure_factor = reference_measure_factor,
            .level_set_residual = root_residual,
            .gradient_norm = gradient_norm};
        return true;
    };

    PlanarLineRuleResult production_rule;
    PlanarLineRuleResult reference_rule;
    if (!buildCertifiedPlanarLineRules(
            request.resolvedInterfaceQuadratureOrder(),
            request.tolerance,
            build_sample,
            production_rule,
            reference_rule)) {
        if (diagnostics.first_curved_fragment_failure_detail.empty()) {
            diagnostics.first_curved_fragment_failure_detail =
                "shape=rectangle; rect=(" + formatReal(rect.xmin) + "," +
                formatReal(rect.xmax) + "," + formatReal(rect.ymin) + "," +
                formatReal(rect.ymax) + "); root_a=" + formatPoint(a) +
                "; root_b=" + formatPoint(b) +
                "; failed_parameter=" +
                formatReal(production_rule.failed_parameter);
        }
        switch (production_rule.failure) {
        case PlanarLineRuleFailure::SearchSegment:
            return fail_search_segment();
        case PlanarLineRuleFailure::RootSolve:
            return fail_root_solve();
        case PlanarLineRuleFailure::Gradient:
            return fail_gradient();
        case PlanarLineRuleFailure::None:
        case PlanarLineRuleFailure::Weight:
        case PlanarLineRuleFailure::MomentConvergence:
            return fail_weight();
        }
    }
    if (production_rule.measure <= request.tolerance) {
        return fail_weight();
    }
    auto quadrature_points = std::move(production_rule.points);
    auto moment_certificate_points = std::move(reference_rule.points);
    const auto accumulated_normal = production_rule.accumulated_normal;
    const Real measure = production_rule.measure;
    const Real max_root_residual = production_rule.max_root_residual;
    const Real min_gradient_norm = production_rule.min_gradient_norm;

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
    fragment.moment_certificate_order =
        request.resolvedInterfaceQuadratureOrder();
    fragment.moment_certificate_points =
        std::move(moment_certificate_points);
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

    const auto fail_edge_root_mismatch = [&]() {
        ++diagnostics.curved_fragment_failure_count;
        ++diagnostics.curved_fragment_edge_root_mismatch_count;
        return false;
    };
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

    int local_iterations = 0;
    auto roots = triangleEdgeRoots(input, request, tri, local_iterations);
    if (roots.size() != 2u) {
        return fail_edge_root_mismatch();
    }
    const auto a = roots[0];
    const auto b = roots[1];
    const Real chord_length = distance2D(a, b);
    if (!(chord_length > Real{1.0e-14})) {
        return fail_seed();
    }
    const std::array<Real, 3> tangent{{
        (b[0] - a[0]) / chord_length,
        (b[1] - a[1]) / chord_length,
        0.0,
    }};
    const std::array<Real, 3> transverse{{-tangent[1], tangent[0], 0.0}};
    auto polishing_request = request;
    polishing_request.tolerance =
        std::min(polishing_request.tolerance, Real{1.0e-14});
    polishing_request.implicit_cut_root_tolerance =
        std::min(polishing_request.implicit_cut_root_tolerance,
                 Real{1.0e-14});
    polishing_request.implicit_cut_root_coordinate_tolerance =
        std::min(polishing_request.implicit_cut_root_coordinate_tolerance,
                 Real{1.0e-14});
    const auto build_sample =
        [&](Real t,
            interfaces::CutInterfaceQuadraturePoint& output,
            PlanarLineRuleFailure& failure) {
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
            failure = PlanarLineRuleFailure::SearchSegment;
            return false;
        }
        std::array<Real, 3> point;
        if (!solveRootAlongSegmentNearGuess(input,
                                            polishing_request,
                                            search_start,
                                            search_end,
                                            guess_fraction,
                                            point,
                                            local_iterations)) {
            failure = PlanarLineRuleFailure::RootSolve;
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
            failure = PlanarLineRuleFailure::Gradient;
            return false;
        }
        const Real transverse_derivative =
            dot3(evaluation.reference_gradient, transverse);
        if (std::abs(transverse_derivative) <= Real{1.0e-14}) {
            failure = PlanarLineRuleFailure::Gradient;
            return false;
        }
        const Real tangent_derivative =
            dot3(evaluation.reference_gradient, tangent);
        const Real height_slope = -tangent_derivative / transverse_derivative;
        const Real reference_measure_factor =
            chord_length * std::sqrt(Real{1.0} + height_slope * height_slope);
        if (!std::isfinite(reference_measure_factor) ||
            !(reference_measure_factor > Real{0.0})) {
            failure = PlanarLineRuleFailure::Weight;
            return false;
        }
        const auto normal = normalizedOrDefault(evaluation.reference_gradient);
        output = interfaces::CutInterfaceQuadraturePoint{
            .point = point,
            .parent_coordinate = point,
            .normal = normal,
            .weight = 0.0,
            .reference_measure_factor = reference_measure_factor,
            .level_set_residual = root_residual,
            .gradient_norm = gradient_norm};
        return true;
    };

    PlanarLineRuleResult production_rule;
    PlanarLineRuleResult reference_rule;
    if (!buildCertifiedPlanarLineRules(
            request.resolvedInterfaceQuadratureOrder(),
            request.tolerance,
            build_sample,
            production_rule,
            reference_rule)) {
        switch (production_rule.failure) {
        case PlanarLineRuleFailure::SearchSegment:
            return fail_search_segment();
        case PlanarLineRuleFailure::RootSolve:
            return fail_root_solve();
        case PlanarLineRuleFailure::Gradient:
            return fail_gradient();
        case PlanarLineRuleFailure::None:
        case PlanarLineRuleFailure::Weight:
        case PlanarLineRuleFailure::MomentConvergence:
            return fail_weight();
        }
    }
    if (production_rule.measure <= request.tolerance) {
        return fail_weight();
    }
    auto quadrature_points = std::move(production_rule.points);
    auto moment_certificate_points = std::move(reference_rule.points);
    const auto accumulated_normal = production_rule.accumulated_normal;
    const Real measure = production_rule.measure;
    const Real max_root_residual = production_rule.max_root_residual;
    const Real min_gradient_norm = production_rule.min_gradient_norm;

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
    fragment.moment_certificate_order =
        request.resolvedInterfaceQuadratureOrder();
    fragment.moment_certificate_points =
        std::move(moment_certificate_points);
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
        fragment.moment_certificate_order = -1;
        fragment.moment_certificate_points.clear();
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

    SurfacePatchRuleResult production_rule;
    SurfacePatchRuleResult reference_rule;
    bool certified = false;
    constexpr std::array<std::pair<int, int>, 3> point_counts{{
        {3, 7},
        {7, 12},
        {12, 16},
    }};
    const int minimum_production_point_count =
        request.resolvedInterfaceQuadratureOrder() <= 2 ? 3 : 7;
    for (int refinement_depth = 0;
         refinement_depth <= 4 && !certified;
         ++refinement_depth) {
        for (const auto& [production_count, reference_count] : point_counts) {
            if (production_count < minimum_production_point_count) {
                continue;
            }
            SurfacePatchRuleResult candidate;
            SurfacePatchRuleResult certificate;
            if (!buildPairedRefinedTetrahedronSurfaceRules(
                    request,
                    input,
                    tet,
                    refinement_depth,
                    production_count,
                    reference_count,
                    local_iterations,
                    candidate,
                    certificate)) {
                production_rule = std::move(candidate);
                continue;
            }
            if (candidate.points.empty()) {
                if (matched_edge_roots && mark_boundary_degenerate()) {
                    return true;
                }
                production_rule = SurfacePatchRuleResult{
                    .failure = PlanarLineRuleFailure::RootSolve};
                continue;
            }
            if (certificate.points.empty()) {
                production_rule = std::move(certificate);
                continue;
            }
            Real maximum_moment_error{0.0};
            Real maximum_moment_scaled_error{0.0};
            if (surfacePatchMomentRulesAgree(
                    candidate,
                    certificate,
                    request.resolvedInterfaceQuadratureOrder(),
                    request.tolerance,
                    maximum_moment_error,
                    maximum_moment_scaled_error)) {
                production_rule = std::move(candidate);
                reference_rule = std::move(certificate);
                certified = true;
                break;
            }
            production_rule = SurfacePatchRuleResult{
                .failure = PlanarLineRuleFailure::MomentConvergence,
                .maximum_moment_error = maximum_moment_error,
                .maximum_moment_scaled_error =
                    maximum_moment_scaled_error};
        }
    }

    if (!certified) {
        if (production_rule.failure !=
                PlanarLineRuleFailure::MomentConvergence &&
            matched_edge_roots && mark_boundary_degenerate()) {
            return true;
        }
        if (diagnostics.first_curved_fragment_failure_detail.empty()) {
            std::ostringstream detail;
            detail << "shape=tetrahedron"
                   << "; tet_vertices="
                   << formatPointList({tet.a, tet.b, tet.c, tet.d})
                   << "; base_vertices=" << formatPointList(base_vertices)
                   << "; edge_roots=" << formatPointList(edge_roots)
                   << "; matched_edge_roots="
                   << (matched_edge_roots ? "true" : "false")
                   << "; seed=" << formatPoint(production_rule.failed_seed)
                   << "; f_seed="
                   << formatReal(
                          signedLevelSetValue(input, production_rule.failed_seed))
                   << "; saw_search_segment="
                   << (production_rule.saw_search_segment ? "true" : "false")
                   << "; saw_root="
                   << (production_rule.saw_root ? "true" : "false")
                   << "; saw_gradient="
                   << (production_rule.saw_gradient ? "true" : "false")
                   << "; moment_convergence="
                   << (production_rule.failure ==
                               PlanarLineRuleFailure::MomentConvergence
                           ? "false"
                           : "not-reached")
                   << "; maximum_moment_error="
                   << formatReal(production_rule.maximum_moment_error)
                   << "; maximum_moment_scaled_error="
                   << formatReal(
                          production_rule.maximum_moment_scaled_error);
            diagnostics.first_curved_fragment_failure_detail = detail.str();
        }
        switch (production_rule.failure) {
        case PlanarLineRuleFailure::SearchSegment:
            return fail_search_segment();
        case PlanarLineRuleFailure::RootSolve:
            if (!matched_edge_roots) {
                ++diagnostics.curved_fragment_root_solve_edge_root_mismatch;
            }
            return fail_root_solve();
        case PlanarLineRuleFailure::Gradient:
            return fail_gradient();
        case PlanarLineRuleFailure::None:
        case PlanarLineRuleFailure::Weight:
        case PlanarLineRuleFailure::MomentConvergence:
            return fail_weight();
        }
    }

    auto quadrature_points = std::move(production_rule.points);
    auto moment_certificate_points = std::move(reference_rule.points);
    const auto accumulated_normal = production_rule.accumulated_normal;
    const Real measure = production_rule.measure;
    const Real max_root_residual = production_rule.max_root_residual;
    const Real min_gradient_norm = production_rule.min_gradient_norm;

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
    fragment.moment_certificate_order =
        request.resolvedInterfaceQuadratureOrder();
    fragment.moment_certificate_points =
        std::move(moment_certificate_points);
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

[[nodiscard]] bool pointInsideRectangle(
    const Rectangle2D& rect,
    const std::array<Real, 3>& point,
    Real tolerance) noexcept
{
    return point[0] >= rect.xmin - tolerance &&
           point[0] <= rect.xmax + tolerance &&
           point[1] >= rect.ymin - tolerance &&
           point[1] <= rect.ymax + tolerance;
}

[[nodiscard]] bool pointInsideBox(
    const Box3D& box,
    const std::array<Real, 3>& point,
    Real tolerance) noexcept
{
    return point[0] >= box.xmin - tolerance &&
           point[0] <= box.xmax + tolerance &&
           point[1] >= box.ymin - tolerance &&
           point[1] <= box.ymax + tolerance &&
           point[2] >= box.zmin - tolerance &&
           point[2] <= box.zmax + tolerance;
}

void appendUniquePoint(std::vector<std::array<Real, 3>>& points,
                       const std::array<Real, 3>& point,
                       Real tolerance)
{
    const Real tol_sq = tolerance * tolerance;
    const auto existing =
        std::find_if(points.begin(), points.end(), [&](const auto& candidate) {
            const Real dx = candidate[0] - point[0];
            const Real dy = candidate[1] - point[1];
            const Real dz = candidate[2] - point[2];
            return dx * dx + dy * dy + dz * dz <= tol_sq;
        });
    if (existing == points.end()) {
        points.push_back(point);
    }
}

void appendHighOrderSamplesInsideRectangle(
    std::vector<std::array<Real, 3>>& samples,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    Real tolerance)
{
    for (const auto& point : input.high_order_sample_points) {
        if (pointInsideRectangle(rect, point, tolerance)) {
            appendUniquePoint(samples, point, tolerance);
        }
    }
}

void appendHighOrderSamplesInsideBox(
    std::vector<std::array<Real, 3>>& samples,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Box3D& box,
    Real tolerance)
{
    for (const auto& point : input.high_order_sample_points) {
        if (pointInsideBox(box, point, tolerance)) {
            appendUniquePoint(samples, point, tolerance);
        }
    }
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
    region.local_region_index =
        static_cast<LocalIndex>(cut.volume_regions.size());
    region.topology_id =
        "cell-" + std::to_string(input.linearized_input.parent_cell) +
        "-volume-" + construction_token + "-" +
        sideTopologyToken(region.side) + "-" +
        std::to_string(region.local_region_index);
    region.stable_id =
        interfaces::cutVolumeStableId(request.interface_marker,
                                      input.linearized_input.parent_cell,
                                      region.local_region_index,
                                      region.side,
                                      request.source.value_revision);
    if (region.achieved_quadrature_order < 0) {
        region.achieved_quadrature_order =
            interfaces::implementedLevelSetCutVolumeExactOrder(
                request.resolvedVolumeQuadratureOrder());
    }
}

void stampGeneratedInterfaceFragmentMetadata(
    interfaces::CutInterfaceFragment& fragment,
    const interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const char* construction_token)
{
    fragment.local_fragment_index =
        static_cast<LocalIndex>(cut.fragments.size());
    fragment.topology_id =
        "cell-" + std::to_string(input.linearized_input.parent_cell) +
        "-interface-" + construction_token + "-" +
        std::to_string(fragment.local_fragment_index);
    fragment.branch_id = fragment.topology_id;
    fragment.stable_id =
        interfaces::cutInterfaceStableId(
            request.interface_marker,
            input.linearized_input.parent_cell,
            fragment.local_fragment_index,
            request.source.value_revision);
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
            region.reference_subcells = rectangleReferenceSubcells(rect);
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
            region.reference_subcells = boxReferenceSubcells(box);
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
            region.reference_subcells = {
                referenceTriangle(tri.a, tri.b, tri.c)};
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
            region.reference_subcells = {
                referenceTetrahedron(tet.a, tet.b, tet.c, tet.d)};
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "tetrahedron");
        cut.volume_regions.push_back(std::move(region));
    }
}

void appendCertifiedAlignedRectangleFragment(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    const CertifiedRegularGraph& graph,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    const bool lower_face = graph.lower_face_zero;
    std::array<Real, 3> a{{rect.xmin, rect.ymin, 0.0}};
    std::array<Real, 3> b{{rect.xmax, rect.ymax, 0.0}};
    if (graph.axis == 0) {
        const Real x = lower_face ? rect.xmin : rect.xmax;
        a = {{x, rect.ymin, 0.0}};
        b = {{x, rect.ymax, 0.0}};
    } else {
        const Real y = lower_face ? rect.ymin : rect.ymax;
        a = {{rect.xmin, y, 0.0}};
        b = {{rect.xmax, y, 0.0}};
    }
    const std::array<Real, 3> midpoint{{
        Real{0.5} * (a[0] + b[0]),
        Real{0.5} * (a[1] + b[1]),
        0.0,
    }};
    const auto evaluation = input.evaluator->evaluate(
        input.linearized_input.parent_cell, midpoint);

    interfaces::CutInterfaceFragment fragment;
    fragment.interface_marker = request.interface_marker;
    fragment.parent_cell = input.linearized_input.parent_cell;
    fragment.kind = interfaces::CutInterfaceFragmentKind::CurvedPatch;
    fragment.normal = normalizedOrDefault(evaluation.reference_gradient);
    const Real dx = b[0] - a[0];
    const Real dy = b[1] - a[1];
    fragment.measure = std::sqrt(dx * dx + dy * dy);
    fragment.min_level_set_value = -request.implicit_cut_root_tolerance;
    fragment.max_level_set_value = request.implicit_cut_root_tolerance;
    fragment.conditioning_diagnostic =
        "certified-aligned-regular-graph";
    fragment.max_root_residual = std::abs(evaluation.value - input.isovalue);
    fragment.min_gradient_norm = norm3(evaluation.reference_gradient);
    fragment.root_polished = true;
    fragment.vertices = {
        interfaces::CutInterfaceVertex{
            .point = a,
            .parent_coordinate = a,
            .level_set_value = 0.0,
            .stable_id = interfaces::cutInterfaceStableId(
                request.interface_marker,
                input.linearized_input.parent_cell,
                static_cast<LocalIndex>(1u),
                request.source.value_revision)},
        interfaces::CutInterfaceVertex{
            .point = b,
            .parent_coordinate = b,
            .level_set_value = 0.0,
            .stable_id = interfaces::cutInterfaceStableId(
                request.interface_marker,
                input.linearized_input.parent_cell,
                static_cast<LocalIndex>(2u),
                request.source.value_revision)},
    };
    const auto make_rule = [&](int order) {
        std::vector<interfaces::CutInterfaceQuadraturePoint> points;
        for (const auto& [parameter, unit_weight] :
             gaussLegendreUnitRule(order)) {
            const std::array<Real, 3> point{{
                (Real{1.0} - parameter) * a[0] + parameter * b[0],
                (Real{1.0} - parameter) * a[1] + parameter * b[1],
                0.0,
            }};
            const auto point_evaluation = input.evaluator->evaluate(
                input.linearized_input.parent_cell, point);
            const Real weight = unit_weight * fragment.measure;
            points.push_back(interfaces::CutInterfaceQuadraturePoint{
                .point = point,
                .parent_coordinate = point,
                .normal = normalizedOrDefault(
                    point_evaluation.reference_gradient),
                .weight = weight,
                .reference_measure_factor = weight,
                .level_set_residual =
                    std::abs(point_evaluation.value - input.isovalue),
                .gradient_norm =
                    norm3(point_evaluation.reference_gradient),
            });
        }
        return points;
    };
    fragment.quadrature_points = make_rule(
        request.resolvedInterfaceQuadratureOrder());
    fragment.moment_certificate_order =
        request.resolvedInterfaceQuadratureOrder();
    fragment.moment_certificate_points = make_rule(
        std::max(5, request.resolvedInterfaceQuadratureOrder() + 2));
    stampGeneratedInterfaceFragmentMetadata(
        fragment, cut, request, input, "certified-aligned-rectangle");
    cut.fragments.push_back(std::move(fragment));
    ++diagnostics.interface_fragment_count;
    ++diagnostics.root_branch_count;
    ++diagnostics.curved_fragment_count;
}

void appendLinearizedRectangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    Real parent_measure,
    SayeHyperrectangleDiagnostics& diagnostics,
    bool retain_interface_fragments = true)
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
    if (retain_interface_fragments) {
        diagnostics.interface_fragment_count +=
            static_cast<int>(leaf_cut.fragments.size());
        for (auto& fragment : leaf_cut.fragments) {
            (void)replaceWithRootPolishedRectangleFragment(
                fragment, request, input, rect, diagnostics);
            fragment.parent_cell = input.linearized_input.parent_cell;
            fragment.interface_marker = request.interface_marker;
            stampGeneratedInterfaceFragmentMetadata(
                fragment, cut, request, input, "rectangle");
            cut.fragments.push_back(std::move(fragment));
        }
    }
    for (auto& region : leaf_cut.volume_regions) {
        region.parent_cell = input.linearized_input.parent_cell;
        region.interface_marker = request.interface_marker;
        region.parent_measure = parent_measure;
        region.volume_fraction =
            parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
        region.full_cell_equivalent =
            parent_measure > Real{0.0} &&
            std::abs(region.measure - parent_measure) <=
                measureTolerance(request.tolerance, parent_measure);
        if (!region.full_cell_equivalent &&
            std::abs(region.measure - rectangleMeasure(rect)) <=
                measureTolerance(request.tolerance, region.measure)) {
            region.quadrature_points = rectangleVolumeQuadraturePoints(
                rect, request.resolvedVolumeQuadratureOrder());
            region.reference_subcells = rectangleReferenceSubcells(rect);
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "linearized-rectangle");
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
        stampGeneratedInterfaceFragmentMetadata(
            fragment, cut, request, input, "triangle");
        cut.fragments.push_back(std::move(fragment));
    }
    for (auto& region : leaf_cut.volume_regions) {
        region.parent_cell = input.linearized_input.parent_cell;
        region.interface_marker = request.interface_marker;
        region.parent_measure = parent_measure;
        region.volume_fraction =
            parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
        region.full_cell_equivalent =
            parent_measure > Real{0.0} &&
            std::abs(region.measure - parent_measure) <=
                measureTolerance(request.tolerance, parent_measure);
        if (!region.full_cell_equivalent &&
            std::abs(region.measure - triangleMeasure(tri)) <=
                measureTolerance(request.tolerance, region.measure)) {
            region.quadrature_points = triangleVolumeQuadraturePoints(
                tri.a,
                tri.b,
                tri.c,
                request.resolvedVolumeQuadratureOrder());
            region.reference_subcells = {
                referenceTriangle(tri.a, tri.b, tri.c)};
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "linearized-triangle");
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

[[nodiscard]] interfaces::LevelSetCellCutResult makeLinearizedTetrahedronCut(
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet)
{
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
    return leaf_cut;
}

void appendLinearizedTetrahedronCutResult(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    Real parent_measure,
    interfaces::LevelSetCellCutResult leaf_cut,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    ++diagnostics.linearized_leaf_count;
    diagnostics.interface_fragment_count +=
        static_cast<int>(leaf_cut.fragments.size());
    for (auto& fragment : leaf_cut.fragments) {
        if (request.implicit_quadrature_backend !=
            "SayeHyperrectangle") {
            (void)replaceWithRootPolishedTetrahedronFragment(
                fragment, request, input, tet, diagnostics);
        }
        fragment.parent_cell = input.linearized_input.parent_cell;
        fragment.interface_marker = request.interface_marker;
        stampGeneratedInterfaceFragmentMetadata(
            fragment, cut, request, input, "tetrahedron");
        cut.fragments.push_back(std::move(fragment));
    }
    for (auto& region : leaf_cut.volume_regions) {
        region.parent_cell = input.linearized_input.parent_cell;
        region.interface_marker = request.interface_marker;
        region.parent_measure = parent_measure;
        region.volume_fraction =
            parent_measure > Real{0.0} ? region.measure / parent_measure : Real{0.0};
        region.full_cell_equivalent =
            parent_measure > Real{0.0} &&
            std::abs(region.measure - parent_measure) <=
                measureTolerance(request.tolerance, parent_measure);
        if (!region.full_cell_equivalent &&
            std::abs(region.measure - tetrahedronMeasure(tet)) <=
                measureTolerance(request.tolerance, region.measure)) {
            region.quadrature_points = tetrahedronVolumeQuadraturePoints(
                tet, request.resolvedVolumeQuadratureOrder());
            region.reference_subcells = {
                referenceTetrahedron(tet.a, tet.b, tet.c, tet.d)};
            for (auto& point : region.quadrature_points) {
                point.normal = region.normal;
            }
        }
        stampGeneratedVolumeRegionMetadata(
            region, cut, request, input, "linearized-tetrahedron");
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
    appendLinearizedTetrahedronCutResult(
        cut,
        request,
        input,
        tet,
        parent_measure,
        makeLinearizedTetrahedronCut(request, input, tet),
        diagnostics);
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
    SayeHyperrectangleDiagnostics& diagnostics,
    bool retain_interface_fragments = true)
{
    const auto first_fragment = cut.fragments.size();
    const int initial_fragment_count = diagnostics.interface_fragment_count;
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
    if (!retain_interface_fragments) {
        cut.fragments.resize(first_fragment);
        diagnostics.interface_fragment_count = initial_fragment_count;
    }
}

void appendUniqueSplitCoordinate(std::vector<Real>& coordinates,
                                 Real coordinate,
                                 Real lower,
                                 Real upper,
                                 Real tolerance)
{
    if (coordinate <= lower + tolerance || coordinate >= upper - tolerance) {
        return;
    }
    const auto existing =
        std::find_if(coordinates.begin(), coordinates.end(), [&](Real value) {
            return std::abs(value - coordinate) <= tolerance;
        });
    if (existing == coordinates.end()) {
        coordinates.push_back(coordinate);
    }
}

void appendTopologyAwareLinearizedRectangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    Real parent_measure,
    int terminal_extra_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    int local_iterations = 0;
    const auto edge_roots =
        rectangleEdgeRoots(input, request, rect, local_iterations);
    diagnostics.root_finder_iteration_count += local_iterations;
    if (edge_roots.size() > 2u &&
        terminal_extra_depth < kTerminalTopologyExtraSubdivisionDepth) {
        ++diagnostics.terminal_topology_refinement_count;
        diagnostics.max_terminal_topology_extra_depth =
            std::max(diagnostics.max_terminal_topology_extra_depth,
                     terminal_extra_depth + 1);
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
            appendTopologyAwareLinearizedRectangleCut(cut,
                                                      request,
                                                      input,
                                                      child,
                                                      parent_measure,
                                                      terminal_extra_depth + 1,
                                                      diagnostics);
        }
        return;
    }

    appendLinearizedRectangleCut(cut, request, input, rect, parent_measure,
                                 diagnostics);
}

[[nodiscard]] bool appendHintRefinedRectangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Rectangle2D& rect,
    Real parent_measure,
    const std::vector<std::array<Real, 3>>& samples,
    int terminal_extra_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    const Real sign_tolerance = request.implicit_cut_root_tolerance;
    const std::array<std::array<Real, 3>, 4> corners{{
        {{rect.xmin, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymin, 0.0}},
        {{rect.xmax, rect.ymax, 0.0}},
        {{rect.xmin, rect.ymax, 0.0}},
    }};

    bool corners_strictly_negative = true;
    bool corners_strictly_positive = true;
    std::array<Real, 4> corner_values{{0.0, 0.0, 0.0, 0.0}};
    std::size_t corner_index = 0u;
    for (const auto& corner : corners) {
        const Real value = signedLevelSetValue(input, corner);
        corner_values[corner_index++] = value;
        corners_strictly_negative =
            corners_strictly_negative && value < -sign_tolerance;
        corners_strictly_positive =
            corners_strictly_positive && value > sign_tolerance;
    }

    const Real coordinate_tolerance = rootCoordinateTolerance(request);
    std::vector<Real> xs{rect.xmin, rect.xmax};
    std::vector<Real> ys{rect.ymin, rect.ymax};
    bool found_opposite_sample = false;
    for (const auto& point : samples) {
        if (!pointInsideRectangle(rect, point, coordinate_tolerance)) {
            continue;
        }
        const Real value = signedLevelSetValue(input, point);
        bool opposite_sample =
            (corners_strictly_positive && value < -sign_tolerance) ||
            (corners_strictly_negative && value > sign_tolerance);
        if (!opposite_sample &&
            !corners_strictly_positive &&
            !corners_strictly_negative) {
            const Real sx =
                (point[0] - rect.xmin) / std::max(rect.xmax - rect.xmin,
                                                  Real{1.0e-30});
            const Real sy =
                (point[1] - rect.ymin) / std::max(rect.ymax - rect.ymin,
                                                  Real{1.0e-30});
            const Real bilinear_value =
                (Real{1.0} - sx) * (Real{1.0} - sy) * corner_values[0] +
                sx * (Real{1.0} - sy) * corner_values[1] +
                sx * sy * corner_values[2] +
                (Real{1.0} - sx) * sy * corner_values[3];
            opposite_sample =
                (bilinear_value > sign_tolerance && value < -sign_tolerance) ||
                (bilinear_value < -sign_tolerance && value > sign_tolerance);
        }
        if (!opposite_sample) {
            continue;
        }
        found_opposite_sample = true;
        appendUniqueSplitCoordinate(
            xs, point[0], rect.xmin, rect.xmax, coordinate_tolerance);
        appendUniqueSplitCoordinate(
            ys, point[1], rect.ymin, rect.ymax, coordinate_tolerance);
    }

    if (!found_opposite_sample || (xs.size() == 2u && ys.size() == 2u)) {
        return false;
    }

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    ++diagnostics.subdivision_count;
    for (std::size_t iy = 0u; iy + 1u < ys.size(); ++iy) {
        for (std::size_t ix = 0u; ix + 1u < xs.size(); ++ix) {
            const Rectangle2D child{xs[ix], xs[ix + 1u], ys[iy], ys[iy + 1u]};
            if (rectangleMeasure(child) <= Real{0.0}) {
                continue;
            }
            appendTopologyAwareLinearizedRectangleCut(cut,
                                                      request,
                                                      input,
                                                      child,
                                                      parent_measure,
                                                      terminal_extra_depth + 1,
                                                      diagnostics);
        }
    }
    return true;
}

[[nodiscard]] bool appendHintRefinedBoxCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Box3D& box,
    Real parent_measure,
    const std::vector<std::array<Real, 3>>& samples,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    const Real sign_tolerance = request.implicit_cut_root_tolerance;
    const auto corners = boxVertices(box);

    bool corners_strictly_negative = true;
    bool corners_strictly_positive = true;
    for (const auto& corner : corners) {
        const Real value = signedLevelSetValue(input, corner);
        corners_strictly_negative =
            corners_strictly_negative && value < -sign_tolerance;
        corners_strictly_positive =
            corners_strictly_positive && value > sign_tolerance;
    }
    if (!corners_strictly_negative && !corners_strictly_positive) {
        return false;
    }

    const Real coordinate_tolerance = rootCoordinateTolerance(request);
    std::vector<Real> xs{box.xmin, box.xmax};
    std::vector<Real> ys{box.ymin, box.ymax};
    std::vector<Real> zs{box.zmin, box.zmax};
    bool found_opposite_sample = false;
    for (const auto& point : samples) {
        if (!pointInsideBox(box, point, coordinate_tolerance)) {
            continue;
        }
        const Real value = signedLevelSetValue(input, point);
        const bool opposite_sample =
            (corners_strictly_positive && value < -sign_tolerance) ||
            (corners_strictly_negative && value > sign_tolerance);
        if (!opposite_sample) {
            continue;
        }
        found_opposite_sample = true;
        appendUniqueSplitCoordinate(
            xs, point[0], box.xmin, box.xmax, coordinate_tolerance);
        appendUniqueSplitCoordinate(
            ys, point[1], box.ymin, box.ymax, coordinate_tolerance);
        appendUniqueSplitCoordinate(
            zs, point[2], box.zmin, box.zmax, coordinate_tolerance);
    }
    if (!found_opposite_sample ||
        (xs.size() == 2u && ys.size() == 2u && zs.size() == 2u)) {
        return false;
    }

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(zs.begin(), zs.end());
    ++diagnostics.subdivision_count;
    for (std::size_t iz = 0u; iz + 1u < zs.size(); ++iz) {
        for (std::size_t iy = 0u; iy + 1u < ys.size(); ++iy) {
            for (std::size_t ix = 0u; ix + 1u < xs.size(); ++ix) {
                const Box3D child{xs[ix], xs[ix + 1u],
                                  ys[iy], ys[iy + 1u],
                                  zs[iz], zs[iz + 1u]};
                if (boxMeasure(child) <= Real{0.0}) {
                    continue;
                }
                appendLinearizedBoxCut(
                    cut, request, input, child, parent_measure, diagnostics);
            }
        }
    }
    return true;
}

void appendAdaptiveRectangleCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const TensorBernsteinCertificate* certificate,
    const Rectangle2D& rect,
    Real parent_measure,
    int depth,
    int max_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    diagnostics.max_depth_reached =
        std::max(diagnostics.max_depth_reached, depth);
    auto samples = rectangleSamplePoints(rect);
    appendHighOrderSamplesInsideRectangle(
        samples, input, rect, rootCoordinateTolerance(request));
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

    std::optional<OutwardInterval> certified_range;
    bool certified_negative = false;
    bool certified_positive = false;
    if (certificate != nullptr) {
        certified_range = certifiedRectangleRange(*certificate, rect);
        ++diagnostics.certified_range_query_count;
        certified_negative =
            certified_range->upper <
            -static_cast<long double>(request.implicit_cut_root_tolerance);
        certified_positive =
            certified_range->lower >
            static_cast<long double>(request.implicit_cut_root_tolerance);
        min_signed = std::min(
            min_signed, intervalLowerAsReal(certified_range->lower));
        max_signed = std::max(
            max_signed, intervalUpperAsReal(certified_range->upper));
    }

    if ((certificate == nullptr && (!has_negative || !has_positive)) ||
        certified_negative || certified_positive) {
        if (certificate != nullptr) {
            ++diagnostics.certified_full_sign_region_count;
        }
        appendFullRectangleRegion(
            cut,
            request,
            input,
            rect,
            (certificate != nullptr ? certified_negative : has_negative)
                ? geometry::CutIntegrationSide::Negative
                : geometry::CutIntegrationSide::Positive,
            parent_measure,
            min_signed,
            max_signed,
            diagnostics);
        return;
    }

    if (certificate != nullptr &&
        request.required_implicit_cut_backend_qualification ==
            "ProductionQualified") {
        ++diagnostics.certified_topology_query_count;
        const auto regular_graph = certifiedRegularGraph(
            *certificate,
            rect,
            request.implicit_cut_root_tolerance);
        const bool topology_resolution_sufficient =
            regular_graph.has_value() &&
            depth >= max_depth + 3;
        if (topology_resolution_sufficient) {
            ++diagnostics.certified_regular_graph_leaf_count;
            const bool retain_interface =
                certifiedGraphLeafOwnsInterface(
                    *certificate,
                    *regular_graph,
                    {{rect.xmin, rect.ymin, 0.0}});
            const auto first_fragment = cut.fragments.size();
            appendLinearizedRectangleCut(
                cut,
                request,
                input,
                rect,
                parent_measure,
                diagnostics,
                retain_interface);
            if (retain_interface &&
                cut.fragments.size() == first_fragment &&
                (regular_graph->lower_face_zero ||
                 regular_graph->upper_face_zero)) {
                appendCertifiedAlignedRectangleFragment(
                    cut,
                    request,
                    input,
                    rect,
                    *regular_graph,
                    diagnostics);
            }
            return;
        }
    }

    if (depth >= max_depth) {
        const int extra_depth = depth - max_depth;
        int local_iterations = 0;
        const auto edge_roots =
            rectangleEdgeRoots(input, request, rect, local_iterations);
        diagnostics.root_finder_iteration_count += local_iterations;
        if (certificate != nullptr &&
            request.required_implicit_cut_backend_qualification ==
                "ProductionQualified") {
            if (extra_depth >= kCertifiedRangeExtraSubdivisionDepth) {
                const auto boundary_touch =
                    certifiedOneSidedBoundaryTouch(
                        *certificate,
                        input,
                        rect,
                        *certified_range,
                        request.implicit_cut_root_tolerance);
                if (boundary_touch.has_value()) {
                    ++diagnostics.certified_full_sign_region_count;
                    appendFullRectangleRegion(
                        cut,
                        request,
                        input,
                        rect,
                        *boundary_touch,
                        parent_measure,
                        intervalLowerAsReal(certified_range->lower),
                        intervalUpperAsReal(certified_range->upper),
                        diagnostics);
                    return;
                }
                if (request.required_implicit_cut_backend_qualification !=
                    "ProductionQualified") {
                    appendLinearizedRectangleCut(
                        cut,
                        request,
                        input,
                        rect,
                        parent_measure,
                        diagnostics);
                    return;
                }
                ++diagnostics.certified_topology_fail_closed_count;
                throw std::invalid_argument(
                    "SayeHyperrectangle could not certify regular graph topology for an ambiguous Q2/Q3 tensor polynomial leaf within the bounded topology-refinement depth; leaf=[" +
                    std::to_string(rect.xmin) + "," +
                    std::to_string(rect.xmax) + "]x[" +
                    std::to_string(rect.ymin) + "," +
                    std::to_string(rect.ymax) + "]; certified_range=[" +
                    formatReal(intervalLowerAsReal(certified_range->lower)) +
                    "," +
                    formatReal(intervalUpperAsReal(certified_range->upper)) +
                    "]");
            }
            ++diagnostics.terminal_topology_refinement_count;
            ++diagnostics.certified_same_sign_refinement_count;
            ++diagnostics.certified_topology_refinement_count;
            diagnostics.max_terminal_topology_extra_depth =
                std::max(diagnostics.max_terminal_topology_extra_depth,
                         extra_depth + 1);
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
                    certificate,
                    child,
                    parent_measure,
                    depth + 1,
                    max_depth,
                    diagnostics);
            }
            return;
        }
        const int topology_extra_depth_limit =
            kTerminalTopologyExtraSubdivisionDepth;
        if (edge_roots.size() > 2u &&
            extra_depth < topology_extra_depth_limit) {
            ++diagnostics.terminal_topology_refinement_count;
            diagnostics.max_terminal_topology_extra_depth =
                std::max(diagnostics.max_terminal_topology_extra_depth,
                         extra_depth + 1);
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
                    certificate,
                    child,
                    parent_measure,
                    depth + 1,
                    max_depth,
                    diagnostics);
            }
            return;
        }
        if (appendHintRefinedRectangleCut(cut,
                                          request,
                                          input,
                                          rect,
                                          parent_measure,
                                          samples,
                                          extra_depth,
                                          diagnostics)) {
            return;
        }
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
            certificate,
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
    const TensorBernsteinCertificate* certificate,
    const Box3D& box,
    Real parent_measure,
    int depth,
    int max_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
    diagnostics.max_depth_reached =
        std::max(diagnostics.max_depth_reached, depth);
    auto samples = boxSamplePoints(box);
    appendHighOrderSamplesInsideBox(
        samples, input, box, rootCoordinateTolerance(request));
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

    std::optional<OutwardInterval> certified_range;
    bool certified_negative = false;
    bool certified_positive = false;
    if (certificate != nullptr) {
        certified_range = certifiedBoxRange(*certificate, box);
        ++diagnostics.certified_range_query_count;
        certified_negative =
            certified_range->upper <
            -static_cast<long double>(request.implicit_cut_root_tolerance);
        certified_positive =
            certified_range->lower >
            static_cast<long double>(request.implicit_cut_root_tolerance);
        min_signed = std::min(
            min_signed, intervalLowerAsReal(certified_range->lower));
        max_signed = std::max(
            max_signed, intervalUpperAsReal(certified_range->upper));
    }

    if ((certificate == nullptr && (!has_negative || !has_positive)) ||
        certified_negative || certified_positive) {
        if (certificate != nullptr) {
            ++diagnostics.certified_full_sign_region_count;
        }
        appendFullBoxRegion(
            cut,
            request,
            input,
            box,
            (certificate != nullptr ? certified_negative : has_negative)
                ? geometry::CutIntegrationSide::Negative
                : geometry::CutIntegrationSide::Positive,
            parent_measure,
            min_signed,
            max_signed,
            diagnostics);
        return;
    }

    if (depth >= max_depth) {
        const int extra_depth = depth - max_depth;
        if (certificate != nullptr &&
            request.required_implicit_cut_backend_qualification ==
                "ProductionQualified") {
            ++diagnostics.certified_topology_query_count;
            const auto regular_graph = certifiedRegularGraph(
                *certificate,
                box,
                request.implicit_cut_root_tolerance);
            if (regular_graph.has_value()) {
                ++diagnostics.certified_regular_graph_leaf_count;
                const bool retain_interface =
                    certifiedGraphLeafOwnsInterface(
                        *certificate,
                        *regular_graph,
                        {{box.xmin, box.ymin, box.zmin}});
                appendLinearizedBoxCut(
                    cut,
                    request,
                    input,
                    box,
                    parent_measure,
                    diagnostics,
                    retain_interface);
                return;
            }
            if (extra_depth >= kCertifiedRangeExtraSubdivisionDepth) {
                if (request.required_implicit_cut_backend_qualification !=
                    "ProductionQualified") {
                    appendLinearizedBoxCut(
                        cut,
                        request,
                        input,
                        box,
                        parent_measure,
                        diagnostics);
                    return;
                }
                ++diagnostics.certified_topology_fail_closed_count;
                throw std::invalid_argument(
                    "SayeHyperrectangle could not certify regular graph topology for an ambiguous Q2/Q3 tensor polynomial leaf within the bounded topology-refinement depth");
            }
            ++diagnostics.terminal_topology_refinement_count;
            ++diagnostics.certified_same_sign_refinement_count;
            ++diagnostics.certified_topology_refinement_count;
            diagnostics.max_terminal_topology_extra_depth =
                std::max(diagnostics.max_terminal_topology_extra_depth,
                         extra_depth + 1);
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
                            certificate,
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
            return;
        }
        if (appendHintRefinedBoxCut(
                cut, request, input, box, parent_measure, samples, diagnostics)) {
            return;
        }
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
                    certificate,
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

void appendTopologyAwareLinearizedTetrahedronCut(
    interfaces::LevelSetCellCutResult& cut,
    const interfaces::CutInterfaceDomainRequest& request,
    const ImplicitCutQuadratureBackendCellInput& input,
    const Tetrahedron3D& tet,
    Real parent_measure,
    int terminal_extra_depth,
    SayeHyperrectangleDiagnostics& diagnostics)
{
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

    auto leaf_cut = makeLinearizedTetrahedronCut(request, input, tet);
    std::size_t linearized_edge_root_count = 0u;
    for (const auto& fragment : leaf_cut.fragments) {
        if (fragment.active()) {
            linearized_edge_root_count += fragment.vertices.size();
        }
    }

    int local_iterations = 0;
    const auto sampled_edge_roots =
        tetrahedronEdgeRoots(input, request, tet, local_iterations);
    diagnostics.root_finder_iteration_count += local_iterations;
    const bool topology_mismatch =
        !leaf_cut.hasActiveFragments() ||
        sampled_edge_roots.size() != linearized_edge_root_count;
    if (topology_mismatch &&
        terminal_extra_depth < kTerminalTopologyExtraSubdivisionDepth) {
        ++diagnostics.terminal_topology_refinement_count;
        diagnostics.max_terminal_topology_extra_depth =
            std::max(diagnostics.max_terminal_topology_extra_depth,
                     terminal_extra_depth + 1);
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
            appendTopologyAwareLinearizedTetrahedronCut(
                cut,
                request,
                input,
                child,
                parent_measure,
                terminal_extra_depth + 1,
                diagnostics);
        }
        return;
    }

    appendLinearizedTetrahedronCutResult(cut,
                                         request,
                                         input,
                                         tet,
                                         parent_measure,
                                         std::move(leaf_cut),
                                         diagnostics);
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
        appendTopologyAwareLinearizedTetrahedronCut(
            cut,
            request,
            input,
            tet,
            parent_measure,
            /*terminal_extra_depth=*/0,
            diagnostics);
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
           "; terminal_topology_refinements=" +
           std::to_string(diagnostics.terminal_topology_refinement_count) +
           "; max_terminal_topology_extra_depth=" +
           std::to_string(diagnostics.max_terminal_topology_extra_depth) +
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
           "; certified_tensor_range_available=" +
           std::to_string(diagnostics.certified_tensor_range_available) +
           "; certified_range_queries=" +
           std::to_string(diagnostics.certified_range_query_count) +
           "; certified_full_sign_regions=" +
           std::to_string(diagnostics.certified_full_sign_region_count) +
           "; certified_same_sign_refinements=" +
           std::to_string(
               diagnostics.certified_same_sign_refinement_count) +
           "; certified_range_fail_closed=" +
           std::to_string(diagnostics.certified_range_fail_closed_count) +
           "; certified_topology_queries=" +
           std::to_string(diagnostics.certified_topology_query_count) +
           "; certified_regular_graph_leaves=" +
           std::to_string(diagnostics.certified_regular_graph_leaf_count) +
           "; certified_topology_refinements=" +
           std::to_string(diagnostics.certified_topology_refinement_count) +
           "; certified_topology_fail_closed=" +
           std::to_string(diagnostics.certified_topology_fail_closed_count) +
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
    if (!diagnostics.first_curved_fragment_failure_detail.empty()) {
        diagnostic += "; first_curved_fragment_failure_detail={" +
                      diagnostics.first_curved_fragment_failure_detail + "}";
    }
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
           "; terminal_topology_refinements=" +
           std::to_string(diagnostics.terminal_topology_refinement_count) +
           "; max_terminal_topology_extra_depth=" +
           std::to_string(diagnostics.max_terminal_topology_extra_depth) +
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
           "; certified_tensor_range_available=" +
           std::to_string(diagnostics.certified_tensor_range_available) +
           "; certified_range_queries=" +
           std::to_string(diagnostics.certified_range_query_count) +
           "; certified_full_sign_regions=" +
           std::to_string(diagnostics.certified_full_sign_region_count) +
           "; certified_same_sign_refinements=" +
           std::to_string(
               diagnostics.certified_same_sign_refinement_count) +
           "; certified_range_fail_closed=" +
           std::to_string(diagnostics.certified_range_fail_closed_count) +
           "; certified_topology_queries=" +
           std::to_string(diagnostics.certified_topology_query_count) +
           "; certified_regular_graph_leaves=" +
           std::to_string(diagnostics.certified_regular_graph_leaf_count) +
           "; certified_topology_refinements=" +
           std::to_string(diagnostics.certified_topology_refinement_count) +
           "; certified_topology_fail_closed=" +
           std::to_string(diagnostics.certified_topology_fail_closed_count) +
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
    if (!diagnostics.first_curved_fragment_failure_detail.empty()) {
        diagnostic += "; first_curved_fragment_failure_detail={" +
                      diagnostics.first_curved_fragment_failure_detail + "}";
    }
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
           "; terminal_topology_refinements=" +
           std::to_string(diagnostics.terminal_topology_refinement_count) +
           "; max_terminal_topology_extra_depth=" +
           std::to_string(diagnostics.max_terminal_topology_extra_depth) +
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
           "; terminal_topology_refinements=" +
           std::to_string(diagnostics.terminal_topology_refinement_count) +
           "; max_terminal_topology_extra_depth=" +
           std::to_string(diagnostics.max_terminal_topology_extra_depth) +
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
    if (!diagnostics.first_curved_fragment_failure_detail.empty()) {
        diagnostic += "; first_curved_fragment_failure_detail={" +
                      diagnostics.first_curved_fragment_failure_detail + "}";
    }
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
        for (auto& subcell : region.reference_subcells) {
            subcell.measure_scale *= scale;
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
            std::max(0, request.implicit_cut_max_subdivision_depth);
        const auto tensor_range_certificate =
            makeTensorBernsteinCertificate(mesh_dimension, input);
        const auto cell_id = input.linearized_input.parent_cell;
        if (request.required_implicit_cut_backend_qualification ==
                "ProductionQualified" &&
            input.evaluator->interpolationOrder(cell_id) > 1 &&
            !tensor_range_certificate.has_value()) {
            setUnavailableOrderMetadata(result, request);
            result.cut.supported = false;
            result.cut.degeneracy =
                interfaces::CutInterfaceDegeneracy::NoCut;
            result.cut.diagnostic =
                "production-qualified SayeHyperrectangle tensor polynomial "
                "range certification is limited to Q2 and Q3; "
                "higher orders require a certified range implementation";
            result.diagnostic_status =
                ImplicitCutQuadratureDiagnosticStatus::Failed;
            appendDetailedBackendDiagnostics(result, request);
            return finalizeBackendResult(
                std::move(result), request, backend_start);
        }
        SayeHyperrectangleDiagnostics diagnostics;
        diagnostics.certified_tensor_range_available =
            tensor_range_certificate.has_value() ? 1 : 0;
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
                tensor_range_certificate.has_value()
                    ? &*tensor_range_certificate
                    : nullptr,
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
                tensor_range_certificate.has_value()
                    ? &*tensor_range_certificate
                    : nullptr,
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
            std::max(0, request.implicit_cut_max_subdivision_depth);
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
        capability.supports_refreshed_frozen_quadrature =
            capability.supports_element_type;
        capability.supports_differentiated_quadrature =
            capability.supports_element_type;
        capability.validation_level_set_order = 1;
        if (!capability.supports_element_type) {
            capability.maximum_reported_interface_order = -1;
            capability.maximum_reported_volume_order = -1;
            return capability;
        }
        capability.maximum_reported_interface_order =
            mesh_dimension == 2 ? 5 : 2;
        capability.maximum_reported_volume_order =
            mesh_dimension == 2 ? 5 : 2;
        return capability;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        capability.implemented = true;
        capability.supports_element_type =
            supportsSayeHyperrectangleMilestone(mesh_dimension, element_type);
        capability.supports_high_order_geometry = true;
        capability.supports_refreshed_frozen_quadrature =
            capability.supports_element_type;
        capability.supports_differentiated_quadrature = false;
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
        capability.supports_refreshed_frozen_quadrature =
            capability.supports_element_type;
        capability.supports_differentiated_quadrature = false;
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
        capability.supports_refreshed_frozen_quadrature = false;
        capability.supports_differentiated_quadrature = false;
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
        capability.supports_refreshed_frozen_quadrature =
            selected_capability.supports_refreshed_frozen_quadrature;
        capability.supports_differentiated_quadrature =
            selected_capability.supports_differentiated_quadrature;
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
        if ((fragment.moment_certificate_order < 0) !=
            fragment.moment_certificate_points.empty()) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend returned incomplete interface moment-certificate evidence");
        }
        Real certificate_weight_sum = Real{0.0};
        for (const auto& point : fragment.moment_certificate_points) {
            geometry::CutQuadraturePoint qp;
            qp.point = point.point;
            qp.normal = point.normal;
            qp.weight = point.weight;
            qp.parent_coordinate = point.parent_coordinate;
            qp.reference_measure_factor = point.reference_measure_factor;
            qp.level_set_residual = point.level_set_residual;
            qp.gradient_norm = point.gradient_norm;
            if (!finitePoint(qp) || !(qp.weight > Real{0.0}) ||
                point.level_set_residual >
                    Real{10.0} * rootResidualTolerance(request) ||
                !(point.gradient_norm > Real{0.0}) ||
                !(norm3(point.normal) > Real{1.0e-30})) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend returned invalid interface moment-certificate evidence");
            }
            certificate_weight_sum += point.weight;
        }
        if (!fragment.moment_certificate_points.empty() &&
            std::abs(certificate_weight_sum - fragment.measure) >
                measureTolerance(request.tolerance, fragment.measure)) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend interface moment-certificate weights do not sum to the fragment measure; topology=" +
                    fragment.topology_id + "; certificate_weight_sum=" +
                    formatReal(certificate_weight_sum) +
                    "; fragment_measure=" + formatReal(fragment.measure));
        }
        if (!fragment.moment_certificate_points.empty()) {
            SurfacePatchRuleResult production_moments;
            production_moments.points = fragment.quadrature_points;
            SurfacePatchRuleResult reference_moments;
            reference_moments.points = fragment.moment_certificate_points;
            Real maximum_moment_error{0.0};
            Real maximum_moment_scaled_error{0.0};
            if (!surfacePatchMomentRulesAgree(
                    production_moments,
                    reference_moments,
                    fragment.moment_certificate_order,
                    request.tolerance,
                    maximum_moment_error,
                    maximum_moment_scaled_error)) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend interface quadrature does not reproduce its independent polynomial-moment reference");
            }
        }
        if (!fragment.quadrature_points.empty() &&
            fragment.measure > Real{0.0} &&
            std::abs(interface_weight_sum - fragment.measure) >
                measureTolerance(request.tolerance, fragment.measure)) {
            const Real abs_error =
                std::abs(interface_weight_sum - fragment.measure);
            const Real tolerance =
                measureTolerance(request.tolerance, fragment.measure);
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend interface quadrature weights do not sum to the fragment measure"
                "; parent_cell=" +
                    std::to_string(fragment.parent_cell) +
                "; fragment_measure=" + formatReal(fragment.measure) +
                "; interface_weight_sum=" + formatReal(interface_weight_sum) +
                "; abs_error=" + formatReal(abs_error) +
                "; tolerance=" + formatReal(tolerance) +
                "; quadrature_points=" +
                    std::to_string(fragment.quadrature_points.size()));
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
        if (fragment.active() &&
            fragment.kind == interfaces::CutInterfaceFragmentKind::CurvedPatch &&
            result.achieved_interface_quadrature_order > 1 &&
            (fragment.moment_certificate_order <
                 result.achieved_interface_quadrature_order ||
             fragment.moment_certificate_points.empty())) {
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend curved interface rule lacks an independent polynomial-moment reference");
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
            const Real abs_error =
                std::abs(volume_weight_sum - region.measure);
            const Real tolerance =
                measureTolerance(request.tolerance, region.measure);
            return failedValidation(
                ImplicitCutQuadratureDiagnosticStatus::Failed,
                "implicit cut backend volume quadrature weights do not sum to the region measure"
                "; parent_cell=" +
                    std::to_string(region.parent_cell) +
                "; side=" + std::string(sideTopologyToken(region.side)) +
                "; region_measure=" + formatReal(region.measure) +
                "; volume_weight_sum=" + formatReal(volume_weight_sum) +
                "; abs_error=" + formatReal(abs_error) +
                "; tolerance=" + formatReal(tolerance) +
                "; parent_measure=" + formatReal(region.parent_measure) +
                "; volume_fraction=" + formatReal(region.volume_fraction) +
                "; quadrature_points=" +
                    std::to_string(region.quadrature_points.size()));
        }
        if (!region.reference_subcells.empty()) {
            const int dimension = element_dimension(linearized_input.element_type);
            Real subcell_measure_sum = Real{0.0};
            for (const auto& subcell : region.reference_subcells) {
                const Real subcell_measure =
                    referenceSimplexMeasure(subcell, dimension);
                if (!(subcell_measure > Real{0.0}) ||
                    !std::isfinite(subcell_measure)) {
                    return failedValidation(
                        ImplicitCutQuadratureDiagnosticStatus::Failed,
                        "implicit cut backend returned an invalid reference-simplex decomposition");
                }
                subcell_measure_sum += subcell_measure;
            }
            if (std::abs(subcell_measure_sum - region.measure) >
                measureTolerance(request.tolerance, region.measure)) {
                return failedValidation(
                    ImplicitCutQuadratureDiagnosticStatus::Failed,
                    "implicit cut backend reference-simplex decomposition does not match the region measure"
                    "; region_measure=" + formatReal(region.measure) +
                    "; subcell_measure_sum=" +
                        formatReal(subcell_measure_sum));
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
