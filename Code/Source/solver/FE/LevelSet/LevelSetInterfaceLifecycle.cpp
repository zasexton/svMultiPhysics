#include "LevelSet/LevelSetInterfaceLifecycle.h"

#include "Basis/NodeOrderingConventions.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {

const char* generatedInterfaceGeometryModeName(
    GeneratedInterfaceGeometryMode mode) noexcept
{
    switch (mode) {
    case GeneratedInterfaceGeometryMode::LinearCorner:
        return "LinearCorner";
    case GeneratedInterfaceGeometryMode::HighOrderImplicit:
        return "HighOrderImplicit";
    }
    return "Unknown";
}

const char* implicitCutQuadratureBackendName(
    ImplicitCutQuadratureBackend backend) noexcept
{
    switch (backend) {
    case ImplicitCutQuadratureBackend::LinearCorner:
        return "LinearCorner";
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        return "SayeHyperrectangle";
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
        return "HighOrderSubcell";
    case ImplicitCutQuadratureBackend::MomentFit:
        return "MomentFit";
    case ImplicitCutQuadratureBackend::Auto:
        return "Auto";
    }
    return "Unknown";
}

const char* implicitCutFallbackPolicyName(
    ImplicitCutFallbackPolicy policy) noexcept
{
    switch (policy) {
    case ImplicitCutFallbackPolicy::Fail:
        return "Fail";
    case ImplicitCutFallbackPolicy::LinearCorner:
        return "LinearCorner";
    }
    return "Unknown";
}

const char* geometryTangentPolicyName(GeometryTangentPolicy policy) noexcept
{
    switch (policy) {
    case GeometryTangentPolicy::RefreshedFrozenQuadrature:
        return "RefreshedFrozenQuadrature";
    case GeometryTangentPolicy::DifferentiatedQuadrature:
        return "DifferentiatedQuadrature";
    }
    return "Unknown";
}

GeometryQuadratureSensitivitySupport geometryQuadratureSensitivitySupport(
    GeometryTangentPolicy policy)
{
    GeometryQuadratureSensitivitySupport support;
    support.policy = policy;
    if (policy == GeometryTangentPolicy::RefreshedFrozenQuadrature) {
        support.diagnostic =
            "RefreshedFrozenQuadrature refreshes generated quadrature before assembly and supports first-order Hadamard cut-volume/interface-measure shape terms where explicitly installed, but treats quadrature points, weights, measures, normals, curvature, and topology as fixed during tangent assembly";
        return support;
    }
    support.diagnostic =
        "DifferentiatedQuadrature is reserved until quadrature point, weight, measure, normal, and topology-transition sensitivities are implemented";
    return support;
}

namespace {

struct GeneratedInterfaceCellDiagnostics {
    std::size_t node_count{0};
    std::size_t corner_count{0};
    bool corner_linearized{false};
    ImplicitCutQuadratureBackend selected_backend{
        ImplicitCutQuadratureBackend::LinearCorner};
    int achieved_interface_quadrature_order{0};
    int achieved_volume_quadrature_order{0};
    bool fallback_used{false};
    std::size_t volume_quadrature_point_count{0};
    std::size_t interface_quadrature_point_count{0};
    double backend_elapsed_seconds{0.0};
    bool linear_full_cell_fast_path{false};
    std::string backend_diagnostic{};
};

[[nodiscard]] std::size_t implicitCutQuadratureBackendIndex(
    ImplicitCutQuadratureBackend backend) noexcept
{
    switch (backend) {
    case ImplicitCutQuadratureBackend::LinearCorner:
        return 0u;
    case ImplicitCutQuadratureBackend::SayeHyperrectangle:
        return 1u;
    case ImplicitCutQuadratureBackend::HighOrderSubcell:
        return 2u;
    case ImplicitCutQuadratureBackend::MomentFit:
        return 3u;
    case ImplicitCutQuadratureBackend::Auto:
        return 4u;
    }
    return 0u;
}

[[nodiscard]] std::string validateRuleProvenance(
    const geometry::CutQuadratureRule& rule)
{
    const auto& provenance = rule.provenance;
    if (provenance.embedded_geometry_id.empty()) {
        return "missing embedded geometry id";
    }
    if (provenance.cut_topology_id.empty()) {
        return "missing cut topology id";
    }
    if (provenance.parent_entity < static_cast<MeshIndex>(0)) {
        return "missing parent entity";
    }
    if (provenance.marker < 0) {
        return "missing interface marker";
    }
    if (provenance.cut_topology_revision == 0u) {
        return "missing cut topology revision";
    }
    if (provenance.implicit_geometry_mode.empty()) {
        return "missing implicit geometry mode";
    }
    if (provenance.implicit_quadrature_backend.empty()) {
        return "missing implicit quadrature backend";
    }
    if (provenance.selected_implicit_quadrature_backend.empty()) {
        return "missing selected implicit quadrature backend";
    }
    if (provenance.implicit_fallback_policy.empty()) {
        return "missing implicit fallback policy";
    }
    if (provenance.implicit_fallback_status.empty() ||
        provenance.implicit_fallback_status == "Unknown") {
        return "missing resolved implicit fallback status";
    }
    if (provenance.geometry_tangent_policy.empty()) {
        return "missing geometry tangent policy";
    }
    if (provenance.implicit_cut_root_tolerance <= Real{0.0}) {
        return "missing implicit cut root tolerance";
    }
    if (provenance.implicit_cut_root_coordinate_tolerance <= Real{0.0}) {
        return "missing implicit cut root coordinate tolerance";
    }
    if (provenance.implicit_cut_root_max_iterations <= 0) {
        return "missing implicit cut root max iterations";
    }
    if (provenance.requested_quadrature_order < 0) {
        return "missing requested quadrature order";
    }
    if (provenance.achieved_quadrature_order < 0) {
        return "missing achieved quadrature order";
    }
    if (provenance.achieved_quadrature_order >
        provenance.requested_quadrature_order) {
        return "achieved quadrature order exceeds requested order";
    }
    return {};
}

void validateGeneratedInterfaceRuleProvenance(
    const interfaces::LevelSetInterfaceDomain& domain)
{
    const auto interface_rules = domain.interfaceQuadratureRules();
    for (const auto& rule : interface_rules) {
        const auto diagnostic = validateRuleProvenance(rule);
        if (!diagnostic.empty()) {
            throw std::invalid_argument(
                "generated level-set interface produced incomplete interface "
                "quadrature provenance: " + diagnostic);
        }
    }

    const auto volume_rules = domain.volumeQuadratureRules();
    for (const auto& rule : volume_rules) {
        const auto diagnostic = validateRuleProvenance(rule);
        if (!diagnostic.empty()) {
            throw std::invalid_argument(
                "generated level-set interface produced incomplete volume "
                "quadrature provenance: " + diagnostic);
        }
    }
}

[[nodiscard]] std::size_t cornerCount(ElementType type)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return 3u;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return 4u;
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return 8u;
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return 6u;
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return 5u;
    default:
        return 0u;
    }
}

[[nodiscard]] const char* elementTypeDiagnosticName(ElementType type) noexcept
{
    switch (type) {
    case ElementType::Triangle3:
        return "Triangle3";
    case ElementType::Triangle6:
        return "Triangle6";
    case ElementType::Quad4:
        return "Quad4";
    case ElementType::Quad8:
        return "Quad8";
    case ElementType::Quad9:
        return "Quad9";
    case ElementType::Tetra4:
        return "Tetra4";
    case ElementType::Tetra10:
        return "Tetra10";
    case ElementType::Hex8:
        return "Hex8";
    case ElementType::Hex20:
        return "Hex20";
    case ElementType::Hex27:
        return "Hex27";
    case ElementType::Wedge6:
        return "Wedge6";
    case ElementType::Wedge15:
        return "Wedge15";
    case ElementType::Wedge18:
        return "Wedge18";
    case ElementType::Pyramid5:
        return "Pyramid5";
    case ElementType::Pyramid13:
        return "Pyramid13";
    case ElementType::Pyramid14:
        return "Pyramid14";
    default:
        return "Unsupported";
    }
}

[[nodiscard]] std::string backendCellDiagnostic(
    const ImplicitCutQuadratureBackendDriver& backend,
    GlobalIndex cell_id,
    ElementType type,
    const std::string& diagnostic)
{
    return levelSetImplicitCutBackendCellDiagnostic(
        backend.kind(), cell_id, type, diagnostic);
}

[[nodiscard]] bool isExplicitGlobalHighOrderBackend(
    ImplicitCutQuadratureBackend backend) noexcept
{
    return backend == ImplicitCutQuadratureBackend::SayeHyperrectangle ||
           backend == ImplicitCutQuadratureBackend::HighOrderSubcell ||
           backend == ImplicitCutQuadratureBackend::MomentFit;
}

[[nodiscard]] bool requestAllowsLinearCornerFallback(
    const interfaces::CutInterfaceDomainRequest& request) noexcept
{
    return request.implicit_fallback_policy ==
           implicitCutFallbackPolicyName(
               ImplicitCutFallbackPolicy::LinearCorner);
}

[[nodiscard]] bool linearCornerSupportsCell(int mesh_dimension,
                                            ElementType type) noexcept
{
    return implicitCutQuadratureBackendDriver(
               ImplicitCutQuadratureBackend::LinearCorner)
        .supports(mesh_dimension, type);
}

[[nodiscard]] std::string unsupportedGlobalBackendDiagnostic(
    const ImplicitCutQuadratureBackendCapability& capability,
    const interfaces::CutInterfaceDomainRequest& request,
    GlobalIndex cell_id,
    ElementType type)
{
    const bool fallback_requested = requestAllowsLinearCornerFallback(request);
    const bool fallback_supported =
        capability.implemented &&
        fallback_requested &&
        linearCornerSupportsCell(capability.mesh_dimension, type);
    std::string diagnostic =
        "global high-order backend validation rejected unsupported cut cell"
        "; backend=" +
        std::string(implicitCutQuadratureBackendName(capability.backend)) +
        "; cell=" + std::to_string(cell_id) +
        "; element_type=" + elementTypeDiagnosticName(type) +
        "; fallback_policy=" + request.implicit_fallback_policy +
        "; linear_corner_fallback_supported=" +
        std::string(fallback_supported ? "true" : "false") +
        "; capability_state=" +
        implicitCutQuadratureBackendQualificationName(capability.qualification) +
        "; production_qualified=" +
        std::string(
            capability.qualification ==
                    ImplicitCutQuadratureBackendQualification::ProductionQualified
                ? "true"
                : "false") +
        "; possible_interface_order=" +
        std::to_string(capability.maximum_reported_interface_order) +
        "; possible_volume_order=" +
        std::to_string(capability.maximum_reported_volume_order);
    if (!fallback_requested) {
        diagnostic +=
            "; use implicit_cut_quadrature_backend=Auto for mixed supported "
            "meshes or configure Implicit_cut_fallback_policy=LinearCorner "
            "for counted linear fallback on supported linear cell families";
    }
    return diagnostic;
}

[[nodiscard]] std::string insufficientBackendQualificationDiagnostic(
    const ImplicitCutQuadratureBackendCapability& capability,
    const interfaces::CutInterfaceDomainRequest& request,
    GlobalIndex cell_id,
    ElementType type)
{
    return "global high-order backend validation rejected non-production "
           "qualified cut cell; required=ProductionQualified; backend=" +
           std::string(implicitCutQuadratureBackendName(capability.backend)) +
           "; cell=" + std::to_string(cell_id) +
           "; element_type=" + elementTypeDiagnosticName(type) +
           "; fallback_policy=" + request.implicit_fallback_policy +
           "; capability_state=" +
           implicitCutQuadratureBackendQualificationName(capability.qualification) +
           "; diagnostic=" + capability.qualification_diagnostic;
}

void validateGlobalHighOrderBackendCellSupport(
    const assembly::IMeshAccess& mesh,
    ImplicitCutQuadratureBackend backend,
    const interfaces::CutInterfaceDomainRequest& request,
    bool require_production_qualified_backend)
{
    const bool validate_support = isExplicitGlobalHighOrderBackend(backend);
    const bool validate_qualification = require_production_qualified_backend;
    if (!validate_support && !validate_qualification) {
        return;
    }

    std::string first_error;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        if (!first_error.empty()) {
            return;
        }
        const auto type = mesh.getCellType(cell_id);
        const auto capability =
            implicitCutQuadratureBackendCapability(
                backend, mesh.dimension(), type);
        const bool supported =
            capability.implemented && capability.supports_element_type;
        const bool fallback_supported =
            capability.implemented &&
            requestAllowsLinearCornerFallback(request) &&
            linearCornerSupportsCell(mesh.dimension(), type);
        if (validate_support && !supported && !fallback_supported) {
            first_error =
                backendCellDiagnostic(
                    implicitCutQuadratureBackendDriver(backend),
                    cell_id,
                    type,
                    unsupportedGlobalBackendDiagnostic(
                        capability, request, cell_id, type));
            return;
        }
        if (validate_qualification &&
            (!supported ||
             capability.qualification !=
                 ImplicitCutQuadratureBackendQualification::ProductionQualified)) {
            first_error =
                backendCellDiagnostic(
                    implicitCutQuadratureBackendDriver(backend),
                    cell_id,
                    type,
                    insufficientBackendQualificationDiagnostic(
                        capability, request, cell_id, type));
            return;
        }
    });

    if (!first_error.empty()) {
        throw std::invalid_argument(first_error);
    }
}

void stampSelectedImplicitBackend(
    interfaces::LevelSetCellCutResult& cut,
    ImplicitCutQuadratureBackend backend)
{
    const std::string backend_name = implicitCutQuadratureBackendName(backend);
    for (auto& fragment : cut.fragments) {
        fragment.implicit_quadrature_backend = backend_name;
    }
    for (auto& region : cut.volume_regions) {
        region.implicit_quadrature_backend = backend_name;
    }
}

void stampImplicitFallbackStatus(
    interfaces::LevelSetCellCutResult& cut,
    bool fallback_used)
{
    const std::string fallback_status = fallback_used ? "Used" : "None";
    for (auto& fragment : cut.fragments) {
        fragment.implicit_fallback_status = fallback_status;
    }
    for (auto& region : cut.volume_regions) {
        region.implicit_fallback_status = fallback_status;
    }
}

[[nodiscard]] Real coefficientAtVertex(const dofs::EntityDofMap& entity_map,
                                       GlobalIndex vertex,
                                       std::span<const Real> coefficients)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::invalid_argument(
            "generated level-set interface requires one scalar DOF per mesh vertex");
    }
    const auto dof = dofs.front();
    if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
        throw std::invalid_argument(
            "generated level-set interface found a vertex DOF outside the coefficient span");
    }
    return coefficients[static_cast<std::size_t>(dof)];
}

[[nodiscard]] std::array<Real, 3> referenceCornerCoordinate(ElementType type,
                                                           std::size_t local_node)
{
    const auto point = basis::NodeOrdering::get_node_coords(type, local_node);
    return {{point[0], point[1], point[2]}};
}

[[nodiscard]] std::array<Real, 3> minReferenceCoordinate(
    const std::vector<std::array<Real, 3>>& coordinates) noexcept
{
    std::array<Real, 3> out{{0.0, 0.0, 0.0}};
    if (coordinates.empty()) {
        return out;
    }
    out = coordinates.front();
    for (const auto& point : coordinates) {
        out[0] = std::min(out[0], point[0]);
        out[1] = std::min(out[1], point[1]);
        out[2] = std::min(out[2], point[2]);
    }
    return out;
}

[[nodiscard]] std::array<Real, 3> maxReferenceCoordinate(
    const std::vector<std::array<Real, 3>>& coordinates) noexcept
{
    std::array<Real, 3> out{{1.0, 1.0, 1.0}};
    if (coordinates.empty()) {
        return out;
    }
    out = coordinates.front();
    for (const auto& point : coordinates) {
        out[0] = std::max(out[0], point[0]);
        out[1] = std::max(out[1], point[1]);
        out[2] = std::max(out[2], point[2]);
    }
    return out;
}

[[nodiscard]] bool linearFullCellFastPathApplies(
    const LevelSetCellEvaluator& evaluator,
    GlobalIndex cell_id,
    const interfaces::LevelSetCellCutInput& input,
    Real isovalue,
    Real tolerance) noexcept
{
    if (evaluator.interpolationOrder(cell_id) > 1) {
        return false;
    }
    bool strictly_negative = true;
    bool strictly_positive = true;
    for (const auto value : input.level_set_values) {
        const auto signed_value = value - isovalue;
        strictly_negative = strictly_negative && signed_value < -tolerance;
        strictly_positive = strictly_positive && signed_value > tolerance;
    }
    return strictly_negative || strictly_positive;
}

[[nodiscard]] std::optional<GeneratedInterfaceCellDiagnostics>
appendLinearFullCellFastPath(
    interfaces::LevelSetInterfaceDomain& domain,
    int mesh_dimension,
    ElementType type,
    const interfaces::LevelSetCellCutInput& input,
    const LevelSetCellEvaluator& evaluator,
    GlobalIndex cell_id,
    std::size_t node_count,
    std::size_t corner_count,
    ImplicitCutQuadratureBackend selected_backend)
{
    const auto& request = domain.request();
    if (!linearFullCellFastPathApplies(
            evaluator, cell_id, input, request.isovalue, request.tolerance)) {
        return std::nullopt;
    }

    interfaces::LevelSetCellCutResult cut;
    if (mesh_dimension == 2 && interfaces::supportsLinearLevelSetCellCut2D(type)) {
        cut = interfaces::cutLinearLevelSetCell2D(request, input);
    } else if (mesh_dimension == 3 &&
               interfaces::supportsLinearLevelSetCellCut3D(type)) {
        cut = interfaces::cutLinearLevelSetCell3D(request, input);
    } else {
        return std::nullopt;
    }
    if (!cut.supported || !cut.fragments.empty() || cut.volume_regions.empty()) {
        return std::nullopt;
    }

    std::size_t volume_quadrature_point_count = 0u;
    int achieved_volume_quadrature_order =
        request.resolvedVolumeQuadratureOrder();
    for (auto& region : cut.volume_regions) {
        region.implicit_quadrature_backend =
            implicitCutQuadratureBackendName(selected_backend);
        region.implicit_fallback_status = "None";
        volume_quadrature_point_count += region.quadraturePointCount();
        const int region_order =
            region.achieved_quadrature_order >= 0
                ? std::min(request.resolvedVolumeQuadratureOrder(),
                           region.achieved_quadrature_order)
                : interfaces::implementedLevelSetCutVolumeExactOrder(
                      request.resolvedVolumeQuadratureOrder());
        achieved_volume_quadrature_order =
            std::min(achieved_volume_quadrature_order, region_order);
        domain.addVolumeRegion(std::move(region));
    }

    return GeneratedInterfaceCellDiagnostics{
        .node_count = node_count,
        .corner_count = corner_count,
        .corner_linearized = false,
        .selected_backend = selected_backend,
        .achieved_interface_quadrature_order =
            request.resolvedInterfaceQuadratureOrder(),
        .achieved_volume_quadrature_order =
            achieved_volume_quadrature_order,
        .fallback_used = false,
        .volume_quadrature_point_count = volume_quadrature_point_count,
        .interface_quadrature_point_count = 0u,
        .backend_elapsed_seconds = 0.0,
        .linear_full_cell_fast_path = true,
        .backend_diagnostic = {}};
}

[[nodiscard]] FieldId resolveLevelSetField(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options)
{
    const auto field = system.findFieldByName(options.level_set_field_name);
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "generated level-set interface references unknown field '" +
            options.level_set_field_name + "'");
    }
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "generated level-set interface field '" +
            options.level_set_field_name + "' must be scalar");
    }
    if (rec.space->space_type() != spaces::SpaceType::H1) {
        throw std::invalid_argument(
            "generated level-set interface field '" +
            options.level_set_field_name + "' must use a scalar H1/C0 space");
    }
    if (rec.space->polynomial_order() < 1) {
        throw std::invalid_argument(
            "generated level-set interface field '" +
            options.level_set_field_name + "' must have polynomial order at least one");
    }
    return field;
}

[[nodiscard]] std::uint64_t generatedInterfaceQuadraturePolicyKey(
    const LevelSetGeneratedInterfaceOptions& options,
    int interface_quadrature_order,
    int volume_quadrature_order) noexcept
{
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    const auto mix_real = [&mix](Real value) noexcept {
        std::uint64_t bits = 0u;
        static_assert(sizeof(value) <= sizeof(bits));
        std::memcpy(&bits, &value, sizeof(value));
        mix(bits);
    };

    mix(static_cast<std::uint64_t>(options.geometry_mode));
    mix(static_cast<std::uint64_t>(options.implicit_cut_quadrature_backend));
    mix(static_cast<std::uint64_t>(options.implicit_cut_fallback_policy));
    mix(static_cast<std::uint64_t>(options.geometry_tangent_policy));
    mix(static_cast<std::uint64_t>(options.quadrature_order));
    mix(static_cast<std::uint64_t>(interface_quadrature_order));
    mix(static_cast<std::uint64_t>(volume_quadrature_order));
    mix_real(options.tolerance);
    mix_real(options.implicit_cut_root_tolerance);
    mix_real(options.implicit_cut_root_coordinate_tolerance);
    mix(static_cast<std::uint64_t>(options.implicit_cut_root_max_iterations));
    mix(static_cast<std::uint64_t>(options.implicit_cut_max_subdivision_depth));
    mix(options.keep_degenerate_fragments ? 1u : 0u);
    mix(options.allow_corner_linearized_geometry ? 1u : 0u);
    mix(options.require_production_qualified_implicit_cut_backend ? 1u : 0u);
    return h;
}

[[nodiscard]] GeneratedInterfaceCellDiagnostics appendGeneratedInterfaceCell(
    interfaces::LevelSetInterfaceDomain& domain,
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    const ImplicitCutQuadratureBackendDriver& backend,
    const LevelSetCellEvaluator& evaluator,
    std::span<const Real> coefficients,
    GlobalIndex cell_id)
{
    const auto type = mesh.getCellType(cell_id);
    const std::size_t count = cornerCount(type);
    if (count == 0u) {
        throw std::invalid_argument(
            backendCellDiagnostic(
                backend,
                cell_id,
                type,
                "unsupported element type"));
    }

    std::vector<GlobalIndex> cell_nodes;
    mesh.getCellNodes(cell_id, cell_nodes);
    if (cell_nodes.size() < count) {
        throw std::invalid_argument(
            "generated level-set interface found incomplete cell geometry");
    }

    interfaces::LevelSetCellCutInput input{};
    input.parent_cell = cell_id;
    input.element_type = type;
    input.node_coordinates.reserve(count);
    input.level_set_values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        input.node_coordinates.push_back(referenceCornerCoordinate(type, i));
        input.level_set_values.push_back(
            coefficientAtVertex(entity_map, cell_nodes[i], coefficients));
    }

    if (mesh.dimension() != 2 && mesh.dimension() != 3) {
        throw std::invalid_argument(
            "generated level-set interface requires a 2D or 3D mesh");
    }
    ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;
    backend_input.isovalue = domain.request().isovalue;
    backend_input.reference_min =
        minReferenceCoordinate(input.node_coordinates);
    backend_input.reference_max =
        maxReferenceCoordinate(input.node_coordinates);

    if (auto full_cell_fast_path =
            appendLinearFullCellFastPath(domain,
                                         mesh.dimension(),
                                         type,
                                         input,
                                         evaluator,
                                         cell_id,
                                         cell_nodes.size(),
                                         count,
                                         backend.kind())) {
        return *full_cell_fast_path;
    }

    auto backend_result =
        backend.cut(mesh.dimension(), domain.request(), backend_input);
    const auto validation =
        validateImplicitCutQuadratureBackendCellResult(
            domain.request(), backend_input, backend_result);
    if (!validation.ok) {
        throw std::invalid_argument(
            backendCellDiagnostic(
                backend, cell_id, type, validation.diagnostic));
    }
    if (!backend_result.cut.supported &&
        backend_result.diagnostic_status ==
            ImplicitCutQuadratureDiagnosticStatus::Unsupported &&
        backend.kind() != ImplicitCutQuadratureBackend::LinearCorner &&
        backend.kind() != ImplicitCutQuadratureBackend::MomentFit &&
        requestAllowsLinearCornerFallback(domain.request()) &&
        linearCornerSupportsCell(mesh.dimension(), type)) {
        const auto& fallback_backend =
            implicitCutQuadratureBackendDriver(
                ImplicitCutQuadratureBackend::LinearCorner);
        auto fallback_result =
            fallback_backend.cut(mesh.dimension(), domain.request(), backend_input);
        const auto fallback_validation =
            validateImplicitCutQuadratureBackendCellResult(
                domain.request(), backend_input, fallback_result);
        if (!fallback_validation.ok) {
            throw std::invalid_argument(
                backendCellDiagnostic(
                    fallback_backend,
                    cell_id,
                    type,
                    fallback_validation.diagnostic));
        }
        if (!fallback_result.cut.supported) {
            throw std::invalid_argument(
                backendCellDiagnostic(
                    fallback_backend,
                    cell_id,
                    type,
                    fallback_result.cut.diagnostic));
        }
        fallback_result.fallback_used = true;
        fallback_result.fallback_reason =
            "explicit LinearCorner fallback for unsupported " +
            std::string(backend.name()) + " element type";
        fallback_result.requested_high_order_downgrade = true;
        fallback_result.selected_backend =
            ImplicitCutQuadratureBackend::LinearCorner;
        fallback_result.cut.diagnostic =
            "fallback_from_backend=" + std::string(backend.name()) +
            "; fallback_reason=" + fallback_result.fallback_reason +
            "; " + fallback_result.cut.diagnostic;
        backend_result = std::move(fallback_result);
    }
    if (!backend_result.cut.supported) {
        throw std::invalid_argument(
            backendCellDiagnostic(
                backend, cell_id, type, backend_result.cut.diagnostic));
    }
    if (backend_result.fallback_used &&
        domain.request().implicit_fallback_policy ==
            implicitCutFallbackPolicyName(ImplicitCutFallbackPolicy::Fail)) {
        throw std::invalid_argument(
            backendCellDiagnostic(
                backend,
                cell_id,
                type,
                "implicit cut fallback policy Fail rejected backend fallback; " +
                    backend_result.cut.diagnostic));
    }
    stampSelectedImplicitBackend(
        backend_result.cut,
        backend_result.selected_backend);
    stampImplicitFallbackStatus(
        backend_result.cut,
        backend_result.fallback_used);
    for (auto& fragment : backend_result.cut.fragments) {
        domain.addFragment(std::move(fragment));
    }
    for (auto& region : backend_result.cut.volume_regions) {
        domain.addVolumeRegion(std::move(region));
    }

    return GeneratedInterfaceCellDiagnostics{
        .node_count = cell_nodes.size(),
        .corner_count = count,
        .corner_linearized =
            backend_result.selected_backend ==
                ImplicitCutQuadratureBackend::LinearCorner &&
            cell_nodes.size() > count,
        .selected_backend = backend_result.selected_backend,
        .achieved_interface_quadrature_order =
            backend_result.achieved_interface_quadrature_order,
        .achieved_volume_quadrature_order =
            backend_result.achieved_volume_quadrature_order,
        .fallback_used = backend_result.fallback_used,
        .volume_quadrature_point_count =
            backend_result.volume_quadrature_point_count,
        .interface_quadrature_point_count =
            backend_result.interface_quadrature_point_count,
        .backend_elapsed_seconds = backend_result.backend_elapsed_seconds,
        .backend_diagnostic = backend_result.cut.diagnostic};
}

} // namespace

struct LevelSetGeneratedInterfaceLifecycle::Cache {
    struct Context {
        FieldId field{INVALID_FIELD_ID};
        int marker{-1};
        Real isovalue{0.0};
        std::uint64_t source_layout_revision{0};
        std::uint64_t mesh_geometry_revision{0};
        std::uint64_t mesh_topology_revision{0};
        std::uint64_t mesh_ownership_revision{0};
        std::uint64_t field_dof_state_revision{0};
        std::uint64_t quadrature_policy_key{0};

        [[nodiscard]] bool operator==(const Context& other) const noexcept
        {
            return field == other.field &&
                   marker == other.marker &&
                   isovalue == other.isovalue &&
                   source_layout_revision == other.source_layout_revision &&
                   mesh_geometry_revision == other.mesh_geometry_revision &&
                   mesh_topology_revision == other.mesh_topology_revision &&
                   mesh_ownership_revision == other.mesh_ownership_revision &&
                   field_dof_state_revision == other.field_dof_state_revision &&
                   quadrature_policy_key == other.quadrature_policy_key;
        }
    };

    struct Cell {
        std::uint64_t signature{0};
        GeneratedInterfaceCellDiagnostics diagnostics{};
        std::vector<interfaces::CutInterfaceFragment> fragments{};
        std::vector<interfaces::CutInterfaceVolumeRegion> volume_regions{};
    };

    struct CellSlot {
        bool valid{false};
        Cell cell{};
    };

    struct DomainSlot {
        bool valid{false};
        std::vector<Real> coefficients{};
        LevelSetGeneratedInterfaceResult result{};
    };

    std::optional<Context> context{};
    std::vector<CellSlot> cells{};
    DomainSlot domain{};
};

namespace {

constexpr std::uint64_t kGeneratedCellCacheHashOffset = 1469598103934665603ull;
constexpr std::uint64_t kGeneratedCellCacheHashPrime = 1099511628211ull;

void mixGeneratedCellCacheHash(std::uint64_t& h, std::uint64_t value) noexcept
{
    h ^= value;
    h *= kGeneratedCellCacheHashPrime;
}

void mixGeneratedCellCacheReal(std::uint64_t& h, Real value) noexcept
{
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    mixGeneratedCellCacheHash(h, bits);
}

struct GeneratedInterfaceCellSignature {
    std::uint64_t value{0};
    Real min_level_set_value{0.0};
    Real max_level_set_value{0.0};
};

[[nodiscard]] GeneratedInterfaceCellSignature generatedInterfaceCellSignature(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& field_dofs,
    const LevelSetCellEvaluator& evaluator,
    std::span<const Real> coefficients,
    Real isovalue,
    Real tolerance,
    GlobalIndex cell_id)
{
    std::uint64_t h = kGeneratedCellCacheHashOffset;
    mixGeneratedCellCacheHash(h, static_cast<std::uint64_t>(cell_id));
    mixGeneratedCellCacheHash(
        h, static_cast<std::uint64_t>(mesh.getCellType(cell_id)));

    const auto dofs = field_dofs.getCellDofs(cell_id);
    if (dofs.empty()) {
        throw std::invalid_argument(
            "generated level-set interface cache found a cell without level-set DOFs");
    }
    Real min_value = std::numeric_limits<Real>::infinity();
    Real max_value = -std::numeric_limits<Real>::infinity();
    mixGeneratedCellCacheHash(h, static_cast<std::uint64_t>(dofs.size()));
    for (const auto dof : dofs) {
        if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
            throw std::invalid_argument(
                "generated level-set interface cache found a cell DOF outside the coefficient span");
        }
        const auto value = coefficients[static_cast<std::size_t>(dof)];
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    const auto interpolation_order = evaluator.interpolationOrder(cell_id);
    const bool linear_full_negative =
        interpolation_order <= 1 && max_value < isovalue - tolerance;
    const bool linear_full_positive =
        interpolation_order <= 1 && min_value > isovalue + tolerance;
    if (linear_full_negative || linear_full_positive) {
        mixGeneratedCellCacheHash(h, linear_full_negative ? 1u : 2u);
    } else {
        mixGeneratedCellCacheHash(h, 0u);
        for (const auto dof : dofs) {
            mixGeneratedCellCacheHash(h, static_cast<std::uint64_t>(dof));
            mixGeneratedCellCacheReal(
                h, coefficients[static_cast<std::size_t>(dof)]);
        }
    }

    return GeneratedInterfaceCellSignature{
        .value = h,
        .min_level_set_value = min_value,
        .max_level_set_value = max_value,
    };
}

[[nodiscard]] LevelSetGeneratedInterfaceLifecycle::Cache::Context
generatedInterfaceCellCacheContext(
    const interfaces::CutInterfaceDomainRequest& request,
    const dofs::DofHandler& field_dofs)
{
    LevelSetGeneratedInterfaceLifecycle::Cache::Context context;
    context.field = request.source.field_id;
    context.marker = request.interface_marker;
    context.isovalue = request.isovalue;
    context.source_layout_revision = request.source.layout_revision;
    context.mesh_geometry_revision = request.mesh_geometry_revision;
    context.mesh_topology_revision = request.mesh_topology_revision;
    context.mesh_ownership_revision = request.ownership_revision;
    context.field_dof_state_revision = field_dofs.getDofStateRevision();
    context.quadrature_policy_key = request.quadrature_policy_key;
    return context;
}

void appendCachedGeneratedInterfaceCell(
    interfaces::LevelSetInterfaceDomain& domain,
    const LevelSetGeneratedInterfaceLifecycle::Cache::Cell& cached,
    const interfaces::CutInterfaceDomainRequest& request,
    const GeneratedInterfaceCellSignature& signature)
{
    for (auto fragment : cached.fragments) {
        fragment.interface_marker = request.interface_marker;
        fragment.min_level_set_value = signature.min_level_set_value;
        fragment.max_level_set_value = signature.max_level_set_value;
        fragment.stable_id = interfaces::cutInterfaceStableId(
            fragment.interface_marker,
            fragment.parent_cell,
            fragment.local_fragment_index,
            request.source.value_revision);
        domain.addFragment(std::move(fragment));
    }
    for (auto region : cached.volume_regions) {
        region.interface_marker = request.interface_marker;
        region.min_level_set_value = signature.min_level_set_value;
        region.max_level_set_value = signature.max_level_set_value;
        region.stable_id = interfaces::cutVolumeStableId(
            region.interface_marker,
            region.parent_cell,
            region.local_region_index,
            region.side,
            request.source.value_revision);
        domain.addVolumeRegion(std::move(region));
    }
}

[[nodiscard]] bool coefficientSnapshotMatches(
    const std::vector<Real>& snapshot,
    std::span<const Real> coefficients)
{
    return snapshot.size() == coefficients.size() &&
           std::equal(snapshot.begin(), snapshot.end(), coefficients.begin());
}

[[nodiscard]] LevelSetGeneratedInterfaceResult retargetCachedGeneratedInterfaceResult(
    const LevelSetGeneratedInterfaceResult& cached,
    const interfaces::CutInterfaceDomainRequest& request,
    std::uint64_t revision)
{
    auto result = cached;
    auto domain_request = request;
    domain_request.achieved_interface_quadrature_order =
        cached.domain.request().achieved_interface_quadrature_order;
    domain_request.achieved_volume_quadrature_order =
        cached.domain.request().achieved_volume_quadrature_order;
    domain_request.implicit_fallback_status =
        cached.domain.request().implicit_fallback_status;

    interfaces::LevelSetInterfaceDomain domain(domain_request);
    for (const auto& fragment : cached.domain.fragments()) {
        auto copied = fragment;
        copied.interface_marker = request.interface_marker;
        copied.stable_id =
            interfaces::cutInterfaceStableId(copied.interface_marker,
                                             copied.parent_cell,
                                             copied.local_fragment_index,
                                             revision);
        domain.addFragment(std::move(copied));
    }
    for (const auto& region : cached.domain.volumeRegions()) {
        auto copied = region;
        copied.interface_marker = request.interface_marker;
        copied.stable_id =
            interfaces::cutVolumeStableId(copied.interface_marker,
                                          copied.parent_cell,
                                          copied.local_region_index,
                                          copied.side,
                                          revision);
        domain.addVolumeRegion(std::move(copied));
    }

    result.interface_marker = request.interface_marker;
    result.value_revision = revision;
    result.domain = std::move(domain);
    result.summary = result.domain.summary();
    result.cell_cache_hits = result.cell_count;
    result.cell_cache_misses = 0u;
    result.domain_cache_hits = 1u;
    result.backend_elapsed_seconds = 0.0;
    validateGeneratedInterfaceRuleProvenance(result.domain);
    return result;
}

} // namespace

LevelSetGeneratedInterfaceLifecycle::LevelSetGeneratedInterfaceLifecycle(
    int marker_base,
    int marker_range)
    : marker_registry_(marker_base, marker_range)
    , cache_(std::make_unique<Cache>())
{
}

LevelSetGeneratedInterfaceLifecycle::~LevelSetGeneratedInterfaceLifecycle() =
    default;

LevelSetGeneratedInterfaceLifecycle::LevelSetGeneratedInterfaceLifecycle(
    LevelSetGeneratedInterfaceLifecycle&&) noexcept = default;

LevelSetGeneratedInterfaceLifecycle& LevelSetGeneratedInterfaceLifecycle::operator=(
    LevelSetGeneratedInterfaceLifecycle&&) noexcept = default;

void LevelSetGeneratedInterfaceLifecycle::restoreValueRevision(
    std::uint64_t value_revision) noexcept
{
    value_revision_ = value_revision;
}

std::string levelSetImplicitCutBackendCellDiagnostic(
    ImplicitCutQuadratureBackend backend,
    GlobalIndex cell_id,
    ElementType type,
    const std::string& diagnostic)
{
    return "generated level-set interface backend failure"
           "; backend=" +
           std::string(implicitCutQuadratureBackendDriver(backend).name()) +
           "; cell=" + std::to_string(cell_id) +
           "; element_type=" + elementTypeDiagnosticName(type) +
           "; diagnostic=" + diagnostic;
}

LevelSetGeneratedInterfaceResult LevelSetGeneratedInterfaceLifecycle::build(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options,
    std::span<const Real> solution)
{
    if (options.domain_id.empty()) {
        throw std::invalid_argument("generated level-set interface requires a non-empty domain_id");
    }
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("generated level-set interface requires a positive tolerance");
    }
    if (options.quadrature_order < 0) {
        throw std::invalid_argument("generated level-set interface requires nonnegative quadrature_order");
    }
    if (!(options.implicit_cut_root_tolerance > Real{0.0})) {
        throw std::invalid_argument(
            "generated level-set interface requires a positive implicit_cut_root_tolerance");
    }
    if (!(options.implicit_cut_root_coordinate_tolerance > Real{0.0})) {
        throw std::invalid_argument(
            "generated level-set interface requires a positive implicit_cut_root_coordinate_tolerance");
    }
    if (options.implicit_cut_root_max_iterations <= 0) {
        throw std::invalid_argument(
            "generated level-set interface requires positive implicit_cut_root_max_iterations");
    }
    if (options.implicit_cut_max_subdivision_depth < 0) {
        throw std::invalid_argument(
            "generated level-set interface requires nonnegative implicit_cut_max_subdivision_depth");
    }
    if (options.geometry_tangent_policy ==
        GeometryTangentPolicy::DifferentiatedQuadrature) {
        throw std::invalid_argument(
            "generated level-set interface geometry_tangent_policy=DifferentiatedQuadrature is reserved until quadrature point, weight, measure, normal, and topology sensitivities are implemented");
    }
    if (options.geometry_mode == GeneratedInterfaceGeometryMode::LinearCorner &&
        options.implicit_cut_quadrature_backend !=
            ImplicitCutQuadratureBackend::LinearCorner) {
        throw std::invalid_argument(
            "generated level-set interface LinearCorner geometry requires the LinearCorner implicit cut quadrature backend");
    }
    if (options.geometry_mode == GeneratedInterfaceGeometryMode::HighOrderImplicit &&
        options.implicit_cut_quadrature_backend ==
            ImplicitCutQuadratureBackend::LinearCorner) {
        throw std::invalid_argument(
            "high-order implicit generated level-set interface geometry requires a high-order implicit cut quadrature backend");
    }
    const auto& backend = implicitCutQuadratureBackendDriver(
        options.implicit_cut_quadrature_backend);
    const int interface_quadrature_order =
        options.interface_quadrature_order >= 0 ? options.interface_quadrature_order
                                                : options.quadrature_order;
    const int volume_quadrature_order =
        options.volume_quadrature_order >= 0 ? options.volume_quadrature_order
                                             : options.quadrature_order;
    if (interface_quadrature_order < 0) {
        throw std::invalid_argument(
            "generated level-set interface requires nonnegative interface_quadrature_order");
    }
    if (volume_quadrature_order < 0) {
        throw std::invalid_argument(
            "generated level-set interface requires nonnegative volume_quadrature_order");
    }

    const auto field = resolveLevelSetField(system, options);
    const auto& field_dofs = system.fieldDofHandler(field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "generated level-set interface received an incompatible system solution span");
    }
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "generated level-set interface requires a scalar nodal field");
    }
    const auto& mesh = system.meshAccess();
    if (entity_map->numVertices() != mesh.numVertices()) {
        throw std::invalid_argument(
            "generated level-set interface requires field and mesh vertex counts to match");
    }

    const auto marker_source = interfaces::LevelSetInterfaceSource::fromField(field);
    interfaces::GeneratedInterfaceMarkerKey marker_key{};
    marker_key.source = marker_source;
    marker_key.domain_id = options.domain_id;
    marker_key.isovalue = options.isovalue;
    marker_key.requested_marker = options.requested_interface_marker;
    const int marker = marker_registry_.assign(marker_key);

    const auto revision = ++value_revision_;
    interfaces::CutInterfaceDomainRequest request{};
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        field,
        field_dofs.getDofStateRevision(),
        revision);
    request.interface_marker = marker;
    request.isovalue = options.isovalue;
    request.tolerance = options.tolerance;
    request.quadrature_order = options.quadrature_order;
    request.interface_quadrature_order = interface_quadrature_order;
    request.volume_quadrature_order = volume_quadrature_order;
    request.mesh_geometry_revision = mesh.geometryRevision();
    request.mesh_topology_revision = mesh.topologyRevision();
    request.ownership_revision = mesh.ownershipRevision();
    request.quadrature_policy_key =
        generatedInterfaceQuadraturePolicyKey(
            options, interface_quadrature_order, volume_quadrature_order);
    request.implicit_geometry_mode =
        generatedInterfaceGeometryModeName(options.geometry_mode);
    request.implicit_quadrature_backend =
        backend.name();
    request.implicit_fallback_policy =
        implicitCutFallbackPolicyName(options.implicit_cut_fallback_policy);
    request.implicit_fallback_status = "Unknown";
    request.geometry_tangent_policy =
        geometryTangentPolicyName(options.geometry_tangent_policy);
    request.implicit_cut_root_tolerance =
        options.implicit_cut_root_tolerance;
    request.implicit_cut_root_coordinate_tolerance =
        options.implicit_cut_root_coordinate_tolerance;
    request.implicit_cut_root_max_iterations =
        options.implicit_cut_root_max_iterations;
    request.implicit_cut_max_subdivision_depth =
        options.implicit_cut_max_subdivision_depth;
    request.keep_degenerate_fragments = options.keep_degenerate_fragments;
    request.required_implicit_cut_backend_qualification =
        options.require_production_qualified_implicit_cut_backend
            ? "ProductionQualified"
            : "";

    validateGlobalHighOrderBackendCellSupport(
        mesh,
        options.implicit_cut_quadrature_backend,
        request,
        options.require_production_qualified_implicit_cut_backend);

    interfaces::LevelSetInterfaceDomain domain(request);
    const auto coefficients = solution.subspan(offset, n_field_dofs);
    const auto evaluator = makeLevelSetCellEvaluator(system, field, solution);
    if (!cache_) {
        cache_ = std::make_unique<Cache>();
    }
    const auto cache_context =
        generatedInterfaceCellCacheContext(request, field_dofs);
    if (!cache_->context.has_value() ||
        !(*cache_->context == cache_context)) {
        cache_->cells.clear();
        cache_->cells.resize(static_cast<std::size_t>(mesh.numCells()));
        cache_->domain = Cache::DomainSlot{};
        cache_->context = cache_context;
    }
    if (cache_->domain.valid &&
        coefficientSnapshotMatches(cache_->domain.coefficients, coefficients)) {
        return retargetCachedGeneratedInterfaceResult(
            cache_->domain.result, request, revision);
    }
    std::size_t cell_count = 0u;
    std::size_t corner_linearized_cell_count = 0u;
    std::size_t max_cell_node_count = 0u;
    std::size_t max_corner_node_count = 0u;
    int achieved_interface_quadrature_order =
        request.resolvedInterfaceQuadratureOrder();
    int achieved_volume_quadrature_order =
        request.resolvedVolumeQuadratureOrder();
    std::size_t implicit_cut_fallback_cell_count = 0u;
    std::size_t backend_volume_quadrature_point_count = 0u;
    std::size_t backend_interface_quadrature_point_count = 0u;
    double backend_elapsed_seconds = 0.0;
    std::size_t backend_diagnostic_cell_count = 0u;
    std::string first_backend_diagnostic;
    std::array<std::size_t, 5> selected_backend_counts{};
    std::size_t cell_cache_hits = 0u;
    std::size_t cell_cache_misses = 0u;
    std::size_t linear_full_cell_fast_path_count = 0u;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        if (cell_id < 0 ||
            static_cast<std::size_t>(cell_id) >= cache_->cells.size()) {
            throw std::invalid_argument(
                "generated level-set interface cache encountered a cell id outside the mesh cell range");
        }
        const auto cell_signature =
            generatedInterfaceCellSignature(mesh,
                                            field_dofs,
                                            evaluator,
                                            coefficients,
                                            request.isovalue,
                                            request.tolerance,
                                            cell_id);
        GeneratedInterfaceCellDiagnostics diagnostics;
        auto& cached_slot = cache_->cells[static_cast<std::size_t>(cell_id)];
        if (cached_slot.valid &&
            cached_slot.cell.signature == cell_signature.value) {
            diagnostics = cached_slot.cell.diagnostics;
            diagnostics.backend_elapsed_seconds = 0.0;
            appendCachedGeneratedInterfaceCell(
                domain, cached_slot.cell, request, cell_signature);
            ++cell_cache_hits;
        } else {
            const auto fragment_begin = domain.fragments().size();
            const auto volume_begin = domain.volumeRegions().size();
            diagnostics =
                appendGeneratedInterfaceCell(
                    domain, mesh, *entity_map, backend, evaluator, coefficients, cell_id);

            Cache::Cell cell_cache_entry;
            cell_cache_entry.signature = cell_signature.value;
            cell_cache_entry.diagnostics = diagnostics;
            const auto& fragments = domain.fragments();
            const auto& volume_regions = domain.volumeRegions();
            cell_cache_entry.fragments.assign(
                fragments.begin() + static_cast<std::ptrdiff_t>(fragment_begin),
                fragments.end());
            cell_cache_entry.volume_regions.assign(
                volume_regions.begin() + static_cast<std::ptrdiff_t>(volume_begin),
                volume_regions.end());
            cached_slot.cell = std::move(cell_cache_entry);
            cached_slot.valid = true;
            ++cell_cache_misses;
        }
        ++cell_count;
        ++selected_backend_counts[
            implicitCutQuadratureBackendIndex(diagnostics.selected_backend)];
        if (diagnostics.corner_linearized) {
            ++corner_linearized_cell_count;
        }
        achieved_interface_quadrature_order =
            std::min(achieved_interface_quadrature_order,
                     diagnostics.achieved_interface_quadrature_order);
        achieved_volume_quadrature_order =
            std::min(achieved_volume_quadrature_order,
                     diagnostics.achieved_volume_quadrature_order);
        if (diagnostics.fallback_used) {
            ++implicit_cut_fallback_cell_count;
        }
        if (diagnostics.linear_full_cell_fast_path) {
            ++linear_full_cell_fast_path_count;
        }
        backend_volume_quadrature_point_count +=
            diagnostics.volume_quadrature_point_count;
        backend_interface_quadrature_point_count +=
            diagnostics.interface_quadrature_point_count;
        backend_elapsed_seconds += diagnostics.backend_elapsed_seconds;
        if (!diagnostics.backend_diagnostic.empty()) {
            ++backend_diagnostic_cell_count;
            if (first_backend_diagnostic.empty()) {
                first_backend_diagnostic = diagnostics.backend_diagnostic;
            }
        }
        max_cell_node_count = std::max(max_cell_node_count, diagnostics.node_count);
        max_corner_node_count = std::max(max_corner_node_count, diagnostics.corner_count);
    });
    if (!options.allow_corner_linearized_geometry &&
        corner_linearized_cell_count > 0u) {
        throw std::invalid_argument(
            "generated level-set interface would corner-linearize high-order cell geometry; "
            "set allow_corner_linearized_geometry=true only for explicitly linearized test cases");
    }

    LevelSetGeneratedInterfaceResult result;
    result.interface_marker = marker;
    result.value_revision = revision;
    result.domain = std::move(domain);
    result.cell_count = cell_count;
    result.corner_linearized_cell_count = corner_linearized_cell_count;
    result.max_cell_node_count = max_cell_node_count;
    result.max_corner_node_count = max_corner_node_count;
    result.implicit_cut_quadrature_backend =
        options.implicit_cut_quadrature_backend;
    result.geometry_tangent_policy =
        options.geometry_tangent_policy;
    result.achieved_interface_quadrature_order =
        achieved_interface_quadrature_order;
    result.achieved_volume_quadrature_order =
        achieved_volume_quadrature_order;
    result.domain.mutableRequest().achieved_interface_quadrature_order =
        achieved_interface_quadrature_order;
    result.domain.mutableRequest().achieved_volume_quadrature_order =
        achieved_volume_quadrature_order;
    result.domain.mutableRequest().implicit_fallback_status =
        implicit_cut_fallback_cell_count > 0u ? "Used" : "None";
    validateGeneratedInterfaceRuleProvenance(result.domain);
    result.summary = result.domain.summary();
    result.implicit_cut_fallback_cell_count =
        implicit_cut_fallback_cell_count;
    result.selected_implicit_cut_quadrature_backend_counts =
        selected_backend_counts;
    result.backend_volume_quadrature_point_count =
        backend_volume_quadrature_point_count;
    result.backend_interface_quadrature_point_count =
        backend_interface_quadrature_point_count;
    result.backend_elapsed_seconds = backend_elapsed_seconds;
    result.cell_cache_hits = cell_cache_hits;
    result.cell_cache_misses = cell_cache_misses;
    result.linear_full_cell_fast_path_count =
        linear_full_cell_fast_path_count;
    result.success =
        result.summary.active_fragment_count > 0u ||
        result.summary.active_volume_region_count > 0u;
    if (!result.success) {
        result.diagnostic = "generated level-set interface has no active fragments or volume regions";
    } else if (!first_backend_diagnostic.empty() &&
               options.implicit_cut_quadrature_backend !=
                   ImplicitCutQuadratureBackend::LinearCorner) {
        result.diagnostic =
            "generated level-set interface backend diagnostics: cells=" +
            std::to_string(backend_diagnostic_cell_count) + "; first_cell=" +
                first_backend_diagnostic;
    }
    cache_->domain.valid = true;
    cache_->domain.coefficients.assign(coefficients.begin(), coefficients.end());
    cache_->domain.result = result;
    return result;
}

} // namespace svmp::FE::level_set
