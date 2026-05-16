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
#include <stdexcept>
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
            "RefreshedFrozenQuadrature refreshes generated quadrature before assembly but treats quadrature points, weights, measures, normals, and topology as fixed during tangent assembly";
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
    int achieved_interface_quadrature_order{0};
    int achieved_volume_quadrature_order{0};
    bool fallback_used{false};
    std::string backend_diagnostic{};
};

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
    mix(static_cast<std::uint64_t>(options.implicit_cut_max_subdivision_depth));
    mix(options.keep_degenerate_fragments ? 1u : 0u);
    mix(options.allow_corner_linearized_geometry ? 1u : 0u);
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
    if (!backend_result.cut.supported) {
        throw std::invalid_argument(
            backendCellDiagnostic(
                backend, cell_id, type, backend_result.cut.diagnostic));
    }
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
            backend.kind() == ImplicitCutQuadratureBackend::LinearCorner &&
            cell_nodes.size() > count,
        .achieved_interface_quadrature_order =
            backend_result.achieved_interface_quadrature_order,
        .achieved_volume_quadrature_order =
            backend_result.achieved_volume_quadrature_order,
        .fallback_used = backend_result.fallback_used,
        .backend_diagnostic = backend_result.cut.diagnostic};
}

} // namespace

LevelSetGeneratedInterfaceLifecycle::LevelSetGeneratedInterfaceLifecycle(
    int marker_base,
    int marker_range)
    : marker_registry_(marker_base, marker_range)
{
}

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
    if (options.implicit_cut_max_subdivision_depth < 0) {
        throw std::invalid_argument(
            "generated level-set interface requires nonnegative implicit_cut_max_subdivision_depth");
    }
    if (options.geometry_tangent_policy ==
        GeometryTangentPolicy::DifferentiatedQuadrature) {
        throw std::invalid_argument(
            "generated level-set interface geometry_tangent_policy=DifferentiatedQuadrature is reserved until quadrature point, weight, normal, and topology sensitivities are implemented");
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
        mesh.fieldLayoutRevision(),
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
    request.geometry_tangent_policy =
        geometryTangentPolicyName(options.geometry_tangent_policy);
    request.implicit_cut_root_tolerance =
        options.implicit_cut_root_tolerance;
    request.implicit_cut_max_subdivision_depth =
        options.implicit_cut_max_subdivision_depth;
    request.keep_degenerate_fragments = options.keep_degenerate_fragments;

    interfaces::LevelSetInterfaceDomain domain(request);
    const auto coefficients = solution.subspan(offset, n_field_dofs);
    const auto evaluator = makeLevelSetCellEvaluator(system, field, solution);
    std::size_t cell_count = 0u;
    std::size_t corner_linearized_cell_count = 0u;
    std::size_t max_cell_node_count = 0u;
    std::size_t max_corner_node_count = 0u;
    int achieved_interface_quadrature_order =
        backend.achievedInterfaceQuadratureOrder(request);
    int achieved_volume_quadrature_order =
        backend.achievedVolumeQuadratureOrder(request);
    std::size_t implicit_cut_fallback_cell_count = 0u;
    std::size_t backend_diagnostic_cell_count = 0u;
    std::string first_backend_diagnostic;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        const auto diagnostics =
            appendGeneratedInterfaceCell(
                domain, mesh, *entity_map, backend, evaluator, coefficients, cell_id);
        ++cell_count;
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
    result.summary = result.domain.summary();
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
    result.implicit_cut_fallback_cell_count =
        implicit_cut_fallback_cell_count;
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
    return result;
}

} // namespace svmp::FE::level_set
