/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GeneratedBoundaryAggregateTraceCertificate.cpp
 * @brief End-to-end aggregate trace certification on generated boundaries.
 */

#include <gtest/gtest.h>

#include "Analysis/GeneratedBoundaryAggregateTraceCertificate.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Forms/BoundaryConditions.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemAssembly.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::analysis::test {
namespace {

constexpr int kInterfaceMarker = 701;
constexpr int kWallMarker = 17;
constexpr int kContactMarker = 702;
constexpr int kActiveBoundaryMarker = 703;
constexpr std::uint64_t kSourceLayoutRevision = 1u;
constexpr std::uint64_t kSourceValueRevision = 1u;
constexpr std::uint64_t kQuadraturePolicyKey = 704u;

void installFormBoundTracePolicy(
    systems::FESystem& system,
    FieldId velocity,
    const spaces::FunctionSpace& velocity_space,
    Real gamma,
    bool symmetric)
{
    const auto u =
        forms::FormExpr::stateField(
            velocity, velocity_space, "u_trace_policy");
    const auto v =
        forms::FormExpr::testFunction(
            velocity, velocity_space, "v_trace_policy");
    std::vector<forms::FormExpr> zero_components(
        static_cast<std::size_t>(
            velocity_space.value_dimension()),
        forms::FormExpr::constant(Real{0.0}));
    const auto zero =
        forms::FormExpr::asVector(
            std::move(zero_components));
    auto terms =
        forms::bc::
            buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                u,
                v,
                zero,
                forms::FormExpr::constant(Real{2.5}),
                kWallMarker,
                kActiveBoundaryMarker,
                forms::bc::TraceNitscheOptions{
                    .gamma = gamma,
                    .variant = symmetric
                        ? forms::bc::NitscheVariant::Symmetric
                        : forms::bc::NitscheVariant::Unsymmetric,
                    .scale_with_p = false});
    systems::FormInstallOptions install;
    install.generated_boundary_nitsche_trace_requests.push_back(
        systems::GeneratedBoundaryNitscheTraceInstallRequest{
            .binding = std::move(terms.binding),
            .volume_interface_marker = kInterfaceMarker,
        });
    (void)systems::installFormulation(
        system,
        "velocity",
        {velocity},
        terms.route_contribution,
        install);
}

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(
        const char* key,
        const char* value)
        : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_ = std::string(prior);
        }
        if (value == nullptr) {
            ::unsetenv(key_);
        } else {
            ::setenv(key_, value, 1);
        }
    }

    ~ScopedEnvironmentVariable()
    {
        if (prior_.has_value()) {
            ::setenv(
                key_,
                prior_->c_str(),
                1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(
        const ScopedEnvironmentVariable&) = delete;

private:
    const char* key_;
    std::optional<std::string> prior_{};
};

[[nodiscard]] std::shared_ptr<Mesh> unitTriangleMesh()
{
    auto base = std::make_shared<MeshBase>();
    CellShape cell_shape{};
    cell_shape.family = CellFamily::Triangle;
    cell_shape.order = 1;
    cell_shape.num_corners = 3;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        std::vector<real_t>{
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        },
        std::vector<offset_t>{0, 3},
        std::vector<index_t>{0, 1, 2},
        std::vector<CellShape>{cell_shape});
    base->finalize();

    const auto& face_to_cell = base->face2cell();
    for (std::size_t face = 0u;
         face < face_to_cell.size();
         ++face) {
        const bool is_boundary =
            (face_to_cell[face][0] == INVALID_INDEX) !=
            (face_to_cell[face][1] == INVALID_INDEX);
        if (!is_boundary) {
            continue;
        }
        const auto [vertices, count] =
            base->face_vertices_span(
                static_cast<index_t>(face));
        bool has_zero = false;
        bool has_two = false;
        for (std::size_t vertex = 0u;
             vertices != nullptr && vertex < count;
             ++vertex) {
            has_zero =
                has_zero || vertices[vertex] == 0;
            has_two =
                has_two || vertices[vertex] == 2;
        }
        if (has_zero && has_two) {
            base->set_boundary_label(
                static_cast<index_t>(face),
                kWallMarker);
            return create_mesh(std::move(base));
        }
    }
    throw std::runtime_error(
        "unit triangle x=0 boundary face was not found");
}

[[nodiscard]] interfaces::LevelSetInterfaceSource
constantNegativeSource()
{
    return interfaces::LevelSetInterfaceSource::fromEvaluator(
        "aggregate-trace-constant-negative",
        kSourceLayoutRevision,
        kSourceValueRevision);
}

void setRequestRevisions(
    const assembly::IMeshAccess& mesh,
    std::uint64_t& geometry_revision,
    std::uint64_t& topology_revision,
    std::uint64_t& ownership_revision)
{
    geometry_revision = mesh.geometryRevision();
    topology_revision = mesh.topologyRevision();
    ownership_revision = mesh.ownershipRevision();
}

[[nodiscard]] interfaces::LevelSetInterfaceDomain
fullNegativeInterfaceDomain(
    const assembly::IMeshAccess& mesh)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = constantNegativeSource();
    request.interface_marker = kInterfaceMarker;
    request.quadrature_order = 2;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    setRequestRevisions(
        mesh,
        request.mesh_geometry_revision,
        request.mesh_topology_revision,
        request.ownership_revision);
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";

    interfaces::LevelSetInterfaceDomain domain(request);
    interfaces::CutInterfaceVolumeRegion region;
    region.interface_marker = kInterfaceMarker;
    region.parent_cell = 0;
    region.parent_cell_global_id =
        mesh.getCellGlobalId(0);
    region.owner_rank =
        mesh.getCellOwnerRank(0);
    region.local_region_index = 0;
    region.side =
        geometry::CutIntegrationSide::Negative;
    region.centroid = {{
        Real{1.0} / Real{3.0},
        Real{1.0} / Real{3.0},
        Real{0.0}}};
    region.parent_measure = Real{0.5};
    region.measure = Real{0.5};
    region.volume_fraction = Real{1.0};
    region.min_level_set_value = Real{-1.0};
    region.max_level_set_value = Real{-1.0};
    region.topology_id = "full-negative-triangle";
    region.implicit_quadrature_backend = "LinearCorner";
    region.implicit_fallback_status = "None";
    region.full_cell_equivalent = true;
    region.achieved_quadrature_order = 2;
    domain.addVolumeRegion(std::move(region));
    return domain;
}

[[nodiscard]]
interfaces::GeneratedInterfaceBoundaryIntersectionRequest
contactRequest(const assembly::IMeshAccess& mesh)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest
        request;
    request.source = constantNegativeSource();
    request.generated_domain_id =
        "aggregate-trace-unit-triangle";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.intersection_marker = kContactMarker;
    request.quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    setRequestRevisions(
        mesh,
        request.mesh_geometry_revision,
        request.mesh_topology_revision,
        request.ownership_revision);
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.source_value_revision =
        kSourceValueRevision;
    return request;
}

[[nodiscard]] interfaces::GeneratedActiveBoundaryRequest
activeBoundaryRequest(
    const assembly::IMeshAccess& mesh)
{
    interfaces::GeneratedActiveBoundaryRequest request;
    request.source = constantNegativeSource();
    request.generated_domain_id =
        "aggregate-trace-unit-triangle";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.active_boundary_marker =
        kActiveBoundaryMarker;
    request.side =
        geometry::CutIntegrationSide::Negative;
    request.quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    setRequestRevisions(
        mesh,
        request.mesh_geometry_revision,
        request.mesh_topology_revision,
        request.ownership_revision);
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.source_value_revision =
        kSourceValueRevision;
    return request;
}

[[nodiscard]]
std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
fullNegativeSnapshot(
    const assembly::IMeshAccess& mesh)
{
    auto interface_domain =
        fullNegativeInterfaceDomain(mesh);
    auto contact_domain =
        interfaces::
            buildGeneratedInterfaceBoundaryIntersectionDomain(
                contactRequest(mesh),
                interface_domain,
                mesh);
    interfaces::GeneratedActiveBoundaryScalarField
        boundary_scalar;
    boundary_scalar.value_at_node =
        [](GlobalIndex) {
            return Real{-1.0};
        };
    auto active_domain =
        interfaces::buildGeneratedActiveBoundaryDomain(
            activeBoundaryRequest(mesh),
            interface_domain,
            contact_domain,
            mesh,
            boundary_scalar);

    std::vector<
        interfaces::
            GeneratedInterfaceBoundaryIntersectionDomain>
        contact_domains;
    contact_domains.push_back(
        std::move(contact_domain));
    std::vector<
        interfaces::GeneratedActiveBoundaryDomain>
        active_domains;
    active_domains.push_back(
        std::move(active_domain));
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition =
        false;
    policy.minimum_retained_volume_fraction =
        assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value =
        [](GlobalIndex,
           const std::array<Real, 3>&,
           const geometry::CutQuadratureProvenance&) {
            return Real{-1.0};
        };
    return interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        std::move(contact_domains),
        std::move(active_domains),
        mesh,
        policy,
        std::move(scalar),
        "aggregate-trace-unit-triangle");
}

[[nodiscard]] std::shared_ptr<Mesh> rootedCutSquareMesh()
{
    auto base = std::make_shared<MeshBase>();
    CellShape cell_shape{};
    cell_shape.family = CellFamily::Triangle;
    cell_shape.order = 1;
    cell_shape.num_corners = 3;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        std::vector<real_t>{
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        },
        std::vector<offset_t>{0, 3, 6},
        std::vector<index_t>{
            0, 1, 2,
            0, 2, 3,
        },
        std::vector<CellShape>{
            cell_shape,
            cell_shape,
        });
    base->finalize();

    std::size_t labeled_faces = 0u;
    const auto& face_to_cell = base->face2cell();
    for (std::size_t face = 0u;
         face < face_to_cell.size();
         ++face) {
        const bool is_boundary =
            (face_to_cell[face][0] == INVALID_INDEX) !=
            (face_to_cell[face][1] == INVALID_INDEX);
        if (!is_boundary) {
            continue;
        }
        const auto [vertices, count] =
            base->face_vertices_span(
                static_cast<index_t>(face));
        bool has_two = false;
        bool has_three = false;
        for (std::size_t vertex = 0u;
             vertices != nullptr && vertex < count;
             ++vertex) {
            has_two =
                has_two || vertices[vertex] == 2;
            has_three =
                has_three || vertices[vertex] == 3;
        }
        if (has_two && has_three) {
            base->set_boundary_label(
                static_cast<index_t>(face),
                kWallMarker);
            ++labeled_faces;
        }
    }
    if (labeled_faces != 1u) {
        throw std::runtime_error(
            "rooted cut square top boundary face was not found");
    }
    return create_mesh(
        std::move(base),
        MeshComm::self());
}

[[nodiscard]] interfaces::LevelSetInterfaceSource
rootedCutSource()
{
    return interfaces::LevelSetInterfaceSource::fromEvaluator(
        "aggregate-trace-rooted-cut",
        kSourceLayoutRevision,
        kSourceValueRevision);
}

[[nodiscard]] interfaces::CutInterfaceDomainRequest
rootedCutInterfaceRequest(
    const assembly::IMeshAccess& mesh)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = rootedCutSource();
    request.interface_marker = kInterfaceMarker;
    request.quadrature_order = 2;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    setRequestRevisions(
        mesh,
        request.mesh_geometry_revision,
        request.mesh_topology_revision,
        request.ownership_revision);
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";
    return request;
}

[[nodiscard]] interfaces::LevelSetInterfaceDomain
rootedCutInterfaceDomain(
    const assembly::IMeshAccess& mesh)
{
    const auto request =
        rootedCutInterfaceRequest(mesh);
    interfaces::LevelSetInterfaceDomain domain(request);
    constexpr std::array<Real, 4> nodal_values{{
        Real{-1.0},
        Real{-1.0},
        Real{-1.0},
        Real{7.0},
    }};
    const std::vector<std::array<Real, 3>>
        reference_nodes{
            {{Real{0.0}, Real{0.0}, Real{0.0}}},
            {{Real{1.0}, Real{0.0}, Real{0.0}}},
            {{Real{0.0}, Real{1.0}, Real{0.0}}},
        };

    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        std::vector<GlobalIndex> cell_nodes;
        mesh.getCellNodes(cell, cell_nodes);
        if (cell_nodes.size() != 3u) {
            throw std::runtime_error(
                "rooted cut square expects Triangle3 cells");
        }
        interfaces::LevelSetCellCutInput input;
        input.parent_cell =
            static_cast<MeshIndex>(cell);
        input.element_type =
            mesh.getCellType(cell);
        input.node_coordinates =
            reference_nodes;
        input.level_set_values.reserve(
            cell_nodes.size());
        for (const auto node : cell_nodes) {
            if (node < 0 ||
                static_cast<std::size_t>(node) >=
                    nodal_values.size()) {
                throw std::runtime_error(
                    "rooted cut square has an invalid node index");
            }
            input.level_set_values.push_back(
                nodal_values[
                    static_cast<std::size_t>(node)]);
        }

        auto cut =
            interfaces::cutLinearLevelSetCell2D(
                request,
                input);
        if (!cut.supported) {
            throw std::runtime_error(
                "rooted cut square builder failed: " +
                cut.diagnostic);
        }
        for (auto& fragment : cut.fragments) {
            fragment.parent_cell_global_id =
                mesh.getCellGlobalId(cell);
            fragment.owner_rank =
                mesh.getCellOwnerRank(cell);
            fragment.stable_id = 0u;
            domain.addFragment(
                std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            region.parent_cell_global_id =
                mesh.getCellGlobalId(cell);
            region.owner_rank =
                mesh.getCellOwnerRank(cell);
            region.stable_id = 0u;
            domain.addVolumeRegion(
                std::move(region));
        }
    }
    return domain;
}

[[nodiscard]]
interfaces::GeneratedInterfaceBoundaryIntersectionRequest
rootedCutContactRequest(
    const assembly::IMeshAccess& mesh)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest
        request;
    request.source = rootedCutSource();
    request.generated_domain_id =
        "aggregate-trace-rooted-cut";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.intersection_marker = kContactMarker;
    request.quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    setRequestRevisions(
        mesh,
        request.mesh_geometry_revision,
        request.mesh_topology_revision,
        request.ownership_revision);
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.source_value_revision =
        kSourceValueRevision;
    return request;
}

[[nodiscard]] interfaces::GeneratedActiveBoundaryRequest
rootedCutActiveBoundaryRequest(
    const assembly::IMeshAccess& mesh)
{
    interfaces::GeneratedActiveBoundaryRequest request;
    request.source = rootedCutSource();
    request.generated_domain_id =
        "aggregate-trace-rooted-cut";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.active_boundary_marker =
        kActiveBoundaryMarker;
    request.side =
        geometry::CutIntegrationSide::Negative;
    request.quadrature_order = 2;
    request.frame =
        geometry::CutGeometryFrame::Reference;
    setRequestRevisions(
        mesh,
        request.mesh_geometry_revision,
        request.mesh_topology_revision,
        request.ownership_revision);
    request.quadrature_policy_key =
        kQuadraturePolicyKey;
    request.source_value_revision =
        kSourceValueRevision;
    return request;
}

[[nodiscard]]
std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
rootedCutSnapshot(
    const assembly::IMeshAccess& mesh)
{
    auto interface_domain =
        rootedCutInterfaceDomain(mesh);
    auto contact_domain =
        interfaces::
            buildGeneratedInterfaceBoundaryIntersectionDomain(
                rootedCutContactRequest(mesh),
                interface_domain,
                mesh);

    interfaces::GeneratedActiveBoundaryScalarField
        boundary_scalar;
    boundary_scalar.value_at_node =
        [](GlobalIndex node) {
            constexpr std::array<Real, 4> values{{
                Real{-1.0},
                Real{-1.0},
                Real{-1.0},
                Real{7.0},
            }};
            if (node < 0 ||
                static_cast<std::size_t>(node) >=
                    values.size()) {
                throw std::runtime_error(
                    "rooted cut scalar received an invalid node");
            }
            return values[
                static_cast<std::size_t>(node)];
        };
    auto active_domain =
        interfaces::buildGeneratedActiveBoundaryDomain(
            rootedCutActiveBoundaryRequest(mesh),
            interface_domain,
            contact_domain,
            mesh,
            boundary_scalar);

    std::vector<
        interfaces::
            GeneratedInterfaceBoundaryIntersectionDomain>
        contact_domains;
    contact_domains.push_back(
        std::move(contact_domain));
    std::vector<
        interfaces::GeneratedActiveBoundaryDomain>
        active_domains;
    active_domains.push_back(
        std::move(active_domain));
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition =
        false;
    policy.minimum_retained_volume_fraction =
        assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();

    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value =
        [](GlobalIndex parent_cell,
           const std::array<Real, 3>& xi,
           const geometry::CutQuadratureProvenance&) {
            if (parent_cell == 0) {
                return Real{-1.0};
            }
            if (parent_cell == 1) {
                return Real{-1.0} +
                       Real{8.0} * xi[1];
            }
            throw std::runtime_error(
                "rooted cut scalar received an invalid cell");
        };
    scalar.reference_gradient =
        [](GlobalIndex parent_cell,
           const std::array<Real, 3>&,
           const geometry::CutQuadratureProvenance&) {
            if (parent_cell == 0) {
                return std::array<Real, 3>{{
                    Real{0.0},
                    Real{0.0},
                    Real{0.0},
                }};
            }
            if (parent_cell == 1) {
                return std::array<Real, 3>{{
                    Real{0.0},
                    Real{8.0},
                    Real{0.0},
                }};
            }
            throw std::runtime_error(
                "rooted cut gradient received an invalid cell");
        };
    return interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        std::move(contact_domains),
        std::move(active_domains),
        mesh,
        policy,
        std::move(scalar),
        "aggregate-trace-rooted-cut");
}

[[nodiscard]]
std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
rootlessCutTriangleSnapshot(
    const assembly::IMeshAccess& mesh)
{
    const auto request =
        rootedCutInterfaceRequest(mesh);
    interfaces::LevelSetInterfaceDomain
        interface_domain(request);
    interfaces::LevelSetCellCutInput input;
    input.parent_cell = 0;
    input.element_type = ElementType::Triangle3;
    input.node_coordinates = {
        {{Real{0.0}, Real{0.0}, Real{0.0}}},
        {{Real{1.0}, Real{0.0}, Real{0.0}}},
        {{Real{0.0}, Real{1.0}, Real{0.0}}},
    };
    input.level_set_values = {
        Real{-1.0},
        Real{7.0},
        Real{-1.0},
    };
    auto cut =
        interfaces::cutLinearLevelSetCell2D(
            request, input);
    if (!cut.supported) {
        throw std::runtime_error(
            "rootless cut triangle builder failed: " +
            cut.diagnostic);
    }
    for (auto& fragment : cut.fragments) {
        fragment.parent_cell_global_id =
            mesh.getCellGlobalId(0);
        fragment.owner_rank =
            mesh.getCellOwnerRank(0);
        fragment.stable_id = 0u;
        interface_domain.addFragment(
            std::move(fragment));
    }
    for (auto& region : cut.volume_regions) {
        region.parent_cell_global_id =
            mesh.getCellGlobalId(0);
        region.owner_rank =
            mesh.getCellOwnerRank(0);
        region.stable_id = 0u;
        interface_domain.addVolumeRegion(
            std::move(region));
    }

    auto contact_domain =
        interfaces::
            buildGeneratedInterfaceBoundaryIntersectionDomain(
                rootedCutContactRequest(mesh),
                interface_domain,
                mesh);
    interfaces::GeneratedActiveBoundaryScalarField
        boundary_scalar;
    boundary_scalar.value_at_node =
        [](GlobalIndex node) {
            constexpr std::array<Real, 3> values{{
                Real{-1.0},
                Real{7.0},
                Real{-1.0},
            }};
            if (node < 0 ||
                static_cast<std::size_t>(node) >=
                    values.size()) {
                throw std::runtime_error(
                    "rootless cut scalar received an invalid node");
            }
            return values[
                static_cast<std::size_t>(node)];
        };
    auto active_domain =
        interfaces::buildGeneratedActiveBoundaryDomain(
            rootedCutActiveBoundaryRequest(mesh),
            interface_domain,
            contact_domain,
            mesh,
            boundary_scalar);

    std::vector<
        interfaces::
            GeneratedInterfaceBoundaryIntersectionDomain>
        contact_domains;
    contact_domains.push_back(
        std::move(contact_domain));
    std::vector<
        interfaces::GeneratedActiveBoundaryDomain>
        active_domains;
    active_domains.push_back(
        std::move(active_domain));
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition =
        false;
    policy.minimum_retained_volume_fraction =
        assembly::CutIntegrationContext::
            minGeneratedCutVolumeFraction();
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value =
        [](GlobalIndex parent_cell,
           const std::array<Real, 3>& xi,
           const geometry::CutQuadratureProvenance&) {
            if (parent_cell != 0) {
                throw std::runtime_error(
                    "rootless cut scalar received an invalid cell");
            }
            return Real{-1.0} +
                   Real{8.0} * xi[0];
        };
    scalar.reference_gradient =
        [](GlobalIndex parent_cell,
           const std::array<Real, 3>&,
           const geometry::CutQuadratureProvenance&) {
            if (parent_cell != 0) {
                throw std::runtime_error(
                    "rootless cut gradient received an invalid cell");
            }
            return std::array<Real, 3>{{
                Real{8.0},
                Real{0.0},
                Real{0.0},
            }};
        };
    return interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        std::move(contact_domains),
        std::move(active_domains),
        mesh,
        policy,
        std::move(scalar),
        "aggregate-trace-rootless-cut");
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     FormBindingRequiresExactlyOneRouteAnchorBeforeMutation)
{
    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);
    enum class InvalidAnchorUse {
        Missing,
        Duplicated,
        Scaled,
    };
    const auto require_rejection =
        [&](InvalidAnchorUse invalid_use) {
            systems::FESystem system(mesh);
            const auto velocity =
                system.addField(
                    systems::FieldSpec{
                        .name = "velocity",
                        .space = velocity_space,
                        .components = 2});
            const auto u =
                forms::FormExpr::stateField(
                    velocity,
                    *velocity_space,
                    "u_binding_negative");
            const auto v =
                forms::FormExpr::testFunction(
                    velocity,
                    *velocity_space,
                    "v_binding_negative");
            const auto zero =
                forms::FormExpr::asVector(
                    std::vector<forms::FormExpr>{
                        forms::FormExpr::constant(Real{0.0}),
                        forms::FormExpr::constant(Real{0.0}),
                    });
            if (invalid_use == InvalidAnchorUse::Missing) {
                const auto discrete_u =
                    forms::FormExpr::discreteField(
                        velocity,
                        *velocity_space,
                        "discrete_u_binding_negative");
                EXPECT_THROW(
                    (void)forms::bc::
                        buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                            discrete_u,
                            v,
                            zero,
                            forms::FormExpr::constant(
                                Real{2.5}),
                            kWallMarker,
                            kActiveBoundaryMarker),
                    std::invalid_argument);
                EXPECT_THROW(
                    (void)forms::bc::
                        buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                            u,
                            discrete_u,
                            zero,
                            forms::FormExpr::constant(
                                Real{2.5}),
                            kWallMarker,
                            kActiveBoundaryMarker),
                    std::invalid_argument);
                EXPECT_THROW(
                    (void)forms::bc::
                        buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                            u,
                            v,
                            zero,
                            forms::FormExpr::constant(
                                Real{2.5}),
                            kWallMarker,
                            kActiveBoundaryMarker,
                            forms::bc::TraceNitscheOptions{
                                .gamma = Real{8.0},
                                .variant =
                                    static_cast<
                                        forms::bc::
                                            NitscheVariant>(
                                        255),
                                .scale_with_p = false}),
                    std::invalid_argument);
                const auto require_prescribed_rejection =
                    [&](const forms::FormExpr&
                            prescribed_value) {
                        EXPECT_THROW(
                            (void)forms::bc::
                                buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                                    u,
                                    v,
                                    prescribed_value,
                                    forms::FormExpr::
                                        constant(
                                            Real{2.5}),
                                    kWallMarker,
                                    kActiveBoundaryMarker),
                            std::invalid_argument);
                    };
                const auto trial =
                    forms::FormExpr::trialFunction(
                        *velocity_space,
                        "trial_prescribed_value_negative");
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            u.component(0),
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            v.component(0),
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            trial.component(0),
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                require_prescribed_rejection(
                    discrete_u);
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            forms::FormExpr::
                                previousSolution(1),
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            forms::FormExpr::auxiliaryInput(
                                "coupled_prescribed_value_negative"),
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            forms::FormExpr::
                                geometryTrialVectorVariation()
                                    .component(0),
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                require_prescribed_rejection(
                    forms::FormExpr::asVector(
                        std::vector<forms::FormExpr>{
                            forms::FormExpr::constant(
                                Real{0.0}),
                        }));
                const auto mismatched_v =
                    forms::FormExpr::testFunction(
                        velocity,
                        *scalar_space,
                        "mismatched_v_binding_negative");
                EXPECT_THROW(
                    (void)forms::bc::
                        buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                            u,
                            mismatched_v,
                            zero,
                            forms::FormExpr::constant(
                                Real{2.5}),
                            kWallMarker,
                            kActiveBoundaryMarker),
                    std::invalid_argument);
                const auto wrong_scalar_space =
                    std::make_shared<spaces::H1Space>(
                        ElementType::Quad4,
                        /*order=*/1);
                const auto wrong_velocity_space =
                    std::make_shared<
                        spaces::ProductSpace>(
                        wrong_scalar_space,
                        /*components=*/2);
                const auto wrong_space_u =
                    forms::FormExpr::stateField(
                        velocity,
                        *wrong_velocity_space,
                        "wrong_space_u_binding_negative");
                const auto wrong_space_v =
                    forms::FormExpr::testFunction(
                        velocity,
                        *wrong_velocity_space,
                        "wrong_space_v_binding_negative");
                auto wrong_space_terms =
                    forms::bc::
                        buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                            wrong_space_u,
                            wrong_space_v,
                            zero,
                            forms::FormExpr::constant(
                                Real{2.5}),
                            kWallMarker,
                            kActiveBoundaryMarker,
                            forms::bc::TraceNitscheOptions{
                                .gamma = Real{8.0},
                                .variant =
                                    forms::bc::
                                        NitscheVariant::Symmetric,
                                .scale_with_p = false});
                systems::FormInstallOptions
                    wrong_space_install;
                wrong_space_install
                    .generated_boundary_nitsche_trace_requests
                    .push_back(
                        systems::
                            GeneratedBoundaryNitscheTraceInstallRequest{
                                .binding = std::move(
                                    wrong_space_terms.binding),
                                .volume_interface_marker =
                                    kInterfaceMarker,
                            });
                EXPECT_THROW(
                    (void)systems::installFormulation(
                        system,
                        "wrong_space_velocity",
                        {velocity},
                        wrong_space_terms.route_contribution,
                        wrong_space_install),
                    std::invalid_argument);
                EXPECT_FALSE(
                    system.hasOperator(
                        "wrong_space_velocity"));
                EXPECT_TRUE(
                    system
                        .generatedBoundaryNitscheTracePolicies()
                        .empty());
                EXPECT_TRUE(
                    system.formulationRecords().empty());
                EXPECT_TRUE(
                    system.contributionDescriptors().empty());
            }
            auto terms =
                forms::bc::
                    buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                        u,
                        v,
                        zero,
                        forms::FormExpr::constant(Real{2.5}),
                        kWallMarker,
                        kActiveBoundaryMarker,
                        forms::bc::TraceNitscheOptions{
                            .gamma = Real{8.0},
                            .variant =
                                forms::bc::NitscheVariant::Symmetric,
                            .scale_with_p = false});
            forms::FormExpr residual;
            switch (invalid_use) {
                case InvalidAnchorUse::Missing:
                    residual = forms::inner(u, v).dx();
                    break;
                case InvalidAnchorUse::Duplicated:
                    residual =
                        terms.route_contribution +
                        terms.route_contribution;
                    break;
                case InvalidAnchorUse::Scaled:
                    residual =
                        forms::FormExpr::constant(
                            Real{2.0}) *
                        terms.route_contribution;
                    break;
            }
            systems::FormInstallOptions install;
            install
                .generated_boundary_nitsche_trace_requests
                .push_back(
                    systems::
                        GeneratedBoundaryNitscheTraceInstallRequest{
                            .binding =
                                std::move(terms.binding),
                            .volume_interface_marker =
                                kInterfaceMarker,
                        });
            EXPECT_THROW(
                (void)systems::installFormulation(
                    system,
                    "velocity",
                    {velocity},
                    residual,
                    install),
                InvalidArgumentException);
            EXPECT_FALSE(system.hasOperator("velocity"));
            EXPECT_TRUE(
                system.generatedBoundaryNitscheTracePolicies()
                    .empty());
            EXPECT_TRUE(
                system.formulationRecords().empty());
            EXPECT_TRUE(
                system.contributionDescriptors().empty());
        };

    require_rejection(InvalidAnchorUse::Missing);
    require_rejection(InvalidAnchorUse::Duplicated);
    require_rejection(InvalidAnchorUse::Scaled);
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     FullActiveUnitTriangleHasAnalyticBoundFour)
{
    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);

    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    installFormBoundTracePolicy(
        system,
        velocity,
        *velocity_space,
        Real{8.0},
        true);
    const auto policies =
        system.generatedBoundaryNitscheTracePolicies();
    ASSERT_EQ(policies.size(), 1u);
    const auto policy_id = policies.front().id;
    EXPECT_NE(policies.front().form_binding_digest, 0u);
    EXPECT_EQ(
        policies.front().source_formulation_record_index,
        0u);
    EXPECT_EQ(
        policies.front().penalty_polynomial_order,
        1);
    EXPECT_EQ(
        policies.front().effective_penalty_multiplier,
        Real{8.0});
    EXPECT_NE(
        policies.front().id,
        systems::
            INVALID_GENERATED_BOUNDARY_NITSCHE_TRACE_POLICY_ID);
    ASSERT_NO_THROW(system.setup());
    EXPECT_TRUE(
        system.generatedBoundaryNitscheTraceCertificates()
            .empty());

    const auto cut_cell_gid =
        system.meshAccess().getCellGlobalId(0);
    const auto snapshot =
        fullNegativeSnapshot(system.meshAccess());
    ASSERT_NE(snapshot, nullptr);
    ASSERT_EQ(
        snapshot->interfaceDomain()
            .volumeRegions()
            .size(),
        1u);
    const auto snapshot_volume_stable_id =
        snapshot->interfaceDomain()
            .volumeRegions()
            .front()
            .stable_id;
    ASSERT_NE(snapshot_volume_stable_id, 0u);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains().size(),
        1u);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains()
            .front()
            .fragments()
            .size(),
        1u);
    const auto boundary_stable_id =
        snapshot->activeBoundaryDomains()
            .front()
            .fragments()
            .front()
            .stable_id;
    ASSERT_NE(boundary_stable_id, 0u);
    EXPECT_EQ(
        snapshot->activeBoundaryDomains()
            .front()
            .fragments()
            .front()
            .parent_cell_global_id,
        cut_cell_gid);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains()
            .front()
            .boundaryQuadratureRules()
            .size(),
        1u);

    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    ASSERT_NO_THROW(
        context->addFreeSurfaceGeometrySnapshot(
            snapshot,
            geometry::CutIntegrationSide::Negative));
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto reports =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(reports.size(), 1u);
    ASSERT_NE(reports.front(), nullptr);
    const auto& report = *reports.front();
    EXPECT_TRUE(report.trace_bound_eligible);
    ASSERT_EQ(report.active_cells.size(), 1u);
    EXPECT_EQ(
        report.active_cells.front().kind,
        constraints::
            SmallCutAggregationActiveCellKind::
                FullActive);
    EXPECT_TRUE(report.rows.empty());
    EXPECT_TRUE(report.patches.empty());
    const auto volume_rule_indices =
        context
            ->generatedVolumeRuleIndexSpanForMarkerAndSide(
                kInterfaceMarker,
                geometry::CutIntegrationSide::Negative);
    ASSERT_EQ(volume_rule_indices.size(), 1u);
    ASSERT_LT(
        volume_rule_indices.front(),
        context->volumeRules().size());
    const auto volume_stable_id =
        context
            ->volumeRules()[
                volume_rule_indices.front()]
            .provenance.cut_topology_revision;
    ASSERT_NE(volume_stable_id, 0u);
    EXPECT_EQ(
        volume_stable_id,
        snapshot_volume_stable_id);
    ASSERT_EQ(
        report.active_cells.front()
            .retained_rule_stable_ids.size(),
        1u);
    EXPECT_EQ(
        report.active_cells.front()
            .retained_rule_stable_ids.front(),
        volume_stable_id);

    GeneratedBoundaryAggregateTraceCertificationOptions
        options;
    options.field = velocity;
    options.physical_boundary_marker = kWallMarker;
    options.volume_interface_marker =
        kInterfaceMarker;
    options.generated_active_boundary_marker =
        kActiveBoundaryMarker;
    options.dynamic_viscosity = Real{2.5};

    GeneratedBoundaryAggregateTraceCertificate
        certificate;
    ASSERT_NO_THROW(
        certificate =
            certifyGeneratedBoundaryAggregateTrace(
                system,
                options));
    EXPECT_EQ(certificate.field, velocity);
    EXPECT_EQ(
        certificate.physical_boundary_marker,
        kWallMarker);
    EXPECT_EQ(certificate.communicator_size, 1);
    EXPECT_NE(
        certificate.aggregation_content_digest,
        0u);
    EXPECT_EQ(
        certificate.aggregation_content_digest,
        report.canonical_content_digest);
    EXPECT_NE(
        certificate.canonical_certificate_digest,
        0u);
    EXPECT_EQ(
        certificate.cut_context_content_revision,
        context->contentRevision());
    EXPECT_EQ(
        certificate.cut_context_content_revision,
        report.revision.cut_context_content_revision);
    EXPECT_EQ(
        certificate.free_surface_snapshot_revision,
        snapshot->revision().snapshot_revision_key);
    EXPECT_EQ(
        certificate.free_surface_snapshot_revision,
        report.revision.free_surface_snapshot_revision);
    EXPECT_EQ(
        certificate.source_value_revision,
        kSourceValueRevision);
    EXPECT_EQ(
        certificate.source_value_revision,
        report.revision.source_value_revision);
    EXPECT_EQ(
        certificate.affine_constraint_layout_revision,
        system.constraints()
            .constraintLayoutRevision());
    EXPECT_EQ(
        certificate.affine_constraint_layout_revision,
        report.revision
            .affine_constraint_layout_revision);
    EXPECT_EQ(certificate.active_cell_count, 1u);
    EXPECT_EQ(
        certificate.generated_boundary_rule_count,
        1u);
    EXPECT_EQ(certificate.certified_patch_count, 1u);
    EXPECT_EQ(certificate.maximum_support_overlap, 1u);
    EXPECT_EQ(
        certificate.maximum_terminal_tangent_dimension,
        6u);
    EXPECT_NEAR(
        certificate.retained_active_physical_volume,
        Real{0.5},
        Real{1.0e-14});
    EXPECT_NEAR(
        certificate.generated_boundary_physical_measure,
        Real{1.0},
        Real{1.0e-14});

    ASSERT_EQ(certificate.patches.size(), 1u);
    const auto& patch = certificate.patches.front();
    EXPECT_TRUE(patch.synthetic_full_active_patch);
    EXPECT_EQ(
        patch.canonical_patch_index,
        std::numeric_limits<std::size_t>::max());
    ASSERT_EQ(patch.support_cell_gids.size(), 1u);
    ASSERT_EQ(
        patch.boundary_rule_stable_ids.size(),
        1u);
    EXPECT_EQ(
        patch.boundary_rule_stable_ids.front(),
        boundary_stable_id);
    EXPECT_EQ(patch.raw_support_dof_count, 6u);
    EXPECT_EQ(
        patch.terminal_tangent_dof_count,
        6u);
    EXPECT_EQ(patch.rigid_mode_candidate_count, 3u);
    EXPECT_EQ(
        patch.structural_rigid_mode_count,
        3u);
    EXPECT_EQ(patch.rigid_mode_constraint_rank, 0u);
    EXPECT_EQ(
        patch.rigid_mode_quotient_status,
        GeneratedBoundaryRigidModeQuotientStatus::
            Applied);
    EXPECT_EQ(patch.maximum_cell_support_overlap, 1u);
    EXPECT_NEAR(
        patch.retained_support_physical_volume,
        Real{0.5},
        Real{1.0e-14});
    EXPECT_NEAR(
        patch.generated_boundary_physical_measure,
        Real{1.0},
        Real{1.0e-14});

    const auto& bound = patch.generalized_bound;
    EXPECT_EQ(bound.dimension, 6u);
    EXPECT_EQ(bound.positive_rank, 3u);
    EXPECT_EQ(bound.nullity, 3u);
    EXPECT_TRUE(bound.denominator_converged);
    EXPECT_TRUE(bound.quotient_converged);
    EXPECT_TRUE(bound.explicit_nullspace.applied);
    EXPECT_TRUE(
        bound.explicit_nullspace
            .exact_binary64_actions_proven);
    EXPECT_TRUE(
        bound.explicit_nullspace
            .exact_binary64_anchor_rank_proven);
    EXPECT_EQ(
        bound.explicit_nullspace.supplied_nullity,
        3u);
    EXPECT_EQ(
        bound.explicit_nullspace.reduced_dimension,
        3u);
    EXPECT_EQ(
        bound.explicit_nullspace
            .eliminated_coordinates.size(),
        3u);
    EXPECT_EQ(
        bound.explicit_nullspace
            .maximum_denominator_action,
        Real{0.0});
    EXPECT_EQ(
        bound.explicit_nullspace
            .maximum_numerator_action,
        Real{0.0});
    EXPECT_TRUE(bound.exact_dyadic.applied);
    EXPECT_TRUE(
        bound.exact_dyadic
            .denominator_positive_definite_proven);
    EXPECT_TRUE(
        bound.exact_dyadic
            .numerator_positive_semidefinite_proven);
    EXPECT_TRUE(
        bound.exact_dyadic.upper_inequality_proven);
    EXPECT_EQ(bound.exact_dyadic.dimension, 3u);
    EXPECT_EQ(bound.exact_dyadic.denominator_rank, 3u);
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.exact_dyadic.directly_proven_upper_bound);
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue,
        Real{4.0},
        Real{1.0e-10});
    EXPECT_GE(
        bound.conservative_upper_bound,
        Real{4.0});
    EXPECT_NEAR(
        bound.conservative_upper_bound,
        Real{4.0},
        Real{1.0e-9});
    EXPECT_GE(
        certificate.global_conservative_upper_bound,
        bound.conservative_upper_bound);
    EXPECT_NEAR(
        certificate.global_conservative_upper_bound,
        Real{4.0},
        Real{1.0e-9});

    const auto eager =
        system.generatedBoundaryNitscheTraceCertificates();
    ASSERT_EQ(eager.size(), 1u);
    EXPECT_EQ(eager.front().policy.id, policy_id);
    EXPECT_EQ(
        eager.front().policy.generated_active_boundary_marker,
        kActiveBoundaryMarker);
    EXPECT_EQ(
        eager.front().certificate.canonical_certificate_digest,
        certificate.canonical_certificate_digest);
    EXPECT_EQ(
        eager.front().aggregation_report,
        reports.front());
    EXPECT_EQ(eager.front().polynomial_order, 1);
    EXPECT_EQ(
        eager.front().effective_penalty_multiplier,
        Real{8.0});
    EXPECT_GE(
        eager.front().trace_to_penalty_ratio,
        certificate.global_conservative_upper_bound /
            Real{8.0});
    EXPECT_LT(
        eager.front().trace_to_penalty_ratio,
        Real{1.0});
    EXPECT_EQ(
        eager.front()
            .grouped_symmetric_trace_to_penalty_ratio,
        eager.front().trace_to_penalty_ratio);
    ASSERT_TRUE(
        eager.front()
            .symmetric_energy_ratio_lower_bound
            .has_value());
    EXPECT_GT(
        *eager.front()
             .symmetric_energy_ratio_lower_bound,
        Real{0.0});

    const auto repeated =
        certifyGeneratedBoundaryAggregateTrace(
            system,
            options);
    EXPECT_EQ(
        repeated.canonical_certificate_digest,
        certificate.canonical_certificate_digest);
    EXPECT_EQ(
        repeated.global_conservative_upper_bound,
        certificate.global_conservative_upper_bound);
    auto wrong_physical_marker = options;
    wrong_physical_marker.physical_boundary_marker =
        kWallMarker + 1;
    EXPECT_THROW(
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            wrong_physical_marker),
        std::runtime_error);
    auto wrong_volume_marker = options;
    wrong_volume_marker.volume_interface_marker =
        kInterfaceMarker + 1;
    EXPECT_THROW(
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            wrong_volume_marker),
        std::runtime_error);
    auto wrong_generated_boundary_marker = options;
    wrong_generated_boundary_marker
        .generated_active_boundary_marker =
        kActiveBoundaryMarker + 1;
    EXPECT_THROW(
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            wrong_generated_boundary_marker),
        std::runtime_error);
    auto invalid_viscosity = options;
    invalid_viscosity.dynamic_viscosity =
        Real{0.0};
    EXPECT_THROW(
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            invalid_viscosity),
        std::runtime_error);
    auto invalid_dimension_cap = options;
    invalid_dimension_cap.maximum_reduced_dimension =
        129u;
    EXPECT_THROW(
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            invalid_dimension_cap),
        std::runtime_error);
    auto restrictive_dimension_cap = options;
    restrictive_dimension_cap.maximum_reduced_dimension =
        5u;
    EXPECT_THROW(
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            restrictive_dimension_cap),
        std::runtime_error);

    assembly::DenseMatrixView untouched(
        system.dofHandler().getNumDofs());
    untouched.addMatrixEntry(0, 0, Real{3.25});
    system.setCutIntegrationContext(context);
    EXPECT_TRUE(
        system.generatedBoundaryNitscheTraceCertificates()
            .empty());
    systems::AssemblyRequest request;
    request.op = "velocity";
    request.want_matrix = true;
    systems::SystemStateView state;
    EXPECT_THROW(
        (void)systems::assembleOperator(
            system,
            request,
            state,
            &untouched,
            nullptr),
        std::runtime_error);
    EXPECT_EQ(
        untouched.getMatrixEntry(0, 0),
        Real{3.25});
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     RootedCutSquareCertifiesActualAggregateProlongation)
{
    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh = rootedCutSquareMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);

    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    ASSERT_NO_THROW(system.setup());

    const auto root_cell_gid =
        system.meshAccess().getCellGlobalId(0);
    const auto cut_cell_gid =
        system.meshAccess().getCellGlobalId(1);
    ASSERT_NE(root_cell_gid, cut_cell_gid);
    std::vector<GlobalIndex> expected_support{
        root_cell_gid,
        cut_cell_gid,
    };
    std::sort(
        expected_support.begin(),
        expected_support.end());

    const auto snapshot =
        rootedCutSnapshot(system.meshAccess());
    ASSERT_NE(snapshot, nullptr);
    ASSERT_EQ(
        snapshot->interfaceDomain()
            .volumeRegions()
            .size(),
        3u);
    ASSERT_EQ(
        snapshot->interfaceDomain()
            .fragments()
            .size(),
        1u);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains().size(),
        1u);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains()
            .front()
            .fragments()
            .size(),
        1u);
    ASSERT_EQ(
        snapshot->activeBoundaryDomains()
            .front()
            .boundaryQuadratureRules()
            .size(),
        1u);
    const auto boundary_stable_id =
        snapshot->activeBoundaryDomains()
            .front()
            .fragments()
            .front()
            .stable_id;
    ASSERT_NE(boundary_stable_id, 0u);

    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    ASSERT_NO_THROW(
        context->addFreeSurfaceGeometrySnapshot(
            snapshot,
            geometry::CutIntegrationSide::Negative));
    ASSERT_NO_THROW(
        context
            ->assertAllFreeSurfaceGeometrySnapshotsCurrent(
                system.meshAccess()));
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());
    ASSERT_NO_THROW(
        context
            ->assertAllFreeSurfaceGeometrySnapshotsCurrent(
                system.meshAccess()));

    const auto reports =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(reports.size(), 1u);
    ASSERT_NE(reports.front(), nullptr);
    const auto& report = *reports.front();
    EXPECT_TRUE(report.trace_bound_eligible);
    EXPECT_NE(report.canonical_content_digest, 0u);
    ASSERT_EQ(report.active_cells.size(), 2u);
    const auto root_cell =
        std::find_if(
            report.active_cells.begin(),
            report.active_cells.end(),
            [](const auto& cell) {
                return cell.kind ==
                    constraints::
                        SmallCutAggregationActiveCellKind::
                            FullActive;
            });
    const auto cut_cell =
        std::find_if(
            report.active_cells.begin(),
            report.active_cells.end(),
            [](const auto& cell) {
                return cell.kind ==
                    constraints::
                        SmallCutAggregationActiveCellKind::
                            Cut;
            });
    ASSERT_NE(root_cell, report.active_cells.end());
    ASSERT_NE(cut_cell, report.active_cells.end());
    EXPECT_EQ(root_cell->cell_gid, root_cell_gid);
    EXPECT_EQ(cut_cell->cell_gid, cut_cell_gid);
    EXPECT_NEAR(
        root_cell->retained_physical_volume,
        Real{0.5},
        Real{1.0e-14});
    EXPECT_NEAR(
        cut_cell->retained_physical_volume,
        Real{15.0} / Real{128.0},
        Real{1.0e-14});

    const auto volume_rule_indices =
        context
            ->generatedVolumeRuleIndexSpanForMarkerAndSide(
                kInterfaceMarker,
                geometry::CutIntegrationSide::Negative);
    ASSERT_EQ(volume_rule_indices.size(), 2u);
    for (const auto& active_cell :
         report.active_cells) {
        ASSERT_EQ(
            active_cell
                .retained_rule_stable_ids.size(),
            1u);
        const auto context_rule =
            std::find_if(
                volume_rule_indices.begin(),
                volume_rule_indices.end(),
                [&](std::size_t index) {
                    return index <
                               context
                                   ->volumeRules()
                                   .size() &&
                           context
                                   ->volumeRules()[index]
                                   .provenance
                                   .parent_entity_global_id ==
                               active_cell.cell_gid;
                });
        ASSERT_NE(
            context_rule,
            volume_rule_indices.end());
        const auto stable_id =
            context
                ->volumeRules()[*context_rule]
                .provenance
                .cut_topology_revision;
        EXPECT_NE(stable_id, 0u);
        EXPECT_EQ(
            active_cell
                .retained_rule_stable_ids
                .front(),
            stable_id);
        const auto snapshot_region =
            std::find_if(
                snapshot
                    ->interfaceDomain()
                    .volumeRegions()
                    .begin(),
                snapshot
                    ->interfaceDomain()
                    .volumeRegions()
                    .end(),
                [&](const auto& region) {
                    return region.side ==
                               geometry::
                                   CutIntegrationSide::
                                       Negative &&
                           region
                                   .parent_cell_global_id ==
                               active_cell.cell_gid;
                });
        ASSERT_NE(
            snapshot_region,
            snapshot
                ->interfaceDomain()
                .volumeRegions()
                .end());
        EXPECT_EQ(
            snapshot_region->stable_id,
            stable_id);
    }

    const auto* entity_map =
        system
            .fieldDofHandler(velocity)
            .getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto vertex_three_local_dofs =
        entity_map->getVertexDofs(3);
    ASSERT_EQ(
        vertex_three_local_dofs.size(),
        2u);
    std::vector<GlobalIndex>
        expected_slave_dofs;
    for (const auto dof :
         vertex_three_local_dofs) {
        expected_slave_dofs.push_back(
            system.fieldDofOffset(velocity) +
            dof);
    }
    std::sort(
        expected_slave_dofs.begin(),
        expected_slave_dofs.end());

    const auto root_local_dofs =
        system
            .fieldDofHandler(velocity)
            .getCellDofs(0);
    std::vector<GlobalIndex> root_dofs;
    for (const auto dof : root_local_dofs) {
        root_dofs.push_back(
            system.fieldDofOffset(velocity) +
            dof);
    }
    std::sort(root_dofs.begin(), root_dofs.end());

    ASSERT_EQ(report.rows.size(), 2u);
    std::vector<GlobalIndex> reported_slaves;
    std::vector<std::size_t> reported_components;
    for (const auto& row : report.rows) {
        reported_slaves.push_back(row.slave_dof);
        reported_components.push_back(
            row.component);
        EXPECT_EQ(
            row.provisional_kind,
            constraints::
                SmallCutAggregationProvisionalRowKind::
                    RootedExtension);
        EXPECT_EQ(
            row.final_kind,
            constraints::
                SmallCutAggregationFinalRowKind::
                    MasterBearing);
        EXPECT_EQ(
            row.root_cell_gid,
            root_cell_gid);
        EXPECT_EQ(row.root_distance, 1u);
        EXPECT_EQ(
            row.final_inhomogeneity,
            Real{0.0});
        ASSERT_EQ(
            row.provisional_entries.size(),
            3u);
        ASSERT_EQ(
            row.final_entries.size(),
            3u);
        Real weight_sum = Real{0.0};
        std::vector<Real> weights;
        for (const auto& entry :
             row.final_entries) {
            EXPECT_TRUE(
                std::binary_search(
                    root_dofs.begin(),
                    root_dofs.end(),
                    entry.master_dof));
            EXPECT_TRUE(
                std::isfinite(entry.weight));
            weight_sum += entry.weight;
            weights.push_back(entry.weight);
        }
        std::sort(weights.begin(), weights.end());
        EXPECT_NEAR(
            weight_sum,
            Real{1.0},
            Real{1.0e-14});
        ASSERT_EQ(weights.size(), 3u);
        EXPECT_NEAR(
            weights[0],
            Real{-1.0},
            Real{1.0e-14});
        EXPECT_NEAR(
            weights[1],
            Real{1.0},
            Real{1.0e-14});
        EXPECT_NEAR(
            weights[2],
            Real{1.0},
            Real{1.0e-14});
    }
    std::sort(
        reported_slaves.begin(),
        reported_slaves.end());
    std::sort(
        reported_components.begin(),
        reported_components.end());
    EXPECT_EQ(
        reported_slaves,
        expected_slave_dofs);
    EXPECT_EQ(
        reported_components,
        (std::vector<std::size_t>{0u, 1u}));
    for (const auto dof :
         expected_slave_dofs) {
        EXPECT_TRUE(
            system.constraints()
                .isConstrained(dof));
    }

    ASSERT_EQ(report.patches.size(), 1u);
    const auto& aggregate_patch =
        report.patches.front();
    EXPECT_EQ(
        aggregate_patch.kind,
        constraints::
            SmallCutAggregationPatchKind::
                Rooted);
    EXPECT_EQ(
        aggregate_patch.root_cell_gid,
        root_cell_gid);
    EXPECT_EQ(
        aggregate_patch.member_cell_gids,
        expected_support);
    EXPECT_EQ(
        aggregate_patch.support_cell_gids,
        expected_support);
    EXPECT_EQ(
        aggregate_patch.slave_dofs,
        expected_slave_dofs);

    EXPECT_EQ(
        report.revision.cut_context_content_revision,
        context->contentRevision());
    EXPECT_EQ(
        report.revision.free_surface_snapshot_revision,
        snapshot->revision()
            .snapshot_revision_key);
    EXPECT_EQ(
        report.revision.source_value_revision,
        kSourceValueRevision);
    EXPECT_EQ(
        report.revision
            .affine_constraint_layout_revision,
        system.constraints()
            .constraintLayoutRevision());

    GeneratedBoundaryAggregateTraceCertificationOptions
        options;
    options.field = velocity;
    options.physical_boundary_marker = kWallMarker;
    options.volume_interface_marker =
        kInterfaceMarker;
    options.generated_active_boundary_marker =
        kActiveBoundaryMarker;
    options.dynamic_viscosity = Real{2.5};

    const auto certificate =
        certifyGeneratedBoundaryAggregateTrace(
            system,
            options);
    EXPECT_EQ(
        certificate.aggregation_content_digest,
        report.canonical_content_digest);
    EXPECT_NE(
        certificate.canonical_certificate_digest,
        0u);
    EXPECT_EQ(
        certificate.cut_context_content_revision,
        context->contentRevision());
    EXPECT_EQ(
        certificate.free_surface_snapshot_revision,
        snapshot->revision()
            .snapshot_revision_key);
    EXPECT_EQ(
        certificate.source_value_revision,
        kSourceValueRevision);
    EXPECT_EQ(
        certificate
            .affine_constraint_layout_revision,
        system.constraints()
            .constraintLayoutRevision());
    EXPECT_EQ(certificate.active_cell_count, 2u);
    EXPECT_EQ(
        certificate.generated_boundary_rule_count,
        1u);
    EXPECT_EQ(certificate.certified_patch_count, 1u);
    EXPECT_EQ(certificate.maximum_support_overlap, 1u);
    EXPECT_EQ(
        certificate
            .maximum_terminal_tangent_dimension,
        6u);
    EXPECT_NEAR(
        certificate.retained_active_physical_volume,
        Real{79.0} / Real{128.0},
        Real{1.0e-14});
    EXPECT_NEAR(
        certificate
            .generated_boundary_physical_measure,
        Real{1.0} / Real{8.0},
        Real{1.0e-14});

    ASSERT_EQ(certificate.patches.size(), 1u);
    const auto& patch =
        certificate.patches.front();
    EXPECT_FALSE(
        patch.synthetic_full_active_patch);
    EXPECT_EQ(patch.canonical_patch_index, 0u);
    EXPECT_EQ(patch.root_cell_gid, root_cell_gid);
    EXPECT_EQ(
        patch.support_cell_gids,
        expected_support);
    EXPECT_EQ(
        patch.boundary_rule_stable_ids,
        (std::vector<std::uint64_t>{
            boundary_stable_id}));
    EXPECT_EQ(patch.raw_support_dof_count, 8u);
    EXPECT_EQ(
        patch.terminal_tangent_dof_count,
        6u);
    EXPECT_EQ(patch.rigid_mode_candidate_count, 3u);
    EXPECT_EQ(
        patch.structural_rigid_mode_count,
        3u);
    EXPECT_EQ(
        patch.rigid_mode_constraint_rank,
        0u);
    EXPECT_EQ(
        patch.rigid_mode_quotient_status,
        GeneratedBoundaryRigidModeQuotientStatus::
            Applied);
    EXPECT_EQ(patch.maximum_cell_support_overlap, 1u);
    EXPECT_NEAR(
        patch.retained_support_physical_volume,
        Real{79.0} / Real{128.0},
        Real{1.0e-14});
    EXPECT_NEAR(
        patch.generated_boundary_physical_measure,
        Real{1.0} / Real{8.0},
        Real{1.0e-14});

    const auto& bound = patch.generalized_bound;
    EXPECT_EQ(bound.dimension, 6u);
    EXPECT_EQ(bound.positive_rank, 3u);
    EXPECT_EQ(bound.nullity, 3u);
    EXPECT_TRUE(bound.denominator_converged);
    EXPECT_TRUE(bound.quotient_converged);
    EXPECT_TRUE(bound.explicit_nullspace.applied);
    EXPECT_TRUE(
        bound.explicit_nullspace
            .exact_binary64_actions_proven);
    EXPECT_TRUE(
        bound.explicit_nullspace
            .exact_binary64_anchor_rank_proven);
    EXPECT_EQ(
        bound.explicit_nullspace
            .supplied_nullity,
        3u);
    EXPECT_EQ(
        bound.explicit_nullspace
            .maximum_denominator_action,
        Real{0.0});
    EXPECT_EQ(
        bound.explicit_nullspace
            .maximum_numerator_action,
        Real{0.0});
    EXPECT_TRUE(bound.exact_dyadic.applied);
    EXPECT_TRUE(
        bound.exact_dyadic
            .denominator_positive_definite_proven);
    EXPECT_TRUE(
        bound.exact_dyadic
            .numerator_positive_semidefinite_proven);
    EXPECT_TRUE(
        bound.exact_dyadic.upper_inequality_proven);
    EXPECT_EQ(bound.exact_dyadic.dimension, 3u);
    EXPECT_EQ(bound.exact_dyadic.denominator_rank, 3u);
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.exact_dyadic.directly_proven_upper_bound);
    EXPECT_NEAR(
        bound.largest_quotient_eigenvalue,
        Real{32.0} / Real{79.0},
        Real{1.0e-10});
    EXPECT_TRUE(
        std::isfinite(
            bound.conservative_upper_bound));
    EXPECT_GT(
        bound.conservative_upper_bound,
        Real{0.0});
    EXPECT_GE(
        bound.conservative_upper_bound,
        bound.largest_quotient_eigenvalue);
    EXPECT_TRUE(
        std::isfinite(
            certificate
                .global_conservative_upper_bound));
    EXPECT_GE(
        certificate
            .global_conservative_upper_bound,
        bound.conservative_upper_bound);

    const auto repeated =
        certifyGeneratedBoundaryAggregateTrace(
            system,
            options);
    EXPECT_EQ(
        repeated.canonical_certificate_digest,
        certificate.canonical_certificate_digest);
    EXPECT_EQ(
        repeated.global_conservative_upper_bound,
        certificate.global_conservative_upper_bound);
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     RootlessAggregateSupportIsRejected)
{
    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);
    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    ASSERT_NO_THROW(system.setup());

    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    context->addFreeSurfaceGeometrySnapshot(
        rootlessCutTriangleSnapshot(
            system.meshAccess()),
        geometry::CutIntegrationSide::Negative);
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto reports =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(reports.size(), 1u);
    ASSERT_NE(reports.front(), nullptr);
    EXPECT_TRUE(reports.front()->trace_bound_eligible);
    ASSERT_EQ(reports.front()->patches.size(), 1u);
    EXPECT_EQ(
        reports.front()->patches.front().kind,
        constraints::
            SmallCutAggregationPatchKind::Rootless);

    try {
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            GeneratedBoundaryAggregateTraceCertificationOptions{
                .field = velocity,
                .physical_boundary_marker = kWallMarker,
                .volume_interface_marker =
                    kInterfaceMarker,
                .generated_active_boundary_marker =
                    kActiveBoundaryMarker,
                .dynamic_viscosity = Real{2.5},
            });
        FAIL() << "rootless aggregate trace certification "
                  "must fail closed";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(
            std::string(error.what()).find(
                "rootless aggregate support cannot certify"),
            std::string::npos)
            << error.what();
    }
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     ImportedGeneratedDomainsWithoutAuthoritativeSnapshotFailClosed)
{
    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);

    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    ASSERT_NO_THROW(system.setup());

    const auto snapshot =
        fullNegativeSnapshot(system.meshAccess());
    ASSERT_NE(snapshot, nullptr);
    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(
        snapshot->interfaceDomain(),
        geometry::CutIntegrationSide::Negative);
    for (const auto& contact :
         snapshot->contactDomains()) {
        context
            ->addGeneratedInterfaceBoundaryIntersectionDomain(
                contact);
    }
    for (const auto& active :
         snapshot->activeBoundaryDomains()) {
        context->addGeneratedActiveBoundaryDomain(
            active);
    }
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto reports =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(reports.size(), 1u);
    ASSERT_NE(reports.front(), nullptr);
    EXPECT_TRUE(reports.front()->trace_bound_eligible);
    EXPECT_FALSE(
        reports.front()
            ->revision.has_free_surface_snapshot_revision);

    try {
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            GeneratedBoundaryAggregateTraceCertificationOptions{
                .field = velocity,
                .physical_boundary_marker = kWallMarker,
                .volume_interface_marker =
                    kInterfaceMarker,
                .generated_active_boundary_marker =
                    kActiveBoundaryMarker,
                .dynamic_viscosity = Real{2.5},
            });
        FAIL() << "snapshot-free trace certification must fail "
                  "closed";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(
            std::string(error.what()).find(
                "aggregation report lacks authoritative "
                "free-surface snapshot revisions"),
            std::string::npos)
            << error.what();
    }
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     ScalarFieldIsRejectedAsAnUnsupportedTraceSpace)
{
    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    systems::FESystem system(mesh);
    const auto scalar =
        system.addField(
            systems::FieldSpec{
                .name = "scalar",
                .space = scalar_space,
                .components = 1});
    system.addOperator("scalar");
    ASSERT_NO_THROW(system.setup());

    try {
        (void)certifyGeneratedBoundaryAggregateTrace(
            system,
            GeneratedBoundaryAggregateTraceCertificationOptions{
                .field = scalar,
                .physical_boundary_marker = kWallMarker,
                .volume_interface_marker =
                    kInterfaceMarker,
                .generated_active_boundary_marker =
                    kActiveBoundaryMarker,
                .dynamic_viscosity = Real{1.0},
            });
        FAIL() << "scalar trace certification must fail closed";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(
            std::string(error.what()).find(
                "field must be an affine P1 Product H1 velocity"),
            std::string::npos)
            << error.what();
    }
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     SymmetricPolicyRejectsAnInsufficientConfiguredPenalty)
{
    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);

    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    installFormBoundTracePolicy(
        system,
        velocity,
        *velocity_space,
        Real{4.0},
        true);
    ASSERT_NO_THROW(system.setup());

    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    context->addFreeSurfaceGeometrySnapshot(
        fullNegativeSnapshot(system.meshAccess()),
        geometry::CutIntegrationSide::Negative);
    system.setCutIntegrationContext(context);
    EXPECT_THROW(
        system.rebuildConstraintState(),
        std::runtime_error);
    EXPECT_TRUE(
        system.generatedBoundaryNitscheTraceCertificates()
            .empty());
}

TEST(GeneratedBoundaryAggregateTraceCertificate,
     UnsymmetricPolicyRetainsTheBoundWithoutACoercivityThreshold)
{
    const ScopedEnvironmentVariable slave_all_cut(
        "SVMP_AGGREGATION_SLAVE_ALL_CUT",
        "0");
    const ScopedEnvironmentVariable linear_extension(
        "SVMP_AGGREGATION_LINEAR_EXTENSION",
        "0");
    const ScopedEnvironmentVariable allow_unaggregated(
        "SVMP_AGGREGATION_ALLOW_UNAGGREGATED",
        "0");
    const ScopedEnvironmentVariable maximum_lines(
        "SVMP_AGGREGATION_MAX_LINES",
        nullptr);

    auto mesh = unitTriangleMesh();
    auto scalar_space =
        std::make_shared<spaces::H1Space>(
            ElementType::Triangle3,
            /*order=*/1);
    auto velocity_space =
        std::make_shared<spaces::ProductSpace>(
            scalar_space,
            /*components=*/2);

    systems::FESystem system(mesh);
    const auto velocity =
        system.addField(
            systems::FieldSpec{
                .name = "velocity",
                .space = velocity_space,
                .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(
        std::make_unique<
            constraints::SmallCutAggregationConstraint>(
            velocity,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    installFormBoundTracePolicy(
        system,
        velocity,
        *velocity_space,
        Real{1.0},
        false);
    ASSERT_NO_THROW(system.setup());

    auto context =
        std::make_shared<
            assembly::CutIntegrationContext>();
    context->addFreeSurfaceGeometrySnapshot(
        fullNegativeSnapshot(system.meshAccess()),
        geometry::CutIntegrationSide::Negative);
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto eager =
        system.generatedBoundaryNitscheTraceCertificates();
    ASSERT_EQ(eager.size(), 1u);
    EXPECT_GT(
        eager.front().trace_to_penalty_ratio,
        Real{1.0});
    EXPECT_EQ(
        eager.front()
            .grouped_symmetric_trace_to_penalty_ratio,
        Real{0.0});
    EXPECT_FALSE(
        eager.front()
            .symmetric_energy_ratio_lower_bound
            .has_value());
}

} // namespace
} // namespace svmp::FE::analysis::test
