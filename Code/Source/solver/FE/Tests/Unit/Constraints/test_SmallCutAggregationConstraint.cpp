/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SmallCutAggregationConstraint.cpp
 * @brief Direct algorithm tests for AgFEM small-cut aggregation.
 *
 * Hand-built generated cut-volume rules on tiny structured meshes pin down
 * the production contract: candidate selection, root BFS through cut cells,
 * emitted extrapolation weights (full-order and linear corner sub-basis),
 * wall/gauge exclusion (including Q2 midside wall nodes), strong-Dirichlet
 * override of master-bearing lines, the fail-closed under-aggregation
 * policy, the sub-parametric rejection, and the pruned-sliver interplay with
 * the inactive-pin constraint path.
 *
 * DOF identity convention: midside/interior dofs are resolved through the
 * DofHandler's nodal cell pairing (getCellDofs in mesh-node order). The
 * pairing itself is independently validated inside the fixtures: corner
 * positions must agree with the EntityDofMap vertex lookup, and nodes shared
 * by two cells must resolve to the same dof from both cells. (The
 * EntityDofMap edge lookup is NOT used: MeshBase topological edge ids and
 * the entity map's edge indexing do not correspond on these fixtures.)
 */

#include <gtest/gtest.h>

#include <span>

#include "Assembly/CutIntegrationContext.h"
#include "Basis/NodeOrderingConventions.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Constraints/SmallCutAggregationCellIndex.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Core/AggregationGuardDiagnostics.h"
#include "Core/Logger.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <bit>
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

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

constexpr int kInterfaceMarker = 7;

// The detailed aggregation diagnostics print only when the logger admits
// DEBUG (FE_LOG_LEVEL=DEBUG); tests that read them raise the level.
class ScopedLogLevel {
public:
    explicit ScopedLogLevel(LogLevel level)
        : prior_(Logger::instance().get_level())
    {
        Logger::instance().set_level(level);
    }

    ~ScopedLogLevel() { Logger::instance().set_level(prior_); }

    ScopedLogLevel(const ScopedLogLevel&) = delete;
    ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;

private:
    LogLevel prior_;
};

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* key, const char* value)
        : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_value_ = std::string(prior);
        }
        ::setenv(key_, value, 1);
    }

    ~ScopedEnvVar()
    {
        if (prior_value_.has_value()) {
            ::setenv(key_, prior_value_->c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* key_;
    std::optional<std::string> prior_value_;
};

/// Label the derived boundary face containing both given corner vertices.
/// Must run AFTER finalize(): explicit BoundaryOnly face storage does not
/// survive the full codim-1 derivation that higher-order (edge-dof) meshes
/// trigger, so fixtures label the derived faces instead.
void labelBoundaryFaceWithCorners(MeshBase& base,
                                  index_t first_vertex,
                                  index_t second_vertex,
                                  int marker)
{
    const auto& f2c = base.face2cell();
    for (std::size_t f = 0; f < f2c.size(); ++f) {
        const bool boundary =
            (f2c[f][0] == INVALID_INDEX) != (f2c[f][1] == INVALID_INDEX);
        if (!boundary) {
            continue;
        }
        const auto [ptr, count] =
            base.face_vertices_span(static_cast<index_t>(f));
        if (ptr == nullptr || count < 2u) {
            continue;
        }
        bool has_first = false;
        bool has_second = false;
        for (std::size_t i = 0; i < count; ++i) {
            has_first = has_first || ptr[i] == first_vertex;
            has_second = has_second || ptr[i] == second_vertex;
        }
        if (has_first && has_second) {
            base.set_boundary_label(static_cast<index_t>(f), marker);
            return;
        }
    }
    ADD_FAILURE() << "boundary face with corners (" << first_vertex << ","
                  << second_vertex << ") not found";
}

/// Label the unique derived boundary face containing every requested corner.
/// This is unambiguous for the non-tensor Wedge/Pyramid fixtures below, where
/// a two-corner lookup could select a neighboring face sharing an edge.
void labelBoundaryFaceWithCornerSet(MeshBase& base,
                                    const std::vector<index_t>& corners,
                                    int marker)
{
    const auto& f2c = base.face2cell();
    for (std::size_t f = 0; f < f2c.size(); ++f) {
        const bool boundary =
            (f2c[f][0] == INVALID_INDEX) != (f2c[f][1] == INVALID_INDEX);
        if (!boundary) {
            continue;
        }
        const auto [ptr, count] =
            base.face_vertices_span(static_cast<index_t>(f));
        if (ptr == nullptr) {
            continue;
        }
        bool contains_all = true;
        for (const auto corner : corners) {
            bool found = false;
            for (std::size_t i = 0; i < count; ++i) {
                found = found || ptr[i] == corner;
            }
            if (!found) {
                contains_all = false;
                break;
            }
        }
        if (contains_all) {
            base.set_boundary_label(static_cast<index_t>(f), marker);
            return;
        }
    }
    ADD_FAILURE() << "boundary face with requested corner set not found";
}

std::shared_ptr<Mesh> buildSingleQuadratic3DCell(ElementType type,
                                                 std::size_t wall_face,
                                                 int wall_marker)
{
    const auto node_count = basis::ReferenceNodeLayout::num_nodes(type);
    std::vector<real_t> x_ref;
    x_ref.reserve(3u * node_count);
    std::vector<index_t> connectivity;
    connectivity.reserve(node_count);
    for (std::size_t node = 0; node < node_count; ++node) {
        const auto point = basis::ReferenceNodeLayout::get_node_coords(type, node);
        x_ref.push_back(static_cast<real_t>(point[0]));
        x_ref.push_back(static_cast<real_t>(point[1]));
        x_ref.push_back(static_cast<real_t>(point[2]));
        connectivity.push_back(static_cast<index_t>(node));
    }

    CellShape shape{};
    if (type == ElementType::Wedge18) {
        shape.family = CellFamily::Wedge;
        shape.num_corners = 6;
    } else if (type == ElementType::Pyramid14) {
        shape.family = CellFamily::Pyramid;
        shape.num_corners = 5;
    } else {
        throw std::invalid_argument(
            "buildSingleQuadratic3DCell requires Wedge18 or Pyramid14");
    }
    shape.order = 2;

    auto base = std::make_shared<MeshBase>();
    base->build_from_arrays(
        /*spatial_dim=*/3,
        x_ref,
        std::vector<offset_t>{0, static_cast<offset_t>(node_count)},
        connectivity,
        std::vector<CellShape>{shape});
    base->finalize();

    const auto reference = elements::ReferenceElement::create(type);
    const auto& face_nodes = reference.face_nodes(wall_face);
    std::vector<index_t> face_corners;
    face_corners.reserve(face_nodes.size());
    for (const auto local : face_nodes) {
        face_corners.push_back(static_cast<index_t>(local));
    }
    labelBoundaryFaceWithCornerSet(*base, face_corners, wall_marker);
    return create_mesh(std::move(base));
}

/// Strip of n unit Q1 quads along x. Vertices: bottom row 0..n, top row
/// n+1..2n+1; cell c = {c, c+1, n+2+c, n+1+c} (CCW). With a wall marker the
/// left edge (x=0) boundary face is labeled after finalize.
std::shared_ptr<Mesh> buildQuadStrip(int n_cells,
                                     std::optional<int> left_wall_marker = std::nullopt,
                                     Real coordinate_scale = Real(1))
{
    auto base = std::make_shared<MeshBase>();

    std::vector<real_t> x_ref;
    for (int row = 0; row < 2; ++row) {
        for (int i = 0; i <= n_cells; ++i) {
            x_ref.push_back(static_cast<real_t>(i) * coordinate_scale);
            x_ref.push_back(static_cast<real_t>(row) * coordinate_scale);
        }
    }
    std::vector<offset_t> cell2vertex_offsets{0};
    std::vector<index_t> cell2vertex;
    for (int c = 0; c < n_cells; ++c) {
        const auto b = static_cast<index_t>(c);
        const auto t = static_cast<index_t>(n_cells + 1 + c);
        cell2vertex.insert(cell2vertex.end(),
                           {b, static_cast<index_t>(b + 1),
                            static_cast<index_t>(t + 1), t});
        cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
    }

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(static_cast<std::size_t>(n_cells), shape));
    base->finalize();

    if (left_wall_marker.has_value()) {
        labelBoundaryFaceWithCorners(*base,
                                     0,
                                     static_cast<index_t>(n_cells + 1),
                                     *left_wall_marker);
    }

    return create_mesh(std::move(base));
}

/// Column of 2*rows unit P1 triangles over [0,1]x[0,rows]: vertex v(i,j) =
/// 2j + i; row j holds cell 2j = {v(0,j), v(1,j), v(1,j+1)} (its first
/// vertex is not the right-angle vertex, so its first-vertex chart is
/// sheared) and cell 2j+1 = {v(0,j), v(1,j+1), v(0,j+1)}.  With
/// root_starts_at_right_angle, cell 0 lists the same vertices from its
/// right-angle vertex, {v(1,0), v(1,1), v(0,0)}.
std::shared_ptr<Mesh> buildTriangleColumn(int rows,
                                          bool root_starts_at_right_angle = false)
{
    auto base = std::make_shared<MeshBase>();
    std::vector<real_t> x_ref;
    for (int j = 0; j <= rows; ++j) {
        for (int i = 0; i < 2; ++i) {
            x_ref.push_back(static_cast<real_t>(i));
            x_ref.push_back(static_cast<real_t>(j));
        }
    }
    const auto v = [](int i, int j) { return static_cast<index_t>(2 * j + i); };
    std::vector<offset_t> cell2vertex_offsets{0};
    std::vector<index_t> cell2vertex;
    for (int j = 0; j < rows; ++j) {
        if (j == 0 && root_starts_at_right_angle) {
            cell2vertex.insert(cell2vertex.end(), {v(1, 0), v(1, 1), v(0, 0)});
        } else {
            cell2vertex.insert(cell2vertex.end(), {v(0, j), v(1, j), v(1, j + 1)});
        }
        cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
        cell2vertex.insert(cell2vertex.end(), {v(0, j), v(1, j + 1), v(0, j + 1)});
        cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
    }
    CellShape shape{};
    shape.family = CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(static_cast<std::size_t>(2 * rows), shape));
    base->finalize();
    return create_mesh(std::move(base));
}

/// Three Q1 quads where c0 and c2 meet only at vertex 0, while c0 and c1
/// share a face. This separates c2 from the c0/c1 active face component
/// without duplicating its C0 vertex DOF.
std::shared_ptr<Mesh> buildVertexTouchQuadPatch()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        -1.0, -1.0,
        0.0, -1.0,
        -1.0, 0.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4, 8, 12};
    const std::vector<index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4,
        6, 7, 0, 8,
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(3, shape));
    base->finalize();
    return create_mesh(std::move(base));
}

/// A face-connected c0/c1 strip plus a detached c2 quad. The detached cell
/// supplies a master in a distinct active feature without initially
/// participating in the rooted aggregation patch.
std::shared_ptr<Mesh> buildDetachedQuadPatch()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        3.0, 0.0,
        4.0, 0.0,
        4.0, 1.0,
        3.0, 1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4, 8, 12};
    const std::vector<index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4,
        6, 7, 8, 9,
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(3, shape));
    base->finalize();
    return create_mesh(std::move(base));
}

/// Two iso-parametric 9-node quads: c0 = [0,1]^2, c1 = [1,2]x[0,1].
/// Corners 0..5 (bottom 0,1,2; top 3,4,5), c0 midsides 6(0.5,0) 7(1,0.5)
/// 8(0.5,1) 9(0,0.5) center 10(0.5,0.5); c1 midsides 11(1.5,0) 12(2,0.5)
/// 13(1.5,1) center 14(1.5,0.5); node 7 is the shared edge midside.
std::shared_ptr<Mesh> buildTwoQuad9Strip(std::optional<int> left_wall_marker = std::nullopt)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        0.5, 0.0,
        1.0, 0.5,
        0.5, 1.0,
        0.0, 0.5,
        0.5, 0.5,
        1.5, 0.0,
        2.0, 0.5,
        1.5, 1.0,
        1.5, 0.5,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 9, 18};
    const std::vector<index_t> cell2vertex = {
        0, 1, 4, 3, 6, 7, 8, 9, 10,
        1, 2, 5, 4, 11, 12, 13, 7, 14,
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(2, shape));
    base->finalize();

    if (left_wall_marker.has_value()) {
        labelBoundaryFaceWithCorners(*base, 0, 3, *left_wall_marker);
    }

    return create_mesh(std::move(base));
}

struct CellRuleSpec {
    GlobalIndex cell{-1};
    Real volume_fraction{0.0};
    bool full_cell_equivalent{false};
    std::optional<std::uint64_t> cut_topology_revision{};
};

void addCellRule(assembly::CutIntegrationContext& context,
                 const CellRuleSpec& spec,
                 geometry::CutIntegrationSide side,
                 std::uint64_t source_value_revision = 0u)
{
    const auto cut_topology_revision =
        spec.cut_topology_revision.value_or(
            interfaces::cutVolumeStableId(
                kInterfaceMarker,
                spec.cell,
                /*local_region_index=*/0,
                side,
                /*source_revision=*/1u));

    assembly::CutCellAssemblyMetadata metadata{};
    metadata.cell = spec.cell;
    metadata.parent_entity = spec.cell;
    metadata.side = side;
    metadata.volume_fraction = spec.volume_fraction;
    metadata.revision_key = cut_topology_revision;
    metadata.cut_topology_revision = cut_topology_revision;
    metadata.source_value_revision = source_value_revision;

    geometry::CutQuadratureRule rule{};
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = side;
    rule.measure = spec.volume_fraction;
    rule.parent_measure = Real{1.0};
    rule.volume_fraction = spec.volume_fraction;
    rule.full_cell_equivalent = spec.full_cell_equivalent;
    rule.frame = geometry::CutGeometryFrame::Current;
    rule.provenance.parent_entity = spec.cell;
    rule.provenance.parent_entity_global_id = spec.cell;
    rule.provenance.marker = kInterfaceMarker;
    rule.provenance.cut_topology_revision =
        cut_topology_revision;
    rule.provenance.source_value_revision = source_value_revision;

    context.addGeneratedVolumeRule(kInterfaceMarker, metadata, rule);
}

std::shared_ptr<assembly::CutIntegrationContext> makeCutContext(
    const std::vector<CellRuleSpec>& specs,
    geometry::CutIntegrationSide side = geometry::CutIntegrationSide::Negative)
{
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    for (const auto& spec : specs) {
        addCellRule(*context, spec, side);
    }
    return context;
}

std::shared_ptr<assembly::CutIntegrationContext> makePublishedCutContext(
    const std::vector<CellRuleSpec>& specs,
    std::uint64_t source_value_revision)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/91,
        /*layout_revision=*/7u,
        source_value_revision);
    request.generated_domain_id =
        "small_cut_class_swap_publication";
    request.interface_marker = kInterfaceMarker;
    request.isovalue = Real{0.0};
    request.quadrature_policy_key = 23u;
    request.frame = geometry::CutGeometryFrame::Current;

    interfaces::LevelSetInterfaceDomain domain(request);
    for (const auto& spec : specs) {
        interfaces::CutInterfaceVolumeRegion region;
        region.interface_marker = kInterfaceMarker;
        region.parent_cell = spec.cell;
        region.parent_cell_global_id = spec.cell;
        region.local_region_index = 0;
        region.stable_id =
            spec.cut_topology_revision.value_or(
                interfaces::cutVolumeStableId(
                    kInterfaceMarker,
                    spec.cell,
                    /*local_region_index=*/0,
                    geometry::CutIntegrationSide::Negative,
                    /*source_revision=*/1u));
        region.side =
            geometry::CutIntegrationSide::Negative;
        region.parent_measure = Real{1.0};
        region.measure = spec.volume_fraction;
        region.volume_fraction = spec.volume_fraction;
        region.full_cell_equivalent =
            spec.full_cell_equivalent;
        domain.addVolumeRegion(std::move(region));
    }

    auto context = std::make_shared<assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(domain);
    return context;
}

[[nodiscard]] GlobalIndex vertexDof(const systems::FESystem& system,
                                    FieldId field,
                                    GlobalIndex vertex,
                                    std::size_t component = 0)
{
    const auto* entity = system.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(entity, nullptr);
    if (entity == nullptr) {
        return GlobalIndex{-1};
    }
    const auto dofs = entity->getVertexDofs(vertex);
    EXPECT_GT(dofs.size(), component);
    if (dofs.size() <= component) {
        return GlobalIndex{-1};
    }
    return system.fieldDofOffset(field) + dofs[component];
}

class TimeDependentVertexPin final : public ISystemConstraint {
public:
    TimeDependentVertexPin(FieldId field,
                           GlobalIndex vertex,
                           Real initial_value)
        : field_(field),
          vertex_(vertex),
          current_value_(initial_value)
    {
    }

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override
    {
        const auto* entity =
            system.fieldDofHandler(field_).getEntityDofMap();
        if (entity == nullptr) {
            throw std::logic_error(
                "TimeDependentVertexPin requires an entity DOF map");
        }
        const auto dofs = entity->getVertexDofs(vertex_);
        if (dofs.size() != 1u) {
            throw std::logic_error(
                "TimeDependentVertexPin requires one scalar vertex DOF");
        }
        resolved_dof_ = system.fieldDofOffset(field_) + dofs.front();
        constraints.addDirichlet(resolved_dof_, current_value_);
    }

    bool updateValues(const systems::FESystem&,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override
    {
        if (resolved_dof_ == INVALID_GLOBAL_INDEX) {
            throw std::logic_error(
                "TimeDependentVertexPin was updated before application");
        }
        const auto next_value = static_cast<Real>(time + dt);
        if (next_value == current_value_) {
            return false;
        }
        constraints.updateInhomogeneity(resolved_dof_, next_value);
        current_value_ = next_value;
        return true;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return true;
    }

    [[nodiscard]] ConstraintDependencyDeclaration
    dependencyDeclaration() const override
    {
        auto out = ISystemConstraint::dependencyDeclaration();
        out.structural.labels = false;
        out.structural.ownership = false;
        return out;
    }

    [[nodiscard]] systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        systems::SetupStorageRequirements requirements;
        requirements.entity_dof_map = true;
        return requirements;
    }

private:
    FieldId field_{INVALID_FIELD_ID};
    GlobalIndex vertex_{INVALID_GLOBAL_INDEX};
    GlobalIndex resolved_dof_{INVALID_GLOBAL_INDEX};
    Real current_value_{0.0};
};

class GeometryDependentVertexPin final : public ISystemConstraint {
public:
    GeometryDependentVertexPin(FieldId field,
                               GlobalIndex vertex,
                               Real initial_value)
        : field_(field),
          vertex_(vertex),
          current_value_(initial_value)
    {
    }

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override
    {
        const auto* entity =
            system.fieldDofHandler(field_).getEntityDofMap();
        if (entity == nullptr) {
            throw std::logic_error(
                "GeometryDependentVertexPin requires an entity DOF map");
        }
        const auto dofs = entity->getVertexDofs(vertex_);
        if (dofs.size() != 1u) {
            throw std::logic_error(
                "GeometryDependentVertexPin requires one scalar vertex DOF");
        }
        resolved_dof_ =
            system.fieldDofOffset(field_) + dofs.front();
        constraints.addDirichlet(resolved_dof_, current_value_);
    }

    bool updateValues(const systems::FESystem&,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override
    {
        if (resolved_dof_ == INVALID_GLOBAL_INDEX) {
            throw std::logic_error(
                "GeometryDependentVertexPin was updated before application");
        }
        const auto next_value = static_cast<Real>(time + dt);
        if (next_value == current_value_) {
            return false;
        }
        constraints.updateInhomogeneity(resolved_dof_, next_value);
        current_value_ = next_value;
        return true;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] ConstraintDependencyDeclaration
    dependencyDeclaration() const override
    {
        auto out = ISystemConstraint::dependencyDeclaration();
        out.value.geometry = true;
        return out;
    }

    [[nodiscard]] systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        systems::SetupStorageRequirements requirements;
        requirements.entity_dof_map = true;
        return requirements;
    }

private:
    FieldId field_{INVALID_FIELD_ID};
    GlobalIndex vertex_{INVALID_GLOBAL_INDEX};
    GlobalIndex resolved_dof_{INVALID_GLOBAL_INDEX};
    Real current_value_{0.0};
};

class VertexAffineTie final : public ISystemConstraint {
public:
    VertexAffineTie(FieldId field,
                    GlobalIndex slave_vertex,
                    GlobalIndex master_vertex)
        : field_(field),
          slave_vertex_(slave_vertex),
          master_vertex_(master_vertex)
    {
    }

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override
    {
        const auto resolve_vertex =
            [&](GlobalIndex vertex) {
                const auto* entity =
                    system.fieldDofHandler(field_).getEntityDofMap();
                if (entity == nullptr) {
                    throw std::logic_error(
                        "VertexAffineTie requires an entity DOF map");
                }
                const auto dofs = entity->getVertexDofs(vertex);
                if (dofs.size() != 1u) {
                    throw std::logic_error(
                        "VertexAffineTie requires one scalar vertex DOF");
                }
                return system.fieldDofOffset(field_) + dofs.front();
            };

        ConstraintLine line;
        line.slave_dof = resolve_vertex(slave_vertex_);
        line.entries.push_back(
            {resolve_vertex(master_vertex_), 1.0});
        constraints.addConstraintLine(line);
    }

    bool updateValues(const systems::FESystem&,
                      AffineConstraints&,
                      double,
                      double) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override
    {
        return false;
    }

    [[nodiscard]] systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        systems::SetupStorageRequirements requirements;
        requirements.entity_dof_map = true;
        return requirements;
    }

private:
    FieldId field_{INVALID_FIELD_ID};
    GlobalIndex slave_vertex_{INVALID_GLOBAL_INDEX};
    GlobalIndex master_vertex_{INVALID_GLOBAL_INDEX};
};

/// Scalar-field dof of a cell-local mesh node via the DofHandler's nodal
/// pairing (cell dofs in mesh-node order).
[[nodiscard]] GlobalIndex cellNodeDof(const systems::FESystem& system,
                                      FieldId field,
                                      GlobalIndex cell,
                                      std::size_t local_node)
{
    const auto dofs = system.fieldDofHandler(field).getCellDofs(cell);
    EXPECT_GT(dofs.size(), local_node);
    if (dofs.size() <= local_node) {
        return GlobalIndex{-1};
    }
    return system.fieldDofOffset(field) + dofs[local_node];
}

/// System-global product-field DOF at a cell-local node/component. Product
/// cell DOFs are component-major in the public DofHandler cell view; deriving
/// expected masters here avoids assuming that EntityDofMap vertex numbering
/// is also the product cell ordering.
[[nodiscard]] GlobalIndex cellNodeComponentDof(
    const systems::FESystem& system,
    FieldId field,
    GlobalIndex cell,
    std::size_t local_node,
    std::size_t component,
    std::size_t basis_count)
{
    const auto dofs = system.fieldDofHandler(field).getCellDofs(cell);
    const auto position = component * basis_count + local_node;
    EXPECT_GT(dofs.size(), position);
    if (dofs.size() <= position) {
        return GlobalIndex{-1};
    }
    return system.fieldDofOffset(field) + dofs[position];
}

/// Independent validation of the nodal cell pairing on the two-quad9 strip:
/// corner slots must agree with the EntityDofMap, and the shared midside
/// node 7 must resolve to the same dof from both incident cells.
void validateQuad9NodalPairing(const systems::FESystem& system, FieldId field)
{
    EXPECT_EQ(cellNodeDof(system, field, 0, 0), vertexDof(system, field, 0));
    EXPECT_EQ(cellNodeDof(system, field, 0, 1), vertexDof(system, field, 1));
    EXPECT_EQ(cellNodeDof(system, field, 0, 2), vertexDof(system, field, 4));
    EXPECT_EQ(cellNodeDof(system, field, 0, 3), vertexDof(system, field, 3));
    EXPECT_EQ(cellNodeDof(system, field, 1, 0), vertexDof(system, field, 1));
    EXPECT_EQ(cellNodeDof(system, field, 1, 1), vertexDof(system, field, 2));
    EXPECT_EQ(cellNodeDof(system, field, 1, 2), vertexDof(system, field, 5));
    EXPECT_EQ(cellNodeDof(system, field, 1, 3), vertexDof(system, field, 4));
    // Shared edge midside (node 7): c0 slot 5 == c1 slot 7.
    EXPECT_EQ(cellNodeDof(system, field, 0, 5), cellNodeDof(system, field, 1, 7));
}

/// Sorted (master_dof, weight) pairs of a constraint line.
[[nodiscard]] std::vector<std::pair<GlobalIndex, double>> lineEntries(
    const systems::FESystem& system,
    GlobalIndex dof)
{
    std::vector<std::pair<GlobalIndex, double>> out;
    const auto view = system.constraints().getConstraint(dof);
    EXPECT_TRUE(view.has_value()) << "dof " << dof << " is not constrained";
    if (!view.has_value()) {
        return out;
    }
    for (const auto& entry : view->entries) {
        out.emplace_back(entry.master_dof, entry.weight);
    }
    std::sort(out.begin(), out.end());
    return out;
}

void expectEntries(const std::vector<std::pair<GlobalIndex, double>>& actual,
                   std::vector<std::pair<GlobalIndex, double>> expected,
                   double tol = 1.0e-9)
{
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].first, expected[i].first) << "entry " << i;
        EXPECT_NEAR(actual[i].second, expected[i].second, tol) << "entry " << i;
    }
}

void expectHomogeneousPin(const systems::FESystem& system, GlobalIndex dof)
{
    const auto view = system.constraints().getConstraint(dof);
    ASSERT_TRUE(view.has_value()) << "dof " << dof << " is not constrained";
    EXPECT_TRUE(view->isDirichlet()) << "dof " << dof;
    EXPECT_NEAR(view->inhomogeneity, 0.0, 1.0e-15) << "dof " << dof;
}

[[nodiscard]] std::vector<std::pair<GlobalIndex, double>>
prolongationEntries(const std::vector<ConstraintEntry>& entries)
{
    std::vector<std::pair<GlobalIndex, double>> out;
    out.reserve(entries.size());
    for (const auto& entry : entries) {
        out.emplace_back(entry.master_dof, entry.weight);
    }
    std::sort(out.begin(), out.end());
    return out;
}

[[nodiscard]] const SmallCutAggregationProlongationRow*
findFinalizedProlongationRow(
    const SmallCutAggregationProlongationReport& report,
    GlobalIndex slave)
{
    const auto found = std::find_if(
        report.rows.begin(),
        report.rows.end(),
        [slave](const auto& row) {
            return row.slave_dof == slave;
        });
    EXPECT_NE(found, report.rows.end())
        << "missing finalized prolongation row for slave " << slave;
    return found == report.rows.end() ? nullptr : &*found;
}

void expectRootlessQuadraticWallFaceExclusion(
    ElementType type,
    std::size_t wall_face,
    const std::vector<std::size_t>& excluded_local_nodes)
{
    constexpr int wall_marker = 11;
    auto mesh = buildSingleQuadratic3DCell(type, wall_face, wall_marker);
    auto space = std::make_shared<spaces::H1Space>(type, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{wall_marker}));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto node_count = basis::ReferenceNodeLayout::num_nodes(type);
    ASSERT_TRUE(std::is_sorted(excluded_local_nodes.begin(),
                               excluded_local_nodes.end()));
    std::size_t expected_pins = 0u;
    for (std::size_t local = 0; local < node_count; ++local) {
        const bool excluded = std::binary_search(excluded_local_nodes.begin(),
                                                 excluded_local_nodes.end(),
                                                 local);
        const auto dof = cellNodeDof(system, pressure, 0, local);
        const auto line = system.constraints().getConstraint(dof);
        if (excluded) {
            EXPECT_FALSE(line.has_value()) << "excluded local node " << local;
            continue;
        }
        ++expected_pins;
        ASSERT_TRUE(line.has_value()) << "rootless local node " << local;
        EXPECT_TRUE(line->isDirichlet()) << "rootless local node " << local;
        EXPECT_TRUE(line->entries.empty()) << "rootless local node " << local;
        EXPECT_NEAR(line->inhomogeneity, 0.0, 1.0e-15)
            << "rootless local node " << local;
    }
    EXPECT_EQ(system.constraints().numConstraints(), expected_pins);
}

} // namespace

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#define SVMP_AGG_TEST_BODY
#else
#define SVMP_AGG_TEST_BODY GTEST_SKIP() << "Requires FE built with Mesh integration.";
#endif

TEST(SmallCutAggregationConstraint,
     PhysicalRootOrderingIgnoresPartitionNumberedMastersAndProviders)
{
    struct RootProposalView {
        detail::SmallCutAggregationPhysicalRootKey key;
        int provider_rank;
        std::vector<GlobalIndex> algebraic_master_dofs;
        std::vector<std::pair<GlobalIndex, double>> physical_line;
    };

    const auto select = [](const std::vector<RootProposalView>& proposals)
        -> const RootProposalView& {
        return *std::min_element(
            proposals.begin(),
            proposals.end(),
            [](const auto& lhs, const auto& rhs) {
                return detail::smallCutAggregationPhysicalRootLess(
                    lhs.key,
                    lhs.provider_rank,
                    rhs.key,
                    rhs.provider_rank);
            });
    };

    const std::vector<std::pair<GlobalIndex, double>> expected_line{
        {900, 2.0},
        {901, -1.0},
    };

    // Equal-distance roots receive opposite owner-contiguous master ordering
    // and opposite providers in these two partition views. A master-DOF key
    // would therefore choose a different physical root in each view.
    const std::vector<RootProposalView> first_partition{
        {{1u, 41}, 1, {100, 101}, expected_line},
        {{1u, 52}, 0, {1, 2}, {{910, 2.0}, {911, -1.0}}},
    };
    const std::vector<RootProposalView> second_partition{
        {{1u, 41}, 0, {1, 2}, expected_line},
        {{1u, 52}, 1, {100, 101}, {{910, 2.0}, {911, -1.0}}},
    };

    const auto& first_choice = select(first_partition);
    const auto& second_choice = select(second_partition);
    EXPECT_EQ(first_choice.key.cell_gid, 41);
    EXPECT_EQ(second_choice.key.cell_gid, 41);
    EXPECT_EQ(first_choice.physical_line, expected_line);
    EXPECT_EQ(second_choice.physical_line, expected_line);
    EXPECT_NE(first_choice.algebraic_master_dofs,
              second_choice.algebraic_master_dofs);
    EXPECT_NE(first_choice.provider_rank, second_choice.provider_rank);

    // Provider rank breaks a tie only after the physical root and physical
    // line are identical.
    const std::vector<RootProposalView> equivalent_providers{
        {{1u, 41}, 3, {300, 301}, expected_line},
        {{1u, 41}, 1, {100, 101}, expected_line},
    };
    const auto& provider_choice = select(equivalent_providers);
    EXPECT_EQ(provider_choice.key.cell_gid, 41);
    EXPECT_EQ(provider_choice.physical_line, expected_line);
    EXPECT_EQ(provider_choice.provider_rank, 1);
}

TEST(SmallCutAggregationConstraint, RejectsInvalidMarkerAndInterfaceActiveSide)
{
    EXPECT_THROW(
        (SmallCutAggregationConstraint(
            FieldId{0}, geometry::CutIntegrationSide::Negative, -1)),
        std::invalid_argument);
    EXPECT_THROW(
        (SmallCutAggregationConstraint(
            FieldId{0}, geometry::CutIntegrationSide::Interface,
            kInterfaceMarker)),
        std::invalid_argument);
    auto invalid_guards = SmallCutAggregationGuardOptions{};
    invalid_guards.maximum_root_path_length = 0u;
    EXPECT_THROW(
        (SmallCutAggregationConstraint(
            FieldId{0},
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker,
            {},
            {},
            invalid_guards)),
        std::invalid_argument);
    invalid_guards = SmallCutAggregationGuardOptions{};
    invalid_guards.maximum_row_l1_norm = 0.5;
    EXPECT_THROW(
        (SmallCutAggregationConstraint(
            FieldId{0},
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker,
            {},
            {},
            invalid_guards)),
        std::invalid_argument);
}

TEST(SmallCutAggregationConstraint,
     AllowsInitialSetupWithoutContextButPostSetupRebuildFailsClosed)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(1);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));

    EXPECT_NO_THROW(system.setup());
    EXPECT_EQ(system.constraints().numConstraints(), 0u);
    try {
        system.rebuildConstraintState();
        FAIL() << "post-setup aggregation rebuild must require a cut context";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("missing_cut_integration_context"),
                  std::string::npos);
    }
    EXPECT_EQ(system.constraints().numConstraints(), 0u);
#endif
}

TEST(SmallCutAggregationConstraint, WrongGeneratedVolumeMarkerFailsClosed)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(1);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        /*wrong marker=*/kInterfaceMarker + 1));
    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    }));

    try {
        system.rebuildConstraintState();
        FAIL() << "wrong aggregation marker must not silently disable the constraint";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(
                      "missing_marker_cell_classification"),
                  std::string::npos);
    }
    EXPECT_EQ(system.constraints().numConstraints(), 0u);
#endif
}

TEST(SmallCutAggregationConstraint, InvalidRetainedVolumeFractionsFailClosed)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(1);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));
    ASSERT_NO_THROW(system.setup());

    for (const Real invalid :
         {std::numeric_limits<Real>::quiet_NaN(), Real{1.25}}) {
        system.setCutIntegrationContext(makeCutContext({
            {.cell = 0,
             .volume_fraction = invalid,
             .full_cell_equivalent = false},
        }));
        try {
            system.rebuildConstraintState();
            FAIL() << "invalid retained volume fraction must fail closed";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(
                          "invalid_retained_volume_fraction"),
                      std::string::npos);
        }
        EXPECT_EQ(system.constraints().numConstraints(), 0u);
    }
#endif
}

TEST(SmallCutAggregationConstraint, SlavesOnlyUnsupportedCutVerticesWithExtrapolatedRootWeights)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    mesh->set_cell_gids({});
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto leading_field = system.addField(
        systems::FieldSpec{
            .name = "leading", .space = space, .components = 1});
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    EXPECT_NE(leading_field, pressure);
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    ASSERT_GT(system.fieldDofOffset(pressure), 0);
    ASSERT_FALSE(system.meshAccess().globalEntityIdsAvailable());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto& constraints = system.constraints();
    // v0 (0,0) and v3 (0,1) touch only the cut cell: slaved. The shared
    // vertices v1/v4 touch the full-active root, v2/v5 belong to it: free.
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 1)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 4)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 5)));

    // Bilinear extension of root cell [1,2]x[0,1] evaluated at (0,0): the
    // y=0 masters carry 2 and -1, the y=1 masters vanish. Same row at y=1
    // for v3. Lines are homogeneous.
    expectEntries(lineEntries(system, vertexDof(system, pressure, 0)),
                  {{vertexDof(system, pressure, 1), 2.0},
                   {vertexDof(system, pressure, 2), -1.0}});
    expectEntries(lineEntries(system, vertexDof(system, pressure, 3)),
                  {{vertexDof(system, pressure, 4), 2.0},
                   {vertexDof(system, pressure, 5), -1.0}});
    EXPECT_NEAR(system.constraints().getInhomogeneity(
                    vertexDof(system, pressure, 0)),
                0.0, 1.0e-15);

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    const auto& prolongation = *prolongations.front();
    EXPECT_EQ(prolongation.field, pressure);
    EXPECT_EQ(prolongation.active_side,
              geometry::CutIntegrationSide::Negative);
    EXPECT_EQ(prolongation.interface_marker, kInterfaceMarker);
    EXPECT_FALSE(prolongation.slave_all_cut);
    EXPECT_FALSE(prolongation.linear_extension);
    EXPECT_FALSE(prolongation.allow_unaggregated);
    EXPECT_TRUE(prolongation.trace_bound_eligible);
    EXPECT_NE(prolongation.canonical_content_digest, 0u);
    EXPECT_EQ(prolongation.revision.local_rank, 0);
    EXPECT_EQ(prolongation.revision.communicator_size, 1);
    EXPECT_TRUE(prolongation.revision.constraint.valid);
    EXPECT_EQ(prolongation.revision.constraint.fe_constraint_layout,
              system.constraintLayoutRevision());
    EXPECT_EQ(prolongation.revision.affine_constraint_layout_revision,
              system.constraints().constraintLayoutRevision());
    ASSERT_NE(system.cutIntegrationContext(), nullptr);
    EXPECT_EQ(prolongation.revision.cut_context_content_revision,
              system.cutIntegrationContext()->contentRevision());

    const auto bottom_slave = vertexDof(system, pressure, 0);
    const auto top_slave = vertexDof(system, pressure, 3);
    const auto* bottom_row =
        findFinalizedProlongationRow(prolongation, bottom_slave);
    const auto* top_row =
        findFinalizedProlongationRow(prolongation, top_slave);
    ASSERT_NE(bottom_row, nullptr);
    ASSERT_NE(top_row, nullptr);
    for (const auto* row : {bottom_row, top_row}) {
        EXPECT_EQ(row->candidate_dof, row->slave_dof);
        EXPECT_EQ(row->component, 0u);
        EXPECT_EQ(row->slave_owner_rank, 0);
        EXPECT_EQ(row->provisional_kind,
                  SmallCutAggregationProvisionalRowKind::RootedExtension);
        EXPECT_EQ(row->final_kind,
                  SmallCutAggregationFinalRowKind::MasterBearing);
        EXPECT_FALSE(row->preconstrained_at_apply);
        EXPECT_NE(row->root_cell_gid, INVALID_GLOBAL_INDEX);
        EXPECT_EQ(row->root_cell_owner_rank, 0);
        EXPECT_EQ(row->root_provider_rank, 0);
        EXPECT_EQ(row->root_distance, 1u);
        EXPECT_NEAR(row->final_inhomogeneity, 0.0, 1.0e-15);
    }
    EXPECT_EQ(bottom_row->root_cell_gid, top_row->root_cell_gid);
    expectEntries(
        prolongationEntries(bottom_row->provisional_entries),
        {{vertexDof(system, pressure, 1), 2.0},
         {vertexDof(system, pressure, 2), -1.0}});
    expectEntries(
        prolongationEntries(bottom_row->final_entries),
        {{vertexDof(system, pressure, 1), 2.0},
         {vertexDof(system, pressure, 2), -1.0}});
    expectEntries(
        prolongationEntries(top_row->provisional_entries),
        {{vertexDof(system, pressure, 4), 2.0},
         {vertexDof(system, pressure, 5), -1.0}});
    expectEntries(
        prolongationEntries(top_row->final_entries),
        {{vertexDof(system, pressure, 4), 2.0},
         {vertexDof(system, pressure, 5), -1.0}});

    ASSERT_EQ(prolongation.active_cells.size(), 2u);
    const auto full_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [](const auto& cell) {
            return cell.kind ==
                SmallCutAggregationActiveCellKind::FullActive;
        });
    const auto cut_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [](const auto& cell) {
            return cell.kind ==
                SmallCutAggregationActiveCellKind::Cut;
        });
    ASSERT_NE(full_cell, prolongation.active_cells.end());
    ASSERT_NE(cut_cell, prolongation.active_cells.end());
    EXPECT_EQ(full_cell->cell_gid, bottom_row->root_cell_gid);
    EXPECT_EQ(full_cell->owner_rank, 0);
    EXPECT_EQ(cut_cell->owner_rank, 0);
    EXPECT_EQ(full_cell->retained_measure_provider_rank, 0);
    EXPECT_EQ(cut_cell->retained_measure_provider_rank, 0);
    EXPECT_NEAR(full_cell->retained_physical_volume, 1.0, 1.0e-14);
    EXPECT_NEAR(cut_cell->retained_physical_volume, 0.3, 1.0e-14);
    EXPECT_EQ(full_cell->retained_rule_stable_ids.size(), 1u);
    EXPECT_EQ(cut_cell->retained_rule_stable_ids.size(), 1u);
    EXPECT_EQ(full_cell->active_feature_id, cut_cell->active_feature_id);
    EXPECT_TRUE(std::binary_search(cut_cell->field_dofs.begin(),
                                   cut_cell->field_dofs.end(),
                                   bottom_slave));
    EXPECT_TRUE(std::binary_search(cut_cell->field_dofs.begin(),
                                   cut_cell->field_dofs.end(),
                                   top_slave));
    const auto pressure_begin = system.fieldDofOffset(pressure);
    const auto pressure_end =
        pressure_begin + system.fieldDofHandler(pressure).getNumDofs();
    for (const auto& cell : prolongation.active_cells) {
        ASSERT_FALSE(cell.field_dofs.empty());
        EXPECT_TRUE(std::all_of(
            cell.field_dofs.begin(),
            cell.field_dofs.end(),
            [pressure_begin, pressure_end](GlobalIndex dof) {
                return dof >= pressure_begin && dof < pressure_end;
            }));
    }

    ASSERT_EQ(prolongation.patches.size(), 1u);
    const auto& patch = prolongation.patches.front();
    EXPECT_EQ(patch.kind, SmallCutAggregationPatchKind::Rooted);
    EXPECT_EQ(patch.root_cell_gid, bottom_row->root_cell_gid);
    EXPECT_EQ(patch.root_cell_owner_rank, 0);
    ASSERT_EQ(patch.active_feature_ids.size(), 1u);
    EXPECT_EQ(patch.active_feature_ids.front(),
              full_cell->active_feature_id);
    EXPECT_EQ(patch.member_cell_gids.size(), 2u);
    EXPECT_EQ(patch.support_cell_gids.size(), 2u);
    EXPECT_EQ(patch.slave_dofs.size(), 2u);
    EXPECT_TRUE(std::binary_search(patch.slave_dofs.begin(),
                                   patch.slave_dofs.end(),
                                   bottom_slave));
    EXPECT_TRUE(std::binary_search(patch.slave_dofs.begin(),
                                   patch.slave_dofs.end(),
                                   top_slave));

    EXPECT_NE(log_output.find("diagnostic=small_cut_aggregation"), std::string::npos);
    EXPECT_NE(log_output.find("candidate_vertices=2"), std::string::npos);
    EXPECT_NE(log_output.find("aggregated_vertices=2"), std::string::npos);
    EXPECT_NE(log_output.find("vertices_without_root=0"), std::string::npos);
    EXPECT_NE(log_output.find("empty_line_failures=0"), std::string::npos);
    EXPECT_NE(log_output.find("maximum_root_path_length=8"),
              std::string::npos);
    EXPECT_NE(log_output.find("maximum_observed_root_path=1"),
              std::string::npos);
    EXPECT_NE(log_output.find(
                  "root_path_search=bounded_candidate_neighborhood"),
              std::string::npos);
    EXPECT_NE(log_output.find("root_path_seed_index_entries=4"),
              std::string::npos);
    EXPECT_NE(log_output.find("root_path_search_cell_visits=4"),
              std::string::npos);
    EXPECT_NE(log_output.find(
                  "maximum_observed_reference_extrapolation=2"),
              std::string::npos);
    EXPECT_NE(log_output.find("maximum_observed_absolute_coefficient=2"),
              std::string::npos);
    EXPECT_NE(log_output.find("maximum_observed_row_l1_norm=3"),
              std::string::npos);
    EXPECT_NE(log_output.find("pruned_volume_rules=0"), std::string::npos);
#endif
}

TEST(SmallCutAggregationConstraint,
     UniformlyTinyCellsRetainExactExtrapolatedRootCoordinates)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    constexpr Real coordinate_scale = Real(1e-16);
    auto mesh = buildQuadStrip(2, std::nullopt, coordinate_scale);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    // A fixed 1e-12 physical residual test would accept the initial xi=0.25
    // on this mesh.  The exact scale-aware inversion instead recovers the
    // same root-cell extrapolation as the unit-sized mesh.
    expectEntries(lineEntries(system, vertexDof(system, pressure, 0)),
                  {{vertexDof(system, pressure, 1), 2.0},
                   {vertexDof(system, pressure, 2), -1.0}});
    expectEntries(lineEntries(system, vertexDof(system, pressure, 3)),
                  {{vertexDof(system, pressure, 4), 2.0},
                   {vertexDof(system, pressure, 5), -1.0}});
#endif
}

TEST(SmallCutAggregationConstraint,
     FinalizedCellLedgerPreservesDistinctTopologyRevisionsWithinOneCell)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0,
         .volume_fraction = Real{0.1},
         .full_cell_equivalent = false,
         .cut_topology_revision = 101u},
        {.cell = 0,
         .volume_fraction = Real{0.2},
         .full_cell_equivalent = false,
         .cut_topology_revision = 102u},
        {.cell = 1,
         .volume_fraction = Real{1.0},
         .full_cell_equivalent = true,
         .cut_topology_revision = 201u},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    const auto& prolongation = *prolongations.front();
    EXPECT_TRUE(prolongation.trace_bound_eligible);
    ASSERT_EQ(prolongation.active_cells.size(), 2u);
    const auto cut_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [](const auto& cell) {
            return cell.kind ==
                SmallCutAggregationActiveCellKind::Cut;
        });
    const auto root_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [](const auto& cell) {
            return cell.kind ==
                SmallCutAggregationActiveCellKind::FullActive;
        });
    ASSERT_NE(cut_cell, prolongation.active_cells.end());
    ASSERT_NE(root_cell, prolongation.active_cells.end());
    EXPECT_NEAR(cut_cell->retained_physical_volume, 0.3, 1.0e-14);
    EXPECT_EQ(cut_cell->retained_rule_stable_ids,
              (std::vector<std::uint64_t>{101u, 102u}));
    EXPECT_NEAR(root_cell->retained_physical_volume, 1.0, 1.0e-14);
    EXPECT_EQ(root_cell->retained_rule_stable_ids,
              (std::vector<std::uint64_t>{201u}));
    ASSERT_EQ(prolongation.patches.size(), 1u);
    EXPECT_EQ(prolongation.patches.front().kind,
              SmallCutAggregationPatchKind::Rooted);
    ASSERT_EQ(prolongation.patches.front().active_feature_ids.size(), 1u);
    EXPECT_EQ(prolongation.patches.front().active_feature_ids.front(),
              cut_cell->active_feature_id);
#endif
}

TEST(SmallCutAggregationConstraint,
     ZeroTopologyRevisionMakesFinalizedTraceMetadataIneligible)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0,
         .volume_fraction = Real{0.3},
         .full_cell_equivalent = false,
         .cut_topology_revision = 0u},
        {.cell = 1,
         .volume_fraction = Real{1.0},
         .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    const auto& prolongation = *prolongations.front();
    EXPECT_FALSE(prolongation.trace_bound_eligible);
    EXPECT_NE(prolongation.canonical_content_digest, 0u);
    const auto cut_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [](const auto& cell) {
            return cell.kind ==
                SmallCutAggregationActiveCellKind::Cut;
        });
    ASSERT_NE(cut_cell, prolongation.active_cells.end());
    EXPECT_EQ(cut_cell->retained_rule_stable_ids,
              (std::vector<std::uint64_t>{0u}));
#endif
}

TEST(SmallCutAggregationConstraint,
     RootedPatchRetainsEveryVertexTouchActiveFeature)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildVertexTouchQuadPatch();
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 2, .volume_fraction = Real{0.4}, .full_cell_equivalent = false},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    const auto& prolongation = *prolongations.front();
    EXPECT_TRUE(prolongation.trace_bound_eligible);
    ASSERT_EQ(prolongation.active_cells.size(), 3u);

    const auto shared_dof = vertexDof(system, pressure, 0);
    const auto* shared_row =
        findFinalizedProlongationRow(prolongation, shared_dof);
    ASSERT_NE(shared_row, nullptr);
    EXPECT_EQ(shared_row->provisional_kind,
              SmallCutAggregationProvisionalRowKind::RootedExtension);
    EXPECT_EQ(shared_row->final_kind,
              SmallCutAggregationFinalRowKind::MasterBearing);

    std::vector<GlobalIndex> shared_active_features;
    for (const auto& cell : prolongation.active_cells) {
        if (cell.kind != SmallCutAggregationActiveCellKind::Cut ||
            !std::binary_search(cell.field_dofs.begin(),
                                cell.field_dofs.end(),
                                shared_dof)) {
            continue;
        }
        shared_active_features.push_back(cell.active_feature_id);
    }
    std::sort(shared_active_features.begin(),
              shared_active_features.end());
    shared_active_features.erase(
        std::unique(shared_active_features.begin(),
                    shared_active_features.end()),
        shared_active_features.end());
    ASSERT_EQ(shared_active_features.size(), 2u);

    const auto rooted_patch = std::find_if(
        prolongation.patches.begin(),
        prolongation.patches.end(),
        [](const auto& patch) {
            return patch.kind ==
                SmallCutAggregationPatchKind::Rooted;
        });
    ASSERT_NE(rooted_patch, prolongation.patches.end());
    EXPECT_TRUE(std::is_sorted(
        rooted_patch->active_feature_ids.begin(),
        rooted_patch->active_feature_ids.end()));
    EXPECT_EQ(std::adjacent_find(
                  rooted_patch->active_feature_ids.begin(),
                  rooted_patch->active_feature_ids.end()),
              rooted_patch->active_feature_ids.end());
    EXPECT_EQ(rooted_patch->active_feature_ids,
              shared_active_features);
    EXPECT_TRUE(std::binary_search(
        rooted_patch->slave_dofs.begin(),
        rooted_patch->slave_dofs.end(),
        shared_dof));
    EXPECT_EQ(rooted_patch->member_cell_gids.size(), 3u);
#endif
}

TEST(SmallCutAggregationConstraint,
     FinalAffineClosureAddsDetachedMasterFeatureToRootedPatch)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildDetachedQuadPatch();
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    system.addSystemConstraint(
        std::make_unique<VertexAffineTie>(
            pressure,
            /*slave_vertex=*/1,
            /*master_vertex=*/6));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    const auto& prolongation = *prolongations.front();
    EXPECT_TRUE(prolongation.trace_bound_eligible);

    const auto bottom_slave = vertexDof(system, pressure, 0);
    const auto tied_root_master = vertexDof(system, pressure, 1);
    const auto other_root_master = vertexDof(system, pressure, 2);
    const auto detached_master = vertexDof(system, pressure, 6);
    const auto* bottom_row =
        findFinalizedProlongationRow(prolongation, bottom_slave);
    ASSERT_NE(bottom_row, nullptr);
    EXPECT_EQ(bottom_row->provisional_kind,
              SmallCutAggregationProvisionalRowKind::RootedExtension);
    EXPECT_EQ(bottom_row->final_kind,
              SmallCutAggregationFinalRowKind::MasterBearing);
    expectEntries(
        prolongationEntries(bottom_row->provisional_entries),
        {{tied_root_master, 2.0},
         {other_root_master, -1.0}});
    expectEntries(
        prolongationEntries(bottom_row->final_entries),
        {{detached_master, 2.0},
         {other_root_master, -1.0}});

    const auto root_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [&](const auto& cell) {
            return cell.cell_gid ==
                bottom_row->root_cell_gid;
        });
    const auto detached_cell = std::find_if(
        prolongation.active_cells.begin(),
        prolongation.active_cells.end(),
        [](const auto& cell) {
            return cell.cell_gid == 2;
        });
    ASSERT_NE(root_cell, prolongation.active_cells.end());
    ASSERT_NE(detached_cell, prolongation.active_cells.end());
    ASSERT_NE(root_cell->active_feature_id,
              detached_cell->active_feature_id);
    EXPECT_TRUE(std::binary_search(
        detached_cell->field_dofs.begin(),
        detached_cell->field_dofs.end(),
        detached_master));

    const auto rooted_patch = std::find_if(
        prolongation.patches.begin(),
        prolongation.patches.end(),
        [&](const auto& patch) {
            return patch.kind ==
                       SmallCutAggregationPatchKind::Rooted &&
                   std::binary_search(
                       patch.slave_dofs.begin(),
                       patch.slave_dofs.end(),
                       bottom_slave);
        });
    ASSERT_NE(rooted_patch, prolongation.patches.end());
    EXPECT_FALSE(std::binary_search(
        rooted_patch->member_cell_gids.begin(),
        rooted_patch->member_cell_gids.end(),
        detached_cell->cell_gid));
    EXPECT_TRUE(std::binary_search(
        rooted_patch->support_cell_gids.begin(),
        rooted_patch->support_cell_gids.end(),
        detached_cell->cell_gid));

    std::vector<GlobalIndex> expected_features{
        root_cell->active_feature_id,
        detached_cell->active_feature_id,
    };
    std::sort(expected_features.begin(),
              expected_features.end());
    ASSERT_EQ(expected_features.size(), 2u);
    EXPECT_EQ(rooted_patch->active_feature_ids,
              expected_features);
#endif
}

TEST(SmallCutAggregationConstraint, RootSearchTraversesCutCellsToNearestFullActiveCell)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // c0 cut, c1 cut, c2 full-active: candidates on c0 must BFS through the
    // cut band to root at c2 and receive its (further-extrapolated) weights.
    auto mesh = buildQuadStrip(3);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.2}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{0.5}, .full_cell_equivalent = false},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    // Bottom row vertices: 0(0,0) 1(1,0) 2(2,0) 3(3,0); top row 4..7.
    const auto& constraints = system.constraints();
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 1)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 4)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 5)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 6)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 7)));

    // Root [2,3]x[0,1]: linear extension along y=0 gives (3-x) and (x-2).
    expectEntries(lineEntries(system, vertexDof(system, pressure, 1)),
                  {{vertexDof(system, pressure, 2), 2.0},
                   {vertexDof(system, pressure, 3), -1.0}});
    expectEntries(lineEntries(system, vertexDof(system, pressure, 0)),
                  {{vertexDof(system, pressure, 2), 3.0},
                   {vertexDof(system, pressure, 3), -2.0}});
#endif
}

TEST(SmallCutAggregationConstraint, RootPathGuardRejectsLongCutBand)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(3);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto guards = SmallCutAggregationGuardOptions{};
    guards.maximum_root_path_length = 1u;
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{},
        std::vector<GlobalIndex>{},
        guards));
    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.2}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{0.5}, .full_cell_equivalent = false},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    try {
        system.rebuildConstraintState();
        FAIL() << "a root beyond the fixed path guard must fail closed";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(
                      "root_path_guard_rejection"),
                  std::string::npos);
        EXPECT_NE(std::string(error.what()).find("maximum_allowed_path=1"),
                  std::string::npos);
        EXPECT_NE(std::string(error.what()).find(
                      "Small_cut_aggregation_rootless_fallback"),
                  std::string::npos);
    }
    EXPECT_EQ(system.constraints().numConstraints(), 0u);
#endif
}

TEST(SmallCutAggregationConstraint, RootlessFallbackPinsCandidatesBeyondRootPathGuard)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // c0 cut, c1 cut, c2 full, path guard 1: the vertices of c0 alone (0, 4)
    // reach c2 only at path 2, so they have no admissible extension.  With
    // the opt-in fallback they get the rootless-island policy (D33) instead
    // of failing; vertices 1 and 5 reach c2 at path 1 and keep their rows.
    auto mesh = buildQuadStrip(3);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto guards = SmallCutAggregationGuardOptions{};
    guards.maximum_root_path_length = 1u;
    guards.rootless_fallback = true;
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{},
        std::vector<GlobalIndex>{},
        guards));
    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.2}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{0.5}, .full_cell_equivalent = false},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    const auto totals_before = diagnostics::aggregationGuardRootlessTotals();
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    for (const GlobalIndex vertex : {0, 4}) {
        expectHomogeneousPin(system, vertexDof(system, pressure, vertex));
    }
    expectEntries(lineEntries(system, vertexDof(system, pressure, 1)),
                  {{vertexDof(system, pressure, 2), 2.0},
                   {vertexDof(system, pressure, 3), -1.0}});
    expectEntries(lineEntries(system, vertexDof(system, pressure, 5)),
                  {{vertexDof(system, pressure, 6), 2.0},
                   {vertexDof(system, pressure, 7), -1.0}});
    const auto reports = system.completedSmallCutAggregationRefreshReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports.front().root_path_guard_rejections, 2u);
    EXPECT_NE(log_output.find("diagnostic=aggregation_guard_rootless "
                              "reason=root_path_guard"),
              std::string::npos);
    EXPECT_NE(log_output.find("maximum_allowed_path=1"), std::string::npos);
    EXPECT_NE(log_output.find("feature_cells=3"), std::string::npos);
    const auto totals_after = diagnostics::aggregationGuardRootlessTotals();
    EXPECT_EQ(totals_after.root_path_guard_candidates_total -
                  totals_before.root_path_guard_candidates_total,
              2u);
    EXPECT_EQ(totals_after.proposal_guard_candidates_total,
              totals_before.proposal_guard_candidates_total);
    EXPECT_NE(diagnostics::aggregationGuardRootlessSummary().find(
                  "root_path_guard_candidates_total="),
              std::string::npos);
#endif
}

TEST(SmallCutAggregationConstraint,
     RootTraversalRetainsOnlyRootsInsideGuardedCandidateNeighborhood)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(7);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto guards = SmallCutAggregationGuardOptions{};
    guards.maximum_root_path_length = 3u;
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{},
        std::vector<GlobalIndex>{},
        guards));
    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 1, .volume_fraction = Real{0.2}, .full_cell_equivalent = false},
        {.cell = 2, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 3, .volume_fraction = Real{0.4}, .full_cell_equivalent = false},
        {.cell = 4, .volume_fraction = Real{0.5}, .full_cell_equivalent = false},
        {.cell = 5, .volume_fraction = Real{0.6}, .full_cell_equivalent = false},
        {.cell = 6, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto reports = system.completedSmallCutAggregationRefreshReports();
    ASSERT_EQ(reports.size(), 1u);
    EXPECT_EQ(reports.front().maximum_root_path_length, 3u);
    EXPECT_EQ(reports.front().maximum_observed_root_path, 3u);
    EXPECT_EQ(reports.front().root_path_guard_rejections, 0u);
    EXPECT_NE(log_output.find(
                  "root_path_search=bounded_candidate_neighborhood"),
              std::string::npos);
    EXPECT_NE(log_output.find("maximum_observed_root_path=3"),
              std::string::npos);
    EXPECT_NE(log_output.find("root_path_guard_rejections=0"),
              std::string::npos);

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    ASSERT_FALSE(prolongations.front()->rows.empty());
    for (const auto& row : prolongations.front()->rows) {
        EXPECT_LE(row.root_distance, 3u);
    }
#endif
}

TEST(SmallCutAggregationConstraint,
     ExtrapolationAndCoefficientGuardsRejectAmplifyingRows)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto run_rejection = [](SmallCutAggregationGuardOptions guards) {
        auto mesh = buildQuadStrip(2);
        auto space = std::make_shared<spaces::H1Space>(
            ElementType::Quad4, /*order=*/1);
        systems::FESystem system(mesh);
        const auto pressure = system.addField(systems::FieldSpec{
            .name = "p", .space = space, .components = 1});
        system.addOperator("pressure");
        system.addSystemConstraint(
            std::make_unique<SmallCutAggregationConstraint>(
                pressure,
                geometry::CutIntegrationSide::Negative,
                kInterfaceMarker,
                std::vector<int>{},
                std::vector<GlobalIndex>{},
                guards));
        EXPECT_NO_THROW(system.setup());
        system.setCutIntegrationContext(makeCutContext({
            {.cell = 0,
             .volume_fraction = Real{0.3},
             .full_cell_equivalent = false},
            {.cell = 1,
             .volume_fraction = Real{1.0},
             .full_cell_equivalent = true},
        }));
        try {
            system.rebuildConstraintState();
            ADD_FAILURE() << "candidates without a root inside the guards must fail closed";
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("diagnostic=aggregation_no_root_inside_guards"),
                      std::string::npos)
                << message;
            EXPECT_EQ(message.find("incomplete_distributed_aggregation_halo"),
                      std::string::npos)
                << message;
            EXPECT_NE(message.find("Small_cut_aggregation_rootless_fallback"),
                      std::string::npos);
        }
        EXPECT_EQ(system.constraints().numConstraints(), 0u);
    };

    auto extrapolation_guards = SmallCutAggregationGuardOptions{};
    extrapolation_guards.maximum_reference_extrapolation_distance = 1.0;
    run_rejection(extrapolation_guards);

    auto coefficient_guards = SmallCutAggregationGuardOptions{};
    coefficient_guards.maximum_absolute_coefficient = 1.5;
    run_rejection(coefficient_guards);
#endif
}

TEST(SmallCutAggregationConstraint,
     RootlessFallbackPinsCandidatesWhoseProposalsFailTheGuards)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // The only root proposal of the candidates 0 and 3 fails the guard, so
    // they have no admissible extension.  With the opt-in fallback they get
    // the rootless-island policy (D33) instead of failing.
    const auto run_rejection = [](SmallCutAggregationGuardOptions guards) {
        auto mesh = buildQuadStrip(2);
        auto space = std::make_shared<spaces::H1Space>(
            ElementType::Quad4, /*order=*/1);
        systems::FESystem system(mesh);
        const auto pressure = system.addField(systems::FieldSpec{
            .name = "p", .space = space, .components = 1});
        system.addOperator("pressure");
        system.addSystemConstraint(
            std::make_unique<SmallCutAggregationConstraint>(
                pressure,
                geometry::CutIntegrationSide::Negative,
                kInterfaceMarker,
                std::vector<int>{},
                std::vector<GlobalIndex>{},
                guards));
        EXPECT_NO_THROW(system.setup());
        system.setCutIntegrationContext(makeCutContext({
            {.cell = 0,
             .volume_fraction = Real{0.3},
             .full_cell_equivalent = false},
            {.cell = 1,
             .volume_fraction = Real{1.0},
             .full_cell_equivalent = true},
        }));
        const auto totals_before = diagnostics::aggregationGuardRootlessTotals();
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        EXPECT_NO_THROW(system.rebuildConstraintState());
        auto log_output = testing::internal::GetCapturedStdout();
        log_output += testing::internal::GetCapturedStderr();
        for (const GlobalIndex vertex : {0, 3}) {
            expectHomogeneousPin(system, vertexDof(system, pressure, vertex));
        }
        EXPECT_EQ(system.constraints().numConstraints(), 2u);
        EXPECT_NE(log_output.find("diagnostic=aggregation_guard_rootless "
                                  "reason=proposal_guard"),
                  std::string::npos);
        const auto totals_after = diagnostics::aggregationGuardRootlessTotals();
        EXPECT_EQ(totals_after.proposal_guard_candidates_total -
                      totals_before.proposal_guard_candidates_total,
                  2u);
        EXPECT_EQ(totals_after.refreshes_with_cases -
                      totals_before.refreshes_with_cases,
                  1u);
    };

    auto extrapolation_guards = SmallCutAggregationGuardOptions{};
    extrapolation_guards.maximum_reference_extrapolation_distance = 1.0;
    extrapolation_guards.rootless_fallback = true;
    run_rejection(extrapolation_guards);

    auto coefficient_guards = SmallCutAggregationGuardOptions{};
    coefficient_guards.maximum_absolute_coefficient = 1.5;
    coefficient_guards.rootless_fallback = true;
    run_rejection(coefficient_guards);
#endif
}

TEST(SmallCutAggregationConstraint,
     RejectedRootProposalDoesNotContaminateAcceptedGuardTelemetry)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(5);
    auto space = std::make_shared<spaces::H1Space>(
        ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(systems::FieldSpec{
        .name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto guards = SmallCutAggregationGuardOptions{};
    guards.maximum_reference_extrapolation_distance = 2.0;
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker,
            std::vector<int>{},
            std::vector<GlobalIndex>{},
            guards));
    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0,
         .volume_fraction = Real{1.0},
         .full_cell_equivalent = true},
        {.cell = 1,
         .volume_fraction = Real{0.2},
         .full_cell_equivalent = false},
        {.cell = 2,
         .volume_fraction = Real{0.3},
         .full_cell_equivalent = false},
        {.cell = 3,
         .volume_fraction = Real{0.4},
         .full_cell_equivalent = false},
        {.cell = 4,
         .volume_fraction = Real{1.0},
         .full_cell_equivalent = true},
    }));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto reports =
        system.completedSmallCutAggregationRefreshReports();
    ASSERT_EQ(reports.size(), 1u);
    const auto& report = reports.front();
    EXPECT_GT(report.extrapolation_guard_rejections, 0u);
    EXPECT_LE(report.maximum_observed_reference_extrapolation,
              report.maximum_reference_extrapolation_distance);
    EXPECT_NE(log_output.find(
                  "maximum_attempted_reference_extrapolation=4"),
              std::string::npos);
#endif
}

TEST(SmallCutAggregationConstraint,
     SimplexExtrapolationGuardDoesNotDependOnRootVertexOrder)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Wetting-wedge geometry: only the bottom triangle of a 4-row column is
    // full, so the top vertices (0,4) and (1,4) must extrapolate from it.
    // In its first-vertex chart (origin (0,0), sheared) they lie 5.0 and
    // 4.24 from the reference simplex, beyond the default guard 4; in the
    // charts of the other vertices 3.16 and 3.0.  The guard must not depend
    // on the vertex order, so both rows are accepted with the default
    // guards and keep the exact P1 extension weights, whichever vertex the
    // mesh lists first.
    const auto build = [](const SmallCutAggregationGuardOptions& guards,
                          systems::FESystem& system) {
        const auto pressure = system.addField(systems::FieldSpec{
            .name = "p",
            .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3,
                                                       /*order=*/1),
            .components = 1});
        system.addOperator("pressure");
        system.addSystemConstraint(
            std::make_unique<SmallCutAggregationConstraint>(
                pressure,
                geometry::CutIntegrationSide::Negative,
                kInterfaceMarker,
                std::vector<int>{},
                std::vector<GlobalIndex>{},
                guards));
        EXPECT_NO_THROW(system.setup());
        std::vector<CellRuleSpec> rules{
            {.cell = 0, .volume_fraction = Real{1.0}, .full_cell_equivalent = true}};
        for (GlobalIndex cell = 1; cell < 8; ++cell) {
            rules.push_back({.cell = cell,
                             .volume_fraction = Real{0.5},
                             .full_cell_equivalent = false});
        }
        system.setCutIntegrationContext(makeCutContext(rules));
        return pressure;
    };

    for (const bool root_starts_at_right_angle : {false, true}) {
        SCOPED_TRACE(root_starts_at_right_angle ? "right-angle first" : "sheared chart first");
        systems::FESystem system(
            buildTriangleColumn(4, root_starts_at_right_angle));
        const auto pressure = build(SmallCutAggregationGuardOptions{}, system);
        ASSERT_NO_THROW(system.rebuildConstraintState());
        // Root {v0=(0,0), v1=(1,0), v3=(1,1)}: weights (1-a-b, a, b) with
        // (x,y) = (a+b, b).
        expectEntries(lineEntries(system, vertexDof(system, pressure, 9)),
                      {{vertexDof(system, pressure, 1), -3.0},
                       {vertexDof(system, pressure, 3), 4.0}});
        expectEntries(lineEntries(system, vertexDof(system, pressure, 8)),
                      {{vertexDof(system, pressure, 0), 1.0},
                       {vertexDof(system, pressure, 1), -4.0},
                       {vertexDof(system, pressure, 3), 4.0}});
        for (const GlobalIndex vertex : {0, 1, 3}) {
            EXPECT_FALSE(system.constraints().isConstrained(
                vertexDof(system, pressure, vertex)));
        }
        const auto reports =
            system.completedSmallCutAggregationRefreshReports();
        ASSERT_EQ(reports.size(), 1u);
        EXPECT_EQ(reports.front().extrapolation_guard_rejections, 0u);
        EXPECT_NEAR(reports.front().maximum_observed_reference_extrapolation,
                    std::sqrt(Real{10.0}), 1.0e-12);
    }
    {
        // The guard still binds on the vertex-order-invariant distance.
        auto guards = SmallCutAggregationGuardOptions{};
        guards.maximum_reference_extrapolation_distance = 2.9;
        systems::FESystem system(buildTriangleColumn(4));
        static_cast<void>(build(guards, system));
        try {
            system.rebuildConstraintState();
            FAIL() << "a root beyond the invariant extrapolation guard must fail closed";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("no_valid_root_proposal"),
                      std::string::npos);
        }
    }
    {
        // With the opt-in fallback the top vertices (3.16 and 3.0) get the
        // rootless-island policy and the others keep their rows.
        auto guards = SmallCutAggregationGuardOptions{};
        guards.maximum_reference_extrapolation_distance = 2.9;
        guards.rootless_fallback = true;
        systems::FESystem system(buildTriangleColumn(4));
        const auto pressure = build(guards, system);
        ASSERT_NO_THROW(system.rebuildConstraintState());
        for (const GlobalIndex vertex : {8, 9}) {
            expectHomogeneousPin(system, vertexDof(system, pressure, vertex));
        }
        expectEntries(lineEntries(system, vertexDof(system, pressure, 7)),
                      {{vertexDof(system, pressure, 1), -2.0},
                       {vertexDof(system, pressure, 3), 3.0}});
    }
#endif
}

TEST(SmallCutAggregationConstraint, ProductFieldSlavesAllComponentsWithSameWeights)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<spaces::ProductSpace>(scalar_space, /*components=*/2);

    systems::FESystem system(mesh);
    const auto velocity = system.addField(
        systems::FieldSpec{.name = "u", .space = vector_space, .components = 2});
    system.addOperator("velocity");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        velocity,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    // Every component of the slave vertex is constrained to the SAME
    // component of the masters with identical geometric weights — exercises
    // the cell-dof layout detection (node-major vs component-major) and the
    // slave-dof copy (a stale span here once rebound component >= 1 slaves
    // to master-node dofs).
    for (std::size_t component = 0; component < 2; ++component) {
        const auto slave = vertexDof(system, velocity, 0, component);
        ASSERT_TRUE(system.constraints().isConstrained(slave))
            << "component " << component;
        expectEntries(lineEntries(system, slave),
                      {{vertexDof(system, velocity, 1, component), 2.0},
                       {vertexDof(system, velocity, 2, component), -1.0}});
    }
    for (const auto vertex : {1, 2, 4, 5}) {
        for (std::size_t component = 0; component < 2; ++component) {
            EXPECT_FALSE(system.constraints().isConstrained(
                vertexDof(system, velocity, vertex, component)))
                << "vertex " << vertex << " component " << component;
        }
    }
#endif
}

TEST(SmallCutAggregationConstraint,
     ProductFieldAtMaximumSupportedComponentCountPreservesComponents)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // ProductSpace currently supports physical dimensions 1..3. Exercise its
    // maximum supported count and assert exact component preservation; a
    // higher-component fixture cannot be constructed through the public API.
    constexpr std::size_t component_count = 3u;
    auto mesh = buildQuadStrip(2);
    auto scalar_space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    auto product_space = std::make_shared<spaces::ProductSpace>(
        scalar_space, static_cast<int>(component_count));

    systems::FESystem system(mesh);
    const auto field = system.addField(systems::FieldSpec{
        .name = "q3",
        .space = product_space,
        .components = static_cast<int>(component_count)});
    system.addOperator("q3");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        field,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    const auto basis_count = static_cast<std::size_t>(
        system.fieldRecord(field).space->element().basis().size());

    for (const auto vertex : {0, 3}) {
        for (std::size_t component = 0; component < component_count;
             ++component) {
            const auto slave = vertexDof(system, field, vertex, component);
            const auto entries = lineEntries(system, slave);
            ASSERT_FALSE(entries.empty())
                << "vertex " << vertex << " component " << component;
            // buildQuadStrip root-cell connectivity is
            // {bottom-near, bottom-far, top-far, top-near}.
            const std::size_t near_root_slot = vertex == 0 ? 0u : 3u;
            const std::size_t far_root_slot = vertex == 0 ? 1u : 2u;
            const auto near_master = cellNodeComponentDof(
                system,
                field,
                /*root cell=*/1,
                near_root_slot,
                component,
                basis_count);
            const auto far_master = cellNodeComponentDof(
                system,
                field,
                /*root cell=*/1,
                far_root_slot,
                component,
                basis_count);
            const auto component_dofs = system.fieldMap().getComponentDofs(
                "q3", static_cast<LocalIndex>(component));
            EXPECT_TRUE(component_dofs.contains(near_master));
            EXPECT_TRUE(component_dofs.contains(far_master));
            expectEntries(
                entries,
                {{near_master, 2.0}, {far_master, -1.0}});
            long double sum = 0.0L;
            long double l1 = 0.0L;
            for (const auto& [master, weight] : entries) {
                EXPECT_GE(master, 0);
                EXPECT_TRUE(std::isfinite(weight));
                sum += static_cast<long double>(weight);
                l1 += std::abs(static_cast<long double>(weight));
            }
            const auto tolerance = 1.0e-10L * std::max(1.0L, l1);
            EXPECT_NEAR(static_cast<double>(sum), 1.0,
                        static_cast<double>(tolerance));
        }
    }
    EXPECT_EQ(system.constraints().numConstraints(),
              2u * component_count);
#endif
}

TEST(SmallCutAggregationConstraint, NoRootIslandCandidatesArePinnedHomogeneously)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Isolated cut island: cut rules exist but no full-active cell is
    // reachable. Breaking free surfaces produce these routinely (312-step
    // d18 sees up to ~48 per refresh), so the production policy is a
    // homogeneous fail-safe PIN — not a throw, and not a free dof.
    auto mesh = buildQuadStrip(2);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    auto context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    });
    addCellRule(*context,
                {.cell = 1,
                 .volume_fraction = Real{1.0},
                 .full_cell_equivalent = true},
                geometry::CutIntegrationSide::Positive);
    system.setCutIntegrationContext(std::move(context));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    // All four island candidates (vertices of the lone cut cell) are pinned
    // with empty-entry homogeneous Dirichlet lines.
    for (const auto vertex : {0, 1, 3, 4}) {
        const auto view = system.constraints().getConstraint(
            vertexDof(system, pressure, vertex));
        ASSERT_TRUE(view.has_value()) << "vertex " << vertex;
        EXPECT_TRUE(view->isDirichlet()) << "vertex " << vertex;
        EXPECT_NEAR(view->inhomogeneity, 0.0, 1.0e-15) << "vertex " << vertex;
    }
    EXPECT_NE(log_output.find("vertices_without_root=4"), std::string::npos);
    EXPECT_NE(log_output.find("island_pinned_dofs=4"), std::string::npos);
    EXPECT_NE(log_output.find("distributed_halo_validation=not_parallel"),
              std::string::npos);
#endif
}

TEST(SmallCutAggregationConstraint,
     CompletedRefreshReportSeparatesRootedAndRootlessCandidates)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto aggregation = std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker);
    const auto* aggregation_view = aggregation.get();
    system.addSystemConstraint(std::move(aggregation));

    ASSERT_NO_THROW(system.setup());
    EXPECT_FALSE(aggregation_view->completedRefreshReport().has_value());
    EXPECT_TRUE(
        system.finalizedSmallCutAggregationProlongations().empty());

    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    {
        const auto& report = aggregation_view->completedRefreshReport();
        ASSERT_TRUE(report.has_value());
        EXPECT_EQ(report->field, pressure);
        EXPECT_EQ(
            report->local_lineage.successful_publication_ordinal, 1u);
        EXPECT_EQ(
            report->geometry_identity.kind,
            SmallCutAggregationGeometryIdentityKind::Unavailable);
        EXPECT_FALSE(report->geometry_identity.available);
        EXPECT_FALSE(report->geometry_identity
                         .communicator_fingerprint_consensus_validated);
        EXPECT_EQ(report->active_side,
                  geometry::CutIntegrationSide::Negative);
        EXPECT_EQ(report->interface_marker, kInterfaceMarker);
        EXPECT_EQ(report->canonical_candidate_vertices, 2u);
        EXPECT_EQ(report->canonical_rooted_candidate_vertices, 2u);
        EXPECT_EQ(report->canonical_rootless_candidate_vertices, 0u);
        EXPECT_EQ(report->canonical_owned_aggregate_dofs, 2u);
        EXPECT_EQ(report->canonical_owned_pinned_dofs, 0u);
        EXPECT_EQ(report->canonical_strong_suppressed_dofs, 0u);
        EXPECT_EQ(report->canonical_active_feature_count, 1u);
        EXPECT_EQ(report->canonical_rooted_active_feature_count, 1u);
        EXPECT_EQ(report->canonical_rootless_active_feature_count, 0u);
        EXPECT_DOUBLE_EQ(
            report->canonical_rootless_active_physical_volume, 0.0);
        ASSERT_EQ(report->canonical_active_features.size(), 1u);
        const auto& feature = report->canonical_active_features.front();
        EXPECT_EQ(feature.stable_feature_id, 0);
        EXPECT_EQ(
            feature.disposition,
            SmallCutAggregationActiveFeatureDisposition::Rooted);
        EXPECT_EQ(feature.canonical_cell_count, 2u);
        EXPECT_EQ(feature.canonical_full_active_cell_count, 1u);
        EXPECT_EQ(feature.canonical_cut_cell_count, 1u);
        EXPECT_NEAR(
            feature.canonical_retained_physical_volume, 1.3, 1.0e-14);
    }
    const auto rooted_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(rooted_prolongations.size(), 1u);
    ASSERT_NE(rooted_prolongations.front(), nullptr);
    EXPECT_TRUE(rooted_prolongations.front()->trace_bound_eligible);
    EXPECT_EQ(rooted_prolongations.front()->rows.size(), 2u);
    ASSERT_EQ(rooted_prolongations.front()->patches.size(), 1u);
    EXPECT_EQ(rooted_prolongations.front()->patches.front().kind,
              SmallCutAggregationPatchKind::Rooted);
    const auto rooted_digest =
        rooted_prolongations.front()->canonical_content_digest;
    EXPECT_NE(rooted_digest, 0u);

    system.setCutIntegrationContext(
        std::make_shared<assembly::CutIntegrationContext>());
    EXPECT_TRUE(
        system.finalizedSmallCutAggregationProlongations().empty());
    EXPECT_THROW(system.rebuildConstraintState(), std::runtime_error);
    EXPECT_FALSE(aggregation_view->completedRefreshReport().has_value());
    EXPECT_TRUE(
        system.finalizedSmallCutAggregationProlongations().empty());

    auto rootless_context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    });
    addCellRule(*rootless_context,
                {.cell = 1,
                 .volume_fraction = Real{1.0},
                 .full_cell_equivalent = true},
                geometry::CutIntegrationSide::Positive);
    system.setCutIntegrationContext(std::move(rootless_context));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto& report = aggregation_view->completedRefreshReport();
    ASSERT_TRUE(report.has_value());
    EXPECT_EQ(report->field, pressure);
    EXPECT_EQ(report->local_lineage.successful_publication_ordinal, 2u);
    EXPECT_FALSE(report->geometry_identity.available);
    EXPECT_FALSE(report->canonical_topology_transition.has_value())
        << "a failed refresh deliberately clears the prior comparison chain";
    EXPECT_EQ(report->active_side, geometry::CutIntegrationSide::Negative);
    EXPECT_EQ(report->interface_marker, kInterfaceMarker);
    EXPECT_EQ(report->canonical_candidate_vertices, 4u);
    EXPECT_EQ(report->canonical_rooted_candidate_vertices, 0u);
    EXPECT_EQ(report->canonical_rootless_candidate_vertices, 4u);
    EXPECT_EQ(report->canonical_owned_aggregate_dofs, 0u);
    EXPECT_EQ(report->canonical_owned_pinned_dofs, 4u);
    EXPECT_EQ(report->canonical_strong_suppressed_dofs, 0u);
    EXPECT_EQ(report->canonical_active_feature_count, 1u);
    EXPECT_EQ(report->canonical_rooted_active_feature_count, 0u);
    EXPECT_EQ(report->canonical_rootless_active_feature_count, 1u);
    EXPECT_NEAR(
        report->canonical_rootless_active_physical_volume, 0.3, 1.0e-14);
    ASSERT_EQ(report->canonical_active_features.size(), 1u);
    const auto& rootless_feature =
        report->canonical_active_features.front();
    EXPECT_EQ(rootless_feature.stable_feature_id, 0);
    EXPECT_EQ(
        rootless_feature.disposition,
        SmallCutAggregationActiveFeatureDisposition::Rootless);
    EXPECT_EQ(rootless_feature.canonical_cell_count, 1u);
    EXPECT_EQ(rootless_feature.canonical_full_active_cell_count, 0u);
    EXPECT_EQ(rootless_feature.canonical_cut_cell_count, 1u);
    EXPECT_NEAR(
        rootless_feature.canonical_retained_physical_volume,
        0.3,
        1.0e-14);

    for (const auto vertex : {0, 1, 3, 4}) {
        const auto view = system.constraints().getConstraint(
            vertexDof(system, pressure, vertex));
        ASSERT_TRUE(view.has_value()) << "vertex " << vertex;
        EXPECT_TRUE(view->isDirichlet()) << "vertex " << vertex;
        EXPECT_TRUE(view->entries.empty()) << "vertex " << vertex;
        EXPECT_NEAR(view->inhomogeneity, 0.0, 1.0e-15) << "vertex " << vertex;
    }

    const auto rootless_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(rootless_prolongations.size(), 1u);
    ASSERT_NE(rootless_prolongations.front(), nullptr);
    const auto& rootless = *rootless_prolongations.front();
    EXPECT_TRUE(rootless.trace_bound_eligible);
    EXPECT_NE(rootless.canonical_content_digest, 0u);
    EXPECT_NE(rootless.canonical_content_digest, rooted_digest);
    ASSERT_EQ(rootless.rows.size(), 4u);
    for (const auto vertex : {0, 1, 3, 4}) {
        const auto dof = vertexDof(system, pressure, vertex);
        const auto* row =
            findFinalizedProlongationRow(rootless, dof);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->candidate_dof, dof);
        EXPECT_EQ(row->component, 0u);
        EXPECT_EQ(row->provisional_kind,
                  SmallCutAggregationProvisionalRowKind::
                      RootlessHomogeneousPin);
        EXPECT_EQ(row->final_kind,
                  SmallCutAggregationFinalRowKind::HomogeneousPin);
        EXPECT_EQ(row->root_cell_gid, INVALID_GLOBAL_INDEX);
        EXPECT_TRUE(row->provisional_entries.empty());
        EXPECT_TRUE(row->final_entries.empty());
        EXPECT_NEAR(row->final_inhomogeneity, 0.0, 1.0e-15);
    }
    ASSERT_EQ(rootless.active_cells.size(), 1u);
    EXPECT_EQ(rootless.active_cells.front().kind,
              SmallCutAggregationActiveCellKind::Cut);
    EXPECT_NEAR(rootless.active_cells.front().retained_physical_volume,
                0.3,
                1.0e-14);
    ASSERT_EQ(rootless.patches.size(), 1u);
    EXPECT_EQ(rootless.patches.front().kind,
              SmallCutAggregationPatchKind::Rootless);
    EXPECT_EQ(rootless.patches.front().member_cell_gids.size(), 1u);
    EXPECT_EQ(rootless.patches.front().support_cell_gids.size(), 1u);
    EXPECT_EQ(rootless.patches.front().slave_dofs.size(), 4u);

    {
        ScopedEnvVar cap("SVMP_AGGREGATION_MAX_LINES", "0");
        ASSERT_NO_THROW(system.rebuildConstraintState());
    }
    EXPECT_FALSE(aggregation_view->completedRefreshReport().has_value());
    EXPECT_TRUE(
        system.finalizedSmallCutAggregationProlongations().empty());
#endif
}

TEST(SmallCutAggregationConstraint,
     SameCountFullCutSwapHasClassSensitiveTransitionAndProvenance)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto aggregation = std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker);
    const auto* aggregation_view = aggregation.get();
    system.addSystemConstraint(std::move(aggregation));
    ASSERT_NO_THROW(system.setup());

    system.setCutIntegrationContext(makePublishedCutContext(
        {
            {.cell = 0,
             .volume_fraction = Real{0.3},
             .full_cell_equivalent = false},
            {.cell = 1,
             .volume_fraction = Real{1.0},
             .full_cell_equivalent = true},
        },
        /*source_value_revision=*/101u));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    const auto first_optional = aggregation_view->completedRefreshReport();
    ASSERT_TRUE(first_optional.has_value());
    const auto first = *first_optional;
    ASSERT_EQ(first.canonical_active_features.size(), 1u);
    const auto& first_feature = first.canonical_active_features.front();
    EXPECT_EQ(first.local_lineage.successful_publication_ordinal, 1u);
    EXPECT_EQ(
        first.geometry_identity.kind,
        SmallCutAggregationGeometryIdentityKind::GeneratedPublicationSource);
    EXPECT_TRUE(first.geometry_identity.available);
    EXPECT_TRUE(
        first.geometry_identity.communicator_fingerprint_consensus_validated);
    EXPECT_EQ(first.geometry_identity.source_value_revision, 101u);
    EXPECT_EQ(first_feature.canonical_cell_count, 2u);
    EXPECT_EQ(first_feature.canonical_full_active_cell_count, 1u);
    EXPECT_EQ(first_feature.canonical_cut_cell_count, 1u);
    EXPECT_NE(first_feature.canonical_full_active_cell_gid_digest, 0u);
    EXPECT_NE(first_feature.canonical_cut_cell_gid_digest, 0u);
    EXPECT_NE(first_feature.canonical_full_active_cell_gid_digest,
              first_feature.canonical_cut_cell_gid_digest);

    system.setCutIntegrationContext(makePublishedCutContext(
        {
            {.cell = 0,
             .volume_fraction = Real{1.0},
             .full_cell_equivalent = true},
            {.cell = 1,
             .volume_fraction = Real{0.4},
             .full_cell_equivalent = false},
        },
        /*source_value_revision=*/102u));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    const auto& second_optional =
        aggregation_view->completedRefreshReport();
    ASSERT_TRUE(second_optional.has_value());
    const auto& second = *second_optional;
    ASSERT_EQ(second.canonical_active_features.size(), 1u);
    const auto& second_feature = second.canonical_active_features.front();

    // Membership and both class counts are unchanged. Only the assignment of
    // physical GIDs to full/cut classes swaps, so the old aggregate digest is
    // intentionally equal while both class-sensitive digests change.
    EXPECT_EQ(second_feature.stable_feature_id,
              first_feature.stable_feature_id);
    EXPECT_EQ(second_feature.canonical_cell_gid_digest,
              first_feature.canonical_cell_gid_digest);
    EXPECT_EQ(second_feature.canonical_cell_count,
              first_feature.canonical_cell_count);
    EXPECT_EQ(second_feature.canonical_full_active_cell_count,
              first_feature.canonical_full_active_cell_count);
    EXPECT_EQ(second_feature.canonical_cut_cell_count,
              first_feature.canonical_cut_cell_count);
    EXPECT_NE(second_feature.canonical_full_active_cell_gid_digest,
              first_feature.canonical_full_active_cell_gid_digest);
    EXPECT_NE(second_feature.canonical_cut_cell_gid_digest,
              first_feature.canonical_cut_cell_gid_digest);
    EXPECT_NE(second.canonical_feature_class_fingerprint,
              first.canonical_feature_class_fingerprint);

    ASSERT_TRUE(second.canonical_topology_transition.has_value());
    const auto& transition = *second.canonical_topology_transition;
    EXPECT_TRUE(transition.canonical_topology_changed);
    EXPECT_EQ(transition.canonical_features_entered, 0u);
    EXPECT_EQ(transition.canonical_features_exited, 0u);
    EXPECT_EQ(transition.canonical_features_persisted, 1u);
    EXPECT_EQ(transition.canonical_feature_classification_changes, 1u);
    ASSERT_EQ(transition.canonical_feature_transitions.size(), 1u);
    const auto& feature_transition =
        transition.canonical_feature_transitions.front();
    EXPECT_TRUE(feature_transition.present_before);
    EXPECT_TRUE(feature_transition.present_after);
    EXPECT_TRUE(feature_transition.cell_classification_changed);
    EXPECT_EQ(
        feature_transition.canonical_full_active_cell_gid_digest_before,
        first_feature.canonical_full_active_cell_gid_digest);
    EXPECT_EQ(
        feature_transition.canonical_full_active_cell_gid_digest_after,
        second_feature.canonical_full_active_cell_gid_digest);
    EXPECT_EQ(feature_transition.canonical_cut_cell_gid_digest_before,
              first_feature.canonical_cut_cell_gid_digest);
    EXPECT_EQ(feature_transition.canonical_cut_cell_gid_digest_after,
              second_feature.canonical_cut_cell_gid_digest);

    EXPECT_EQ(
        transition.local_lineage_before.successful_publication_ordinal,
        1u);
    EXPECT_EQ(
        transition.local_lineage_after.successful_publication_ordinal,
        2u);
    EXPECT_EQ(transition.geometry_identity_before.source_value_revision,
              101u);
    EXPECT_EQ(transition.geometry_identity_after.source_value_revision,
              102u);
    EXPECT_EQ(transition.geometry_identity_before.source_id,
              transition.geometry_identity_after.source_id);
    EXPECT_EQ(transition.geometry_identity_before.domain_id,
              transition.geometry_identity_after.domain_id);
    EXPECT_TRUE(
        transition.geometry_identity_before
            .communicator_fingerprint_consensus_validated);
    EXPECT_TRUE(
        transition.geometry_identity_after
            .communicator_fingerprint_consensus_validated);
    EXPECT_EQ(transition.canonical_feature_class_fingerprint_before,
              first.canonical_feature_class_fingerprint);
    EXPECT_EQ(transition.canonical_feature_class_fingerprint_after,
              second.canonical_feature_class_fingerprint);
#endif
}

TEST(SmallCutAggregationConstraint,
     CutContextTransactionRollbackRestoresFinalizedSnapshotAndConstraintState)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    auto aggregation = std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker);
    const auto* aggregation_view = aggregation.get();
    system.addSystemConstraint(std::move(aggregation));

    ASSERT_NO_THROW(system.setup());
    auto rooted_context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    });
    system.setCutIntegrationContext(rooted_context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto prior_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prior_prolongations.size(), 1u);
    ASSERT_NE(prior_prolongations.front(), nullptr);
    const auto prior_prolongation = prior_prolongations.front();
    const auto prior_digest =
        prior_prolongation->canonical_content_digest;
    ASSERT_NE(prior_digest, 0u);
    const auto& prior_refresh_optional =
        aggregation_view->completedRefreshReport();
    ASSERT_TRUE(prior_refresh_optional.has_value());
    const auto prior_refresh = *prior_refresh_optional;
    EXPECT_EQ(
        prior_refresh.local_lineage.successful_publication_ordinal, 1u);
    EXPECT_EQ(prior_refresh.canonical_rooted_candidate_vertices, 2u);
    EXPECT_EQ(prior_refresh.canonical_rootless_candidate_vertices, 0u);

    const auto bottom_slave = vertexDof(system, pressure, 0);
    const auto top_slave = vertexDof(system, pressure, 3);
    const auto prior_bottom_entries =
        lineEntries(system, bottom_slave);
    const auto prior_top_entries =
        lineEntries(system, top_slave);
    const auto prior_constraint_count =
        system.constraints().numConstraints();
    const auto prior_affine_revision =
        system.constraints().constraintLayoutRevision();
    const auto prior_fe_constraint_revision =
        system.constraintLayoutRevision();
    const auto* prior_context = system.cutIntegrationContext();
    ASSERT_NE(prior_context, nullptr);

    ASSERT_NO_THROW(system.beginCutIntegrationContextTransaction());
    EXPECT_TRUE(system.cutIntegrationContextTransactionActive());
    auto rootless_context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    });
    addCellRule(*rootless_context,
                {.cell = 1,
                 .volume_fraction = Real{1.0},
                 .full_cell_equivalent = true},
                geometry::CutIntegrationSide::Positive);
    system.setCutIntegrationContext(std::move(rootless_context));
    EXPECT_TRUE(
        system.finalizedSmallCutAggregationProlongations().empty());
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto trial_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(trial_prolongations.size(), 1u);
    ASSERT_NE(trial_prolongations.front(), nullptr);
    EXPECT_NE(trial_prolongations.front().get(),
              prior_prolongation.get());
    EXPECT_NE(trial_prolongations.front()->canonical_content_digest,
              prior_digest);
    const auto& trial_refresh =
        aggregation_view->completedRefreshReport();
    ASSERT_TRUE(trial_refresh.has_value());
    EXPECT_EQ(
        trial_refresh->local_lineage.successful_publication_ordinal, 2u);
    EXPECT_EQ(trial_refresh->canonical_rooted_candidate_vertices, 0u);
    EXPECT_EQ(trial_refresh->canonical_rootless_candidate_vertices, 4u);
    EXPECT_EQ(system.constraints().numConstraints(), 4u);

    ASSERT_NO_THROW(system.rollbackCutIntegrationContextTransaction());
    EXPECT_FALSE(system.cutIntegrationContextTransactionActive());
    EXPECT_EQ(system.cutIntegrationContext(), prior_context);
    const auto restored_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(restored_prolongations.size(), 1u);
    EXPECT_EQ(restored_prolongations.front().get(),
              prior_prolongation.get());
    EXPECT_EQ(restored_prolongations.front()->canonical_content_digest,
              prior_digest);

    const auto& restored_refresh =
        aggregation_view->completedRefreshReport();
    ASSERT_TRUE(restored_refresh.has_value());
    EXPECT_EQ(
        restored_refresh->local_lineage.successful_publication_ordinal,
        prior_refresh.local_lineage.successful_publication_ordinal);
    EXPECT_EQ(restored_refresh->field, prior_refresh.field);
    EXPECT_EQ(restored_refresh->active_side,
              prior_refresh.active_side);
    EXPECT_EQ(restored_refresh->interface_marker,
              prior_refresh.interface_marker);
    EXPECT_EQ(restored_refresh->canonical_candidate_vertices,
              prior_refresh.canonical_candidate_vertices);
    EXPECT_EQ(restored_refresh->canonical_rooted_candidate_vertices,
              prior_refresh.canonical_rooted_candidate_vertices);
    EXPECT_EQ(restored_refresh->canonical_rootless_candidate_vertices,
              prior_refresh.canonical_rootless_candidate_vertices);
    EXPECT_EQ(restored_refresh->canonical_owned_aggregate_dofs,
              prior_refresh.canonical_owned_aggregate_dofs);
    EXPECT_EQ(restored_refresh->canonical_owned_pinned_dofs,
              prior_refresh.canonical_owned_pinned_dofs);
    EXPECT_EQ(restored_refresh->canonical_active_features.size(),
              prior_refresh.canonical_active_features.size());
    ASSERT_EQ(restored_refresh->canonical_active_features.size(), 1u);
    ASSERT_EQ(prior_refresh.canonical_active_features.size(), 1u);
    EXPECT_EQ(
        restored_refresh->canonical_active_features.front().stable_feature_id,
        prior_refresh.canonical_active_features.front().stable_feature_id);
    EXPECT_EQ(
        restored_refresh->canonical_active_features.front().disposition,
        prior_refresh.canonical_active_features.front().disposition);
    EXPECT_EQ(
        restored_refresh->canonical_active_features.front()
            .canonical_cell_gid_digest,
        prior_refresh.canonical_active_features.front()
            .canonical_cell_gid_digest);

    EXPECT_EQ(system.constraints().numConstraints(),
              prior_constraint_count);
    EXPECT_EQ(system.constraints().constraintLayoutRevision(),
              prior_affine_revision);
    EXPECT_EQ(system.constraintLayoutRevision(),
              prior_fe_constraint_revision);
    expectEntries(lineEntries(system, bottom_slave),
                  prior_bottom_entries);
    expectEntries(lineEntries(system, top_slave),
                  prior_top_entries);
    EXPECT_NEAR(system.constraints().getInhomogeneity(bottom_slave),
                0.0,
                1.0e-15);
    EXPECT_NEAR(system.constraints().getInhomogeneity(top_slave),
                0.0,
                1.0e-15);
#endif
}

TEST(SmallCutAggregationConstraint,
     SamePointerCutContextMutationInvalidatesFinalizedSnapshot)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    auto context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    });
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto published =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(published.size(), 1u);
    ASSERT_NE(published.front(), nullptr);
    const auto published_revision = context->contentRevision();
    EXPECT_EQ(
        published.front()->revision.cut_context_content_revision,
        published_revision);
    const auto* installed_context = system.cutIntegrationContext();
    ASSERT_EQ(installed_context, context.get());

    addCellRule(*context,
                {.cell = 0,
                 .volume_fraction = Real{0.7},
                 .full_cell_equivalent = false},
                geometry::CutIntegrationSide::Positive);
    ASSERT_GT(context->contentRevision(), published_revision);
    ASSERT_EQ(system.cutIntegrationContext(), installed_context);

    system.setCutIntegrationContext(context);
    EXPECT_EQ(system.cutIntegrationContext(), installed_context);
    EXPECT_TRUE(
        system.finalizedSmallCutAggregationProlongations().empty());
    EXPECT_EQ(
        published.front()->revision.cut_context_content_revision,
        published_revision);
#endif
}

TEST(SmallCutAggregationConstraint,
     CutContextTransactionRollbackDetachesMutatedOriginalContext)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    auto context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    });
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto prior_content_revision = context->contentRevision();
    const auto prior_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prior_prolongations.size(), 1u);
    ASSERT_NE(prior_prolongations.front(), nullptr);
    const auto prior_prolongation = prior_prolongations.front();
    const auto prior_bottom_slave = vertexDof(system, pressure, 0);
    const auto prior_top_slave = vertexDof(system, pressure, 3);
    const auto prior_bottom_entries =
        lineEntries(system, prior_bottom_slave);
    const auto prior_top_entries =
        lineEntries(system, prior_top_slave);
    const auto prior_constraint_count =
        system.constraints().numConstraints();

    ASSERT_NO_THROW(system.beginCutIntegrationContextTransaction());
    context->clear();
    addCellRule(
        *context,
        {.cell = 0,
         .volume_fraction = Real{0.3},
         .full_cell_equivalent = false},
        geometry::CutIntegrationSide::Negative);
    addCellRule(
        *context,
        {.cell = 1,
         .volume_fraction = Real{1.0},
         .full_cell_equivalent = true},
        geometry::CutIntegrationSide::Positive);
    ASSERT_GT(context->contentRevision(), prior_content_revision);
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());
    ASSERT_EQ(system.cutIntegrationContext(), context.get());
    ASSERT_EQ(
        system.finalizedSmallCutAggregationProlongations().size(),
        1u);
    EXPECT_NE(
        system.finalizedSmallCutAggregationProlongations()
            .front()
            .get(),
        prior_prolongation.get());

    ASSERT_NO_THROW(system.rollbackCutIntegrationContextTransaction());
    ASSERT_NE(system.cutIntegrationContext(), nullptr);
    EXPECT_NE(system.cutIntegrationContext(), context.get());
    EXPECT_EQ(system.cutIntegrationContext()->contentRevision(),
              prior_content_revision);
    EXPECT_EQ(system.constraints().numConstraints(),
              prior_constraint_count);
    expectEntries(lineEntries(system, prior_bottom_slave),
                  prior_bottom_entries);
    expectEntries(lineEntries(system, prior_top_slave),
                  prior_top_entries);

    const auto restored_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(restored_prolongations.size(), 1u);
    EXPECT_EQ(restored_prolongations.front().get(),
              prior_prolongation.get());
    EXPECT_EQ(
        restored_prolongations.front()
            ->revision.cut_context_content_revision,
        prior_content_revision);
#endif
}

TEST(SmallCutAggregationConstraint,
     GeometryValueRefreshRepublishesCurrentFinalizedSnapshot)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    system.addSystemConstraint(
        std::make_unique<GeometryDependentVertexPin>(
            pressure,
            /*vertex=*/2,
            Real{0.25}));

    ASSERT_NO_THROW(system.setup());
    auto context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    });
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());
    const auto initial_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(initial_prolongations.size(), 1u);
    ASSERT_NE(initial_prolongations.front(), nullptr);
    const auto initial_prolongation =
        initial_prolongations.front();

    const auto pinned_dof = vertexDof(system, pressure, 2);
    ASSERT_TRUE(system.constraints().isConstrained(pinned_dof));
    EXPECT_NEAR(system.constraints().getInhomogeneity(pinned_dof),
                0.25,
                1.0e-15);
    const auto prior_geometry_revision =
        mesh->local_mesh().geometry_revision();
    auto moved_vertex =
        mesh->local_mesh().get_vertex_coords(/*vertex=*/2);
    moved_vertex[0] += Real{0.125};
    mesh->local_mesh().set_vertex_coords(
        /*vertex=*/2,
        moved_vertex);
    ASSERT_GT(mesh->local_mesh().geometry_revision(),
              prior_geometry_revision);

    const auto refresh =
        system.refreshConstraintStateForCurrentRevisions(
            /*time=*/3.0,
            /*dt=*/0.25,
            /*allow_structural_rebuild=*/true);
    EXPECT_TRUE(refresh.dependency_changed);
    EXPECT_FALSE(refresh.structural_rebuild);
    EXPECT_TRUE(refresh.value_update);
    EXPECT_NEAR(system.constraints().getInhomogeneity(pinned_dof),
                3.25,
                1.0e-15);
    const auto refreshed_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(refreshed_prolongations.size(), 1u);
    ASSERT_NE(refreshed_prolongations.front(), nullptr);
    EXPECT_NE(refreshed_prolongations.front().get(),
              initial_prolongation.get());
    EXPECT_GT(
        refreshed_prolongations.front()
            ->revision.constraint.time_epoch,
        initial_prolongation->revision.constraint.time_epoch);
    EXPECT_EQ(
        refreshed_prolongations.front()
            ->revision.constraint.geometry,
        mesh->local_mesh().geometry_revision());
    const auto unchanged_refresh =
        system.refreshConstraintStateForCurrentRevisions(
            /*time=*/4.0,
            /*dt=*/0.5,
            /*allow_structural_rebuild=*/true);
    EXPECT_FALSE(unchanged_refresh.dependency_changed);
    EXPECT_FALSE(unchanged_refresh.structural_rebuild);
    EXPECT_FALSE(unchanged_refresh.value_update);
    const auto unchanged_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(unchanged_prolongations.size(), 1u);
    EXPECT_EQ(unchanged_prolongations.front().get(),
              refreshed_prolongations.front().get());
#endif
}

TEST(SmallCutAggregationConstraint, AllowUnaggregatedEnvRestoresFailOpenBehavior)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    ScopedEnvVar allow("SVMP_AGGREGATION_ALLOW_UNAGGREGATED", "1");
    auto mesh = buildQuadStrip(2);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    auto context = makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    });
    addCellRule(*context,
                {.cell = 1,
                 .volume_fraction = Real{1.0},
                 .full_cell_equivalent = true},
                geometry::CutIntegrationSide::Positive);
    system.setCutIntegrationContext(std::move(context));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_NE(log_output.find("continuing fail-open"), std::string::npos);
    for (const auto vertex : {0, 1, 3, 4}) {
        EXPECT_FALSE(system.constraints().isConstrained(
            vertexDof(system, pressure, vertex)))
            << "vertex " << vertex;
    }
#endif
}

TEST(SmallCutAggregationConstraint, MaxLinesDebugCapSkipsFailClosedGate)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // The bisection cap intentionally leaves candidates unconstrained; the
    // fail-closed gate must not fire while it is engaged.
    ScopedEnvVar cap("SVMP_AGGREGATION_MAX_LINES", "0");
    auto mesh = buildQuadStrip(2);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 0)));
#endif
}

TEST(SmallCutAggregationConstraint, WallMarkerVerticesAreNeverSlaved)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    constexpr int wall_marker = 11;
    auto mesh = buildQuadStrip(2, wall_marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{wall_marker}));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    // Both would-be candidates (v0, v3) sit on the wall: excluded before
    // candidacy, left to the strong BC, and NOT a fail-closed violation.
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_NE(log_output.find("candidate_vertices=0"), std::string::npos);
    EXPECT_NE(log_output.find("aggregated_vertices=0"), std::string::npos);
#endif
}

TEST(SmallCutAggregationConstraint, WallExclusionCoversQ2MidsideWallNodes)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    constexpr int wall_marker = 11;
    auto mesh = buildTwoQuad9Strip(wall_marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad9, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{wall_marker}));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    validateQuad9NodalPairing(system, pressure);

    // Wall = left edge x=0: corners v0/v3 AND the Q2 midside wall node 9
    // (0,0.5) are never slaves (the reference-coordinate discriminator must
    // catch the midside node the corner-only face list misses).
    const auto& constraints = system.constraints();
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(constraints.isConstrained(cellNodeDof(system, pressure, 0, 7)));

    // The interior unsupported nodes of the cut cell are still slaved:
    // bottom midside (slot 4), top midside (slot 6), center (slot 8).
    EXPECT_TRUE(constraints.isConstrained(cellNodeDof(system, pressure, 0, 4)));
    EXPECT_TRUE(constraints.isConstrained(cellNodeDof(system, pressure, 0, 6)));
    EXPECT_TRUE(constraints.isConstrained(cellNodeDof(system, pressure, 0, 8)));
    EXPECT_EQ(constraints.numConstraints(), 3u);
#endif
}

TEST(SmallCutAggregationConstraint,
     Wedge18ObliqueWallExclusionCoversEveryQuadraticFaceNode)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Wedge face 3 is the oblique x+y=1 quadrilateral. Its four corners,
    // four edge nodes, and Q2 face-center node must stay available to the
    // strong wall BC; every other node of this rootless cut cell is pinned.
    expectRootlessQuadraticWallFaceExclusion(
        ElementType::Wedge18,
        /*wall_face=*/3,
        std::vector<std::size_t>{1, 2, 4, 5, 7, 10, 13, 14, 16});
#endif
}

TEST(SmallCutAggregationConstraint,
     Pyramid14SlopingWallExclusionCoversEveryQuadraticFaceNode)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Pyramid face 1 is a sloping triangular side. The generic affine-hull
    // classifier must retain its three corners and three edge nodes.
    expectRootlessQuadraticWallFaceExclusion(
        ElementType::Pyramid14,
        /*wall_face=*/1,
        std::vector<std::size_t>{0, 1, 4, 5, 9, 10});
#endif
}

TEST(SmallCutAggregationConstraint, GaugeExcludedPressureVertexKeepsItsPin)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // Production wiring: the gauge vertex is passed as excluded_vertices to
    // aggregation AND pinned by a VertexDirichletConstraint registered after
    // it. The pin must win and keep removing the pressure null mode.
    auto mesh = buildQuadStrip(2);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker,
        std::vector<int>{},
        std::vector<GlobalIndex>{0}));
    system.addSystemConstraint(std::make_unique<VertexDirichletConstraint>(
        pressure,
        std::vector<VertexDirichletValue>{{.vertex_id = 0, .value = Real{5.0}}},
        VertexIdMode::LocalVertexId));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto gauge_dof = vertexDof(system, pressure, 0);
    const auto view = system.constraints().getConstraint(gauge_dof);
    ASSERT_TRUE(view.has_value());
    EXPECT_TRUE(view->isDirichlet());
    EXPECT_NEAR(view->inhomogeneity, 5.0, 1.0e-12);

    // The non-gauge candidate still aggregates normally.
    EXPECT_TRUE(system.constraints().isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(lineEntries(system, vertexDof(system, pressure, 3)).empty());
#endif
}

TEST(SmallCutAggregationConstraint, DirichletInstalledAfterAggregationReplacesMasterBearingLine)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // No exclusion list here: aggregation slaves the vertex first, then the
    // strong pin applies on top. AffineConstraints::addDirichlet must
    // REPLACE the master-bearing line (global strong-BC precedence).
    auto mesh = buildQuadStrip(2);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));
    system.addSystemConstraint(std::make_unique<VertexDirichletConstraint>(
        pressure,
        std::vector<VertexDirichletValue>{{.vertex_id = 0, .value = Real{2.5}}},
        VertexIdMode::LocalVertexId));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto pinned_dof = vertexDof(system, pressure, 0);
    const auto view = system.constraints().getConstraint(pinned_dof);
    ASSERT_TRUE(view.has_value());
    EXPECT_TRUE(view->isDirichlet()) << "master-bearing line was not replaced";
    EXPECT_NEAR(view->inhomogeneity, 2.5, 1.0e-12);

    // The other candidate keeps its master-bearing aggregation line.
    expectEntries(lineEntries(system, vertexDof(system, pressure, 3)),
                  {{vertexDof(system, pressure, 4), 2.0},
                   {vertexDof(system, pressure, 5), -1.0}});

    const auto prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(prolongations.size(), 1u);
    ASSERT_NE(prolongations.front(), nullptr);
    const auto& prolongation = *prolongations.front();
    EXPECT_TRUE(prolongation.trace_bound_eligible);
    EXPECT_NE(prolongation.canonical_content_digest, 0u);
    const auto* pinned_row =
        findFinalizedProlongationRow(prolongation, pinned_dof);
    ASSERT_NE(pinned_row, nullptr);
    EXPECT_EQ(pinned_row->provisional_kind,
              SmallCutAggregationProvisionalRowKind::RootedExtension);
    EXPECT_EQ(pinned_row->final_kind,
              SmallCutAggregationFinalRowKind::FixedValue);
    EXPECT_FALSE(pinned_row->preconstrained_at_apply);
    EXPECT_EQ(pinned_row->provisional_entries.size(), 2u);
    EXPECT_TRUE(pinned_row->final_entries.empty());
    EXPECT_NEAR(pinned_row->final_inhomogeneity, 2.5, 1.0e-12);

    const auto other_slave = vertexDof(system, pressure, 3);
    const auto* aggregate_row =
        findFinalizedProlongationRow(prolongation, other_slave);
    ASSERT_NE(aggregate_row, nullptr);
    EXPECT_EQ(aggregate_row->provisional_kind,
              SmallCutAggregationProvisionalRowKind::RootedExtension);
    EXPECT_EQ(aggregate_row->final_kind,
              SmallCutAggregationFinalRowKind::MasterBearing);
    EXPECT_FALSE(aggregate_row->preconstrained_at_apply);
    expectEntries(
        prolongationEntries(aggregate_row->final_entries),
        {{vertexDof(system, pressure, 4), 2.0},
         {vertexDof(system, pressure, 5), -1.0}});
    EXPECT_NEAR(aggregate_row->final_inhomogeneity, 0.0, 1.0e-15);
#endif
}

TEST(SmallCutAggregationConstraint,
     ValueOnlyConstraintUpdateRepublishesCurrentFinalizedSnapshot)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildQuadStrip(2);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<SmallCutAggregationConstraint>(
            pressure,
            geometry::CutIntegrationSide::Negative,
            kInterfaceMarker));
    constexpr Real initial_pin = Real{1.25};
    system.addSystemConstraint(
        std::make_unique<TimeDependentVertexPin>(
            pressure, 0, initial_pin));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto pinned_dof = vertexDof(system, pressure, 0);
    EXPECT_NEAR(system.constraints().getInhomogeneity(pinned_dof),
                initial_pin,
                1.0e-15);
    const auto initial_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(initial_prolongations.size(), 1u);
    ASSERT_NE(initial_prolongations.front(), nullptr);
    const auto initial_prolongation =
        initial_prolongations.front();
    const auto initial_digest =
        initial_prolongation->canonical_content_digest;
    ASSERT_NE(initial_digest, 0u);
    const auto* initial_row =
        findFinalizedProlongationRow(
            *initial_prolongation, pinned_dof);
    ASSERT_NE(initial_row, nullptr);
    EXPECT_EQ(initial_row->final_kind,
              SmallCutAggregationFinalRowKind::FixedValue);
    EXPECT_NEAR(initial_row->final_inhomogeneity,
                initial_pin,
                1.0e-15);

    constexpr double updated_time = 2.0;
    constexpr double updated_dt = 0.5;
    constexpr Real updated_pin =
        static_cast<Real>(updated_time + updated_dt);
    ASSERT_NO_THROW(
        system.updateConstraints(updated_time, updated_dt));
    EXPECT_NEAR(system.constraints().getInhomogeneity(pinned_dof),
                updated_pin,
                1.0e-15);
    const auto updated_prolongations =
        system.finalizedSmallCutAggregationProlongations();
    ASSERT_EQ(updated_prolongations.size(), 1u);
    ASSERT_NE(updated_prolongations.front(), nullptr);
    EXPECT_NE(updated_prolongations.front().get(),
              initial_prolongation.get());
    EXPECT_NE(updated_prolongations.front()->canonical_content_digest,
              initial_digest);
    EXPECT_GT(
        updated_prolongations.front()
            ->revision.constraint.time_epoch,
        initial_prolongation->revision.constraint.time_epoch);
    const auto* updated_row =
        findFinalizedProlongationRow(
            *updated_prolongations.front(), pinned_dof);
    ASSERT_NE(updated_row, nullptr);
    EXPECT_EQ(updated_row->final_kind,
              SmallCutAggregationFinalRowKind::FixedValue);
    EXPECT_NEAR(updated_row->final_inhomogeneity,
                updated_pin,
                1.0e-15);
    EXPECT_NEAR(initial_row->final_inhomogeneity,
                initial_pin,
                1.0e-15);
    EXPECT_NEAR(system.constraints().getInhomogeneity(pinned_dof),
                updated_pin,
                1.0e-15);
#endif
}

TEST(SmallCutAggregationConstraint, Q2FullOrderExtensionEmitsMidsideSlavesAndMasters)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto mesh = buildTwoQuad9Strip();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad9, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    validateQuad9NodalPairing(system, pressure);
    const auto& constraints = system.constraints();

    // c0 mesh-node slots: 0..3 corners (v0,v1,v4,v3), 4 bottom mid, 5 shared
    // right mid, 6 top mid, 7 left mid, 8 center. c1 slots: 0..3 corners
    // (v1,v2,v5,v4), 4 bottom mid, 5 right mid, 6 top mid, 7 shared mid, 8
    // center. Candidates = every c0 node without full-active support:
    // v0, v3, bottom/top/left midsides, center — six slaves.
    const auto v0 = vertexDof(system, pressure, 0);
    const auto v1 = vertexDof(system, pressure, 1);
    const auto v2 = vertexDof(system, pressure, 2);
    const auto v3 = vertexDof(system, pressure, 3);
    const auto v4 = vertexDof(system, pressure, 4);
    const auto v5 = vertexDof(system, pressure, 5);
    const auto c0_bottom_mid = cellNodeDof(system, pressure, 0, 4);
    const auto c0_top_mid = cellNodeDof(system, pressure, 0, 6);
    const auto c0_left_mid = cellNodeDof(system, pressure, 0, 7);
    const auto c0_center = cellNodeDof(system, pressure, 0, 8);
    const auto shared_mid = cellNodeDof(system, pressure, 0, 5);
    const auto root_bottom_mid = cellNodeDof(system, pressure, 1, 4);
    const auto root_right_mid = cellNodeDof(system, pressure, 1, 5);
    const auto root_top_mid = cellNodeDof(system, pressure, 1, 6);
    const auto root_center = cellNodeDof(system, pressure, 1, 8);

    EXPECT_EQ(constraints.numConstraints(), 6u);
    EXPECT_TRUE(constraints.isConstrained(v0));
    EXPECT_TRUE(constraints.isConstrained(v3));
    EXPECT_TRUE(constraints.isConstrained(c0_bottom_mid));
    EXPECT_TRUE(constraints.isConstrained(c0_top_mid));
    EXPECT_TRUE(constraints.isConstrained(c0_left_mid));
    EXPECT_TRUE(constraints.isConstrained(c0_center));
    EXPECT_FALSE(constraints.isConstrained(v1));
    EXPECT_FALSE(constraints.isConstrained(v4));
    EXPECT_FALSE(constraints.isConstrained(shared_mid));
    EXPECT_FALSE(constraints.isConstrained(root_bottom_mid));

    // Full Q2 extension of root [1,2]x[0,1]. 1D quadratic Lagrange values on
    // x-nodes {1, 1.5, 2}: at x=0.5 -> {3, -3, 1}; at x=0 -> {6, -8, 3}.
    // y-rows select the bottom (y=0), middle (y=0.5), or top (y=1) master
    // row. MIDSIDE masters must appear with these weights.
    expectEntries(lineEntries(system, c0_bottom_mid),
                  {{v1, 3.0}, {root_bottom_mid, -3.0}, {v2, 1.0}});
    expectEntries(lineEntries(system, v0),
                  {{v1, 6.0}, {root_bottom_mid, -8.0}, {v2, 3.0}});
    expectEntries(lineEntries(system, c0_top_mid),
                  {{v4, 3.0}, {root_top_mid, -3.0}, {v5, 1.0}});
    expectEntries(lineEntries(system, v3),
                  {{v4, 6.0}, {root_top_mid, -8.0}, {v5, 3.0}});
    expectEntries(lineEntries(system, c0_left_mid),
                  {{shared_mid, 6.0}, {root_center, -8.0}, {root_right_mid, 3.0}});
    expectEntries(lineEntries(system, c0_center),
                  {{shared_mid, 3.0}, {root_center, -3.0}, {root_right_mid, 1.0}});
#endif
}

TEST(SmallCutAggregationConstraint, LinearExtensionKnobRestrictsMastersToCornerSubBasis)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    ScopedEnvVar linear("SVMP_AGGREGATION_LINEAR_EXTENSION", "1");
    auto mesh = buildTwoQuad9Strip();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad9, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));
    ASSERT_NO_THROW(system.rebuildConstraintState());
    const auto& constraints = system.constraints();

    // The slave SET is unchanged (six candidates), only the extension basis
    // shrinks to the root's linear corner sub-basis: corner masters only,
    // bilinear extrapolation weights, NO midside master entries anywhere.
    EXPECT_EQ(constraints.numConstraints(), 6u);

    const auto v1 = vertexDof(system, pressure, 1);
    const auto v2 = vertexDof(system, pressure, 2);
    const auto v4 = vertexDof(system, pressure, 4);
    const auto v5 = vertexDof(system, pressure, 5);
    const auto c0_bottom_mid = cellNodeDof(system, pressure, 0, 4);
    const auto c0_left_mid = cellNodeDof(system, pressure, 0, 7);

    expectEntries(lineEntries(system, c0_bottom_mid),
                  {{v1, 1.5}, {v2, -0.5}});
    expectEntries(lineEntries(system, vertexDof(system, pressure, 0)),
                  {{v1, 2.0}, {v2, -1.0}});
    // Mid-row candidate (0,0.5) engages all four corners.
    expectEntries(lineEntries(system, c0_left_mid),
                  {{v1, 1.0}, {v2, -0.5}, {v4, 1.0}, {v5, -0.5}});
#endif
}

TEST(SmallCutAggregationConstraint, SubParametricFieldIsRejected)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // A Q2 field on a 4-node quad mesh stores its edge/interior dofs on
    // entities that are NOT mesh nodes: candidate discovery could never
    // slave them, and corner-only extensions would not be a partition of
    // unity. The constraint must reject the configuration loudly.
    auto mesh = buildQuadStrip(2);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());
    system.setCutIntegrationContext(makeCutContext({
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));

    try {
        system.rebuildConstraintState();
        FAIL() << "expected sub-parametric rejection";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("sub-parametric"), std::string::npos);
        EXPECT_NE(message.find("mesh nodes"), std::string::npos);
    }
#endif
}

TEST(SmallCutAggregationConstraint, PrunedSliverFallsToInactivePinPolicy)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // 3-cell strip: c0 carries a below-threshold active sliver (pruned from
    // the generated rules BEFORE aggregation classifies cells), c1 is a real
    // cut cell, c2 is full-active. The intended production policy: the
    // sliver's unsupported dofs fall to the level-set inactive-pin path (no
    // retained active support), real cut candidates still aggregate, and the
    // aggregation diagnostics surface the pruned count for auditability.
    auto mesh = buildQuadStrip(3);
    {
        const auto phi_handle = MeshFields::attach_field(
            mesh->local_mesh(),
            EntityKind::Vertex,
            "phi",
            FieldScalarType::Float64,
            1);
        auto* phi = MeshFields::field_data_as<real_t>(mesh->local_mesh(), phi_handle);
        for (int v = 0; v < 8; ++v) {
            phi[v] = -1.0;  // everything wet: pins must come from missing
                            // retained support, not from the phi sign
        }
    }
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0},
            kInterfaceMarker));
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure,
        geometry::CutIntegrationSide::Negative,
        kInterfaceMarker));

    ASSERT_NO_THROW(system.setup());

    auto context = makeCutContext({
        {.cell = 0,
         .volume_fraction =
             assembly::CutIntegrationContext::minGeneratedCutVolumeFraction() *
             Real{0.5},
         .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{0.4}, .full_cell_equivalent = false},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    });
    EXPECT_EQ(context->generatedPrunedVolumeRuleCount(), 1u);
    // Faithful complement of the pruned negative sliver: the positive side
    // occupies almost the whole parent but is still a cut (non-full) rule.
    // It certifies complete two-sided classification only; it must not make
    // c0 traversable by active-domain aggregation.
    addCellRule(
        *context,
        {.cell = 0,
         .volume_fraction =
             Real{1.0} -
             assembly::CutIntegrationContext::minGeneratedCutVolumeFraction() *
                 Real{0.5},
         .full_cell_equivalent = false},
        geometry::CutIntegrationSide::Positive);
    system.setCutIntegrationContext(std::move(context));

    const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    // Aggregation saw only c1 (cut) + c2 (full): two candidates (v1, v5),
    // both aggregated; the pruned sliver is reported.
    EXPECT_NE(log_output.find("candidate_vertices=2"), std::string::npos);
    EXPECT_NE(log_output.find("aggregated_vertices=2"), std::string::npos);
    EXPECT_NE(log_output.find("pruned_volume_rules=1"), std::string::npos);

    // Sliver-only vertices (v0, v4): wet but WITHOUT retained active
    // support -> pinned inactive (empty-entry Dirichlet lines), not free.
    for (const auto vertex : {0, 4}) {
        const auto view = system.constraints().getConstraint(
            vertexDof(system, pressure, vertex));
        ASSERT_TRUE(view.has_value()) << "vertex " << vertex;
        EXPECT_TRUE(view->isDirichlet()) << "vertex " << vertex;
    }

    // Real cut candidates (v1, v5) carry master-bearing aggregation lines
    // rooted at c2 = [2,3]x[0,1].
    expectEntries(lineEntries(system, vertexDof(system, pressure, 1)),
                  {{vertexDof(system, pressure, 2), 2.0},
                   {vertexDof(system, pressure, 3), -1.0}});
    expectEntries(lineEntries(system, vertexDof(system, pressure, 5)),
                  {{vertexDof(system, pressure, 6), 2.0},
                   {vertexDof(system, pressure, 7), -1.0}});

    // Vertices with retained support stay free.
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 6)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 7)));
#endif
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace {

// Bitwise summary of everything a refresh publishes: closed constraint
// lines, finalized prolongation reports (whose canonical digest covers rows,
// cells with retained volumes and patches) and the refresh reports without
// their rank-local lineage.
struct AggregationPublicationSummary {
    std::vector<std::uint64_t> line_words{};
    std::vector<std::uint64_t> prolongation_words{};
    std::vector<std::uint64_t> report_words{};

    [[nodiscard]] bool operator==(
        const AggregationPublicationSummary&) const = default;
};

[[nodiscard]] std::uint64_t realBits(Real value)
{
    return std::bit_cast<std::uint64_t>(static_cast<double>(value));
}

[[nodiscard]] AggregationPublicationSummary summarizeAggregationPublication(
    const systems::FESystem& system)
{
    AggregationPublicationSummary summary;
    std::vector<std::vector<std::uint64_t>> lines;
    system.constraints().forEach(
        [&](const AffineConstraints::ConstraintView& line) {
            std::vector<std::uint64_t> words{
                static_cast<std::uint64_t>(line.slave_dof),
                std::bit_cast<std::uint64_t>(line.inhomogeneity)};
            for (const auto& entry : line.entries) {
                words.push_back(static_cast<std::uint64_t>(entry.master_dof));
                words.push_back(std::bit_cast<std::uint64_t>(entry.weight));
            }
            lines.push_back(std::move(words));
        });
    std::sort(lines.begin(), lines.end());
    for (const auto& words : lines) {
        summary.line_words.push_back(words.size());
        summary.line_words.insert(
            summary.line_words.end(), words.begin(), words.end());
    }
    for (const auto& prolongation :
         system.finalizedSmallCutAggregationProlongations()) {
        EXPECT_NE(prolongation, nullptr);
        if (!prolongation) {
            continue;
        }
        summary.prolongation_words.push_back(
            static_cast<std::uint64_t>(prolongation->field));
        summary.prolongation_words.push_back(
            prolongation->canonical_content_digest);
        summary.prolongation_words.push_back(
            prolongation->trace_bound_eligible ? 1u : 0u);
        for (const auto& cell : prolongation->active_cells) {
            summary.prolongation_words.push_back(
                static_cast<std::uint64_t>(cell.cell_gid));
            summary.prolongation_words.push_back(
                realBits(cell.retained_physical_volume));
        }
    }
    for (const auto& report :
         system.completedSmallCutAggregationRefreshReports()) {
        auto& words = summary.report_words;
        words.push_back(static_cast<std::uint64_t>(report.field));
        words.push_back(report.canonical_feature_class_fingerprint);
        words.push_back(report.canonical_slave_set_fingerprint);
        words.push_back(report.maximum_observed_root_path);
        words.push_back(realBits(
            report.maximum_observed_reference_extrapolation));
        words.push_back(realBits(report.maximum_observed_absolute_coefficient));
        words.push_back(realBits(report.maximum_observed_row_l1_norm));
        words.push_back(report.canonical_candidate_vertices);
        words.push_back(report.canonical_rooted_candidate_vertices);
        words.push_back(report.canonical_rootless_candidate_vertices);
        words.push_back(report.canonical_owned_aggregate_dofs);
        words.push_back(report.canonical_owned_pinned_dofs);
        words.push_back(report.canonical_strong_suppressed_dofs);
        words.push_back(report.canonical_active_feature_count);
        words.push_back(realBits(
            report.canonical_rootless_active_physical_volume));
        words.push_back(report.local_lineage.successful_publication_ordinal);
        for (const auto& feature : report.canonical_active_features) {
            words.push_back(
                static_cast<std::uint64_t>(feature.stable_feature_id));
            words.push_back(feature.canonical_cell_gid_digest);
            words.push_back(feature.canonical_cell_count);
            words.push_back(static_cast<std::uint64_t>(feature.disposition));
            words.push_back(
                realBits(feature.canonical_retained_physical_volume));
        }
        words.push_back(report.canonical_topology_transition.has_value());
        if (report.canonical_topology_transition.has_value()) {
            const auto& transition = *report.canonical_topology_transition;
            words.push_back(transition.canonical_topology_changed);
            words.push_back(transition.canonical_aggregate_slaves_entered);
            words.push_back(transition.canonical_aggregate_slaves_left);
            words.push_back(transition.canonical_features_entered);
            words.push_back(transition.canonical_features_exited);
            words.push_back(realBits(
                transition.canonical_rootless_active_physical_volume_delta));
        }
    }
    return summary;
}

struct AggregationReuseRun {
    std::vector<AggregationPublicationSummary> summaries{};
    std::vector<std::string> logs{};
};

// Velocity (two components) and pressure on a four-cell strip, both with
// small-cut aggregation, refreshed once per context in `contexts`.
[[nodiscard]] AggregationReuseRun runAggregationReuseSequence(
    const std::vector<std::vector<CellRuleSpec>>& contexts,
    bool disable_reuse)
{
    std::optional<ScopedEnvVar> disable;
    if (disable_reuse) {
        disable.emplace("SVMP_DISABLE_SMALL_CUT_AGGREGATION_REUSE", "1");
    }
    auto mesh = buildQuadStrip(4);
    auto scalar_space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space =
        std::make_shared<spaces::ProductSpace>(scalar_space, /*components=*/2);
    systems::FESystem system(mesh);
    const auto velocity = system.addField(
        systems::FieldSpec{.name = "u", .space = vector_space, .components = 2});
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = scalar_space, .components = 1});
    system.addOperator("equations");
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        velocity, geometry::CutIntegrationSide::Negative, kInterfaceMarker));
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));
    system.setup();

    AggregationReuseRun run;
    for (const auto& specs : contexts) {
        system.setCutIntegrationContext(makeCutContext(specs));
        const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        system.rebuildConstraintState();
        auto log = testing::internal::GetCapturedStdout();
        log += testing::internal::GetCapturedStderr();
        run.logs.push_back(std::move(log));
        run.summaries.push_back(summarizeAggregationPublication(system));
    }
    return run;
}

[[nodiscard]] std::size_t countOccurrences(const std::string& text,
                                           const std::string& needle)
{
    std::size_t count = 0u;
    for (auto position = text.find(needle); position != std::string::npos;
         position = text.find(needle, position + needle.size())) {
        ++count;
    }
    return count;
}

} // namespace
#endif

TEST(SmallCutAggregationConstraint,
     ReusedRefreshPublishesBitwiseTheSameAsAFullRecomputation)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const std::vector<CellRuleSpec> cut_left{
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 3, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    };
    // Same cut topology, different retained volume of the cut cell.
    auto moved_left = cut_left;
    moved_left[0].volume_fraction = Real{0.45};
    // Same cut topology and volumes, new retained-rule identities (generated
    // rules embed the source value revision in their identity).
    auto relabeled_left = moved_left;
    for (auto& spec : relabeled_left) {
        spec.cut_topology_revision =
            std::uint64_t{1000} + static_cast<std::uint64_t>(spec.cell);
    }
    // Different cut topology: the cut moves to the right end.
    const std::vector<CellRuleSpec> cut_right{
        {.cell = 0, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 3, .volume_fraction = Real{0.2}, .full_cell_equivalent = false},
    };
    const std::vector<std::vector<CellRuleSpec>> sequence{
        cut_left, cut_left, moved_left, relabeled_left,
        cut_right, cut_right, cut_left};

    const auto reused = runAggregationReuseSequence(sequence, false);
    const auto recomputed = runAggregationReuseSequence(sequence, true);
    ASSERT_EQ(reused.summaries.size(), sequence.size());
    ASSERT_EQ(recomputed.summaries.size(), sequence.size());
    for (std::size_t step = 0; step < sequence.size(); ++step) {
        EXPECT_FALSE(reused.summaries[step].line_words.empty());
        EXPECT_EQ(reused.summaries[step], recomputed.summaries[step])
            << "step " << step;
        EXPECT_EQ(countOccurrences(recomputed.logs[step],
                                   "decision=rebuilt reason=not_admissible"),
                  2u)
            << recomputed.logs[step];
        // The refresh diagnostic and churn lines are emitted on reuse too.
        EXPECT_EQ(countOccurrences(reused.logs[step],
                                   "diagnostic=small_cut_aggregation "),
                  2u);
        EXPECT_EQ(countOccurrences(reused.logs[step],
                                   "diagnostic=small_cut_aggregation_churn"),
                  2u);
    }

    const auto expect_decisions = [&](std::size_t step,
                                      const std::string& velocity,
                                      const std::string& pressure) {
        const auto& log = reused.logs[step];
        EXPECT_NE(log.find("field='u' decision=" + velocity),
                  std::string::npos)
            << "step " << step << "\n" << log;
        EXPECT_NE(log.find("field='p' decision=" + pressure),
                  std::string::npos)
            << "step " << step << "\n" << log;
    };
    expect_decisions(0, "rebuilt reason=no_previous_refresh",
                     "rebuilt reason=no_previous_refresh");
    // Unchanged input: the pressure constraint shares the retained measures
    // that the velocity constraint refreshed.
    expect_decisions(1, "reused", "reused retained_measures=shared");
    // Volume-only change: still reused, with the new retained volumes.
    expect_decisions(2, "reused retained_measures=recomputed",
                     "reused retained_measures=shared");
    // Identity-only change: reused, with the new identities.
    expect_decisions(3, "reused retained_measures=recomputed",
                     "reused retained_measures=shared");
    // Topology change: both rebuild.
    expect_decisions(4, "rebuilt reason=cut_topology_changed",
                     "rebuilt reason=cut_topology_changed");
    expect_decisions(5, "reused", "reused retained_measures=shared");
    expect_decisions(6, "rebuilt reason=cut_topology_changed",
                     "rebuilt reason=cut_topology_changed");
    EXPECT_NE(reused.summaries[1].prolongation_words,
              reused.summaries[2].prolongation_words);
    EXPECT_NE(reused.summaries[2].prolongation_words,
              reused.summaries[3].prolongation_words);
    EXPECT_EQ(reused.summaries[1].line_words,
              reused.summaries[3].line_words);
#endif
}

TEST(SmallCutAggregationConstraint, ReuseFollowsChangesOfTheIncomingConstraintSet)
{
    SVMP_AGG_TEST_BODY
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    // A strong pin applied before aggregation removes a candidate. Toggling
    // it must rebuild; an unchanged pin must reuse.
    class ToggleableVertexPin final : public ISystemConstraint {
    public:
        ToggleableVertexPin(FieldId field, GlobalIndex vertex, bool* active)
            : field_(field), vertex_(vertex), active_(active)
        {
        }
        void apply(const systems::FESystem& system,
                   AffineConstraints& constraints) override
        {
            if (*active_) {
                constraints.addDirichlet(
                    vertexDof(system, field_, vertex_), Real{0.0});
            }
        }
        bool updateValues(const systems::FESystem&,
                          AffineConstraints&,
                          double,
                          double) override
        {
            return false;
        }
        [[nodiscard]] bool isTimeDependent() const noexcept override
        {
            return false;
        }
        [[nodiscard]] systems::SetupStorageRequirements
        storageRequirements() const noexcept override
        {
            systems::SetupStorageRequirements requirements;
            requirements.entity_dof_map = true;
            return requirements;
        }

    private:
        FieldId field_{INVALID_FIELD_ID};
        GlobalIndex vertex_{-1};
        bool* active_{nullptr};
    };

    auto mesh = buildQuadStrip(3);
    auto space =
        std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    bool pin_active = true;
    system.addSystemConstraint(
        std::make_unique<ToggleableVertexPin>(pressure, 0, &pin_active));
    system.addSystemConstraint(std::make_unique<SmallCutAggregationConstraint>(
        pressure, geometry::CutIntegrationSide::Negative, kInterfaceMarker));
    ASSERT_NO_THROW(system.setup());
    const std::vector<CellRuleSpec> specs{
        {.cell = 0, .volume_fraction = Real{0.3}, .full_cell_equivalent = false},
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
        {.cell = 2, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    };
    const auto refresh = [&]() {
        system.setCutIntegrationContext(makeCutContext(specs));
        const ScopedLogLevel detailed_diagnostics(LogLevel::DEBUG);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        system.rebuildConstraintState();
        auto log = testing::internal::GetCapturedStdout();
        log += testing::internal::GetCapturedStderr();
        return log;
    };
    const auto pinned_slave = vertexDof(system, pressure, 0);
    const auto aggregated_slave = vertexDof(system, pressure, 4);

    EXPECT_NE(refresh().find("decision=rebuilt reason=no_previous_refresh"),
              std::string::npos);
    const auto pinned = summarizeAggregationPublication(system);
    ASSERT_TRUE(system.constraints().getConstraint(pinned_slave).has_value());
    EXPECT_TRUE(system.constraints().getConstraint(pinned_slave)->isDirichlet());
    EXPECT_NE(refresh().find("decision=reused"), std::string::npos);
    EXPECT_EQ(summarizeAggregationPublication(system).line_words,
              pinned.line_words);

    pin_active = false;
    EXPECT_NE(refresh().find("decision=rebuilt reason=inputs_changed"),
              std::string::npos);
    ASSERT_TRUE(system.constraints().getConstraint(pinned_slave).has_value());
    EXPECT_FALSE(
        system.constraints().getConstraint(pinned_slave)->isDirichlet());
    EXPECT_TRUE(system.constraints().isConstrained(aggregated_slave));
    const auto unpinned = summarizeAggregationPublication(system);
    EXPECT_NE(unpinned.line_words, pinned.line_words);
    EXPECT_NE(refresh().find("decision=reused"), std::string::npos);
    EXPECT_EQ(summarizeAggregationPublication(system).line_words,
              unpinned.line_words);
#endif
}

// ============================================================================
// Sorted cell-key tables of one aggregation refresh
// ============================================================================

TEST(SmallCutAggregationCellIndex, LocalTableIteratesAndFindsInKeyOrder)
{
    detail::SmallCutAggregationLocalCellTable table;
    table.append({7, 9, 12}, 3);
    table.append({1, 4, 5}, 11);
    table.append({1, 4, 6}, 0);
    table.append({2}, 8);
    ASSERT_TRUE(table.finalize());
    ASSERT_EQ(table.size(), 4u);

    // Ascending lexicographic key order, as a std::map<CellKey, ...>.
    std::vector<GlobalIndex> cells;
    detail::SmallCutAggregationCellKey previous;
    for (const auto& [key, cell] : table) {
        EXPECT_TRUE(previous.empty() || previous < key);
        previous = key;
        cells.push_back(cell);
    }
    EXPECT_EQ(cells, (std::vector<GlobalIndex>{11, 0, 8, 3}));

    const auto hit = table.find({1, 4, 6});
    ASSERT_NE(hit, table.end());
    EXPECT_EQ(hit->second, 0);
    EXPECT_EQ(table.find({1, 4}), table.end());         // prefix of a key
    EXPECT_EQ(table.find({1, 4, 5, 6}), table.end());   // extension of a key
    EXPECT_EQ(table.find({0}), table.end());            // before every key
    EXPECT_EQ(table.find({9}), table.end());            // after every key
}

TEST(SmallCutAggregationCellIndex, LocalTableRejectsSharedKeys)
{
    detail::SmallCutAggregationLocalCellTable table;
    table.append({3, 4}, 1);
    table.append({1, 2}, 2);
    table.append({3, 4}, 5);
    EXPECT_FALSE(table.finalize());

    detail::SmallCutAggregationLocalCellTable empty;
    EXPECT_TRUE(empty.finalize());
    EXPECT_EQ(empty.find({1}), empty.end());
}

TEST(SmallCutAggregationCellIndex, GlobalIndexMatchesKeyPositions)
{
    const std::vector<detail::SmallCutAggregationCellKey> keys = {
        {1, 2, 3}, {1, 2, 4}, {1, 5}, {2, 3, 4}, {10}};
    detail::SmallCutAggregationCellIndex index(keys);
    ASSERT_EQ(index.size(), keys.size());
    EXPECT_FALSE(index.empty());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(index.find(keys[i]), i);
        EXPECT_EQ(index.key(i), keys[i]);
    }
    EXPECT_EQ(index.find({1, 2}), detail::SmallCutAggregationCellIndex::npos);
    EXPECT_EQ(index.find({1, 3}), detail::SmallCutAggregationCellIndex::npos);
    EXPECT_EQ(index.find({11}), detail::SmallCutAggregationCellIndex::npos);

    // Lookups by view into a received buffer agree with lookups by key.
    const std::vector<GlobalIndex> words = {3, 1, 2, 4, 2, 1, 5, 1, 2};
    EXPECT_EQ(index.find(std::span<const GlobalIndex>(words.data() + 1, 3)), 1u);
    EXPECT_EQ(index.find(std::span<const GlobalIndex>(words.data() + 5, 2)), 2u);
    EXPECT_EQ(index.find(std::span<const GlobalIndex>(words.data() + 7, 2)),
              detail::SmallCutAggregationCellIndex::npos);  // {1, 2}: prefix only
    EXPECT_EQ(index.find(std::span<const GlobalIndex>(words.data(), 1)),
              detail::SmallCutAggregationCellIndex::npos);  // {3}

    const detail::SmallCutAggregationCellIndex none;
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(none.find({1}), detail::SmallCutAggregationCellIndex::npos);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
