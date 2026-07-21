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

#include "Assembly/CutIntegrationContext.h"
#include "Basis/NodeOrderingConventions.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/CutQuadrature.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
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
};

void addCellRule(assembly::CutIntegrationContext& context,
                 const CellRuleSpec& spec,
                 geometry::CutIntegrationSide side)
{
    assembly::CutCellAssemblyMetadata metadata{};
    metadata.cell = spec.cell;
    metadata.parent_entity = spec.cell;
    metadata.side = side;
    metadata.volume_fraction = spec.volume_fraction;

    geometry::CutQuadratureRule rule{};
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = side;
    rule.measure = spec.volume_fraction;
    rule.parent_measure = Real{1.0};
    rule.volume_fraction = spec.volume_fraction;
    rule.full_cell_equivalent = spec.full_cell_equivalent;

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
        {.cell = 1, .volume_fraction = Real{1.0}, .full_cell_equivalent = true},
    }));

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
    }
    EXPECT_EQ(system.constraints().numConstraints(), 0u);
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
        EXPECT_THROW(system.rebuildConstraintState(), std::runtime_error);
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

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
