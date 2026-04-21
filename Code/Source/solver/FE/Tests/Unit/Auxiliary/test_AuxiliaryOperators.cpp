/**
 * @file test_AuxiliaryOperators.cpp
 * @brief Unit tests for AuxiliaryCouplingGraph, AuxiliaryOperatorBuilder,
 *        and AuxiliaryOperatorRegistry.
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryCouplingGraph.h"
#include "Auxiliary/AuxiliaryOperatorBuilder.h"
#include "Auxiliary/AuxiliaryOperatorRegistry.h"
#include "Auxiliary/AuxiliaryRowOwnership.h"

#include <algorithm>
#include <span>
#include <string>
#include <vector>

using svmp::FE::Real;
using namespace svmp::FE::systems;

// ============================================================================
//  AuxiliaryCouplingGraph tests
// ============================================================================

TEST(AuxiliaryCouplingGraph, EmptyGraph)
{
    AuxiliaryCouplingGraph graph;
    EXPECT_EQ(graph.edgeCount(), 0u);
    EXPECT_TRUE(graph.auxiliaryVertices().empty());
    EXPECT_TRUE(graph.fieldVertices().empty());
}

TEST(AuxiliaryCouplingGraph, SelfCoupling)
{
    AuxiliaryCouplingGraph graph;
    graph.addSelfCoupling("ionic", "self_op");

    EXPECT_EQ(graph.edgeCount(), 1u);
    EXPECT_EQ(graph.edges()[0].type, AuxiliaryCouplingType::AuxSelf);
    EXPECT_EQ(graph.edges()[0].source, "ionic");
    EXPECT_EQ(graph.edges()[0].target, "ionic");
}

TEST(AuxiliaryCouplingGraph, AuxToAuxCoupling)
{
    AuxiliaryCouplingGraph graph;
    graph.addAuxToAux("block_A", "block_B", "coupling_op");

    auto incoming = graph.incomingEdges("block_B");
    ASSERT_EQ(incoming.size(), 1u);
    EXPECT_EQ(incoming[0].source, "block_A");
    EXPECT_EQ(incoming[0].type, AuxiliaryCouplingType::AuxToAux);

    auto outgoing = graph.outgoingEdges("block_A");
    ASSERT_EQ(outgoing.size(), 1u);
    EXPECT_EQ(outgoing[0].target, "block_B");
}

TEST(AuxiliaryCouplingGraph, FieldToAuxCoupling)
{
    AuxiliaryCouplingGraph graph;
    graph.addFieldToAux("velocity", "ionic", "field_coupling");

    EXPECT_TRUE(graph.hasCouplingToFields("ionic"));
    EXPECT_FALSE(graph.hasCouplingToAux("ionic"));

    auto fields = graph.fieldVertices();
    ASSERT_EQ(fields.size(), 1u);
    EXPECT_EQ(fields[0], "velocity");
}

TEST(AuxiliaryCouplingGraph, AuxToFieldCoupling)
{
    AuxiliaryCouplingGraph graph;
    graph.addAuxToField("rcr", "pressure", "bc_coupling");

    EXPECT_TRUE(graph.hasCouplingToFields("rcr"));

    auto fields = graph.fieldVertices();
    ASSERT_EQ(fields.size(), 1u);
    EXPECT_EQ(fields[0], "pressure");
}

TEST(AuxiliaryCouplingGraph, AuxVertices)
{
    AuxiliaryCouplingGraph graph;
    graph.addSelfCoupling("A", "op1");
    graph.addAuxToAux("B", "C", "op2");

    auto verts = graph.auxiliaryVertices();
    EXPECT_EQ(verts.size(), 3u);

    // Check all names present (order may vary due to unordered_set)
    auto has = [&](const std::string& n) {
        return std::find(verts.begin(), verts.end(), n) != verts.end();
    };
    EXPECT_TRUE(has("A"));
    EXPECT_TRUE(has("B"));
    EXPECT_TRUE(has("C"));
}

TEST(AuxiliaryCouplingGraph, Clear)
{
    AuxiliaryCouplingGraph graph;
    graph.addSelfCoupling("A", "op");
    graph.clear();
    EXPECT_EQ(graph.edgeCount(), 0u);
}

// ============================================================================
//  AuxiliaryOperatorBuilder tests
// ============================================================================

TEST(AuxiliaryOperatorBuilder, BuildSelfCoupling)
{
    auto desc = AuxiliaryOperatorBuilder("self_op")
        .source("ionic")
        .target("ionic")
        .topology(AuxiliaryCouplingTopology::Dense)
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .build();

    EXPECT_EQ(desc.name, "self_op");
    EXPECT_EQ(desc.coupling_type, AuxiliaryCouplingType::AuxSelf);
    EXPECT_EQ(desc.source_name, "ionic");
    EXPECT_EQ(desc.target_name, "ionic");
    EXPECT_EQ(desc.topology, AuxiliaryCouplingTopology::Dense);
    EXPECT_TRUE(desc.residual_fn);
}

TEST(AuxiliaryOperatorBuilder, BuildFieldToAux)
{
    auto desc = AuxiliaryOperatorBuilder("field_coupling")
        .source("field:velocity")
        .target("ionic")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .build();

    EXPECT_EQ(desc.coupling_type, AuxiliaryCouplingType::FieldToAux);
}

TEST(AuxiliaryOperatorBuilder, BuildAuxToField)
{
    auto desc = AuxiliaryOperatorBuilder("bc_coupling")
        .source("rcr")
        .target("field:pressure")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .build();

    EXPECT_EQ(desc.coupling_type, AuxiliaryCouplingType::AuxToField);
}

TEST(AuxiliaryOperatorBuilder, BuildAuxToAux)
{
    auto desc = AuxiliaryOperatorBuilder("cross_coupling")
        .source("block_A")
        .target("block_B")
        .topology(AuxiliaryCouplingTopology::Sparse)
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .jacobian([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .build();

    EXPECT_EQ(desc.coupling_type, AuxiliaryCouplingType::AuxToAux);
    EXPECT_TRUE(desc.jacobian_fn);
}

TEST(AuxiliaryOperatorBuilder, OptionalMassAndTransfer)
{
    auto desc = AuxiliaryOperatorBuilder("with_mass")
        .source("A")
        .target("A")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .mass([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .transfer([](const AuxiliaryOperatorContext&, std::span<const Real>,
                     std::span<Real>) {})
        .build();

    EXPECT_TRUE(desc.mass_fn);
    EXPECT_TRUE(desc.transfer_fn);
}

TEST(AuxiliaryOperatorBuilder, DerivativePolicy)
{
    AuxiliaryDerivativePolicy policy;
    policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;

    auto desc = AuxiliaryOperatorBuilder("with_deriv")
        .source("A")
        .target("A")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .derivatives(policy)
        .build();

    EXPECT_TRUE(desc.has_derivative_policy);
    EXPECT_EQ(desc.derivative_policy.jacobian_source,
              AuxiliaryDerivativeSource::FiniteDifference);
}

TEST(AuxiliaryOperatorBuilder, MissingSourceThrows)
{
    auto builder = AuxiliaryOperatorBuilder("bad")
        .target("A")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {});

    EXPECT_THROW(builder.build(), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryOperatorBuilder, MissingResidualThrows)
{
    auto builder = AuxiliaryOperatorBuilder("bad")
        .source("A")
        .target("A");

    EXPECT_THROW(builder.build(), svmp::FE::InvalidArgumentException);
}

// ============================================================================
//  AuxiliaryOperatorRegistry tests
// ============================================================================

TEST(AuxiliaryOperatorRegistry, RegisterAndRetrieve)
{
    AuxiliaryOperatorRegistry reg;

    auto desc = AuxiliaryOperatorBuilder("op1")
        .source("A")
        .target("A")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .build();

    reg.registerOperator(desc);

    EXPECT_EQ(reg.operatorCount(), 1u);
    EXPECT_TRUE(reg.hasOperator("op1"));
    EXPECT_EQ(reg.getOperator("op1").name, "op1");
}

TEST(AuxiliaryOperatorRegistry, DuplicateThrows)
{
    AuxiliaryOperatorRegistry reg;

    auto desc = AuxiliaryOperatorBuilder("op1")
        .source("A")
        .target("A")
        .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
        .build();

    reg.registerOperator(desc);
    EXPECT_THROW(reg.registerOperator(desc), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryOperatorRegistry, OperatorNames)
{
    AuxiliaryOperatorRegistry reg;

    auto mk = [](const std::string& name) {
        return AuxiliaryOperatorBuilder(name)
            .source("A")
            .target("A")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build();
    };

    reg.registerOperator(mk("alpha"));
    reg.registerOperator(mk("beta"));

    auto names = reg.operatorNames();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "beta");
}

TEST(AuxiliaryOperatorRegistry, CouplingGraphUpdatedOnRegister)
{
    AuxiliaryOperatorRegistry reg;

    reg.registerOperator(
        AuxiliaryOperatorBuilder("coupling")
            .source("block_A")
            .target("block_B")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build());

    EXPECT_EQ(reg.couplingGraph().edgeCount(), 1u);
    auto edges = reg.couplingGraph().edges();
    EXPECT_EQ(edges[0].source, "block_A");
    EXPECT_EQ(edges[0].target, "block_B");
    EXPECT_EQ(edges[0].operator_name, "coupling");
}

// ============================================================================
//  Monolithic unknown layout tests
// ============================================================================

TEST(AuxiliaryOperatorRegistry, MonolithicUnknownRegistration)
{
    AuxiliaryOperatorRegistry reg;

    reg.registerMonolithicUnknowns("ionic", 100, 4, AuxiliaryStateScope::Node);
    reg.registerMonolithicUnknowns("rcr", 1, 3, AuxiliaryStateScope::Global);

    reg.finalizeLayout();

    EXPECT_TRUE(reg.isLayoutFinalized());

    const auto& layout = reg.auxiliaryLayout();
    ASSERT_EQ(layout.blocks.size(), 2u);

    EXPECT_EQ(layout.blocks[0].name, "ionic");
    EXPECT_EQ(layout.blocks[0].offset, 0u);
    EXPECT_EQ(layout.blocks[0].n_unknowns, 400u);
    EXPECT_EQ(layout.blocks[0].scope, AuxiliaryStateScope::Node);

    EXPECT_EQ(layout.blocks[1].name, "rcr");
    EXPECT_EQ(layout.blocks[1].offset, 400u);
    EXPECT_EQ(layout.blocks[1].n_unknowns, 3u);
    EXPECT_EQ(layout.blocks[1].single_owner_rank, 0);
    EXPECT_EQ(layout.blocks[1].row_owner_ranks, std::vector<int>({0, 0, 0}));

    EXPECT_EQ(layout.total_aux_unknowns, 403u);
}

TEST(AuxiliaryOperatorRegistry, ScopeRowOwnershipHelpersExpandToRows)
{
    const std::vector<int> node_owners{0, 1, 1};
    const auto node_rows = buildAuxiliaryRowOwnerRanks(
        AuxiliaryRowOwnershipSpec{
            .scope = AuxiliaryStateScope::Node,
            .policy = svmp::FE::backends::MixedRowOwnershipPolicy::BackendDofOwner,
            .entity_count = node_owners.size(),
            .stride = 2,
            .entity_owner_ranks =
                std::span<const int>{node_owners.data(), node_owners.size()}});
    EXPECT_EQ(node_rows, std::vector<int>({0, 0, 1, 1, 1, 1}));

    const std::vector<int> cell_owners{0, 2};
    const auto cell_rows = buildAuxiliaryRowOwnerRanks(
        AuxiliaryRowOwnershipSpec{
            .scope = AuxiliaryStateScope::Cell,
            .policy = svmp::FE::backends::MixedRowOwnershipPolicy::CellOwner,
            .entity_count = cell_owners.size(),
            .stride = 1,
            .entity_owner_ranks =
                std::span<const int>{cell_owners.data(), cell_owners.size()}});
    EXPECT_EQ(cell_rows, std::vector<int>({0, 2}));

    const std::vector<std::size_t> qp_offsets{0, 2, 3};
    const auto qp_rows =
        buildQuadraturePointRowOwnerRanks(
            std::span<const int>{cell_owners.data(), cell_owners.size()},
            std::span<const std::size_t>{qp_offsets.data(), qp_offsets.size()},
            /*stride=*/2);
    EXPECT_EQ(qp_rows, std::vector<int>({0, 0, 0, 0, 2, 2}));

    const std::vector<int> cell_regions{1, 0, 1};
    const std::vector<int> region_cell_owners{2, 0, 1};
    const auto region_owners = buildRegionEntityOwnerRanksFromCells(
        std::span<const int>{cell_regions.data(), cell_regions.size()},
        std::span<const int>{region_cell_owners.data(), region_cell_owners.size()},
        /*n_regions=*/2);
    EXPECT_EQ(region_owners, std::vector<int>({0, 2}));
    const auto region_rows = buildAuxiliaryRowOwnerRanks(
        AuxiliaryRowOwnershipSpec{
            .scope = AuxiliaryStateScope::Region,
            .policy = svmp::FE::backends::MixedRowOwnershipPolicy::RegionOwner,
            .entity_count = region_owners.size(),
            .stride = 3,
            .entity_owner_ranks =
                std::span<const int>{region_owners.data(), region_owners.size()}});
    EXPECT_EQ(region_rows, std::vector<int>({0, 0, 0, 2, 2, 2}));
}

TEST(AuxiliaryOperatorRegistry, ConcreteRowOwnersRoundTripThroughMixedLayout)
{
    AuxiliaryOperatorRegistry reg;
    reg.registerMonolithicUnknowns("node_aux", 3, 2, AuxiliaryStateScope::Node);

    const std::vector<int> row_owners{0, 0, 1, 1, 1, 1};
    reg.setBlockRowOwnerRanks("node_aux", row_owners);
    reg.finalizeLayout();

    const auto mixed = reg.composeMixedLayout(/*n_field_unknowns=*/10);
    ASSERT_EQ(mixed.aux_layout.blocks.size(), 1u);
    const auto& block = mixed.aux_layout.blocks.front();
    EXPECT_EQ(block.row_ownership,
              svmp::FE::backends::MixedRowOwnershipPolicy::BackendDofOwner);
    EXPECT_EQ(block.row_owner_ranks, row_owners);
    EXPECT_THROW(reg.setBlockRowOwnerRanks("node_aux", {0}),
                 svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryOperatorRegistry, ComposeMixedLayout)
{
    AuxiliaryOperatorRegistry reg;

    reg.registerMonolithicUnknowns("aux_A", 50, 2, AuxiliaryStateScope::Node);
    reg.finalizeLayout();

    auto mixed = reg.composeMixedLayout(/*n_field_unknowns=*/1000);

    EXPECT_EQ(mixed.n_field_unknowns, 1000u);
    EXPECT_EQ(mixed.n_aux_unknowns, 100u);
    EXPECT_EQ(mixed.total_unknowns, 1100u);
    EXPECT_EQ(mixed.aux_layout.mixed_system_offset, 1000u);
}

TEST(AuxiliaryOperatorRegistry, SolverMetadataRoundTrip)
{
    AuxiliaryOperatorRegistry reg;

    AuxiliaryBlockSolverMetadata meta;
    meta.block_name = "lambda";
    meta.role = AuxiliaryBlockRole::Constraint;
    meta.block_diagonal_suitable = false;
    meta.schur_eliminable = false;

    reg.setBlockSolverMetadata("lambda", meta);

    const auto* stored = reg.getBlockSolverMetadata("lambda");
    ASSERT_NE(stored, nullptr);
    EXPECT_EQ(stored->block_name, "lambda");
    EXPECT_EQ(stored->role, AuxiliaryBlockRole::Constraint);
    EXPECT_FALSE(stored->block_diagonal_suitable);
}

TEST(AuxiliaryOperatorRegistry, MonolithicUnknownRegistrationCarriesSolverMetadata)
{
    AuxiliaryOperatorRegistry reg;

    AuxiliaryBlockSolverMetadata meta;
    meta.block_name = "lambda";
    meta.role = AuxiliaryBlockRole::Constraint;
    meta.block_diagonal_suitable = false;
    meta.schur_complement_partner = "velocity";

    reg.registerMonolithicUnknowns("lambda",
                                   2,
                                   1,
                                   AuxiliaryStateScope::Global,
                                   &meta,
                                   {{0}, {1}});
    reg.finalizeLayout();

    const auto* blk = reg.findLayoutBlock("lambda");
    ASSERT_NE(blk, nullptr);
    EXPECT_EQ(blk->role, AuxiliaryBlockRole::Constraint);
    EXPECT_EQ(blk->backend_role, svmp::FE::backends::BlockRole::ConstraintField);
    EXPECT_FALSE(blk->block_diagonal_suitable);
    EXPECT_EQ(blk->schur_complement_partner, "velocity");
    ASSERT_EQ(blk->constraint_groups.size(), 2u);
    EXPECT_EQ(blk->constraint_groups[0][0], 0);
    EXPECT_EQ(blk->constraint_groups[1][0], 1);
}

TEST(AuxiliaryOperatorRegistry, RoleQueriesUseMixedLayoutMetadata)
{
    AuxiliaryOperatorRegistry reg;

    AuxiliaryBlockSolverMetadata lambda;
    lambda.block_name = "lambda";
    lambda.role = AuxiliaryBlockRole::Constraint;

    AuxiliaryBlockSolverMetadata condensed;
    condensed.block_name = "condensed";
    condensed.role = AuxiliaryBlockRole::SchurEliminable;
    condensed.schur_eliminable = true;
    condensed.schur_complement_partner = "u";

    AuxiliaryBlockSolverMetadata stiff;
    stiff.block_name = "stiff";
    stiff.role = AuxiliaryBlockRole::SpecialPrecondition;

    reg.registerMonolithicUnknowns("lambda", 1, 1, AuxiliaryStateScope::Global, &lambda);
    reg.registerMonolithicUnknowns("condensed", 4, 2, AuxiliaryStateScope::Cell, &condensed);
    reg.registerMonolithicUnknowns("stiff", 3, 1, AuxiliaryStateScope::Node, &stiff);
    reg.finalizeLayout();

    EXPECT_EQ(reg.constraintLikeBlocks(), std::vector<std::string>({"lambda"}));
    EXPECT_EQ(reg.schurEliminableBlocks(), std::vector<std::string>({"condensed"}));
    EXPECT_EQ(reg.specialPreconditionBlocks(), std::vector<std::string>({"stiff"}));
}

TEST(AuxiliaryOperatorRegistry, RegisterAfterFinalizeThrows)
{
    AuxiliaryOperatorRegistry reg;
    reg.registerMonolithicUnknowns("A", 10, 1, AuxiliaryStateScope::Global);
    reg.finalizeLayout();

    EXPECT_THROW(
        reg.registerMonolithicUnknowns("B", 5, 1, AuxiliaryStateScope::Global),
        svmp::FE::systems::InvalidStateException);
}

// ============================================================================
//  Mixed field/auxiliary coupling scenario
// ============================================================================

TEST(AuxiliaryOperatorRegistry, FullMixedCouplingScenario)
{
    AuxiliaryOperatorRegistry reg;

    // Register operators for a mixed field/auxiliary system.
    // Ionic model: field→aux (voltage drives gates)
    reg.registerOperator(
        AuxiliaryOperatorBuilder("field_to_ionic")
            .source("field:voltage")
            .target("ionic")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build());

    // Ionic model: aux→field (gates affect field equation)
    reg.registerOperator(
        AuxiliaryOperatorBuilder("ionic_to_field")
            .source("ionic")
            .target("field:voltage")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build());

    // Ionic self-coupling (diagonal block)
    reg.registerOperator(
        AuxiliaryOperatorBuilder("ionic_self")
            .source("ionic")
            .target("ionic")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .jacobian([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build());

    EXPECT_EQ(reg.operatorCount(), 3u);
    EXPECT_EQ(reg.couplingGraph().edgeCount(), 3u);

    // Coupling graph queries
    EXPECT_TRUE(reg.couplingGraph().hasCouplingToFields("ionic"));
    // Self-coupling is AuxSelf, not AuxToAux — hasCouplingToAux checks AuxToAux only.
    EXPECT_FALSE(reg.couplingGraph().hasCouplingToAux("ionic"));

    // Register monolithic unknowns and compose layout
    reg.registerMonolithicUnknowns("ionic", 100, 4, AuxiliaryStateScope::Node);
    reg.finalizeLayout();

    auto mixed = reg.composeMixedLayout(5000);
    EXPECT_EQ(mixed.total_unknowns, 5400u); // 5000 field + 400 auxiliary
    EXPECT_EQ(mixed.aux_layout.mixed_system_offset, 5000u);
}

// ============================================================================
//  Partitioned blocks stay out of layout
// ============================================================================

TEST(AuxiliaryOperatorRegistry, PartitionedBlocksNotInLayout)
{
    AuxiliaryOperatorRegistry reg;

    // Only monolithic blocks are registered for layout.
    // Partitioned blocks are not added here at all.
    reg.registerMonolithicUnknowns("mono_A", 10, 2, AuxiliaryStateScope::Node);
    reg.finalizeLayout();

    EXPECT_EQ(reg.auxiliaryLayout().blocks.size(), 1u);
    EXPECT_EQ(reg.auxiliaryLayout().total_aux_unknowns, 20u);
}

// ============================================================================
//  Clear
// ============================================================================

TEST(AuxiliaryOperatorRegistry, Clear)
{
    AuxiliaryOperatorRegistry reg;

    reg.registerOperator(
        AuxiliaryOperatorBuilder("op")
            .source("A")
            .target("A")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build());
    reg.registerMonolithicUnknowns("A", 10, 1, AuxiliaryStateScope::Global);
    reg.finalizeLayout();

    reg.clear();

    EXPECT_EQ(reg.operatorCount(), 0u);
    EXPECT_EQ(reg.couplingGraph().edgeCount(), 0u);
    EXPECT_FALSE(reg.isLayoutFinalized());
}

// ============================================================================
//  Coupling topology varieties
// ============================================================================

TEST(AuxiliaryOperatorBuilder, AllTopologies)
{
    auto mkOp = [](AuxiliaryCouplingTopology topo) {
        return AuxiliaryOperatorBuilder("op")
            .source("A")
            .target("B")
            .topology(topo)
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build();
    };

    EXPECT_EQ(mkOp(AuxiliaryCouplingTopology::Dense).topology,
              AuxiliaryCouplingTopology::Dense);
    EXPECT_EQ(mkOp(AuxiliaryCouplingTopology::Sparse).topology,
              AuxiliaryCouplingTopology::Sparse);
    EXPECT_EQ(mkOp(AuxiliaryCouplingTopology::PointToPoint).topology,
              AuxiliaryCouplingTopology::PointToPoint);
    EXPECT_EQ(mkOp(AuxiliaryCouplingTopology::Reduction).topology,
              AuxiliaryCouplingTopology::Reduction);
}
