#include "Systems/InterfaceOperators.h"
#include "Assembly/InterfacePairContext.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include <gtest/gtest.h>

namespace {

svmp::search::InterfaceMap make_map() {
    svmp::search::InterfaceMap map;
    map.name = "operator-map";
    map.state = svmp::search::InterfaceMapState::Committed;

    svmp::search::InterfacePair p0;
    p0.source_face = 1;
    p0.target_face = 0;
    p0.source_cell = 7;
    p0.target_cell = 8;
    p0.source_measure = 2.0;
    p0.target_measure = 2.0;
    p0.source_point = {{1.0, 0.25, 0.0}};
    p0.target_point = {{1.1, 0.25, 0.0}};
    p0.source_face_xi = {{-0.5, 0.0, 0.0}};
    p0.target_face_xi = {{-0.5, 0.0, 0.0}};
    p0.source_cell_xi = {{1.0, -0.5, 0.0}};
    p0.target_cell_xi = {{-1.0, -0.5, 0.0}};
    p0.source_normal = {{1.0, 0.0, 0.0}};
    p0.target_normal = {{-1.0, 0.0, 0.0}};

    svmp::search::InterfacePair p1 = p0;
    p1.source_face = 2;
    p1.target_face = 1;
    p1.source_cell = 9;
    p1.target_cell = 10;
    p1.source_measure = 3.0;
    p1.target_measure = 3.0;
    p1.source_point = {{1.0, 0.75, 0.0}};
    p1.target_point = {{1.1, 0.75, 0.0}};
    p1.source_face_xi = {{0.5, 0.0, 0.0}};
    p1.target_face_xi = {{0.5, 0.0, 0.0}};

    map.pairs = {p0, p1};
    return map;
}

} // namespace

TEST(MultiMeshInterfaceOperatorsTest, PointwiseInterpolationTransfersFaceValues) {
    const auto map = make_map();
    const std::vector<svmp::FE::Real> source_values = {0.0, 4.0, 6.0};

    const auto op = svmp::FE::systems::makeInterfaceTransferOperator(
        svmp::FE::systems::InterfaceOperatorKind::PointwiseInterpolation);
    const auto result = op->apply(map, source_values);

    ASSERT_EQ(result.target_values.size(), 2u);
    EXPECT_DOUBLE_EQ(result.target_values[0], 4.0);
    EXPECT_DOUBLE_EQ(result.target_values[1], 6.0);
    EXPECT_DOUBLE_EQ(result.target_weights[0], 1.0);
    EXPECT_DOUBLE_EQ(result.target_weights[1], 1.0);
}

TEST(MultiMeshInterfaceOperatorsTest, ConservativeAndMortarOperatorsPreserveIntegratedQuantity) {
    const auto map = make_map();
    const std::vector<svmp::FE::Real> source_values = {0.0, 4.0, 6.0};

    const auto conservative = svmp::FE::systems::makeInterfaceTransferOperator(
        svmp::FE::systems::InterfaceOperatorKind::ConservativeProjection);
    const auto mortar = svmp::FE::systems::makeInterfaceTransferOperator(
        svmp::FE::systems::InterfaceOperatorKind::Mortar);

    const auto conservative_result = conservative->apply(map, source_values);
    const auto mortar_result = mortar->apply(map, source_values);

    EXPECT_DOUBLE_EQ(conservative_result.source_integral, 26.0);
    EXPECT_DOUBLE_EQ(conservative_result.target_integral, 26.0);
    EXPECT_DOUBLE_EQ(conservative_result.target_values[0], 4.0);
    EXPECT_DOUBLE_EQ(conservative_result.target_values[1], 6.0);
    EXPECT_EQ(mortar_result.kind, svmp::FE::systems::InterfaceOperatorKind::Mortar);
    EXPECT_DOUBLE_EQ(mortar_result.target_integral, conservative_result.target_integral);
}

TEST(MultiMeshInterfaceOperatorsTest, AssemblyContextExposesPairedSideGeometry) {
    const auto map = make_map();
    svmp::FE::assembly::InterfacePairContext context(map);

    ASSERT_EQ(context.quadraturePairs().size(), 2u);
    const auto& qp = context.quadraturePairs().front();
    EXPECT_EQ(qp.source_face, 1);
    EXPECT_EQ(qp.target_face, 0);
    EXPECT_DOUBLE_EQ(qp.source_point[0], 1.0);
    EXPECT_DOUBLE_EQ(qp.target_point[0], 1.1);
    EXPECT_DOUBLE_EQ(qp.weight, 2.0);
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
