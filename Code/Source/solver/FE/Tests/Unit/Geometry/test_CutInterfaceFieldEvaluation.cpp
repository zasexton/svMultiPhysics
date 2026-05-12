#include "Interfaces/CutInterfaceFieldEvaluation.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

CutInterfaceDomainRequest field_eval_request()
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/9,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 90;
    return request;
}

} // namespace

TEST(CutInterfaceFieldEvaluation, EvaluatesScalarH1FieldOnCutInterfacePoints)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 1,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25}});
    ASSERT_EQ(domain.fragments().size(), 1u);

    H1NodalFieldData field;
    field.element_type = ElementType::Triangle3;
    field.components = 1;
    field.nodal_values = {
        2.0,
        3.0,
        5.0};

    const auto values =
        evaluateH1FieldValuesOnFragment(field, domain.fragments().front());
    ASSERT_EQ(values.size(), 1u);
    ASSERT_EQ(values.front().components.size(), 1u);
    EXPECT_NEAR(values.front().components.front(), 3.375, 1.0e-14);
}

TEST(CutInterfaceFieldEvaluation, EvaluatesVectorH1FieldOnCutInterfacePoints)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 2,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    ASSERT_EQ(domain.fragments().size(), 1u);

    H1NodalFieldData field;
    field.element_type = ElementType::Quad4;
    field.components = 2;
    field.nodal_values = {
        0.0, 10.0,
        1.0, 10.0,
        1.0, 11.0,
        0.0, 11.0};

    const auto values =
        evaluateH1FieldValuesOnFragment(field, domain.fragments().front());
    ASSERT_EQ(values.size(), 1u);
    ASSERT_EQ(values.front().components.size(), 2u);
    EXPECT_NEAR(values.front().components[0], 0.5, 1.0e-14);
    EXPECT_NEAR(values.front().components[1], 10.5, 1.0e-14);
}

TEST(CutInterfaceFieldEvaluation, EvaluatesScalarH1GradientOnCutInterfacePoints)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 3,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25}});
    ASSERT_EQ(domain.fragments().size(), 1u);

    H1NodalFieldData field;
    field.element_type = ElementType::Triangle3;
    field.components = 1;
    field.node_coordinates = {{{0.0, 0.0, 0.0}},
                              {{1.0, 0.0, 0.0}},
                              {{0.0, 1.0, 0.0}}};
    field.nodal_values = {
        2.0,
        3.0,
        5.0};

    const auto gradients =
        evaluateH1FieldGradientsOnFragment(field, domain.fragments().front());
    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_EQ(gradients.front().components.size(), 1u);
    EXPECT_NEAR(gradients.front().components.front()[0], 1.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components.front()[1], 3.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components.front()[2], 0.0, 1.0e-14);
}

TEST(CutInterfaceFieldEvaluation, EvaluatesVectorH1GradientOnCutInterfacePoints)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 4,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    ASSERT_EQ(domain.fragments().size(), 1u);

    H1NodalFieldData field;
    field.element_type = ElementType::Quad4;
    field.components = 2;
    field.node_coordinates = {{{0.0, 0.0, 0.0}},
                              {{1.0, 0.0, 0.0}},
                              {{1.0, 1.0, 0.0}},
                              {{0.0, 1.0, 0.0}}};
    field.nodal_values = {
        0.0, 10.0,
        1.0, 13.0,
        3.0, 12.0,
        2.0, 9.0};

    const auto gradients =
        evaluateH1FieldGradientsOnFragment(field, domain.fragments().front());
    ASSERT_EQ(gradients.size(), 1u);
    ASSERT_EQ(gradients.front().components.size(), 2u);
    EXPECT_NEAR(gradients.front().components[0][0], 1.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components[0][1], 2.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components[0][2], 0.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components[1][0], 3.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components[1][1], -1.0, 1.0e-14);
    EXPECT_NEAR(gradients.front().components[1][2], 0.0, 1.0e-14);
}
