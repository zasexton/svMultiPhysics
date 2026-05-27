#include "Interfaces/CutInterfaceFieldEvaluation.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

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

TEST(CutInterfaceFieldEvaluation,
     EvaluatesTwoSidedScalarH1ValuesOnCutInterfacePoints)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 6,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25}});
    ASSERT_EQ(domain.fragments().size(), 1u);
    const auto bindings = domain.twoSidedParentCellBindings();
    ASSERT_EQ(bindings.size(), 1u);

    H1NodalFieldData negative_field;
    negative_field.element_type = ElementType::Triangle3;
    negative_field.components = 1;
    negative_field.nodal_values = {2.0, 3.0, 5.0};

    H1NodalFieldData positive_field;
    positive_field.element_type = ElementType::Triangle3;
    positive_field.components = 1;
    positive_field.nodal_values = {7.0, 11.0, 13.0};

    const auto values = evaluateH1TwoSidedFieldValuesOnFragment(
        negative_field, positive_field, bindings.front(), domain.fragments().front());
    ASSERT_EQ(values.size(), 1u);
    const auto& value = values.front();
    EXPECT_EQ(value.interface_marker, domain.request().interface_marker);
    EXPECT_EQ(value.parent_cell, 6);
    EXPECT_EQ(value.interface_stable_id, domain.fragments().front().stable_id);
    EXPECT_EQ(value.minus_side, CutInterfaceSideTag::Negative);
    EXPECT_EQ(value.plus_side, CutInterfaceSideTag::Positive);
    ASSERT_EQ(value.minus.components.size(), 1u);
    ASSERT_EQ(value.plus.components.size(), 1u);
    ASSERT_EQ(value.jump.components.size(), 1u);
    ASSERT_EQ(value.average.components.size(), 1u);
    EXPECT_NEAR(value.minus.components.front(), 3.375, 1.0e-14);
    EXPECT_NEAR(value.plus.components.front(), 10.25, 1.0e-14);
    EXPECT_NEAR(value.jump.components.front(), 6.875, 1.0e-14);
    EXPECT_NEAR(value.average.components.front(), 6.8125, 1.0e-14);
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

TEST(CutInterfaceFieldEvaluation,
     EvaluatesTwoSidedScalarH1GradientsOnCutInterfacePoints)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 7,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25}});
    ASSERT_EQ(domain.fragments().size(), 1u);
    const auto bindings = domain.twoSidedParentCellBindings();
    ASSERT_EQ(bindings.size(), 1u);

    H1NodalFieldData negative_field;
    negative_field.element_type = ElementType::Triangle3;
    negative_field.components = 1;
    negative_field.node_coordinates = {{{0.0, 0.0, 0.0}},
                                       {{1.0, 0.0, 0.0}},
                                       {{0.0, 1.0, 0.0}}};
    negative_field.nodal_values = {2.0, 3.0, 5.0};

    H1NodalFieldData positive_field = negative_field;
    positive_field.nodal_values = {7.0, 11.0, 13.0};

    const auto gradients = evaluateH1TwoSidedFieldGradientsOnFragment(
        negative_field, positive_field, bindings.front(), domain.fragments().front());
    ASSERT_EQ(gradients.size(), 1u);
    const auto& gradient = gradients.front();
    ASSERT_EQ(gradient.minus.components.size(), 1u);
    ASSERT_EQ(gradient.plus.components.size(), 1u);
    ASSERT_EQ(gradient.jump.components.size(), 1u);
    ASSERT_EQ(gradient.average.components.size(), 1u);
    EXPECT_NEAR(gradient.minus.components.front()[0], 1.0, 1.0e-14);
    EXPECT_NEAR(gradient.minus.components.front()[1], 3.0, 1.0e-14);
    EXPECT_NEAR(gradient.minus.components.front()[2], 0.0, 1.0e-14);
    EXPECT_NEAR(gradient.plus.components.front()[0], 4.0, 1.0e-14);
    EXPECT_NEAR(gradient.plus.components.front()[1], 6.0, 1.0e-14);
    EXPECT_NEAR(gradient.plus.components.front()[2], 0.0, 1.0e-14);
    EXPECT_NEAR(gradient.jump.components.front()[0], 3.0, 1.0e-14);
    EXPECT_NEAR(gradient.jump.components.front()[1], 3.0, 1.0e-14);
    EXPECT_NEAR(gradient.jump.components.front()[2], 0.0, 1.0e-14);
    EXPECT_NEAR(gradient.average.components.front()[0], 2.5, 1.0e-14);
    EXPECT_NEAR(gradient.average.components.front()[1], 4.5, 1.0e-14);
    EXPECT_NEAR(gradient.average.components.front()[2], 0.0, 1.0e-14);
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

TEST(CutInterfaceFieldEvaluation, IntegratesScalarH1GradientsOnCutInterfaceQuadrature)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 5,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    ASSERT_EQ(domain.fragments().size(), 1u);
    const auto& fragment = domain.fragments().front();

    H1NodalFieldData field;
    field.element_type = ElementType::Quad4;
    field.components = 1;
    field.node_coordinates = {{{0.0, 0.0, 0.0}},
                              {{1.0, 0.0, 0.0}},
                              {{1.0, 1.0, 0.0}},
                              {{0.0, 1.0, 0.0}}};
    field.nodal_values = {
        4.0,
        6.0,
        3.0,
        1.0};

    const auto gradients = evaluateH1FieldGradientsOnFragment(field, fragment);
    ASSERT_EQ(gradients.size(), fragment.quadrature_points.size());

    std::array<Real, 3> integral{{0.0, 0.0, 0.0}};
    for (std::size_t q = 0; q < gradients.size(); ++q) {
        ASSERT_EQ(gradients[q].components.size(), 1u);
        for (std::size_t d = 0; d < integral.size(); ++d) {
            integral[d] +=
                fragment.quadrature_points[q].weight * gradients[q].components.front()[d];
        }
    }

    EXPECT_NEAR(fragment.measure, 1.0, 1.0e-14);
    EXPECT_NEAR(integral[0], 2.0, 1.0e-14);
    EXPECT_NEAR(integral[1], -3.0, 1.0e-14);
    EXPECT_NEAR(integral[2], 0.0, 1.0e-14);
}

TEST(CutInterfaceFieldEvaluation,
     RejectsTwoSidedH1EvaluationWhenBindingIsIncomplete)
{
    LevelSetInterfaceDomain domain(field_eval_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 8,
                             .element_type = ElementType::Triangle3,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25}});
    ASSERT_EQ(domain.fragments().size(), 1u);
    auto bindings = domain.twoSidedParentCellBindings();
    ASSERT_EQ(bindings.size(), 1u);
    bindings.front().positive_volume_region_stable_ids.clear();

    H1NodalFieldData field;
    field.element_type = ElementType::Triangle3;
    field.components = 1;
    field.nodal_values = {2.0, 3.0, 5.0};

    EXPECT_THROW(evaluateH1TwoSidedFieldValuesOnFragment(
                     field, field, bindings.front(), domain.fragments().front()),
                 std::invalid_argument);
}
