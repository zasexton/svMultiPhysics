#include "Coupling/MonolithicCouplingBuilder.h"

#include "Analysis/ProblemAnalysisTypes.h"
#include "Coupling/CouplingFormBuilder.h"
#include "Core/FEException.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

struct BuilderFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem system;
    FieldId left_field{INVALID_FIELD_ID};
    FieldId right_field{INVALID_FIELD_ID};
    CouplingContext context;

    BuilderFixture()
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , system(mesh)
    {
        left_field = system.addField(systems::FieldSpec{
            .name = "left_primary",
            .space = space,
            .components = 1,
        });
        right_field = system.addField(systems::FieldSpec{
            .name = "right_primary",
            .space = space,
            .components = 1,
        });

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "shared_system",
            .system = &system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "shared_system",
            .system = &system,
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "shared_system",
            .system = &system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "shared_system",
            .system = &system,
            .field_name = "primary",
            .field_id = right_field,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

const CouplingInstalledDependency* findDependency(
    const CouplingFormAnalysisMetadata& metadata,
    analysis::VariableKey row,
    analysis::VariableKey dependency)
{
    const auto it = std::find_if(
        metadata.installed_dependencies.begin(),
        metadata.installed_dependencies.end(),
        [&](const CouplingInstalledDependency& installed) {
            return installed.residual_row == row &&
                   installed.dependency == dependency;
        });
    return it == metadata.installed_dependencies.end() ? nullptr : &*it;
}

} // namespace

TEST(MonolithicCouplingBuilder, ResolvesFormContributionThroughContext)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "generic_cell_coupling";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{.participant_name = "left", .field_name = "primary"}};
    contribution.residual =
        (forms.state("left", "primary", "a") *
         forms.test("right", "primary", "w")).dx();

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    ASSERT_EQ(resolved.fields.size(), 1u);
    ASSERT_EQ(resolved.extra_trial_fields.size(), 1u);
    EXPECT_EQ(resolved.fields[0], fixture.right_field);
    EXPECT_EQ(resolved.extra_trial_fields[0], fixture.left_field);
    ASSERT_EQ(resolved.install_options.extra_trial_fields.size(), 1u);
    EXPECT_EQ(resolved.install_options.extra_trial_fields[0], fixture.left_field);
    EXPECT_EQ(resolved.system_name, "shared_system");
}

TEST(MonolithicCouplingBuilder, RejectsOverlappingPrimaryAndExtraTrialFields)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "bad_overlap";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();

    EXPECT_THROW(static_cast<void>(builder.resolveFormContribution(fixture.context, contribution)),
                 InvalidArgumentException);
}

TEST(MonolithicCouplingBuilder, InstallsResolvedFormAndAdaptsBridgeMetadata)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "generic_cell_coupling";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.operator_name = "equations";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{.participant_name = "left", .field_name = "primary"}};
    contribution.residual =
        (forms.state("left", "primary", "a") *
         forms.test("right", "primary", "w")).dx();

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    const auto metadata =
        builder.installResolvedFormContribution(fixture.system, resolved);

    EXPECT_EQ(metadata.contribution_name, "generic_cell_coupling");
    EXPECT_EQ(metadata.origin, "MonolithicCouplingBuilderTest");
    EXPECT_EQ(metadata.system_name, "shared_system");
    EXPECT_NE(std::find(metadata.installed_fields.begin(),
                        metadata.installed_fields.end(),
                        fixture.right_field),
              metadata.installed_fields.end());
    EXPECT_NE(std::find(metadata.installed_fields.begin(),
                        metadata.installed_fields.end(),
                        fixture.left_field),
              metadata.installed_fields.end());

    const auto* dependency = findDependency(
        metadata,
        analysis::VariableKey::field(fixture.right_field),
        analysis::VariableKey::field(fixture.left_field));
    ASSERT_NE(dependency, nullptr);
    EXPECT_EQ(dependency->domain, analysis::DomainKind::Cell);
    EXPECT_TRUE(dependency->contributes_matrix_block);
    EXPECT_TRUE(dependency->contributes_vector);

    const auto block_it = std::find_if(
        metadata.installed_blocks.begin(),
        metadata.installed_blocks.end(),
        [&](const CouplingInstalledBlockProvenance& block) {
            return block.residual_row == analysis::VariableKey::field(fixture.right_field) &&
                   block.dependency == analysis::VariableKey::field(fixture.left_field);
        });
    ASSERT_NE(block_it, metadata.installed_blocks.end());
    EXPECT_TRUE(block_it->has_matrix);
    EXPECT_TRUE(block_it->has_vector);
}
