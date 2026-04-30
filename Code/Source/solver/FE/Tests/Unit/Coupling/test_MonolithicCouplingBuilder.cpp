#include "Coupling/MonolithicCouplingBuilder.h"

#include "Analysis/ProblemAnalysisTypes.h"
#include "Coupling/CouplingGraph.h"
#include "Coupling/CouplingFormBuilder.h"
#include "Core/FEException.h"
#include "Forms/Vocabulary.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

constexpr int kInterfaceMarker = 17;

struct BuilderFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<spaces::ProductSpace> vector_space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem system;
    FieldId left_field{INVALID_FIELD_ID};
    FieldId right_field{INVALID_FIELD_ID};
    FieldId mesh_motion_field{INVALID_FIELD_ID};
    CouplingContext context;

    BuilderFixture()
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , vector_space(std::make_shared<spaces::ProductSpace>(space, 3))
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
        mesh_motion_field = system.addField(systems::FieldSpec{
            .name = "left_mesh_displacement",
            .space = vector_space,
            .components = 3,
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
        builder.addField({
            .participant_name = "left",
            .system_name = "shared_system",
            .system = &system,
            .field_name = "mesh_displacement",
            .field_id = mesh_motion_field,
            .space = vector_space,
            .components = 3,
        });
        context = builder.build();
    }
};

CouplingContext interfaceContext(BuilderFixture& fixture, int marker)
{
    CouplingRegionRef left_region{
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = CouplingInterfaceSide::Minus,
    };
    CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = CouplingInterfaceSide::Plus,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
        .field_name = "primary",
        .field_id = fixture.left_field,
        .space = fixture.space,
        .components = 1,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
        .field_name = "primary",
        .field_id = fixture.right_field,
        .space = fixture.space,
        .components = 1,
    });
    builder.addRegion(left_region);
    builder.addRegion(right_region);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_region, right_region},
    });
    return builder.build();
}

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

template <typename T, typename = void>
struct HasSymbolicAdMode : std::false_type {};
template <typename T>
struct HasSymbolicAdMode<T, std::void_t<decltype(std::declval<T&>().ad_mode)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSymbolicGeometrySensitivity : std::false_type {};
template <typename T>
struct HasSymbolicGeometrySensitivity<
    T,
    std::void_t<decltype(std::declval<T&>().geometry_sensitivity)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSymbolicGeometryTangentPath : std::false_type {};
template <typename T>
struct HasSymbolicGeometryTangentPath<
    T,
    std::void_t<decltype(std::declval<T&>().geometry_tangent_path)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSymbolicUseSymbolicTangent : std::false_type {};
template <typename T>
struct HasSymbolicUseSymbolicTangent<
    T,
    std::void_t<decltype(std::declval<T&>().use_symbolic_tangent)>>
    : std::true_type {};

static_assert(!HasSymbolicAdMode<CouplingSymbolicOptionsDeclaration>::value);
static_assert(
    !HasSymbolicGeometrySensitivity<CouplingSymbolicOptionsDeclaration>::value);
static_assert(
    !HasSymbolicGeometryTangentPath<CouplingSymbolicOptionsDeclaration>::value);
static_assert(
    !HasSymbolicUseSymbolicTangent<CouplingSymbolicOptionsDeclaration>::value);

class GenericTwoParticipantContract final : public CouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "generic";
    }

    [[nodiscard]] CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "generic_instance";
        declaration.participants.push_back({.participant_name = "left"});
        declaration.participants.push_back({.participant_name = "right"});
        declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
        declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
        declaration.dependencies.push_back(CouplingResidualDependency{
            .residual_row = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "right",
                .name = "primary",
            },
            .dependency = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "left",
                .name = "primary",
            },
        });
        declaration.expected_blocks.push_back(CouplingBlockExpectation{
            .residual_row = declaration.dependencies.back().residual_row,
            .dependency = declaration.dependencies.back().dependency,
        });
        return declaration;
    }

    [[nodiscard]] std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext&,
        const CouplingFormBuilder& forms) const override
    {
        CouplingFormContribution contribution;
        contribution.contribution_name = "generic_cell_coupling";
        contribution.origin = "GenericTwoParticipantContract";
        contribution.operator_name = "equations";
        contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
        contribution.extra_trial_field_uses = {{
            .participant_name = "left",
            .field_name = "primary",
        }};
        contribution.residual =
            (forms.state("left", "primary", "a") *
             forms.test("right", "primary", "w")).dx();
        return {contribution};
    }
};

class GenericInterfaceContract final : public CouplingContract {
public:
    [[nodiscard]] std::string name() const override
    {
        return "generic_interface";
    }

    [[nodiscard]] CouplingContractDeclaration declare() const override
    {
        CouplingContractDeclaration declaration;
        declaration.contract_type = name();
        declaration.contract_name = "generic_interface_instance";
        declaration.participants.push_back({.participant_name = "left"});
        declaration.participants.push_back({.participant_name = "right"});
        declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
        declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
        declaration.shared_regions.push_back({
            .shared_region_name = "interface",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
        });
        declaration.dependencies.push_back(CouplingResidualDependency{
            .residual_row = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "right",
                .name = "primary",
            },
            .dependency = {
                .kind = CouplingVariableKind::Field,
                .participant_name = "left",
                .name = "primary",
            },
        });
        declaration.expected_blocks.push_back(CouplingBlockExpectation{
            .residual_row = declaration.dependencies.back().residual_row,
            .dependency = declaration.dependencies.back().dependency,
        });
        return declaration;
    }

    [[nodiscard]] std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext& context,
        const CouplingFormBuilder& forms) const override
    {
        const auto marker = context.sharedRegion("interface", "right").marker;
        CouplingFormContribution contribution;
        contribution.contribution_name = "generic_interface_coupling";
        contribution.origin = "GenericInterfaceContract";
        contribution.operator_name = "equations";
        contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
        contribution.extra_trial_field_uses = {{
            .participant_name = "left",
            .field_name = "primary",
        }};
        contribution.residual =
            (forms.state("left", "primary", "a") *
             forms.test("right", "primary", "w")).dI(marker);
        return {contribution};
    }
};

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

TEST(MonolithicCouplingBuilder, ResolvesDeclaredFormInstallOptions)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "declared_options";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{
        .participant_name = "left",
        .field_name = "mesh_displacement",
    }};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();

    svmp::FE::forms::JITOptions jit_options;
    jit_options.enable = false;
    jit_options.optimization_level = 1;
    contribution.install_options_declaration.ad_mode =
        svmp::FE::forms::ADMode::Reverse;
    contribution.install_options_declaration.compiler_options.jit = jit_options;
    contribution.install_options_declaration.compiler_options.simplify_expressions =
        false;
    contribution.install_options_declaration.compiler_options.verbose = true;
    contribution.install_options_declaration.geometry_sensitivity =
        CouplingGeometrySensitivityDeclaration{
            .mode = svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns,
            .mesh_motion_field = CouplingFieldUse{
                .participant_name = "left",
                .field_name = "mesh_displacement",
            },
            .tangent_path =
                svmp::FE::forms::GeometryTangentPath::SymbolicRequired,
            .use_symbolic_tangent = true,
        };
    fixture.system.bindMeshMotionField(
        systems::MeshMotionFieldRole::Displacement,
        fixture.mesh_motion_field);

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    EXPECT_EQ(resolved.install_options.ad_mode,
              svmp::FE::forms::ADMode::Reverse);
    EXPECT_EQ(resolved.install_options.compiler_options.ad_mode,
              svmp::FE::forms::ADMode::None);
    EXPECT_FALSE(resolved.install_options.compiler_options.jit.enable);
    EXPECT_EQ(resolved.install_options.compiler_options.jit.optimization_level, 1);
    EXPECT_FALSE(resolved.install_options.compiler_options.simplify_expressions);
    EXPECT_TRUE(resolved.install_options.compiler_options.verbose);
    EXPECT_EQ(resolved.install_options.compiler_options.geometry_sensitivity.mode,
              svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns);
    EXPECT_EQ(resolved.install_options.compiler_options.geometry_sensitivity
                  .mesh_motion_field,
              fixture.mesh_motion_field);
    EXPECT_EQ(resolved.install_options.compiler_options.geometry_tangent_path,
              svmp::FE::forms::GeometryTangentPath::SymbolicRequired);
    EXPECT_TRUE(resolved.install_options.compiler_options.use_symbolic_tangent);
}

TEST(MonolithicCouplingBuilder, RequiresMeshMotionSensitivityFieldUse)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "missing_mesh_motion_trial";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();
    contribution.install_options_declaration.geometry_sensitivity =
        CouplingGeometrySensitivityDeclaration{
            .mode = svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns,
            .mesh_motion_field = CouplingFieldUse{
                .participant_name = "left",
                .field_name = "mesh_displacement",
            },
        };
    fixture.system.bindMeshMotionField(
        systems::MeshMotionFieldRole::Displacement,
        fixture.mesh_motion_field);

    EXPECT_THROW(static_cast<void>(
                     builder.resolveFormContribution(fixture.context,
                                                     contribution)),
                 InvalidArgumentException);
}

TEST(MonolithicCouplingBuilder, RejectsInvalidGeometryConstantTangentPolicies)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    auto make_contribution = [&]() {
        CouplingFormContribution contribution;
        contribution.contribution_name = "geometry_constant_policy";
        contribution.origin = "MonolithicCouplingBuilderTest";
        contribution.field_uses = {{.participant_name = "right",
                                    .field_name = "primary"}};
        contribution.residual =
            (forms.state("right", "primary", "u") *
             forms.test("right", "primary", "w")).dx();
        contribution.install_options_declaration.geometry_sensitivity =
            CouplingGeometrySensitivityDeclaration{
                .mode = svmp::FE::forms::GeometrySensitivityMode::GeometryConstant,
            };
        return contribution;
    };

    auto mesh_field = make_contribution();
    mesh_field.install_options_declaration.geometry_sensitivity->mesh_motion_field =
        CouplingFieldUse{
            .participant_name = "left",
            .field_name = "mesh_displacement",
        };
    EXPECT_THROW(static_cast<void>(
                     builder.resolveFormContribution(fixture.context,
                                                     mesh_field)),
                 InvalidArgumentException);

    auto symbolic_required = make_contribution();
    symbolic_required.install_options_declaration.geometry_sensitivity
        ->tangent_path =
        svmp::FE::forms::GeometryTangentPath::SymbolicRequired;
    EXPECT_THROW(static_cast<void>(
                     builder.resolveFormContribution(fixture.context,
                                                     symbolic_required)),
                 InvalidArgumentException);

    auto symbolic_check = make_contribution();
    symbolic_check.install_options_declaration.geometry_sensitivity->tangent_path =
        svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck;
    EXPECT_THROW(static_cast<void>(
                     builder.resolveFormContribution(fixture.context,
                                                     symbolic_check)),
                 InvalidArgumentException);

    auto ordinary_symbolic = make_contribution();
    ordinary_symbolic.install_options_declaration.geometry_sensitivity
        ->use_symbolic_tangent = true;
    const auto resolved = builder.resolveFormContribution(fixture.context,
                                                          ordinary_symbolic);
    EXPECT_TRUE(resolved.install_options.compiler_options.use_symbolic_tangent);
}

TEST(MonolithicCouplingBuilder, RequiresMeshMotionSensitivityBinding)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "missing_mesh_motion_binding";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{
        .participant_name = "left",
        .field_name = "mesh_displacement",
    }};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();
    contribution.install_options_declaration.geometry_sensitivity =
        CouplingGeometrySensitivityDeclaration{
            .mode = svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns,
            .mesh_motion_field = CouplingFieldUse{
                .participant_name = "left",
                .field_name = "mesh_displacement",
            },
        };

    EXPECT_THROW(static_cast<void>(
                     builder.resolveFormContribution(fixture.context,
                                                     contribution)),
                 InvalidArgumentException);
}

TEST(MonolithicCouplingBuilder, ForcesSymbolicTangentForMeshMotionGeometryPath)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "mesh_motion_symbolic_path";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{
        .participant_name = "left",
        .field_name = "mesh_displacement",
    }};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();
    contribution.install_options_declaration.geometry_sensitivity =
        CouplingGeometrySensitivityDeclaration{
            .mode = svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns,
            .mesh_motion_field = CouplingFieldUse{
                .participant_name = "left",
                .field_name = "mesh_displacement",
            },
            .tangent_path =
                svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck,
            .use_symbolic_tangent = false,
        };
    fixture.system.bindMeshMotionField(
        systems::MeshMotionFieldRole::Displacement,
        fixture.mesh_motion_field);

    const auto resolved = builder.resolveFormContribution(fixture.context,
                                                          contribution);
    EXPECT_EQ(resolved.install_options.compiler_options.geometry_tangent_path,
              svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck);
    EXPECT_TRUE(resolved.install_options.compiler_options.use_symbolic_tangent);
}

TEST(MonolithicCouplingBuilder, InstallsMeshMotionGeometrySensitivityProvenance)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "mesh_motion_sensitivity";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.extra_trial_field_uses = {{
        .participant_name = "left",
        .field_name = "mesh_displacement",
    }};
    contribution.residual =
        (forms.state("right", "primary", "u") *
         forms.test("right", "primary", "w")).dx();
    contribution.install_options_declaration.geometry_sensitivity =
        CouplingGeometrySensitivityDeclaration{
            .mode = svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns,
            .mesh_motion_field = CouplingFieldUse{
                .participant_name = "left",
                .field_name = "mesh_displacement",
            },
        };
    fixture.system.bindMeshMotionField(
        systems::MeshMotionFieldRole::Displacement,
        fixture.mesh_motion_field);

    const auto resolved = builder.resolveFormContribution(fixture.context,
                                                          contribution);
    const auto metadata =
        builder.installResolvedFormContribution(fixture.system, resolved);

    EXPECT_EQ(metadata.geometry_sensitivity.mode,
              svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns);
    EXPECT_EQ(metadata.geometry_sensitivity.mesh_motion_field,
              fixture.mesh_motion_field);
    const auto field_it = std::find_if(
        metadata.field_uses.begin(),
        metadata.field_uses.end(),
        [&](const CouplingFormFieldProvenance& field) {
            return field.field == fixture.mesh_motion_field &&
                   field.appears_as_geometry_sensitivity;
        });
    EXPECT_NE(field_it, metadata.field_uses.end());

    const auto provenance_it = std::find_if(
        metadata.geometry_sensitivity_provenance.begin(),
        metadata.geometry_sensitivity_provenance.end(),
        [&](const CouplingGeometrySensitivityProvenance& provenance) {
            return provenance.kind ==
                       CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns &&
                   provenance.mesh_motion_field == fixture.mesh_motion_field;
        });
    EXPECT_NE(provenance_it,
              metadata.geometry_sensitivity_provenance.end());
}

TEST(MonolithicCouplingBuilder, RejectsRawFormInstallOptionOverrides)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    auto make_contribution = [&]() {
        CouplingFormContribution contribution;
        contribution.contribution_name = "raw_options";
        contribution.origin = "MonolithicCouplingBuilderTest";
        contribution.field_uses = {{.participant_name = "right",
                                    .field_name = "primary"}};
        contribution.residual =
            (forms.state("right", "primary", "u") *
             forms.test("right", "primary", "w")).dx();
        return contribution;
    };

    auto expect_rejected = [&](CouplingFormContribution contribution) {
        EXPECT_THROW(static_cast<void>(
                         builder.resolveFormContribution(fixture.context,
                                                         contribution)),
                     InvalidArgumentException);
    };

    auto raw_top_level_ad = make_contribution();
    raw_top_level_ad.install_options.ad_mode =
        svmp::FE::forms::ADMode::Reverse;
    expect_rejected(std::move(raw_top_level_ad));

    auto raw_compiler_ad = make_contribution();
    raw_compiler_ad.install_options.compiler_options.ad_mode =
        svmp::FE::forms::ADMode::Forward;
    expect_rejected(std::move(raw_compiler_ad));

    auto raw_symbolic_tangent = make_contribution();
    raw_symbolic_tangent.install_options.compiler_options.use_symbolic_tangent =
        true;
    expect_rejected(std::move(raw_symbolic_tangent));

    auto raw_geometry_tangent = make_contribution();
    raw_geometry_tangent.install_options.compiler_options.geometry_tangent_path =
        svmp::FE::forms::GeometryTangentPath::SymbolicRequired;
    expect_rejected(std::move(raw_geometry_tangent));

    auto raw_geometry_sensitivity = make_contribution();
    raw_geometry_sensitivity.install_options.compiler_options
        .geometry_sensitivity.mode =
        svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
    raw_geometry_sensitivity.install_options.compiler_options
        .geometry_sensitivity.mesh_motion_field = fixture.left_field;
    expect_rejected(std::move(raw_geometry_sensitivity));

    auto raw_extra_trial_field = make_contribution();
    raw_extra_trial_field.install_options.extra_trial_fields.push_back(
        fixture.left_field);
    expect_rejected(std::move(raw_extra_trial_field));
}

TEST(MonolithicCouplingBuilder, AdaptsNativeBridgeMetadataProvenance)
{
    analysis::FormContributionAnalysisMetadata native;
    native.contribution_name = "bridge_native";
    native.origin = "bridge_test";
    native.system_name = "fluid_system";
    native.operator_tag = "equations";
    native.installed_fields = {1, 9};
    native.geometry_sensitivity.mode =
        svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
    native.geometry_sensitivity.mesh_motion_field = 9;

    analysis::FormTerminalMetadata state_terminal;
    state_terminal.kind = analysis::FormTerminalKind::StateField;
    state_terminal.field_id = 1;
    state_terminal.owner_system_name = "fluid_system";
    state_terminal.owner_participant_name = "fluid";
    native.terminals.push_back(state_terminal);

    analysis::FormTerminalMetadata parameter_terminal;
    parameter_terminal.kind = analysis::FormTerminalKind::ParameterSymbol;
    parameter_terminal.symbol_name = "penalty";
    parameter_terminal.domain = analysis::DomainKind::Boundary;
    parameter_terminal.boundary_marker = 12;
    parameter_terminal.owner_system_name = "fluid_system";
    parameter_terminal.owner_participant_name = "fluid";
    native.terminals.push_back(parameter_terminal);

    native.installed_dependencies.push_back(
        analysis::FormInstalledDependencyMetadata{
            .residual_row = analysis::VariableKey::field(1),
            .dependency = analysis::VariableKey::named(
                analysis::VariableKind::GlobalScalar, "lambda"),
            .domain = analysis::DomainKind::Global,
            .contributes_matrix_block = true,
            .contributes_vector = true,
            .provider = "analysis",
        });
    native.installed_blocks.push_back(analysis::FormInstalledBlockMetadata{
        .residual_row = analysis::VariableKey::field(1),
        .dependency = analysis::VariableKey::named(
            analysis::VariableKind::GlobalScalar, "lambda"),
        .domains = {analysis::DomainKind::Global},
        .has_matrix = true,
        .has_vector = true,
        .provider = "analysis",
    });

    const auto adapted =
        MonolithicCouplingBuilder::adaptFormAnalysisMetadata(native);

    EXPECT_EQ(adapted.contribution_name, "bridge_native");
    EXPECT_EQ(adapted.origin, "bridge_test");
    EXPECT_EQ(adapted.system_name, "fluid_system");
    ASSERT_EQ(adapted.field_uses.size(), 2u);
    const auto state_field = std::find_if(
        adapted.field_uses.begin(),
        adapted.field_uses.end(),
        [](const CouplingFormFieldProvenance& field) {
            return field.field == 1;
        });
    ASSERT_NE(state_field, adapted.field_uses.end());
    EXPECT_TRUE(state_field->appears_as_state_field);
    const auto geometry_field = std::find_if(
        adapted.field_uses.begin(),
        adapted.field_uses.end(),
        [](const CouplingFormFieldProvenance& field) {
            return field.field == 9;
        });
    ASSERT_NE(geometry_field, adapted.field_uses.end());
    EXPECT_TRUE(geometry_field->appears_as_geometry_sensitivity);

    ASSERT_EQ(adapted.non_field_dependencies.size(), 1u);
    EXPECT_EQ(adapted.non_field_dependencies[0].kind,
              CouplingFormNonFieldDependencyKind::Parameter);
    EXPECT_EQ(adapted.non_field_dependencies[0].name, "penalty");
    EXPECT_EQ(adapted.non_field_dependencies[0].marker, 12);

    ASSERT_EQ(adapted.geometry_sensitivity_provenance.size(), 1u);
    EXPECT_EQ(adapted.geometry_sensitivity_provenance[0].kind,
              CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns);
    EXPECT_EQ(adapted.geometry_sensitivity_provenance[0].mesh_motion_field, 9);
    ASSERT_EQ(adapted.geometry_sensitivity_provenance[0].geometry_fields.size(),
              1u);
    EXPECT_EQ(adapted.geometry_sensitivity_provenance[0].geometry_fields[0], 9);

    ASSERT_EQ(adapted.installed_dependencies.size(), 1u);
    EXPECT_TRUE(adapted.installed_dependencies[0].contributes_matrix_block);
    EXPECT_EQ(adapted.installed_dependencies[0].dependency,
              analysis::VariableKey::named(analysis::VariableKind::GlobalScalar,
                                           "lambda"));
    ASSERT_EQ(adapted.installed_blocks.size(), 1u);
    EXPECT_TRUE(adapted.installed_blocks[0].has_matrix);
    EXPECT_EQ(adapted.installed_blocks[0].domains[0],
              analysis::DomainKind::Global);
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

TEST(MonolithicCouplingBuilder, RejectsPreviousSolutionProvenanceOutsideTrialFields)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "bad_history_owner";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.terminal_provenance.push_back(CouplingFormTerminalProvenanceDeclaration{
        .kind = CouplingFormTerminalProvenanceKind::PreviousSolution,
        .terminal_sequence = 1,
        .field = CouplingFieldUse{
            .participant_name = "left",
            .field_name = "primary",
        },
        .temporal_quantity = CouplingTemporalQuantity::FieldHistoryValue,
        .history_index = 1,
    });
    contribution.residual =
        (forms.previousSolution("left", "primary", 1) *
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
    contribution.terminal_provenance.push_back(CouplingFormTerminalProvenanceDeclaration{
        .kind = CouplingFormTerminalProvenanceKind::PreviousSolution,
        .terminal_sequence = 1,
        .field = CouplingFieldUse{
            .participant_name = "left",
            .field_name = "primary",
        },
        .temporal_quantity = CouplingTemporalQuantity::FieldHistoryValue,
        .history_index = 1,
    });
    contribution.residual =
        (forms.state("left", "primary", "a") *
         forms.test("right", "primary", "w")).dx();

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    ASSERT_EQ(resolved.terminal_provenance.size(), 1u);
    EXPECT_EQ(resolved.terminal_provenance[0].terminal_sequence, 1u);

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
    ASSERT_EQ(metadata.declaration_terminal_provenance.size(), 1u);
    EXPECT_EQ(metadata.declaration_terminal_provenance[0].kind,
              CouplingFormTerminalProvenanceKind::PreviousSolution);
    EXPECT_EQ(metadata.declaration_terminal_provenance[0].history_index, 1);
    const auto right_field_use = std::find_if(
        metadata.field_uses.begin(),
        metadata.field_uses.end(),
        [&](const CouplingFormFieldProvenance& field) {
            return field.field == fixture.right_field;
        });
    ASSERT_NE(right_field_use, metadata.field_uses.end());
    EXPECT_TRUE(right_field_use->appears_as_test_field);
    const auto left_field_use = std::find_if(
        metadata.field_uses.begin(),
        metadata.field_uses.end(),
        [&](const CouplingFormFieldProvenance& field) {
            return field.field == fixture.left_field;
        });
    ASSERT_NE(left_field_use, metadata.field_uses.end());
    EXPECT_TRUE(left_field_use->appears_as_state_field);

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

TEST(MonolithicCouplingBuilder, ResolvesGeometryTerminalProvenanceMetadata)
{
    BuilderFixture fixture;
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const CouplingFormBuilder forms(context);
    const MonolithicCouplingBuilder builder;

    const CouplingGeometryTerminalScope scope{
        .participant_name = "left",
        .region = CouplingRegionEndpointDeclaration{
            .participant_name = "left",
            .shared_region_name = "interface",
        },
        .location = CouplingGeometryTerminalLocationDeclaration{
            .region_kind = CouplingRegionKind::Boundary,
            .shared_region_name = "interface",
            .side = CouplingInterfaceSide::Plus,
            .coordinate_configuration = forms::GeometryConfiguration::Current,
            .transform_from_configuration =
                forms::GeometryConfiguration::Reference,
            .transform_to_configuration = forms::GeometryConfiguration::Current,
        },
    };

    CouplingFormContribution contribution;
    contribution.contribution_name = "interface_normal";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.residual =
        (forms.geometryTerminal(CouplingGeometryTerminalQuantity::CurrentNormal,
                                scope)
             .component(0) *
         forms.test("right", "primary", "w"))
            .dI(kInterfaceMarker);
    contribution = forms.attachTerminalProvenance(std::move(contribution));

    const auto resolved = builder.resolveFormContribution(context, contribution);
    ASSERT_EQ(resolved.geometry_terminals.size(), 1u);
    const auto& terminal = resolved.geometry_terminals[0];
    EXPECT_EQ(terminal.quantity, CouplingGeometryTerminalQuantity::CurrentNormal);
    EXPECT_EQ(terminal.analysis_domain, analysis::DomainKind::InterfaceFace);
    EXPECT_TRUE(terminal.normal_available);
    EXPECT_EQ(terminal.provider, "forms");
    EXPECT_EQ(terminal.location.region_kind, CouplingRegionKind::InterfaceFace);
    ASSERT_TRUE(terminal.location.shared_region_name.has_value());
    EXPECT_EQ(*terminal.location.shared_region_name, "interface");
    EXPECT_EQ(terminal.location.marker, kInterfaceMarker);
    EXPECT_EQ(terminal.location.side, CouplingInterfaceSide::Minus);
    EXPECT_EQ(terminal.location.coordinate_configuration,
              forms::GeometryConfiguration::Current);
    ASSERT_TRUE(terminal.location.transform_from_configuration.has_value());
    EXPECT_EQ(*terminal.location.transform_from_configuration,
              forms::GeometryConfiguration::Reference);
    ASSERT_TRUE(terminal.location.transform_to_configuration.has_value());
    EXPECT_EQ(*terminal.location.transform_to_configuration,
              forms::GeometryConfiguration::Current);
    ASSERT_TRUE(terminal.owner.has_value());
    EXPECT_EQ(terminal.owner->participant_name, "left");
    EXPECT_EQ(terminal.owner->system_name, "shared_system");
    ASSERT_TRUE(terminal.owner->region_name.has_value());
    EXPECT_EQ(*terminal.owner->region_name, "interface");
    ASSERT_TRUE(terminal.owner->shared_region_name.has_value());
    EXPECT_EQ(*terminal.owner->shared_region_name, "interface");

    const auto metadata =
        builder.installResolvedFormContribution(fixture.system, resolved);
    const auto metadata_terminal = std::find_if(
        metadata.geometry_terminals.begin(),
        metadata.geometry_terminals.end(),
        [](const CouplingFormGeometryTerminalProvenance& provenance) {
            return provenance.quantity ==
                       CouplingGeometryTerminalQuantity::CurrentNormal &&
                   provenance.owner.has_value() &&
                   provenance.owner->participant_name == "left";
        });
    ASSERT_NE(metadata_terminal, metadata.geometry_terminals.end());
    EXPECT_EQ(metadata_terminal->analysis_domain,
              analysis::DomainKind::InterfaceFace);
    ASSERT_TRUE(metadata_terminal->owner.has_value());
    EXPECT_EQ(metadata_terminal->owner->participant_name, "left");
}

TEST(MonolithicCouplingBuilder, RejectsBridgeMetadataForUndeclaredStateField)
{
    BuilderFixture fixture;
    const CouplingFormBuilder forms(fixture.context);
    const MonolithicCouplingBuilder builder;

    CouplingFormContribution contribution;
    contribution.contribution_name = "undeclared_state";
    contribution.origin = "MonolithicCouplingBuilderTest";
    contribution.field_uses = {{.participant_name = "right", .field_name = "primary"}};
    contribution.residual =
        (forms.state("left", "primary", "a") *
         forms.test("right", "primary", "w")).dx();

    const auto resolved = builder.resolveFormContribution(fixture.context, contribution);
    EXPECT_THROW(static_cast<void>(
                     builder.installResolvedFormContribution(fixture.system,
                                                             resolved)),
                 InvalidArgumentException);
}

TEST(MonolithicCouplingBuilder, RegistersContractOwnedInterfaceAdditionalFields)
{
    BuilderFixture fixture;
    fixture.system.setInterfaceMesh(kInterfaceMarker,
                                    std::make_shared<const svmp::InterfaceMesh>());
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const MonolithicCouplingBuilder builder;

    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .space = fixture.space,
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .shared_region_name = "interface",
    });

    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = builder.validateDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto resolved = builder.registerAdditionalFields(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(resolved.size(), 1u);
    EXPECT_EQ(resolved[0].system_name, "shared_system");
    EXPECT_EQ(resolved[0].field_spec.name, "generic_instance.lambda");
    EXPECT_EQ(resolved[0].field_spec.scope, systems::FieldScope::InterfaceFace);
    EXPECT_EQ(resolved[0].field_spec.interface_marker, kInterfaceMarker);
    ASSERT_NE(resolved[0].field_id, INVALID_FIELD_ID);
    EXPECT_EQ(fixture.system.findFieldByName("generic_instance.lambda"),
              resolved[0].field_id);
    EXPECT_TRUE(fixture.system.hasField("generic_instance.lambda"));
}

TEST(MonolithicCouplingBuilder, SkipsDisabledOptionalAdditionalFields)
{
    BuilderFixture fixture;
    const MonolithicCouplingBuilder builder;

    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .requirement = CouplingRequirement::Optional,
        .enabled = false,
    });

    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = builder.validateDeclarations(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto resolved = builder.resolveAdditionalFields(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations));
    EXPECT_TRUE(resolved.empty());

    const auto registered = builder.registerAdditionalFields(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations));
    EXPECT_TRUE(registered.empty());
    EXPECT_FALSE(fixture.system.hasField("generic_instance.lambda"));
}

TEST(MonolithicCouplingBuilder, GenericContractInstallsAndFinalizesGraph)
{
    BuilderFixture fixture;
    const GenericTwoParticipantContract contract;
    const MonolithicCouplingBuilder builder;
    const CouplingFormBuilder forms(fixture.context);

    const std::array<CouplingContractDeclaration, 1> declarations{contract.declare()};
    const auto declaration_validation = builder.validateDeclarations(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(declaration_validation.ok()) << formatDiagnostics(declaration_validation);

    const auto contributions = contract.buildMonolithicForms(fixture.context, forms);
    const auto installed = builder.installFormContributions(
        fixture.system,
        fixture.context,
        std::span<const CouplingFormContribution>(contributions));

    CouplingGraph graph;
    const auto finalized_validation = graph.buildFinalizedGraph(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed));

    EXPECT_TRUE(finalized_validation.ok()) << formatDiagnostics(finalized_validation);
    ASSERT_EQ(installed.size(), 1u);
    EXPECT_NE(findDependency(installed[0],
                             analysis::VariableKey::field(fixture.right_field),
                             analysis::VariableKey::field(fixture.left_field)),
              nullptr);
}

TEST(MonolithicCouplingBuilder, GenericInterfaceContractInstallsAndFinalizesGraph)
{
    BuilderFixture fixture;
    fixture.system.setInterfaceMesh(kInterfaceMarker,
                                    std::make_shared<const svmp::InterfaceMesh>());
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const GenericInterfaceContract contract;
    const MonolithicCouplingBuilder builder;
    const CouplingFormBuilder forms(context);

    const std::array<CouplingContractDeclaration, 1> declarations{contract.declare()};
    const auto declaration_validation = builder.validateDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(declaration_validation.ok()) << formatDiagnostics(declaration_validation);

    const auto contributions = contract.buildMonolithicForms(context, forms);
    const auto installed = builder.installFormContributions(
        fixture.system,
        context,
        std::span<const CouplingFormContribution>(contributions));

    CouplingGraph graph;
    const auto finalized_validation = graph.buildFinalizedGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed));

    EXPECT_TRUE(finalized_validation.ok()) << formatDiagnostics(finalized_validation);
    ASSERT_EQ(installed.size(), 1u);
    const auto* dependency = findDependency(
        installed[0],
        analysis::VariableKey::field(fixture.right_field),
        analysis::VariableKey::field(fixture.left_field));
    ASSERT_NE(dependency, nullptr);
    EXPECT_EQ(dependency->domain, analysis::DomainKind::InterfaceFace);
}

TEST(MonolithicCouplingBuilder, RejectsInterfaceContractWithoutRegisteredTopology)
{
    BuilderFixture fixture;
    const auto context = interfaceContext(fixture, kInterfaceMarker);
    const GenericInterfaceContract contract;
    const MonolithicCouplingBuilder builder;

    const std::array<CouplingContractDeclaration, 1> declarations{contract.declare()};
    const auto validation = builder.validateDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "interface-face coupling region is missing registered interface topology"),
              std::string::npos);
}
