/**
 * @file test_ConstraintPlan.cpp
 * @brief Phase-ordered registration of system constraints
 */

#include <gtest/gtest.h>

#include "Constraints/ConstraintPlan.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemManifest.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::constraints::test {
namespace {

/// A system constraint that only carries a label for the manifest.
class LabelConstraint final : public ISystemConstraint {
public:
    explicit LabelConstraint(std::string label)
        : label_(std::move(label))
    {
    }

    void apply(const systems::FESystem&, AffineConstraints&) override {}
    bool updateValues(const systems::FESystem&, AffineConstraints&, double, double) override
    {
        return false;
    }
    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }
    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override
    {
        return {};
    }
    [[nodiscard]] std::string manifestDescription() const override { return "label=" + label_; }

private:
    std::string label_;
};

struct PlanSystem {
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh{
        std::make_shared<forms::test::SingleTetraMeshAccess>()};
    std::shared_ptr<spaces::H1Space> scalar{
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1)};
    std::shared_ptr<spaces::ProductSpace> vector{
        std::make_shared<spaces::ProductSpace>(scalar, 3)};
    std::unique_ptr<systems::FESystem> system{};
    FieldId u{INVALID_FIELD_ID};
    FieldId p{INVALID_FIELD_ID};
    FieldId phi{INVALID_FIELD_ID};

    PlanSystem()
    {
        system = std::make_unique<systems::FESystem>(mesh);
        u = system->addField(systems::FieldSpec{.name = "u", .space = vector, .components = 3});
        p = system->addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
        phi = system->addField(systems::FieldSpec{.name = "phi", .space = scalar, .components = 1});
    }

    [[nodiscard]] std::vector<std::string> registered() const
    {
        std::vector<std::string> out;
        for (const auto& line : system->constraintManifestLines()) {
            if (line.rfind("constraint system#", 0) == 0) {
                out.push_back(line);
            }
        }
        return out;
    }

    [[nodiscard]] std::vector<std::string> labels() const
    {
        std::vector<std::string> out;
        for (const auto& line : registered()) {
            const auto at = line.find("label=");
            out.push_back(at == std::string::npos ? std::string("?") : line.substr(at + 6));
        }
        return out;
    }
};

std::unique_ptr<ISystemConstraint> label(std::string text)
{
    return std::make_unique<LabelConstraint>(std::move(text));
}

TEST(ConstraintPlan, CommitRegistersInPhaseThenDeclarationOrder)
{
    PlanSystem s;
    ConstraintPlan plan(*s.system);
    plan.declare(ConstraintPhase::PointLaw, label("point"))
        .declare(ConstraintPhase::Aggregation, label("aggregate_u"))
        .declare(ConstraintPhase::InactiveSupport, label("inactive_u"))
        .declare(ConstraintPhase::Aggregation, label("aggregate_p"))
        .declare(ConstraintPhase::InactiveSupport, label("inactive_p"))
        .declare(ConstraintPhase::VertexValues, label("vertex"));
    EXPECT_EQ(plan.pendingCount(), 6u);
    EXPECT_TRUE(s.registered().empty());

    plan.commit();
    EXPECT_EQ(plan.pendingCount(), 0u);
    EXPECT_EQ(plan.registeredCount(), 6u);
    EXPECT_EQ(plan.phase(), ConstraintPhase::PointLaw);
    EXPECT_EQ(s.labels(), (std::vector<std::string>{
                              "vertex", "inactive_u", "inactive_p", "aggregate_u", "aggregate_p", "point"}));
}

TEST(ConstraintPlan, EssentialsRunAfterTheEarlierPhasesAndBeforeTheLaterOnes)
{
    PlanSystem s;
    ConstraintPlan plan(*s.system);
    plan.declare(ConstraintPhase::InactiveSupport, label("inactive"));
    plan.declare(ConstraintPhase::VertexValues, label("vertex"));
    plan.commitThrough(ConstraintPhase::VertexValues);
    EXPECT_EQ(s.labels(), (std::vector<std::string>{"vertex"}));
    EXPECT_EQ(plan.pendingCount(), 1u);

    plan.essentials([](systems::FESystem& system) {
        system.addSystemConstraint(std::make_unique<LabelConstraint>("velocity_essential"));
    });
    plan.essentials([](systems::FESystem& system) {
        system.addSystemConstraint(std::make_unique<LabelConstraint>("pressure_essential"));
    });
    EXPECT_EQ(plan.phase(), ConstraintPhase::Essential);
    EXPECT_EQ(s.labels(),
              (std::vector<std::string>{"vertex", "velocity_essential", "pressure_essential"}));

    plan.declare(ConstraintPhase::PointLaw, label("point"));
    plan.commit();
    EXPECT_EQ(s.labels(), (std::vector<std::string>{
                              "vertex", "velocity_essential", "pressure_essential", "inactive", "point"}));
}

TEST(ConstraintPlan, PassedPhasesRejectDeclarations)
{
    PlanSystem s;
    ConstraintPlan plan(*s.system);
    plan.essentials([](systems::FESystem&) {});
    EXPECT_THROW(plan.declare(ConstraintPhase::VertexValues, label("late")), std::logic_error);
    plan.commitThrough(ConstraintPhase::Aggregation);
    EXPECT_THROW(plan.essentials([](systems::FESystem&) {}), std::logic_error);
    EXPECT_THROW(plan.declare(ConstraintPhase::InactiveSupport, label("late")), std::logic_error);
    EXPECT_NO_THROW(plan.declare(ConstraintPhase::Aggregation, label("same_phase")));
    EXPECT_THROW(plan.declare(ConstraintPhase::PointLaw, nullptr), std::invalid_argument);
}

TEST(ConstraintPlan, TypedResidualEssentialsMatchCallbackIncludingWeakTermsAndMetadata)
{
    PlanSystem typed;
    PlanSystem callback;
    const auto populate = [](systems::BoundaryConditionManager& manager) {
        manager.add(std::make_unique<forms::bc::EssentialBC>(2, forms::FormExpr::constant(3.0), "p"));
        manager.add(std::make_unique<forms::bc::NaturalBC>(4, forms::FormExpr::constant(5.0)));
        manager.add(std::make_unique<forms::bc::NaturalBC>(4, forms::FormExpr::constant(7.0)));
    };
    systems::BoundaryConditionManager typed_manager;
    systems::BoundaryConditionManager callback_manager;
    populate(typed_manager);
    populate(callback_manager);
    const auto state = forms::StateField(typed.p, *typed.scalar, "p");
    const auto test = forms::TestField(typed.p, *typed.scalar, "q");
    const auto seed = (state * test).dx();
    auto typed_residual = seed;
    auto callback_residual = seed;
    typed.system->addOperator("essential_op");
    callback.system->addOperator("essential_op");
    ConstraintPlan typed_plan(*typed.system);
    ConstraintPlan callback_plan(*callback.system);
    typed_plan.declare(ConstraintPhase::VertexValues, label("vertex"))
        .declare(ConstraintPhase::Essential, label("held_essential"))
        .declare(ConstraintPhase::InactiveSupport, label("inactive"));
    callback_plan.declare(ConstraintPhase::VertexValues, label("vertex"))
        .declare(ConstraintPhase::Essential, label("held_essential"))
        .declare(ConstraintPhase::InactiveSupport, label("inactive"));

    EXPECT_EQ(&typed_plan.essentials(typed_manager, typed_residual, state, test, typed.p, "essential_op"),
              &typed_plan);
    callback_plan.essentials([&](systems::FESystem& system) {
        callback_manager.applyAll(system, callback_residual, state, test, callback.p, "essential_op");
    });
    EXPECT_EQ(typed_residual.toString(), callback_residual.toString());
    EXPECT_EQ(typed_residual.node()->type(), forms::FormExprType::Subtract);
    EXPECT_EQ(typed_residual.node()->children()[0]->children()[0], seed.node());
    EXPECT_EQ(typed.registered(), callback.registered());
    ASSERT_EQ(typed.registered().size(), 4u);
    EXPECT_EQ(typed.labels()[0], "vertex");
    EXPECT_EQ(typed.labels()[1], "held_essential");
    EXPECT_EQ(typed_plan.phase(), ConstraintPhase::Essential);
    EXPECT_EQ(typed_plan.registeredCount(), 2u);
    EXPECT_EQ(typed_plan.pendingCount(), 1u);
    ASSERT_EQ(typed.system->boundaryConditionDescriptors().size(), 3u);
    const auto typed_tags = typed.system->boundaryConditionDescriptorOperatorTags();
    const auto callback_tags = callback.system->boundaryConditionDescriptorOperatorTags();
    EXPECT_EQ((std::vector<std::string>{typed_tags.begin(), typed_tags.end()}),
              (std::vector<std::string>{callback_tags.begin(), callback_tags.end()}));
    EXPECT_EQ((std::vector<std::string>{typed_tags.begin(), typed_tags.end()}),
              (std::vector<std::string>{"essential_op", "essential_op", "essential_op"}));
    const auto& descriptors = typed.system->boundaryConditionDescriptors();
    EXPECT_EQ(descriptors[0].boundary_marker, 2);
    EXPECT_EQ(descriptors[1].boundary_marker, 4);
    EXPECT_EQ(descriptors[2].boundary_marker, 4);
    EXPECT_EQ(descriptors[2].source, "NaturalBC on marker 4");
    EXPECT_EQ(typed.system->contributionDescriptors().size(), callback.system->contributionDescriptors().size());
    EXPECT_EQ(typed.system->gaugeRegistry().anchoring().size(), callback.system->gaugeRegistry().anchoring().size());
    typed_plan.commit();
    callback_plan.commit();
    EXPECT_EQ(typed.registered(), callback.registered());
    EXPECT_EQ(typed.labels().back(), "inactive");
}

TEST(ConstraintPlan, TypedFieldEssentialsMatchCallbackAndAllowSamePhaseOperations)
{
    PlanSystem typed;
    PlanSystem callback;
    ConstraintPlan typed_plan(*typed.system);
    ConstraintPlan callback_plan(*callback.system);
    for (const int marker : {3, 6}) {
        systems::BoundaryConditionManager typed_manager;
        systems::BoundaryConditionManager callback_manager;
        typed_manager.add(std::make_unique<forms::bc::EssentialBC>(marker, forms::FormExpr::constant(2.0), "p"));
        callback_manager.add(std::make_unique<forms::bc::EssentialBC>(marker, forms::FormExpr::constant(2.0), "p"));
        EXPECT_EQ(&typed_plan.essentials(typed_manager, typed.p), &typed_plan);
        callback_plan.essentials([&](systems::FESystem& system) { callback_manager.applyAll(system, callback.p); });
    }
    EXPECT_EQ(typed.registered(), callback.registered());
    ASSERT_EQ(typed.registered().size(), 4u);
    EXPECT_NE(typed.registered()[1].find("marker=3"), std::string::npos);
    EXPECT_NE(typed.registered()[3].find("marker=6"), std::string::npos);
    EXPECT_EQ(typed_plan.phase(), ConstraintPhase::Essential);
    EXPECT_EQ(typed_plan.registeredCount(), 0u);
    EXPECT_EQ(typed.system->boundaryConditionDescriptors().size(), 2u);
}

TEST(ConstraintPlan, TypedStrongBatchPreservesDeclarationOrderAndLateFailureProgress)
{
    PlanSystem typed;
    PlanSystem callback;
    const std::array<forms::bc::StrongDirichlet, 3> declarations{
        forms::bc::strongDirichlet(typed.u, 5, forms::FormExpr::constant(3.0), "u", 2),
        forms::bc::strongDirichlet(typed.u, 4, forms::FormExpr::constant(7.0), "u", 0),
        forms::bc::StrongDirichlet{},
    };
    ConstraintPlan typed_plan(*typed.system);
    ConstraintPlan callback_plan(*callback.system);
    typed_plan.declare(ConstraintPhase::VertexValues, label("vertex"));
    callback_plan.declare(ConstraintPhase::VertexValues, label("vertex"));
    EXPECT_THROW(typed_plan.essentials(declarations), InvalidArgumentException);
    EXPECT_THROW(callback_plan.essentials([&](systems::FESystem& system) {
        systems::installStrongDirichlet(system, declarations);
    }), InvalidArgumentException);
    EXPECT_EQ(typed.registered(), callback.registered());
    ASSERT_EQ(typed.registered().size(), 3u);
    EXPECT_EQ(typed.labels()[0], "vertex");
    EXPECT_NE(typed.registered()[1].find("marker=5 component=2"), std::string::npos);
    EXPECT_NE(typed.registered()[2].find("marker=4 component=0"), std::string::npos);
    EXPECT_EQ(typed_plan.registeredCount(), 1u);
    EXPECT_EQ(typed_plan.pendingCount(), 0u);
    EXPECT_EQ(typed_plan.phase(), ConstraintPhase::Essential);
    EXPECT_EQ(&typed_plan.essentials(std::span<const forms::bc::StrongDirichlet>{}), &typed_plan);
}

TEST(ConstraintPlan, TypedFieldWeakRejectionStillCommitsEarlierPlanConstraints)
{
    PlanSystem s;
    systems::BoundaryConditionManager manager;
    manager.add(std::make_unique<forms::bc::EssentialBC>(2, forms::FormExpr::constant(1.0), "p"));
    manager.add(std::make_unique<forms::bc::NaturalBC>(3, forms::FormExpr::constant(2.0)));
    ConstraintPlan plan(*s.system);
    plan.vertexValues(s.p, {{0, 9.0}})
        .declare(ConstraintPhase::Essential, label("held_essential"))
        .declare(ConstraintPhase::InactiveSupport, label("later"));
    EXPECT_THROW(plan.essentials(manager, s.p), std::invalid_argument);
    ASSERT_EQ(s.registered().size(), 2u);
    EXPECT_NE(s.registered()[0].find("VertexDirichletConstraint"), std::string::npos);
    EXPECT_EQ(s.labels()[1], "held_essential");
    EXPECT_EQ(plan.registeredCount(), 2u);
    EXPECT_EQ(plan.pendingCount(), 1u);
    EXPECT_EQ(plan.phase(), ConstraintPhase::Essential);
    EXPECT_TRUE(s.system->boundaryConditionDescriptors().empty());
    EXPECT_TRUE(s.system->contributionDescriptors().empty());
}

TEST(ConstraintPlan, TypedEssentialsPreserveMarkerConflictPrecheck)
{
    for (const bool strong_first : {false, true}) {
        PlanSystem s;
        systems::BoundaryConditionManager manager;
        auto strong = std::make_unique<forms::bc::EssentialBC>(2, forms::FormExpr::constant(1.0), "p");
        auto weak = std::make_unique<forms::bc::NaturalBC>(2, forms::FormExpr::constant(2.0));
        if (strong_first) {
            manager.add(std::move(strong));
            manager.add(std::move(weak));
        } else {
            manager.add(std::move(weak));
            manager.add(std::move(strong));
        }
        const auto state = forms::StateField(s.p, *s.scalar, "p");
        const auto test = forms::TestField(s.p, *s.scalar, "q");
        auto residual = (state * test).dx();
        const auto seed = residual.nodeShared();
        ConstraintPlan plan(*s.system);
        plan.declare(ConstraintPhase::VertexValues, label("vertex"));
        EXPECT_THROW(plan.essentials(manager, residual, state, test, s.p), std::invalid_argument);
        EXPECT_EQ(residual.nodeShared(), seed);
        EXPECT_EQ(s.labels(), (std::vector<std::string>{"vertex"}));
        EXPECT_TRUE(s.system->boundaryConditionDescriptors().empty());
        EXPECT_TRUE(s.system->contributionDescriptors().empty());
        EXPECT_EQ(plan.phase(), ConstraintPhase::Essential);
    }
}

TEST(ConstraintPlan, TypedEssentialsRejectPassedPhaseBeforeMutatingManagersOrBatches)
{
    PlanSystem s;
    ConstraintPlan plan(*s.system);
    plan.commitThrough(ConstraintPhase::InactiveSupport);
    systems::BoundaryConditionManager manager;
    manager.add(std::make_unique<forms::bc::EssentialBC>(2, forms::FormExpr::constant(1.0), "p"));
    const auto state = forms::StateField(s.p, *s.scalar, "p");
    const auto test = forms::TestField(s.p, *s.scalar, "q");
    auto residual = (state * test).dx();
    const auto seed = residual.nodeShared();
    const std::array declarations{forms::bc::strongDirichlet(s.p, 3, forms::FormExpr::constant(2.0))};
    EXPECT_THROW(plan.essentials(manager, s.p), std::logic_error);
    EXPECT_THROW(plan.essentials(manager, residual, state, test, s.p), std::logic_error);
    EXPECT_THROW(plan.essentials(declarations), std::logic_error);
    EXPECT_TRUE(s.registered().empty());
    EXPECT_TRUE(s.system->boundaryConditionDescriptors().empty());
    EXPECT_EQ(residual.nodeShared(), seed);
    // The manager remains usable because the rejected plan operations did not consume it.
    manager.applyAll(*s.system, s.p);
    EXPECT_EQ(s.registered().size(), 2u);
}

TEST(ConstraintPlan, UncommittedDeclarationsAreDropped)
{
    PlanSystem s;
    {
        ConstraintPlan plan(*s.system);
        plan.declare(ConstraintPhase::Aggregation, label("dropped"));
    }
    EXPECT_TRUE(s.registered().empty());
}

TEST(ConstraintPlan, LevelSetDeclarationsBuildTheExistingConstraints)
{
    PlanSystem s;
    const auto G = level_set::LevelSetGeometry::bind(
        *s.system, {.level_set_field = s.phi, .domain_id = "drop", .isovalue = 0.5});
    const auto L = G.side(forms::CutVolumeSide::Positive);
    SmallCutAggregationGuardOptions guards;
    guards.maximum_root_path_length = 6u;

    ContactPointVelocityLaw law;
    law.wall_boundary_markers = {2};
    law.level_set_field_name = "phi";
    law.contact_speed = [](Real cos_d) { return Real{0.5} * (Real{0.25} - cos_d); };
    law.name = "contact";

    ConstraintPlan plan(*s.system);
    plan.contactPointVelocity(s.u, law)
        .aggregate({s.u, s.p}, L, guards)
        .inactiveSide({s.u, s.p}, L);
    plan.commit();

    const auto lines = s.registered();
    ASSERT_EQ(lines.size(), 5u);
    const std::string marker = "interface_marker=" + std::to_string(G.interfaceMarker());
    for (std::size_t i = 0; i < 2u; ++i) {
        EXPECT_NE(lines[i].find("LevelSetActiveSideVertexDirichletConstraint"), std::string::npos) << lines[i];
        EXPECT_NE(lines[i].find("level_set='phi' active_side=Positive"), std::string::npos) << lines[i];
        EXPECT_NE(lines[i].find(marker), std::string::npos) << lines[i];
    }
    EXPECT_NE(lines[0].find("field=" + std::to_string(s.u)), std::string::npos);
    EXPECT_NE(lines[1].find("field=" + std::to_string(s.p)), std::string::npos);
    for (std::size_t i = 2; i < 4u; ++i) {
        EXPECT_NE(lines[i].find("SmallCutAggregationConstraint"), std::string::npos) << lines[i];
        EXPECT_NE(lines[i].find("active_side=Positive " + marker), std::string::npos) << lines[i];
        EXPECT_NE(lines[i].find("max_root_path=6"), std::string::npos) << lines[i];
    }
    EXPECT_NE(lines[4].find("ContactPointVelocityConstraint"), std::string::npos);
    EXPECT_NE(lines[4].find("name='contact'"), std::string::npos);
}

TEST(ConstraintPlan, ContactPointVelocityFromAGeometryBuildsTheSameLaw)
{
    const auto speed = [](Real cos_d) { return Real{0.5} * (Real{0.25} - cos_d); };

    PlanSystem explicit_law;
    ContactPointVelocityLaw law;
    law.wall_boundary_markers = {2};
    law.wall_normal = {0.0, -1.0, 0.0};
    law.level_set_field_name = "phi";
    law.isovalue = 0.5;
    law.contact_speed = speed;
    law.name = "ren_e_contact_point:2";
    ConstraintPlan(*explicit_law.system).contactPointVelocity(explicit_law.u, law).commit();

    PlanSystem from_geometry;
    const auto G = level_set::LevelSetGeometry::bind(
        *from_geometry.system, {.level_set_field = from_geometry.phi, .domain_id = "drop", .isovalue = 0.5});
    ConstraintPlan(*from_geometry.system)
        .contactPointVelocity(from_geometry.u, G, 2, {0.0, -1.0, 0.0}, speed, "ren_e_contact_point:2")
        .commit();

    ASSERT_EQ(explicit_law.registered().size(), 1u);
    EXPECT_EQ(from_geometry.registered(), explicit_law.registered());
}

TEST(ConstraintPlan, AnchorGaugeRecordsTheEvidenceOnceAndAddsTheMeasureKernel)
{
    PlanSystem s;
    s.system->addOperator("op");
    InterfaceMeasureGaugeAnchor anchor;
    anchor.field = s.p;
    anchor.interface_marker = 7;
    anchor.level_set_field_name = "phi";
    anchor.domain_id = "drop";
    anchor.texts.kernel_name = "UnitAnchor";
    InterfaceMeasureGaugeAnchor second = anchor;
    second.interface_marker = 8;

    // Two interfaces anchor the same field: one evidence record, and a kernel
    // and a cut-context callback for each interface.
    ConstraintPlan plan(*s.system);
    plan.anchorGauge("op", anchor).anchorGauge("op", second);

    std::size_t evidence = 0u;
    for (const auto& e : s.system->gaugeRegistry().anchoring()) {
        if (e.field == s.p && e.source == anchor.texts.anchor_source) {
            ++evidence;
        }
    }
    EXPECT_EQ(evidence, 1u);
    const auto count = [](const std::vector<std::string>& lines, const std::string& text) {
        std::size_t n = 0u;
        for (const auto& line : lines) {
            if (line.find(text) != std::string::npos) {
                ++n;
            }
        }
        return n;
    };
    const std::vector<std::string> kernel_lines = systems::kernelManifestLines(*s.system);
    EXPECT_EQ(count(kernel_lines, "name=UnitAnchor:7"), 1u);
    EXPECT_EQ(count(kernel_lines, "name=UnitAnchor:8"), 1u);
    const std::vector<std::string> constraint_lines = s.system->constraintManifestLines();
    EXPECT_EQ(count(constraint_lines, "name='interface_measure_gauge_anchor:7'"), 1u);
    EXPECT_EQ(count(constraint_lines, "name='interface_measure_gauge_anchor:8'"), 1u);

    // An interface is anchored once: the second anchor of marker 7 finds its
    // callback name registered.
    EXPECT_THROW(plan.anchorGauge("op", anchor), InvalidArgumentException);

    const InterfaceMeasureGaugeKernel kernel(anchor);
    EXPECT_EQ(kernel.name(), "UnitAnchor:7");
    try {
        kernel.validateContextUpdate(*s.system, nullptr);
        ADD_FAILURE() << "expected a missing-measure rejection";
    } catch (const std::runtime_error& error) {
        const std::string text = error.what();
        EXPECT_EQ(text.rfind("InterfaceMeasureGaugeAnchor: interface-measure gauge anchor requires a positive "
                             "generated interface measure before a solve validation_stage=cut_context_update "
                             "marker=7 level_set_field='phi' domain_id='drop' cut_context=missing",
                             0),
                  0u)
            << text;
        EXPECT_NE(text.find("gauge_policy=reject_zero_interface_no_dynamic_gauge_insertion"), std::string::npos);
    }
}

TEST(ConstraintPlan, AnchorKernelShowsItsDeclaredManifestTypeName)
{
    PlanSystem s;
    s.system->addOperator("op");
    InterfaceMeasureGaugeAnchor anchor;
    anchor.field = s.p;
    anchor.interface_marker = 7;
    anchor.texts.kernel_name = "UnitAnchor";
    InterfaceMeasureGaugeAnchor labelled = anchor;
    labelled.interface_marker = 8;
    labelled.texts.manifest_type_name = "unit::(anonymous namespace)::LabelledAnchor";

    ConstraintPlan plan(*s.system);
    plan.anchorGauge("op", anchor).anchorGauge("op", labelled);

    std::size_t fe_type = 0u;
    std::size_t label = 0u;
    for (const auto& line : systems::kernelManifestLines(*s.system)) {
        if (line.find("type=svmp::FE::constraints::InterfaceMeasureGaugeKernel name=UnitAnchor:7") !=
            std::string::npos) {
            ++fe_type;
        }
        if (line.find("type=unit::(anonymous namespace)::LabelledAnchor name=UnitAnchor:8") != std::string::npos) {
            ++label;
        }
    }
    EXPECT_EQ(fe_type, 1u);
    EXPECT_EQ(label, 1u);
}

} // namespace
} // namespace svmp::FE::constraints::test
