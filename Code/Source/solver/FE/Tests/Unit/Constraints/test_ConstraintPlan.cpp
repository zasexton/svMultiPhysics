/**
 * @file test_ConstraintPlan.cpp
 * @brief Phase-ordered registration of system constraints
 */

#include <gtest/gtest.h>

#include "Constraints/ConstraintPlan.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
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

} // namespace
} // namespace svmp::FE::constraints::test
