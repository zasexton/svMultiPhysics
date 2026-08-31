/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constraints/CoupledFieldGaugeConstraint.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

std::shared_ptr<Mesh> buildSingleQuad()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> coordinates = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<offset_t> offsets = {0, 4};
    const std::vector<index_t> vertices = {0, 1, 2, 3};
    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(2, coordinates, offsets, vertices, {shape});
    base->finalize();
    return create_mesh(std::move(base));
}

class WholeFieldConstraint final : public ISystemConstraint {
public:
    explicit WholeFieldConstraint(FieldId field) : field_(field) {}

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override
    {
        const auto offset = system.fieldDofOffset(field_);
        const auto count = system.fieldDofHandler(field_).getNumDofs();
        const auto& owned = system.dofHandler().getPartition().locallyOwned();
        for (GlobalIndex local = 0; local < count; ++local) {
            const auto dof = offset + local;
            if (owned.contains(dof)) {
                constraints.addDirichlet(dof, Real{0.0});
            }
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
        return {};
    }

private:
    FieldId field_{INVALID_FIELD_ID};
};

struct TwoScalarFields {
    FieldId first{INVALID_FIELD_ID};
    FieldId second{INVALID_FIELD_ID};
};

TwoScalarFields addScalarFields(systems::FESystem& system)
{
    auto space = std::make_shared<spaces::H1Space>(
        ElementType::Quad4, 1);
    const auto first = system.addField(
        systems::FieldSpec{
            .name = "p_first", .space = space, .components = 1});
    const auto second = system.addField(
        systems::FieldSpec{
            .name = "p_second", .space = space, .components = 1});
    system.addOperator("coupled_pressure");
    return TwoScalarFields{first, second};
}

} // namespace

TEST(CoupledFieldGaugeConstraint, PinsOneCommonModeInFirstFreeField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    systems::FESystem system(buildSingleQuad());
    const auto fields = addScalarFields(system);
    auto gauge = std::make_unique<CoupledFieldGaugeConstraint>(
        fields.first, fields.second, Real{2.5});
    const auto* gauge_view = gauge.get();
    system.addSystemConstraint(std::move(gauge));

    ASSERT_NO_THROW(system.setup());
    ASSERT_TRUE(gauge_view->pinnedDof().has_value());
    const auto selected = *gauge_view->pinnedDof();
    EXPECT_EQ(selected, system.fieldDofOffset(fields.first));
    EXPECT_TRUE(system.constraints().isConstrained(selected));
    EXPECT_DOUBLE_EQ(system.constraints().getInhomogeneity(selected), 2.5);
    EXPECT_EQ(system.constraints().getConstrainedDofs().size(), 1u);
#endif
}

TEST(CoupledFieldGaugeConstraint, UsesSecondFieldWhenFirstHasNoFreeDof)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    systems::FESystem system(buildSingleQuad());
    const auto fields = addScalarFields(system);
    system.addSystemConstraint(
        std::make_unique<WholeFieldConstraint>(fields.first));
    auto gauge = std::make_unique<CoupledFieldGaugeConstraint>(
        fields.first, fields.second, Real{-3.0});
    const auto* gauge_view = gauge.get();
    system.addSystemConstraint(std::move(gauge));

    ASSERT_NO_THROW(system.setup());
    ASSERT_TRUE(gauge_view->pinnedDof().has_value());
    const auto selected = *gauge_view->pinnedDof();
    EXPECT_EQ(selected, system.fieldDofOffset(fields.second));
    EXPECT_TRUE(system.constraints().isConstrained(selected));
    EXPECT_DOUBLE_EQ(system.constraints().getInhomogeneity(selected), -3.0);
#endif
}

TEST(CoupledFieldGaugeConstraint, RejectsExhaustedFieldPair)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    systems::FESystem system(buildSingleQuad());
    const auto fields = addScalarFields(system);
    system.addSystemConstraint(
        std::make_unique<WholeFieldConstraint>(fields.first));
    system.addSystemConstraint(
        std::make_unique<WholeFieldConstraint>(fields.second));
    system.addSystemConstraint(
        std::make_unique<CoupledFieldGaugeConstraint>(
            fields.first, fields.second));

    EXPECT_THROW(system.setup(), std::runtime_error);
#endif
}

TEST(CoupledFieldGaugeConstraint, RejectsInvalidDefinition)
{
    EXPECT_THROW(
        CoupledFieldGaugeConstraint(INVALID_FIELD_ID, FieldId{0}),
        std::invalid_argument);
    EXPECT_THROW(
        CoupledFieldGaugeConstraint(FieldId{0}, FieldId{0}),
        std::invalid_argument);
    EXPECT_THROW(
        CoupledFieldGaugeConstraint(
            FieldId{0}, FieldId{1},
            std::numeric_limits<Real>::infinity()),
        std::invalid_argument);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
