/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Interfaces/SnapshotFieldEvaluator.h"
#include "Basis/LagrangeBasis.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "Spaces/AdaptiveSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SnapshotFieldAccess.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::ProductSpace;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;

// The vertices of the single reference tetrahedron, which is also the mesh
// cell (SingleTetraMeshAccess): reference and physical points coincide.
const std::array<std::array<Real, 3>, 4> kVertices{{
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};

// A linear field f_c(x) = a_c + sum_j b_cj x_j for components c.
constexpr std::array<Real, 3> kOffset{0.25, -1.5, 2.0};
constexpr std::array<std::array<Real, 3>, 3> kSlope{{
    {1.0, -2.0, 0.5}, {0.75, 3.0, -1.25}, {-0.5, 0.125, 4.0}}};

Real linear(std::size_t component, const std::array<Real, 3>& x)
{
    return kOffset[component] + kSlope[component][0] * x[0] +
           kSlope[component][1] * x[1] + kSlope[component][2] * x[2];
}

struct Fixture {
    std::shared_ptr<H1Space> scalar_space =
        std::make_shared<H1Space>(ElementType::Tetra4, /*order=*/1);
    std::shared_ptr<ProductSpace> vector_space =
        std::make_shared<ProductSpace>(scalar_space, /*components=*/3);
    FESystem system{std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>()};
    FieldId velocity{};
    FieldId pressure{};
    std::vector<Real> solution{};

    Fixture()
    {
        velocity = system.addField(
            FieldSpec{.name = "u", .space = vector_space, .components = 3});
        pressure = system.addField(
            FieldSpec{.name = "p", .space = scalar_space, .components = 1});
        system.setup();
        solution.assign(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);

        // The vertex of each scalar basis function, found from the basis
        // itself (basis i is 1 at its vertex), so the test does not assume a
        // DOF order.
        std::array<std::size_t, 4> vertex_of{};
        for (std::size_t i = 0; i < 4; ++i) {
            std::vector<Real> unit(4, 0.0);
            unit[i] = 1.0;
            for (std::size_t v = 0; v < 4; ++v) {
                const svmp::FE::spaces::FunctionSpace::Value xi{
                    kVertices[v][0], kVertices[v][1], kVertices[v][2]};
                if (std::abs(scalar_space->evaluate(xi, unit)[0] - 1.0) < 1e-12) {
                    vertex_of[i] = v;
                }
            }
        }
        const auto velocity_dofs = system.fieldDofHandler(velocity).getCellDofs(0);
        const auto velocity_offset = system.fieldDofOffset(velocity);
        EXPECT_EQ(velocity_dofs.size(), 12u);
        for (std::size_t c = 0; c < 3; ++c) {
            for (std::size_t i = 0; i < 4; ++i) {
                solution[static_cast<std::size_t>(velocity_offset + velocity_dofs[c * 4 + i])] =
                    linear(c, kVertices[vertex_of[i]]);
            }
        }
        const auto pressure_dofs = system.fieldDofHandler(pressure).getCellDofs(0);
        const auto pressure_offset = system.fieldDofOffset(pressure);
        EXPECT_EQ(pressure_dofs.size(), 4u);
        for (std::size_t i = 0; i < 4; ++i) {
            solution[static_cast<std::size_t>(pressure_offset + pressure_dofs[i])] =
                linear(0, kVertices[vertex_of[i]]);
        }
    }
};

const std::vector<std::array<Real, 3>> kPoints{
    {0.25, 0.25, 0.25}, {0.1, 0.2, 0.3}, {0.6, 0.1, 0.05}, {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}};

std::uint64_t bitsOf(Real value)
{
    return std::bit_cast<std::uint64_t>(value);
}

} // namespace

TEST(SnapshotFieldEvaluator, FieldAccessReadsTheSystem)
{
    Fixture f;
    const auto access = svmp::FE::systems::snapshotFieldAccess(f.system, f.velocity);
    EXPECT_EQ(access.space, f.system.fieldRecord(f.velocity).space);
    EXPECT_EQ(access.components, 3);
    EXPECT_EQ(access.dof_offset, f.system.fieldDofOffset(f.velocity));
    EXPECT_EQ(&access.mesh(), &f.system.meshAccess());
    const auto dofs = access.cell_dofs(0);
    const auto expected = f.system.fieldDofHandler(f.velocity).getCellDofs(0);
    ASSERT_EQ(dofs.size(), expected.size());
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        EXPECT_EQ(dofs[i], expected[i]);
    }
}

TEST(SnapshotFieldEvaluator, VectorFieldValuesAndGradientsOfALinearField)
{
    Fixture f;
    const auto velocity = svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator(
        svmp::FE::systems::snapshotFieldAccess(f.system, f.velocity), f.solution, "unit");
    ASSERT_TRUE(velocity.canEvaluateValue());
    const svmp::FE::geometry::CutQuadratureProvenance provenance{};
    for (const auto& point : kPoints) {
        const auto value = velocity.value(0, point, provenance);
        const auto gradient = velocity.physical_gradient(0, point, provenance);
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(value[c], linear(c, point), 1e-13);
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_NEAR(gradient[c][j], kSlope[c][j], 1e-13);
            }
        }
    }
}

TEST(SnapshotFieldEvaluator, ScalarProductFastPathIsBitwiseTheGenericPath)
{
    Fixture f;
    const auto access = svmp::FE::systems::snapshotFieldAccess(f.system, f.velocity);
    const auto fast = svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator(
        access, f.solution, "unit", /*allow_scalar_product_fast_path=*/true);
    const auto generic = svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator(
        access, f.solution, "unit", /*allow_scalar_product_fast_path=*/false);
    const svmp::FE::geometry::CutQuadratureProvenance provenance{};
    for (const auto& point : kPoints) {
        const auto a = fast.value(0, point, provenance);
        const auto b = generic.value(0, point, provenance);
        const auto ga = fast.physical_gradient(0, point, provenance);
        const auto gb = generic.physical_gradient(0, point, provenance);
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_EQ(bitsOf(a[c]), bitsOf(b[c]));
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(bitsOf(ga[c][j]), bitsOf(gb[c][j]));
            }
        }
    }
}

TEST(SnapshotFieldEvaluator, ScalarFieldValuesAndReferenceGradients)
{
    Fixture f;
    const auto pressure = svmp::FE::interfaces::makeSnapshotScalarFieldEvaluator(
        svmp::FE::systems::snapshotFieldAccess(f.system, f.pressure), f.solution, "unit");
    ASSERT_TRUE(pressure.canEvaluateValue());
    const svmp::FE::geometry::CutQuadratureProvenance provenance{};
    for (const auto& point : kPoints) {
        EXPECT_NEAR(pressure.value(0, point, provenance), linear(0, point), 1e-13);
        const auto gradient = pressure.reference_gradient(0, point, provenance);
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(gradient[j], kSlope[0][j], 1e-13);
        }
    }
}

TEST(SnapshotFieldEvaluator, UsesTheParentCellOrderForScalarAndVectorTraces)
{
    using namespace svmp::FE;
    auto p1 = std::make_shared<H1Space>(ElementType::Tetra4, 1);
    auto p2 = std::make_shared<H1Space>(ElementType::Tetra4, 2);
    auto scalar_space = std::make_shared<spaces::AdaptiveSpace>();
    auto vector_space = std::make_shared<spaces::AdaptiveSpace>();
    scalar_space->add_level(1, p1);
    scalar_space->add_level(2, p2);
    vector_space->add_level(1, std::make_shared<ProductSpace>(p1, 3));
    vector_space->add_level(2, std::make_shared<ProductSpace>(p2, 3));
    for (const auto& space : {scalar_space, vector_space}) {
        space->resize_element_orders(2, 1);
        space->set_element_order(0, 2);
    }
    FESystem system(std::make_shared<forms::test::TwoTetraSharedFaceMeshAccess>());
    const auto scalar_id = system.addField({.name = "scalar", .space = scalar_space});
    const auto vector_id = system.addField({.name = "vector", .space = vector_space, .components = 3});
    system.setup();
    std::vector<Real> solution(system.dofHandler().getNumDofs(), 0.0);
    const auto& nodes = dynamic_cast<const basis::LagrangeBasis&>(p2->element().basis()).nodes();
    const auto scalar_rows = system.fieldDofHandler(scalar_id).getCellDofs(0);
    const auto vector_rows = system.fieldDofHandler(vector_id).getCellDofs(0);
    ASSERT_EQ(scalar_rows.size(), nodes.size());
    ASSERT_EQ(vector_rows.size(), 3 * nodes.size());
    // x(1-x-y-z) vanishes on the shared face. The P1 neighbor stays zero.
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& x = nodes[i];
        const Real value = x[0] * (1 - x[0] - x[1] - x[2]);
        solution[system.fieldDofOffset(scalar_id) + scalar_rows[i]] = value;
        for (int c = 0; c < 3; ++c) {
            solution[system.fieldDofOffset(vector_id) + vector_rows[c * nodes.size() + i]] = (c + 1) * value;
        }
    }
    const auto cell_field = level_set::makeLevelSetCellEvaluator(system, scalar_id, solution);
    const auto scalar = interfaces::makeSnapshotScalarFieldEvaluator(
        systems::snapshotFieldAccess(system, scalar_id), solution, "mixed-order test");
    const auto vector = interfaces::makeSnapshotVectorFieldEvaluator(
        systems::snapshotFieldAccess(system, vector_id), solution, "mixed-order test");
    constexpr std::array<Real, 3> point{0.25, 0.25, 0.125};
    const geometry::CutQuadratureProvenance provenance{};
    for (int cell : {0, 1, 0}) {
        SCOPED_TRACE(cell);
        const Real value = cell == 0 ? 3.0 / 32 : 0;
        const std::array<Real, 3> gradient = cell == 0 ? std::array<Real, 3>{0.125, -0.25, -0.25}
                                                       : std::array<Real, 3>{};
        const auto result = cell_field.evaluate(cell, point);
        EXPECT_EQ(result.interpolation_order, cell == 0 ? 2 : 1);
        EXPECT_DOUBLE_EQ(result.value, value);
        EXPECT_EQ(result.reference_gradient, gradient);
        EXPECT_DOUBLE_EQ(scalar.value(cell, point, provenance), value);
        EXPECT_EQ(scalar.reference_gradient(cell, point, provenance), gradient);
        const auto v = vector.value(cell, point, provenance);
        const auto grad_v = vector.physical_gradient(cell, point, provenance);
        for (int c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(v[c], (c + 1) * value);
            for (int d = 0; d < 3; ++d) EXPECT_DOUBLE_EQ(grad_v[c][d], (c + 1) * gradient[d]);
        }
    }
}

TEST(SnapshotFieldEvaluator, RejectsIncompatibleFieldsAndShortSolutions)
{
    Fixture f;
    const auto scalar = svmp::FE::systems::snapshotFieldAccess(f.system, f.pressure);
    try {
        (void)svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator(scalar, f.solution, "unit context");
        ADD_FAILURE() << "a scalar field was accepted as a velocity";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()),
                  "[svMultiPhysics::Application] unit context velocity field is "
                  "incompatible with its level-set declaration.");
    }
    const auto vector = svmp::FE::systems::snapshotFieldAccess(f.system, f.velocity);
    try {
        (void)svmp::FE::interfaces::makeSnapshotScalarFieldEvaluator(vector, f.solution, "unit context");
        ADD_FAILURE() << "a vector field was accepted as a scalar";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()),
                  "[svMultiPhysics::Application] unit context scalar field is "
                  "incompatible with its diagnostic declaration.");
    }

    const std::vector<Real> short_solution(1, 0.0);
    const auto velocity = svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator(
        vector, short_solution, "unit context");
    const svmp::FE::geometry::CutQuadratureProvenance provenance{};
    try {
        (void)velocity.value(0, kPoints[0], provenance);
        ADD_FAILURE() << "a short solution was read";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()),
                  "[svMultiPhysics::Application] unit context solution is too "
                  "small for the velocity field.");
    }
}
