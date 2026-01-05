/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/AssemblyContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Core/FEException.h"
#include "Spaces/H1Space.h"

#include "Dofs/DofHandler.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

class StateCountingKernel final : public assembly::AssemblyKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::Standard | assembly::RequiredData::MaterialState;
    }

    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return assembly::MaterialStateSpec{sizeof(int), alignof(int)};
    }

    void computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& /*output*/) override
    {
        last_values.clear();
        last_values.reserve(static_cast<std::size_t>(ctx.numQuadraturePoints()));

        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            auto bytes = ctx.materialState(q);
            auto* counter = reinterpret_cast<int*>(bytes.data());
            *counter += 1;
            last_values.push_back(*counter);
        }
    }

    mutable std::vector<int> last_values{};
};

class OldPlusOneKernel final : public assembly::AssemblyKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::Standard | assembly::RequiredData::MaterialState;
    }

    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return assembly::MaterialStateSpec{sizeof(int), alignof(int)};
    }

    void computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& /*output*/) override
    {
        last_old.clear();
        last_work.clear();
        last_old.reserve(static_cast<std::size_t>(ctx.numQuadraturePoints()));
        last_work.reserve(static_cast<std::size_t>(ctx.numQuadraturePoints()));

        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            auto old_bytes = ctx.materialStateOld(q);
            auto work_bytes = ctx.materialStateWork(q);

            const auto* old_ptr = reinterpret_cast<const int*>(old_bytes.data());
            auto* work_ptr = reinterpret_cast<int*>(work_bytes.data());

            *work_ptr = *old_ptr + 1;
            last_old.push_back(*old_ptr);
            last_work.push_back(*work_ptr);
        }
    }

    mutable std::vector<int> last_old{};
    mutable std::vector<int> last_work{};
};

class BoundaryStateCountingKernel final : public assembly::AssemblyKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::Standard | assembly::RequiredData::MaterialState;
    }

    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return assembly::MaterialStateSpec{sizeof(int), alignof(int)};
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    void computeCell(const assembly::AssemblyContext& /*ctx*/, assembly::KernelOutput& /*output*/) override
    {
    }

    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int /*boundary_marker*/,
                             assembly::KernelOutput& /*output*/) override
    {
        last_values.clear();
        last_values.reserve(static_cast<std::size_t>(ctx.numQuadraturePoints()));

        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            auto bytes = ctx.materialStateWork(q);
            auto* counter = reinterpret_cast<int*>(bytes.data());
            *counter += 1;
            last_values.push_back(*counter);
        }
    }

    mutable std::vector<int> last_values{};
};

class InteriorStateCountingKernel final : public assembly::AssemblyKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::Standard | assembly::RequiredData::MaterialState;
    }

    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return assembly::MaterialStateSpec{sizeof(int), alignof(int)};
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }

    void computeCell(const assembly::AssemblyContext& /*ctx*/, assembly::KernelOutput& /*output*/) override
    {
    }

    void computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                             const assembly::AssemblyContext& ctx_plus,
                             assembly::KernelOutput& /*output_minus*/,
                             assembly::KernelOutput& /*output_plus*/,
                             assembly::KernelOutput& /*coupling_mp*/,
                             assembly::KernelOutput& /*coupling_pm*/) override
    {
        shared_across_sides = true;
        last_values.clear();
        last_values.reserve(static_cast<std::size_t>(ctx_minus.numQuadraturePoints()));

        FE_THROW_IF(ctx_minus.numQuadraturePoints() != ctx_plus.numQuadraturePoints(), FEException,
                    "InteriorStateCountingKernel: mismatched quadrature point counts");

        for (LocalIndex q = 0; q < ctx_minus.numQuadraturePoints(); ++q) {
            auto bytes_minus = ctx_minus.materialStateWork(q);
            auto bytes_plus = ctx_plus.materialStateWork(q);
            if (bytes_minus.data() != bytes_plus.data()) {
                shared_across_sides = false;
            }
            auto* counter = reinterpret_cast<int*>(bytes_minus.data());
            *counter += 1;
            last_values.push_back(*counter);
        }
    }

    mutable bool shared_across_sides{false};
    mutable std::vector<int> last_values{};
};

static dofs::MeshTopologyInfo makeSingleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

static dofs::MeshTopologyInfo makeTwoTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3,
                             1, 2, 3, 4};
    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

TEST(MaterialStatePlumbingTest, PersistsAcrossSystemAssemblyCalls)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    systems::FESystem system(mesh);

    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    systems::FieldSpec field;
    field.name = "u";
    field.space = space;
    field.components = 1;
    const auto u = system.addField(field);

    auto kernel = std::make_shared<StateCountingKernel>();
    system.addCellKernel("residual", u, kernel);

    systems::SetupInputs inputs;
    inputs.topology_override = makeSingleTetraTopology();
    system.setup({}, inputs);

    assembly::DenseVectorView rhs(system.dofHandler().getNumDofs());
    rhs.zero();

    systems::SystemStateView state;

    systems::AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    req.zero_outputs = true;

    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_values.empty());
    for (const int v : kernel->last_values) {
        EXPECT_EQ(v, 1);
    }

    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_values.empty());
    for (const int v : kernel->last_values) {
        EXPECT_EQ(v, 2);
    }
}

TEST(MaterialStatePlumbingTest, SupportsBeginAndCommitTimeStepHistory)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    systems::FESystem system(mesh);

    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    systems::FieldSpec field;
    field.name = "u";
    field.space = space;
    field.components = 1;
    const auto u = system.addField(field);

    auto kernel = std::make_shared<OldPlusOneKernel>();
    system.addCellKernel("residual", u, kernel);

    systems::SetupInputs inputs;
    inputs.topology_override = makeSingleTetraTopology();
    system.setup({}, inputs);

    assembly::DenseVectorView rhs(system.dofHandler().getNumDofs());
    rhs.zero();

    systems::SystemStateView state;

    systems::AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    req.zero_outputs = true;

    system.beginTimeStep();
    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_old.empty());
    ASSERT_EQ(kernel->last_old.size(), kernel->last_work.size());
    for (std::size_t i = 0; i < kernel->last_old.size(); ++i) {
        EXPECT_EQ(kernel->last_old[i], 0);
        EXPECT_EQ(kernel->last_work[i], 1);
    }

    system.commitTimeStep();

    system.beginTimeStep();
    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_old.empty());
    ASSERT_EQ(kernel->last_old.size(), kernel->last_work.size());
    for (std::size_t i = 0; i < kernel->last_old.size(); ++i) {
        EXPECT_EQ(kernel->last_old[i], 1);
        EXPECT_EQ(kernel->last_work[i], 2);
    }
}

TEST(MaterialStatePlumbingTest, SupportsBoundaryFaceMaterialState)
{
    constexpr int kMarker = 3;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(kMarker);
    systems::FESystem system(mesh);

    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    systems::FieldSpec field;
    field.name = "u";
    field.space = space;
    field.components = 1;
    const auto u = system.addField(field);

    auto kernel = std::make_shared<BoundaryStateCountingKernel>();
    system.addBoundaryKernel("residual", kMarker, u, kernel);

    systems::SetupInputs inputs;
    inputs.topology_override = makeSingleTetraTopology();
    system.setup({}, inputs);

    assembly::DenseVectorView rhs(system.dofHandler().getNumDofs());
    rhs.zero();

    systems::SystemStateView state;

    systems::AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    req.zero_outputs = true;

    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_values.empty());
    for (const int v : kernel->last_values) {
        EXPECT_EQ(v, 1);
    }

    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_values.empty());
    for (const int v : kernel->last_values) {
        EXPECT_EQ(v, 2);
    }
}

TEST(MaterialStatePlumbingTest, SupportsInteriorFaceMaterialState)
{
    auto mesh = std::make_shared<forms::test::TwoTetraSharedFaceMeshAccess>();
    systems::FESystem system(mesh);

    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    systems::FieldSpec field;
    field.name = "u";
    field.space = space;
    field.components = 1;
    const auto u = system.addField(field);

    auto kernel = std::make_shared<InteriorStateCountingKernel>();
    system.addInteriorFaceKernel("residual", u, kernel);

    systems::SetupInputs inputs;
    inputs.topology_override = makeTwoTetraTopology();
    system.setup({}, inputs);

    assembly::DenseVectorView rhs(system.dofHandler().getNumDofs());
    rhs.zero();

    systems::SystemStateView state;

    systems::AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    req.zero_outputs = true;

    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_values.empty());
    EXPECT_TRUE(kernel->shared_across_sides);
    for (const int v : kernel->last_values) {
        EXPECT_EQ(v, 1);
    }

    (void)system.assemble(req, state, nullptr, &rhs);
    ASSERT_FALSE(kernel->last_values.empty());
    EXPECT_TRUE(kernel->shared_across_sides);
    for (const int v : kernel->last_values) {
        EXPECT_EQ(v, 2);
    }
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp
