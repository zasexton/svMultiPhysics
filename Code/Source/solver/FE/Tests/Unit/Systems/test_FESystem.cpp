/**
 * @file test_FESystem.cpp
 * @brief Unit tests for Systems::FESystem (Mesh-driven setup + assembly)
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/MeshAccess.h"
#include "Assembly/StandardAssembler.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <numeric>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

using svmp::FE::assembly::AssemblyKernel;
using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::DenseVectorView;
using svmp::FE::assembly::MassKernel;
using svmp::FE::assembly::StandardAssembler;

using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::L2Space;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_two_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_two_segment_line_mesh()
{
    auto base = std::make_shared<MeshBase>();

    // Three vertices, two line cells: (0--1--2).
    const std::vector<svmp::real_t> X_ref = {0.0, 1.0, 2.0};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 2, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 1, 2};

    CellShape shape{};
    shape.family = CellFamily::Line;
    shape.num_corners = 2;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/1, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

class SolutionProbeKernel final : public svmp::FE::assembly::LinearFormKernel {
public:
    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        return svmp::FE::assembly::RequiredData::IntegrationWeights |
               svmp::FE::assembly::RequiredData::BasisValues |
               svmp::FE::assembly::RequiredData::SolutionValues;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& out) override
    {
        const auto n = ctx.numTestDofs();
        out.reserve(n, n, /*need_matrix=*/false, /*need_vector=*/true);
        const Real u0 = ctx.solutionValue(0);
        for (svmp::FE::LocalIndex i = 0; i < n; ++i) {
            out.vectorEntry(i) = u0;
        }
    }

    [[nodiscard]] std::string name() const override { return "SolutionProbeKernel"; }
};

class TwoMasterConstraint final : public svmp::FE::constraints::Constraint {
public:
    TwoMasterConstraint(GlobalIndex slave, GlobalIndex master0, GlobalIndex master1)
        : slave_(slave), master0_(master0), master1_(master1)
    {}

    void apply(svmp::FE::constraints::AffineConstraints& constraints) const override
    {
        constraints.addLine(slave_);
        constraints.addEntry(slave_, master0_, 0.5);
        constraints.addEntry(slave_, master1_, 0.5);
    }

    [[nodiscard]] svmp::FE::constraints::ConstraintType getType() const noexcept override
    {
        return svmp::FE::constraints::ConstraintType::MultiPoint;
    }

    [[nodiscard]] svmp::FE::constraints::ConstraintInfo getInfo() const override
    {
        svmp::FE::constraints::ConstraintInfo info;
        info.name = "TwoMasterConstraint";
        info.type = getType();
        info.num_constrained_dofs = 1;
        info.is_homogeneous = true;
        return info;
    }

    [[nodiscard]] std::unique_ptr<svmp::FE::constraints::Constraint> clone() const override
    {
        return std::make_unique<TwoMasterConstraint>(*this);
    }

private:
    GlobalIndex slave_;
    GlobalIndex master0_;
    GlobalIndex master1_;
};

class DGInteriorFaceProbeKernel final : public svmp::FE::assembly::LinearFormKernel {
public:
    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        return svmp::FE::assembly::RequiredData::IntegrationWeights |
               svmp::FE::assembly::RequiredData::BasisValues;
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }

    void computeCell(const svmp::FE::assembly::AssemblyContext&,
                     svmp::FE::assembly::KernelOutput&) override
    {
        FE_THROW(svmp::FE::NotImplementedException,
                 "DGInteriorFaceProbeKernel::computeCell is not implemented (interior-face-only kernel)");
    }

    void computeInteriorFace(const svmp::FE::assembly::AssemblyContext& ctx_minus,
                             const svmp::FE::assembly::AssemblyContext& ctx_plus,
                             svmp::FE::assembly::KernelOutput& output_minus,
                             svmp::FE::assembly::KernelOutput& output_plus,
                             svmp::FE::assembly::KernelOutput& coupling_minus_plus,
                             svmp::FE::assembly::KernelOutput& coupling_plus_minus) override
    {
        // Reset outputs each call (KernelOutput::reserve does not clear has_* flags).
        output_minus = {};
        output_plus = {};
        coupling_minus_plus = {};
        coupling_plus_minus = {};

        const auto n_minus = ctx_minus.numTestDofs();
        output_minus.reserve(n_minus, ctx_minus.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        for (svmp::FE::LocalIndex i = 0; i < n_minus; ++i) {
            output_minus.vectorEntry(i) = 1.0;
        }

        const auto n_plus = ctx_plus.numTestDofs();
        output_plus.reserve(n_plus, ctx_plus.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        for (svmp::FE::LocalIndex i = 0; i < n_plus; ++i) {
            output_plus.vectorEntry(i) = 2.0;
        }
    }

    [[nodiscard]] std::string name() const override { return "DGInteriorFaceProbeKernel"; }
};

} // namespace

TEST(FESystem, SetupDistributesDofsFromMesh)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    sys.setup();

    EXPECT_TRUE(sys.isSetup());
    EXPECT_EQ(sys.dofHandler().getNumDofs(), 4);
    EXPECT_EQ(sys.fieldMap().numFields(), 1u);
    EXPECT_EQ(sys.sparsity("mass").numRows(), 4);
}

TEST(FESystem, AssembleAccumulatesMultipleCellTerms)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    // Build reference mass matrix using StandardAssembler directly
    svmp::FE::dofs::DofHandler dh;
    dh.distributeDofs(*mesh, *space);
    dh.finalize();

    DenseMatrixView reference(dh.getNumDofs());
    StandardAssembler assembler;
    assembler.setDofHandler(dh);
    assembler.initialize();

    MassKernel mass_kernel(1.0);
    assembler.assembleMatrix(svmp::FE::assembly::MeshAccess(*mesh),
                             *space, *space, mass_kernel, reference);
    assembler.finalize(&reference, nullptr);

    // Systems mass operator with two identical terms -> 2x reference
    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    auto result = sys.assembleMass(state, mass);
    EXPECT_TRUE(result.success);

    for (GlobalIndex i = 0; i < mass.numRows(); ++i) {
        for (GlobalIndex j = 0; j < mass.numCols(); ++j) {
            EXPECT_NEAR(mass.getMatrixEntry(i, j), 2.0 * reference.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(FESystem, AssembleVectorUsesSolutionInjectionWhenRequested)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");
    sys.addCellKernel("residual", u, std::make_shared<SolutionProbeKernel>());
    sys.setup();

    DenseVectorView rhs(sys.dofHandler().getNumDofs());
    std::vector<Real> uvec(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state;
    state.u = uvec;

    auto result = sys.assembleResidual(state, rhs);
    EXPECT_TRUE(result.success);
    for (GlobalIndex i = 0; i < rhs.numRows(); ++i) {
        EXPECT_DOUBLE_EQ(rhs.getVectorEntry(i), 1.0);
    }
}

TEST(FESystem, DefinitionAfterSetupInvalidatesAndRequiresReseup)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_TRUE(sys.isSetup());

    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    EXPECT_TRUE(sys.assembleMass(state, mass).success);

    sys.addOperator("dummy");
    EXPECT_FALSE(sys.isSetup());
    EXPECT_THROW((void)sys.assembleMass(state, mass), svmp::FE::systems::InvalidStateException);

    sys.setup();
    EXPECT_TRUE(sys.isSetup());

    sys.addField(FieldSpec{.name = "v", .space = space, .components = 1});
    EXPECT_FALSE(sys.isSetup());

    sys.setup();
    EXPECT_TRUE(sys.isSetup());
    EXPECT_EQ(sys.fieldMap().numFields(), 2u);
    ASSERT_NE(sys.blockMap(), nullptr);
    EXPECT_EQ(sys.blockMap()->numBlocks(), 2u);
    EXPECT_EQ(sys.dofHandler().getNumDofs(), 8);
}

TEST(FESystem, AssembleVectorOnlyInteriorFaceTerm)
{
    auto mesh = build_two_quad_mesh();
    auto space = std::make_shared<L2Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("dg_residual");
    sys.addInteriorFaceKernel("dg_residual", u, std::make_shared<DGInteriorFaceProbeKernel>());
    sys.setup();

    DenseVectorView rhs(sys.dofHandler().getNumDofs());
    SystemStateView state;

    AssemblyRequest req;
    req.op = "dg_residual";
    req.want_vector = true;

    auto result = sys.assemble(req, state, nullptr, &rhs);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.interior_faces_assembled, 1);

    // Identify the minus/plus cells for the only interior face.
    const auto& base = mesh->local_mesh();
    const auto& f2c = base.face2cell();
    svmp::index_t cell_minus = svmp::INVALID_INDEX;
    svmp::index_t cell_plus = svmp::INVALID_INDEX;
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(f2c.size()); ++f) {
        const auto& fc = f2c[static_cast<std::size_t>(f)];
        if (fc[0] != svmp::INVALID_INDEX && fc[1] != svmp::INVALID_INDEX) {
            cell_minus = fc[0];
            cell_plus = fc[1];
            break;
        }
    }
    ASSERT_NE(cell_minus, svmp::INVALID_INDEX);
    ASSERT_NE(cell_plus, svmp::INVALID_INDEX);

    const auto minus_dofs = sys.dofHandler().getDofMap().getCellDofs(static_cast<GlobalIndex>(cell_minus));
    const auto plus_dofs = sys.dofHandler().getDofMap().getCellDofs(static_cast<GlobalIndex>(cell_plus));

    for (auto dof : minus_dofs) {
        EXPECT_NEAR(rhs.getVectorEntry(dof), 1.0, 1e-12);
    }
    for (auto dof : plus_dofs) {
        EXPECT_NEAR(rhs.getVectorEntry(dof), 2.0, 1e-12);
    }
}

TEST(FESystem, ConstraintSparsityAugmentationAddsMasterCouplings)
{
    auto mesh = build_two_segment_line_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Line2, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    // Setup once to discover the shared DOF between the two cells.
    sys.setup();
    auto dofs0 = sys.dofHandler().getDofMap().getCellDofs(/*cell_id=*/0);
    auto dofs1 = sys.dofHandler().getDofMap().getCellDofs(/*cell_id=*/1);
    ASSERT_EQ(dofs0.size(), 2u);
    ASSERT_EQ(dofs1.size(), 2u);

    GlobalIndex shared = -1;
    for (auto a : dofs0) {
        for (auto b : dofs1) {
            if (a == b) {
                shared = a;
                break;
            }
        }
    }
    ASSERT_GE(shared, 0);

    const GlobalIndex end0 = (dofs0[0] == shared) ? dofs0[1] : dofs0[0];
    const GlobalIndex end1 = (dofs1[0] == shared) ? dofs1[1] : dofs1[0];

    sys.addConstraint(std::make_unique<TwoMasterConstraint>(shared, end0, end1));

    svmp::FE::systems::SetupOptions opts_no_aug;
    opts_no_aug.use_constraints_in_assembly = false;
    sys.setup(opts_no_aug);
    const bool has_without = sys.sparsity("mass").hasEntry(end0, end1);
    EXPECT_FALSE(has_without);

    svmp::FE::systems::SetupOptions opts_aug;
    opts_aug.use_constraints_in_assembly = true;
    sys.setup(opts_aug);
    const bool has_with = sys.sparsity("mass").hasEntry(end0, end1);
    EXPECT_TRUE(has_with);
}
