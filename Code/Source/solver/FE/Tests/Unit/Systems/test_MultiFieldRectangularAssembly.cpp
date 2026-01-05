/**
 * @file test_MultiFieldRectangularAssembly.cpp
 * @brief Unit tests for multi-field + rectangular/block assembly in Systems
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/AssemblyContext.h"
#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::EntityKind;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::DenseVectorView;
using svmp::FE::assembly::KernelOutput;
using svmp::FE::assembly::MassKernel;
using svmp::FE::assembly::RequiredData;

using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::L2Space;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SetupOptions;
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

std::shared_ptr<Mesh> build_single_quad_mesh_with_left_edge_marker(int marker)
{
    auto mesh = build_single_quad_mesh();
    auto& base = mesh->local_mesh();

    // Find left edge (vertices {0,3}) and mark it.
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base.n_faces()); ++f) {
        const auto verts = base.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has0 = (verts[0] == 0 || verts[1] == 0);
        const bool has3 = (verts[0] == 3 || verts[1] == 3);
        if (has0 && has3) {
            base.set_boundary_label(f, marker);
            base.add_to_set(EntityKind::Face, "left", f);
            break;
        }
    }

    return mesh;
}

std::shared_ptr<Mesh> build_two_quad_mesh_with_left_edge_marker(int marker)
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
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 4, 3, 1, 2, 5, 4};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    auto mesh = svmp::create_mesh(std::move(base));

    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has0 = (verts[0] == 0 || verts[1] == 0);
        const bool has3 = (verts[0] == 3 || verts[1] == 3);
        if (has0 && has3) {
            mbase.set_boundary_label(f, marker);
            break;
        }
    }

    return mesh;
}

class BoundaryMassKernel final : public svmp::FE::assembly::BilinearFormKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override
    {
        return RequiredData::BasisValues | RequiredData::IntegrationWeights;
    }

    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    void computeCell(const svmp::FE::assembly::AssemblyContext&,
                     KernelOutput&) override
    {
        FE_THROW(svmp::FE::NotImplementedException,
                 "BoundaryMassKernel::computeCell is not implemented (boundary-only kernel)");
    }

    void computeBoundaryFace(const svmp::FE::assembly::AssemblyContext& ctx,
                             int /*boundary_marker*/,
                             KernelOutput& out) override
    {
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        const auto n_qpts = ctx.numQuadraturePoints();

        out.reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/false);

        for (svmp::FE::LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);
            for (svmp::FE::LocalIndex i = 0; i < n_test; ++i) {
                const Real phi_i = ctx.basisValue(i, q);
                for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                    const Real phi_j = ctx.trialBasisValue(j, q);
                    out.matrixEntry(i, j) += w * phi_i * phi_j;
                }
            }
        }
    }

    [[nodiscard]] std::string name() const override { return "BoundaryMassKernel"; }
};

class TrialSolutionProbeKernel final : public svmp::FE::assembly::LinearFormKernel {
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
        out.reserve(n, ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        const Real u0 = ctx.solutionValue(0);
        for (svmp::FE::LocalIndex i = 0; i < n; ++i) {
            out.vectorEntry(i) = u0;
        }
    }

    [[nodiscard]] std::string name() const override { return "TrialSolutionProbeKernel"; }
};

double blockAbsSum(const DenseMatrixView& A,
                   GlobalIndex r0, GlobalIndex r1,
                   GlobalIndex c0, GlobalIndex c1)
{
    double sum = 0.0;
    for (GlobalIndex i = r0; i < r1; ++i) {
        for (GlobalIndex j = c0; j < c1; ++j) {
            sum += std::abs(static_cast<double>(A.getMatrixEntry(i, j)));
        }
    }
    return sum;
}

} // namespace

TEST(FESystemMultiField, SetupBuildsMonolithicOffsetsAndBlocks)
{
    auto mesh = build_single_quad_mesh();
    auto u_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/2);
    auto p_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = u_space});
    const auto p = sys.addField(FieldSpec{.name = "p", .space = p_space});
    sys.addOperator("couple");
    sys.addCellKernel("couple", u, p, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_EQ(sys.fieldMap().numFields(), 2u);
    ASSERT_NE(sys.blockMap(), nullptr);
    EXPECT_EQ(sys.blockMap()->numBlocks(), 2u);

    const auto u_off = sys.fieldDofOffset(u);
    const auto p_off = sys.fieldDofOffset(p);
    const auto n_u = sys.fieldDofHandler(u).getNumDofs();
    const auto n_p = sys.fieldDofHandler(p).getNumDofs();

    EXPECT_EQ(u_off, 0);
    EXPECT_EQ(p_off, n_u);
    EXPECT_EQ(sys.dofHandler().getNumDofs(), n_u + n_p);
}

TEST(FESystemMultiField, AssembleRectangularCellTermFillsOffDiagonalBlock)
{
    auto mesh = build_single_quad_mesh();
    auto u_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/2);
    auto p_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = u_space});
    const auto p = sys.addField(FieldSpec{.name = "p", .space = p_space});
    sys.addOperator("couple");
    sys.addCellKernel("couple", u, p, std::make_shared<MassKernel>(1.0));
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    SystemStateView state;

    AssemblyRequest req;
    req.op = "couple";
    req.want_matrix = true;
    auto result = sys.assemble(req, state, &A, nullptr);
    EXPECT_TRUE(result.success);

    const auto u_off = sys.fieldDofOffset(u);
    const auto p_off = sys.fieldDofOffset(p);
    const auto n_u = sys.fieldDofHandler(u).getNumDofs();
    const auto n_p = sys.fieldDofHandler(p).getNumDofs();

    const auto uu = blockAbsSum(A, u_off, u_off + n_u, u_off, u_off + n_u);
    const auto pp = blockAbsSum(A, p_off, p_off + n_p, p_off, p_off + n_p);
    const auto pu = blockAbsSum(A, p_off, p_off + n_p, u_off, u_off + n_u);
    const auto up = blockAbsSum(A, u_off, u_off + n_u, p_off, p_off + n_p);

    EXPECT_NEAR(uu, 0.0, 1e-12);
    EXPECT_NEAR(pp, 0.0, 1e-12);
    EXPECT_NEAR(pu, 0.0, 1e-12);
    EXPECT_GT(up, 0.0);
}

TEST(FESystemMultiField, AssembleRectangularBoundaryTermFillsOffDiagonalBlock)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto u_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/2);
    auto p_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = u_space});
    const auto p = sys.addField(FieldSpec{.name = "p", .space = p_space});
    sys.addOperator("bndry");
    sys.addBoundaryKernel("bndry", marker, u, p, std::make_shared<BoundaryMassKernel>());
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    SystemStateView state;

    AssemblyRequest req;
    req.op = "bndry";
    req.want_matrix = true;
    auto result = sys.assemble(req, state, &A, nullptr);
    EXPECT_TRUE(result.success);

    const auto u_off = sys.fieldDofOffset(u);
    const auto p_off = sys.fieldDofOffset(p);
    const auto n_u = sys.fieldDofHandler(u).getNumDofs();
    const auto n_p = sys.fieldDofHandler(p).getNumDofs();

    const auto up = blockAbsSum(A, u_off, u_off + n_u, p_off, p_off + n_p);
    EXPECT_GT(up, 0.0);
}

TEST(FESystemMultiField, AssembleVectorOnlyRectangularTermUsesTrialFieldSolution)
{
    auto mesh = build_single_quad_mesh();
    auto u_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/2);
    auto p_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = u_space});
    const auto p = sys.addField(FieldSpec{.name = "p", .space = p_space});
    sys.addOperator("rect_residual");
    sys.addCellKernel("rect_residual", u, p, std::make_shared<TrialSolutionProbeKernel>());
    sys.setup();

    DenseVectorView rhs(sys.dofHandler().getNumDofs());
    SystemStateView state;
    std::vector<Real> uvec(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);

    const auto p_off = sys.fieldDofOffset(p);
    const auto n_p = sys.fieldDofHandler(p).getNumDofs();
    for (GlobalIndex i = 0; i < n_p; ++i) {
        uvec[static_cast<std::size_t>(p_off + i)] = 1.0;
    }
    state.u = uvec;

    AssemblyRequest req;
    req.op = "rect_residual";
    req.want_vector = true;
    auto result = sys.assemble(req, state, nullptr, &rhs);
    EXPECT_TRUE(result.success);

    const auto u_off = sys.fieldDofOffset(u);
    const auto n_u = sys.fieldDofHandler(u).getNumDofs();
    for (GlobalIndex i = 0; i < n_u; ++i) {
        EXPECT_NEAR(rhs.getVectorEntry(u_off + i), 1.0, 1e-12);
    }
    for (GlobalIndex i = 0; i < n_p; ++i) {
        EXPECT_NEAR(rhs.getVectorEntry(p_off + i), 0.0, 1e-12);
    }
}

TEST(FESystemSparsity, BoundarySparsityDoesNotIncludeUnmarkedCells)
{
    const int marker = 7;
    auto mesh = build_two_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<L2Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space});
    sys.addOperator("bndry");
    sys.addBoundaryKernel("bndry", marker, u, std::make_shared<BoundaryMassKernel>());

    SetupOptions opts;
    opts.sparsity_options.ensure_diagonal = false;
    opts.sparsity_options.ensure_non_empty_rows = false;
    sys.setup(opts);

    const auto& pattern = sys.sparsity("bndry");
    const auto right_cell_dofs = sys.dofHandler().getDofMap().getCellDofs(/*cell_id=*/1);
    for (auto dof : right_cell_dofs) {
        EXPECT_EQ(pattern.getRowNnz(dof), 0);
    }
}
