/**
 * @file test_FESystemEdgeCases.cpp
 * @brief Unit tests for Systems FESystem edge cases and error paths
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/SystemConstraints.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <memory>
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
using svmp::FE::assembly::MassKernel;
using svmp::FE::assembly::SourceKernel;

using svmp::FE::constraints::AffineConstraints;
using svmp::FE::constraints::ConstraintType;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SetupOptions;
using svmp::FE::systems::SystemStateView;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh_with_left_edge_marker(int marker, std::string set_name)
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

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        if (verts.size() != 2u) continue;
        const bool has0 = (verts[0] == 0 || verts[1] == 0);
        const bool has3 = (verts[0] == 3 || verts[1] == 3);
        if (has0 && has3) {
            base->set_boundary_label(f, marker);
            base->add_to_set(EntityKind::Face, set_name, f);
            break;
        }
    }

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_two_segment_line_mesh()
{
    auto base = std::make_shared<MeshBase>();

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

class TwoMasterConstraint final : public svmp::FE::constraints::Constraint {
public:
    TwoMasterConstraint(GlobalIndex slave, GlobalIndex master0, GlobalIndex master1)
        : slave_(slave), master0_(master0), master1_(master1)
    {}

    void apply(AffineConstraints& constraints) const override
    {
        constraints.addLine(slave_);
        constraints.addEntry(slave_, master0_, 0.5);
        constraints.addEntry(slave_, master1_, 0.5);
    }

    [[nodiscard]] ConstraintType getType() const noexcept override { return ConstraintType::MultiPoint; }

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

class SingleDofDirichlet final : public svmp::FE::constraints::Constraint {
public:
    explicit SingleDofDirichlet(GlobalIndex dof, double value)
        : dof_(dof), value_(value)
    {}

    void apply(AffineConstraints& constraints) const override
    {
        constraints.addLine(dof_);
        constraints.setInhomogeneity(dof_, value_);
    }

    [[nodiscard]] ConstraintType getType() const noexcept override { return ConstraintType::Dirichlet; }

    [[nodiscard]] svmp::FE::constraints::ConstraintInfo getInfo() const override
    {
        svmp::FE::constraints::ConstraintInfo info;
        info.name = "SingleDofDirichlet";
        info.type = getType();
        info.num_constrained_dofs = 1;
        info.is_homogeneous = (value_ == 0.0);
        return info;
    }

    [[nodiscard]] std::unique_ptr<svmp::FE::constraints::Constraint> clone() const override
    {
        return std::make_unique<SingleDofDirichlet>(*this);
    }

private:
    GlobalIndex dof_{-1};
    double value_{0.0};
};

class MaterialStateProbeKernel final : public svmp::FE::assembly::LinearFormKernel {
public:
    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        return svmp::FE::assembly::RequiredData::MaterialState;
    }

    [[nodiscard]] svmp::FE::assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        svmp::FE::assembly::MaterialStateSpec spec;
        spec.bytes_per_qpt = 8;
        spec.alignment = 8;
        return spec;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& out) override
    {
        EXPECT_TRUE(ctx.hasMaterialState());
        EXPECT_EQ(ctx.materialStateBytesPerQpt(), 8u);
        ASSERT_GT(ctx.numQuadraturePoints(), 0);

        const auto state = ctx.materialState(0);
        EXPECT_EQ(state.size(), 8u);

        const auto n = ctx.numTestDofs();
        out.reserve(n, ctx.numTrialDofs(), /*need_matrix=*/false, /*need_vector=*/true);
        out.clear();
    }

    [[nodiscard]] std::string name() const override { return "MaterialStateProbeKernel"; }
};

} // namespace

TEST(FESystemEdgeCases, FESystem_MultipleSetupCalls_RebuildsCorrectly)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    sys.setup();
    const auto n0 = sys.dofHandler().getNumDofs();
    EXPECT_TRUE(sys.isSetup());

    sys.setup();
    const auto n1 = sys.dofHandler().getNumDofs();
    EXPECT_TRUE(sys.isSetup());
    EXPECT_EQ(n0, n1);
}

TEST(FESystemEdgeCases, FESystem_SetupWithDifferentOptions_AppliesNewOptions)
{
    const int marker = 7;
    const std::string set_name = "left";

    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker, set_name);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    sys.setup();
    sys.addConstraint(svmp::FE::systems::makeDirichletConstantByFaceSet(*mesh, sys.dofHandler(), set_name, /*value=*/0.0));

    // With constraints disabled in assembly, constrained rows are not modified.
    SetupOptions opts_no_constraints;
    opts_no_constraints.use_constraints_in_assembly = false;
    sys.setup(opts_no_constraints);

    DenseMatrixView mass_no(sys.dofHandler().getNumDofs());
    mass_no.zero();
    SystemStateView state;
    (void)sys.assembleMass(state, mass_no);

    const auto constrained = svmp::FE::systems::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), set_name);
    ASSERT_EQ(constrained.size(), 2u);
    for (auto dof : constrained) {
        EXPECT_NE(mass_no.getMatrixEntry(dof, dof), 1.0);
    }

    // With constraints enabled, constrained rows become identity-like.
    SetupOptions opts_constraints;
    opts_constraints.use_constraints_in_assembly = true;
    sys.setup(opts_constraints);

    DenseMatrixView mass_yes(sys.dofHandler().getNumDofs());
    mass_yes.zero();
    (void)sys.assembleMass(state, mass_yes);

    for (auto dof : constrained) {
        EXPECT_NEAR(mass_yes.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (GlobalIndex j = 0; j < mass_yes.numCols(); ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mass_yes.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
    }
}

TEST(FESystemEdgeCases, FESystem_AssemblyBeforeSetup_Throws)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    DenseMatrixView A(4);
    A.zero();

    SystemStateView state;
    EXPECT_THROW((void)sys.assembleMass(state, A), svmp::FE::systems::InvalidStateException);
}

TEST(FESystemEdgeCases, FESystem_ZeroComponentField_Behavior)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 0});
    EXPECT_EQ(sys.fieldRecord(u).components, space->value_dimension());
}

TEST(FESystemEdgeCases, FESystem_VeryLargeFieldCount_HandlesCorrectly)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    constexpr int n_fields = 100;
    for (int i = 0; i < n_fields; ++i) {
        sys.addField(FieldSpec{.name = "f" + std::to_string(i), .space = space, .components = 1});
    }
    sys.setup();

    EXPECT_EQ(sys.fieldMap().numFields(), static_cast<std::size_t>(n_fields));
    EXPECT_EQ(sys.dofHandler().getNumDofs(), 4 * n_fields);
}

TEST(FESystemEdgeCases, FESystem_FieldDofOffset_MultipleFields_Correct)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto v = sys.addField(FieldSpec{.name = "v", .space = space, .components = 1});
    const auto w = sys.addField(FieldSpec{.name = "w", .space = space, .components = 1});
    sys.setup();

    EXPECT_EQ(sys.fieldDofOffset(u), 0);
    EXPECT_EQ(sys.fieldDofOffset(v), 4);
    EXPECT_EQ(sys.fieldDofOffset(w), 8);
}

TEST(FESystemEdgeCases, FESystem_ConstraintAfterSetup_InvalidatesSetup)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();
    EXPECT_TRUE(sys.isSetup());

    sys.addConstraint(std::make_unique<SingleDofDirichlet>(0, 0.0));
    EXPECT_FALSE(sys.isSetup());
}

TEST(FESystemEdgeCases, FESystem_MixedConstraintTypes_AppliedCorrectly)
{
    auto mesh = build_two_segment_line_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Line2, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

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
            }
        }
    }
    ASSERT_GE(shared, 0);

    const GlobalIndex end0 = (dofs0[0] == shared) ? dofs0[1] : dofs0[0];
    const GlobalIndex end1 = (dofs1[0] == shared) ? dofs1[1] : dofs1[0];

    sys.addConstraint(std::make_unique<SingleDofDirichlet>(end0, 0.0));
    sys.addConstraint(std::make_unique<TwoMasterConstraint>(shared, end0, end1));
    sys.setup();

    const auto& constraints = sys.constraints();
    EXPECT_TRUE(constraints.isClosed());
    EXPECT_TRUE(constraints.isConstrained(end0));
    EXPECT_TRUE(constraints.isConstrained(shared));

    const auto c0 = constraints.getConstraint(end0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_TRUE(c0->entries.empty());

    const auto c_shared = constraints.getConstraint(shared);
    ASSERT_TRUE(c_shared.has_value());
    ASSERT_EQ(c_shared->entries.size(), 1u);
    EXPECT_EQ(c_shared->entries[0].master_dof, end1);
    EXPECT_NEAR(c_shared->entries[0].weight, 0.5, 1e-15);
}

TEST(FESystemEdgeCases, FESystem_AssembleMass_NoMassOperator_Throws)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    A.zero();
    SystemStateView state;
    EXPECT_THROW((void)sys.assembleMass(state, A), svmp::FE::InvalidArgumentException);
}

TEST(FESystemEdgeCases, FESystem_AssembleResidual_ReturnsCorrectResult)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("residual");
    sys.addCellKernel("residual", u, u, std::make_shared<SourceKernel>(1.0));
    sys.setup();

    DenseVectorView b0(sys.dofHandler().getNumDofs());
    DenseVectorView b1(sys.dofHandler().getNumDofs());
    b0.zero();
    b1.zero();

    SystemStateView state;
    (void)sys.assembleResidual(state, b0);

    AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    (void)sys.assemble(req, state, nullptr, &b1);

    for (GlobalIndex i = 0; i < b0.numRows(); ++i) {
        EXPECT_NEAR(b0.getVectorEntry(i), b1.getVectorEntry(i), 1e-12);
    }
}

TEST(FESystemEdgeCases, FESystem_AssembleJacobian_ReturnsCorrectResult)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("jacobian");
    sys.addCellKernel("jacobian", u, u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    DenseMatrixView A0(sys.dofHandler().getNumDofs());
    DenseMatrixView A1(sys.dofHandler().getNumDofs());
    A0.zero();
    A1.zero();

    SystemStateView state;
    (void)sys.assembleJacobian(state, A0);

    AssemblyRequest req;
    req.op = "jacobian";
    req.want_matrix = true;
    (void)sys.assemble(req, state, &A1, nullptr);

    for (GlobalIndex i = 0; i < A0.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A0.numCols(); ++j) {
            EXPECT_NEAR(A0.getMatrixEntry(i, j), A1.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(FESystemEdgeCases, FESystem_EvaluateFieldAtPoint_NoSearchAccess_ReturnsNullopt)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup({}, svmp::FE::systems::SetupInputs{.topology_override = svmp::FE::dofs::MeshTopologyInfo{
                                                   .n_cells = 1,
                                                   .n_vertices = 4,
                                                   .dim = 3,
                                                   .cell2vertex_offsets = {0, 4},
                                                   .cell2vertex_data = {0, 1, 2, 3},
                                                   .vertex_gids = {0, 1, 2, 3},
                                                   .cell_gids = {0},
                                                   .cell_owner_ranks = {0},
                                               }});

    std::vector<Real> U = {0.0, 0.0, 0.0, 0.0};
    SystemStateView state;
    state.u = U;

    const auto val = sys.evaluateFieldAtPoint(u, state, {0.2, 0.2, 0.2});
    EXPECT_FALSE(val.has_value());
}

TEST(FESystemEdgeCases, FESystem_EvaluateFieldAtPoint_PointOutsideMesh_ReturnsNullopt)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();
    ASSERT_NE(sys.searchAccess(), nullptr);
    sys.searchAccess()->build();

    std::vector<Real> U(sys.dofHandler().getNumDofs(), 0.0);
    SystemStateView state;
    state.u = U;

    const auto val = sys.evaluateFieldAtPoint(u, state, {2.0, 2.0, 0.0});
    EXPECT_FALSE(val.has_value());
}

TEST(FESystemEdgeCases, FESystem_EvaluateFieldAtPoint_VectorField_ReturnsAllComponents)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/2);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 2});
    sys.setup();

    ASSERT_NE(sys.searchAccess(), nullptr);
    sys.searchAccess()->build();

    // ProductSpace stores coefficients as [comp0 block][comp1 block].
    std::vector<Real> U = {
        // comp0: x at vertices
        0.0, 1.0, 1.0, 0.0,
        // comp1: y at vertices
        0.0, 0.0, 1.0, 1.0,
    };
    ASSERT_EQ(U.size(), static_cast<std::size_t>(sys.dofHandler().getNumDofs()));

    SystemStateView state;
    state.u = U;

    const std::array<Real, 3> x{0.3, 0.7, 0.0};
    const auto val = sys.evaluateFieldAtPoint(u, state, x);
    ASSERT_TRUE(val.has_value());
    EXPECT_NEAR((*val)[0], x[0], 1e-12);
    EXPECT_NEAR((*val)[1], x[1], 1e-12);
}

TEST(FESystemEdgeCases, FESystem_BeginCommitTimeStep_WithoutMaterialState_NoOp)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_NO_THROW(sys.beginTimeStep());
    EXPECT_NO_THROW(sys.commitTimeStep());
}

TEST(FESystemEdgeCases, FESystem_DoubleBeginTimeStep_Behavior)
{
    auto mesh = build_single_quad_mesh_with_left_edge_marker(/*marker=*/7, /*set_name=*/"left");
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("stateful");
    sys.addCellKernel("stateful", u, u, std::make_shared<MaterialStateProbeKernel>());
    sys.setup();

    EXPECT_NO_THROW(sys.beginTimeStep());
    EXPECT_NO_THROW(sys.beginTimeStep());
    EXPECT_NO_THROW(sys.commitTimeStep());
}
