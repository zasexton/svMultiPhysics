/**
 * @file test_FESystem.cpp
 * @brief Unit tests for Systems::FESystem (Mesh-driven setup + assembly)
 */

#include <gtest/gtest.h>

#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/MeshDisplacementBinding.h"
#include "Systems/OperatorBackends.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/MeshAccess.h"
#include "Assembly/StandardAssembler.h"

#include "Dofs/EntityDofMap.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Motion/MotionFields.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <memory>
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
using svmp::FE::spaces::ProductSpace;

using svmp::FE::systems::ALEBindingOptions;
using svmp::FE::systems::ALEMeshVelocitySource;
using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::DerivedFieldRole;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::FieldSourceKind;
using svmp::FE::systems::MeshDisplacementBindingOptions;
using svmp::FE::systems::MeshCoordinateUpdateMode;
using svmp::FE::systems::MeshCoordinateUpdateOptions;
using svmp::FE::systems::MeshCoordinateUpdateStage;
using svmp::FE::systems::MeshMotionFieldRole;
using svmp::FE::systems::SystemStateView;
using svmp::FE::systems::resolveALEBinding;
using svmp::FE::systems::resolveMeshDisplacementBinding;

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

class GeometryValueConstraint final : public svmp::FE::constraints::Constraint {
public:
    explicit GeometryValueConstraint(std::shared_ptr<int> updates)
        : updates_(std::move(updates))
    {}

    void apply(svmp::FE::constraints::AffineConstraints& constraints) const override
    {
        constraints.addDirichlet(0, 1.0);
    }

    [[nodiscard]] svmp::FE::constraints::ConstraintType getType() const noexcept override
    {
        return svmp::FE::constraints::ConstraintType::Dirichlet;
    }

    [[nodiscard]] svmp::FE::constraints::ConstraintInfo getInfo() const override
    {
        svmp::FE::constraints::ConstraintInfo info;
        info.name = "GeometryValueConstraint";
        info.type = getType();
        info.num_constrained_dofs = 1;
        info.is_homogeneous = false;
        return info;
    }

    [[nodiscard]] svmp::FE::constraints::ConstraintDependencyDeclaration
    dependencyDeclaration() const override
    {
        svmp::FE::constraints::ConstraintDependencyDeclaration out;
        out.value.geometry = true;
        return out;
    }

    [[nodiscard]] bool updateValues(svmp::FE::constraints::AffineConstraints& constraints,
                                    double time) const override
    {
        (void)time;
        constraints.updateInhomogeneity(0, 2.0);
        ++(*updates_);
        return true;
    }

    [[nodiscard]] std::unique_ptr<svmp::FE::constraints::Constraint> clone() const override
    {
        return std::make_unique<GeometryValueConstraint>(*this);
    }

private:
    std::shared_ptr<int> updates_;
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

TEST(FESystem, CoordinateConfigurationIsPropagatedToMeshAndSearchAccess)
{
    auto mesh = build_single_quad_mesh();

    // Make current coordinates different from reference.
    auto X_cur = mesh->local_mesh().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] += 10.0; // shift x only
    }
    mesh->set_current_coords(X_cur);

    // Reference-configured system should deterministically use X_ref for mesh access/search.
    FESystem sys_ref(mesh, svmp::Configuration::Reference);
    EXPECT_EQ(sys_ref.coordinateConfiguration(), svmp::Configuration::Reference);
    const auto x0_ref = sys_ref.meshAccess().getNodeCoordinates(0);
    EXPECT_NEAR(x0_ref[0], Real(0.0), 1e-12);

    const std::array<Real, 3> p_cur{10.25, 0.25, 0.0};
    ASSERT_NE(sys_ref.searchAccess(), nullptr);
    EXPECT_FALSE(sys_ref.searchAccess()->locatePoint(p_cur).found);

    // Current-configured system should deterministically use X_cur.
    FESystem sys_cur(mesh, svmp::Configuration::Current);
    EXPECT_EQ(sys_cur.coordinateConfiguration(), svmp::Configuration::Current);
    const auto x0_cur = sys_cur.meshAccess().getNodeCoordinates(0);
    EXPECT_NEAR(x0_cur[0], Real(10.0), 1e-12);

    ASSERT_NE(sys_cur.searchAccess(), nullptr);
    EXPECT_TRUE(sys_cur.searchAccess()->locatePoint(p_cur).found);
}

TEST(FESystem, ExplicitCoordinateConfigurationIgnoresMutableActiveConfigurationSwitch)
{
    auto mesh = build_single_quad_mesh();
    auto x_cur = mesh->local_mesh().X_ref();
    for (std::size_t i = 0; i < x_cur.size(); i += 2) {
        x_cur[i] += 4.0;
    }
    mesh->set_current_coords(x_cur);
    mesh->use_reference_configuration();

    FESystem sys_ref(mesh, svmp::Configuration::Reference);
    FESystem sys_cur(mesh, svmp::Configuration::Current);

    EXPECT_EQ(sys_ref.meshAccess().activeConfigurationEpoch(), 0u);
    EXPECT_EQ(sys_cur.meshAccess().activeConfigurationEpoch(), 0u);

    const auto ref_before = sys_ref.meshAccess().getNodeCoordinates(0);
    const auto cur_before = sys_cur.meshAccess().getNodeCoordinates(0);
    EXPECT_NEAR(ref_before[0], 0.0, 1.0e-12);
    EXPECT_NEAR(cur_before[0], 4.0, 1.0e-12);

    const auto epoch_before = mesh->active_configuration_epoch();
    mesh->use_current_configuration();
    EXPECT_GT(mesh->active_configuration_epoch(), epoch_before);

    const auto ref_after = sys_ref.meshAccess().getNodeCoordinates(0);
    const auto cur_after = sys_cur.meshAccess().getNodeCoordinates(0);
    EXPECT_NEAR(ref_after[0], ref_before[0], 1.0e-12);
    EXPECT_NEAR(cur_after[0], cur_before[0], 1.0e-12);

    ASSERT_NE(sys_ref.searchAccess(), nullptr);
    ASSERT_NE(sys_cur.searchAccess(), nullptr);
    EXPECT_TRUE(sys_ref.searchAccess()->locatePoint({0.25, 0.25, 0.0}).found);
    EXPECT_FALSE(sys_ref.searchAccess()->locatePoint({4.25, 0.25, 0.0}).found);
    EXPECT_FALSE(sys_cur.searchAccess()->locatePoint({0.25, 0.25, 0.0}).found);
    EXPECT_TRUE(sys_cur.searchAccess()->locatePoint({4.25, 0.25, 0.0}).found);
}

TEST(FESystem, MassAssemblyRespectsCoordinateConfiguration)
{
    auto mesh = build_single_quad_mesh();

    // Scale x by 2 in the current configuration so the cell area doubles.
    auto X_cur = mesh->local_mesh().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] *= 2.0;
    }
    mesh->set_current_coords(X_cur);

    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    auto build_mass_system = [&](svmp::Configuration cfg) {
        FESystem sys(mesh, cfg);
        const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("mass");
        sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
        sys.setup();
        return sys;
    };

    auto sys_ref = build_mass_system(svmp::Configuration::Reference);
    auto sys_cur = build_mass_system(svmp::Configuration::Current);
    auto sys_def = build_mass_system(svmp::Configuration::Deformed);

    DenseMatrixView M_ref(sys_ref.dofHandler().getNumDofs());
    DenseMatrixView M_cur(sys_cur.dofHandler().getNumDofs());
    DenseMatrixView M_def(sys_def.dofHandler().getNumDofs());
    SystemStateView state;

    ASSERT_TRUE(sys_ref.assembleMass(state, M_ref).success);
    ASSERT_TRUE(sys_cur.assembleMass(state, M_cur).success);
    ASSERT_TRUE(sys_def.assembleMass(state, M_def).success);

    // With x scaled by 2, the mass matrix scales by 2. Deformed is an alias of Current.
    for (GlobalIndex i = 0; i < M_ref.numRows(); ++i) {
        for (GlobalIndex j = 0; j < M_ref.numCols(); ++j) {
            EXPECT_NEAR(M_cur.getMatrixEntry(i, j), 2.0 * M_ref.getMatrixEntry(i, j), 1e-12);
            EXPECT_NEAR(M_def.getMatrixEntry(i, j), M_cur.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(FESystem, PrescribedMovingMeshVectorMassRespectsCurrentGeometry)
{
    auto mesh = build_single_quad_mesh();

    auto x_cur = mesh->local_mesh().X_ref();
    for (std::size_t i = 0; i < x_cur.size(); i += 2u) {
        x_cur[i] *= 2.0;
    }
    mesh->set_current_coords(x_cur);

    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    auto assemble_vector_mass = [&](svmp::Configuration cfg) {
        FESystem sys(mesh, cfg);
        const auto u = sys.addField(FieldSpec{.name = "u", .space = vector_space, .components = 2});
        sys.addOperator("mass");
        sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
        sys.setup();

        DenseMatrixView mass(sys.dofHandler().getNumDofs());
        SystemStateView state;
        EXPECT_TRUE(sys.assembleMass(state, mass).success);
        return mass;
    };

    const auto reference_mass = assemble_vector_mass(svmp::Configuration::Reference);
    const auto current_mass = assemble_vector_mass(svmp::Configuration::Current);

    ASSERT_EQ(reference_mass.numRows(), current_mass.numRows());
    ASSERT_EQ(reference_mass.numCols(), current_mass.numCols());
    for (GlobalIndex i = 0; i < reference_mass.numRows(); ++i) {
        for (GlobalIndex j = 0; j < reference_mass.numCols(); ++j) {
            EXPECT_NEAR(current_mass.getMatrixEntry(i, j),
                        Real(2.0) * reference_mass.getMatrixEntry(i, j),
                        1.0e-12);
        }
    }
}

TEST(FESystem, ReusedCurrentSystemTracksCoordinateMutationWithoutSetup)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh, svmp::Configuration::Current);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    DenseMatrixView reference(sys.dofHandler().getNumDofs());
    DenseMatrixView moved(sys.dofHandler().getNumDofs());
    SystemStateView state;

    ASSERT_TRUE(sys.assembleMass(state, reference).success);

    auto X_cur = mesh->local_mesh().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] *= 2.0;
    }
    mesh->set_current_coords(X_cur);

    ASSERT_TRUE(sys.assembleMass(state, moved).success);

    for (GlobalIndex i = 0; i < reference.numRows(); ++i) {
        for (GlobalIndex j = 0; j < reference.numCols(); ++j) {
            EXPECT_NEAR(moved.getMatrixEntry(i, j),
                        2.0 * reference.getMatrixEntry(i, j),
                        1e-12);
        }
    }
}

TEST(FESystem, MeshGeometryAdvanceNotificationPreservesLayoutAndTopologyNotificationInvalidatesSetup)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh, svmp::Configuration::Current);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    DenseMatrixView reference(sys.dofHandler().getNumDofs());
    DenseMatrixView moved(sys.dofHandler().getNumDofs());
    SystemStateView state;
    ASSERT_TRUE(sys.assembleMass(state, reference).success);

    const auto layout_before = sys.feLayoutRevisionState();
    auto X_cur = mesh->local_mesh().X_ref();
    for (std::size_t i = 0; i < X_cur.size(); i += 2) {
        X_cur[i] *= 2.0;
    }
    mesh->set_current_coords(X_cur);
    sys.notifyMeshGeometryAdvanced();

    EXPECT_TRUE(sys.isSetup());
    EXPECT_EQ(sys.dofLayoutRevision(), layout_before.dof_layout);
    EXPECT_EQ(sys.constraintLayoutRevision(), layout_before.constraint_layout);
    EXPECT_EQ(sys.blockLayoutRevision(), layout_before.block_layout);

    ASSERT_TRUE(sys.assembleMass(state, moved).success);
    for (GlobalIndex i = 0; i < reference.numRows(); ++i) {
        for (GlobalIndex j = 0; j < reference.numCols(); ++j) {
            EXPECT_NEAR(moved.getMatrixEntry(i, j),
                        2.0 * reference.getMatrixEntry(i, j),
                        1e-12);
        }
    }

    sys.notifyMeshTopologyLayoutChanged();
    EXPECT_FALSE(sys.isSetup());
    EXPECT_GT(sys.dofLayoutRevision(), layout_before.dof_layout);
    EXPECT_GT(sys.constraintLayoutRevision(), layout_before.constraint_layout);
    EXPECT_GT(sys.blockLayoutRevision(), layout_before.block_layout);
}

TEST(FESystem, MeshMotionFieldBindingsArePhysicsNeutral)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto displacement = sys.addField(
        FieldSpec{.name = "mesh_displacement", .space = vector_space, .components = 2});
    const auto velocity = sys.addField(
        FieldSpec{.name = "mesh_velocity", .space = vector_space, .components = 2});
    const auto previous = sys.addField(
        FieldSpec{.name = "previous_coordinates", .space = vector_space, .components = 2});
    const auto acceleration = sys.addField(
        FieldSpec{.name = "mesh_acceleration", .space = vector_space, .components = 2});
    const auto previous_velocity = sys.addField(
        FieldSpec{.name = "previous_mesh_velocity", .space = vector_space, .components = 2});
    const auto predicted_velocity = sys.addField(
        FieldSpec{.name = "predicted_mesh_velocity", .space = vector_space, .components = 2});

    sys.bindMeshMotionField(MeshMotionFieldRole::Displacement, displacement);
    sys.bindMeshMotionField("velocity", "mesh_velocity");
    sys.bindMeshMotionField("previous_coordinates", previous);
    sys.bindMeshMotionField("acceleration", acceleration);
    sys.bindMeshMotionField("previous_velocity", previous_velocity);
    sys.bindMeshMotionField("predicted_velocity", predicted_velocity);

    const auto access = sys.meshMotionFieldAccess();
    EXPECT_EQ(access.mesh_displacement, displacement);
    EXPECT_EQ(access.mesh_velocity, velocity);
    EXPECT_EQ(access.previous_coordinates, previous);
    EXPECT_EQ(access.mesh_acceleration, acceleration);
    EXPECT_EQ(access.previous_mesh_velocity, previous_velocity);
    EXPECT_EQ(access.predicted_mesh_velocity, predicted_velocity);

    const auto bound_displacement = sys.meshMotionField(MeshMotionFieldRole::Displacement);
    ASSERT_TRUE(bound_displacement.has_value());
    EXPECT_EQ(*bound_displacement, displacement);
}

TEST(FESystem, PrescribedMeshMotionFieldIsExcludedFromUnknownLayout)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto u = sys.addField(
        FieldSpec{.name = "u", .space = scalar_space, .components = 1});
    const auto p = sys.addField(
        FieldSpec{.name = "p", .space = scalar_space, .components = 1});
    const auto velocity = sys.addMeshMotionDataField("mesh_velocity", vector_space);
    sys.bindMeshMotionField(MeshMotionFieldRole::Velocity, velocity);

    EXPECT_EQ(sys.fieldRecord(u).source_kind, FieldSourceKind::Unknown);
    EXPECT_EQ(sys.fieldRecord(p).source_kind, FieldSourceKind::Unknown);
    EXPECT_EQ(sys.fieldRecord(velocity).source_kind, FieldSourceKind::PrescribedData);
    EXPECT_TRUE(sys.fieldParticipatesInUnknownVector(u));
    EXPECT_TRUE(sys.fieldParticipatesInUnknownVector(p));
    EXPECT_FALSE(sys.fieldParticipatesInUnknownVector(velocity));

    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.addCellKernel("mass", p, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_EQ(sys.dofHandler().getNumDofs(), 8);
    EXPECT_EQ(sys.fieldDofHandler(velocity).getNumDofs(), 8);
    EXPECT_EQ(sys.fieldMap().numFields(), 2u);
    ASSERT_NE(sys.blockMap(), nullptr);
    EXPECT_EQ(sys.blockMap()->numBlocks(), 2u);
}

TEST(FESystem, DerivedMeshVelocityIsExcludedFromUnknownLayout)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto temperature = sys.addField(
        FieldSpec{.name = "temperature", .space = scalar_space, .components = 1});
    const auto displacement = sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    const auto velocity =
        sys.addDerivedMeshVelocityField("mesh_velocity", vector_space, displacement);

    EXPECT_EQ(sys.fieldRecord(temperature).source_kind, FieldSourceKind::Unknown);
    EXPECT_EQ(sys.fieldRecord(displacement).source_kind, FieldSourceKind::Unknown);
    EXPECT_EQ(sys.fieldRecord(velocity).source_kind, FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(sys.fieldRecord(velocity).derived.source_field, displacement);
    EXPECT_EQ(sys.fieldRecord(velocity).derived.role, DerivedFieldRole::TimeDerivative);
    EXPECT_TRUE(sys.fieldParticipatesInUnknownVector(temperature));
    EXPECT_TRUE(sys.fieldParticipatesInUnknownVector(displacement));
    EXPECT_FALSE(sys.fieldParticipatesInUnknownVector(velocity));

    sys.addOperator("mass");
    sys.addCellKernel("mass", temperature, std::make_shared<MassKernel>(1.0));
    sys.addCellKernel("mass", displacement, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_EQ(sys.dofHandler().getNumDofs(), 12);
    EXPECT_EQ(sys.fieldDofHandler(velocity).getNumDofs(), 8);
    EXPECT_EQ(sys.fieldMap().numFields(), 2u);
    ASSERT_NE(sys.blockMap(), nullptr);
    EXPECT_EQ(sys.blockMap()->numBlocks(), 2u);
}

TEST(FESystem, ALEBindingRegistersPrescribedMeshVelocityAsData)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto u = sys.addField(
        FieldSpec{.name = "u", .space = scalar_space, .components = 1});

    const auto ale = resolveALEBinding(
        sys,
        ALEBindingOptions{
            .enabled = true,
            .dimension = 2,
            .mesh_velocity_source = ALEMeshVelocitySource::PrescribedData,
            .mesh_velocity_field_name = "mesh_velocity",
            .mesh_displacement_field_name = "mesh_displacement",
            .mesh_velocity_space = vector_space,
        });

    ASSERT_TRUE(ale.enabled);
    EXPECT_FALSE(ale.coupled());
    ASSERT_NE(ale.mesh_velocity_field, svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Velocity),
              ale.mesh_velocity_field);
    EXPECT_EQ(sys.fieldRecord(ale.mesh_velocity_field).source_kind,
              FieldSourceKind::PrescribedData);
    EXPECT_FALSE(sys.fieldParticipatesInUnknownVector(ale.mesh_velocity_field));

    svmp::FE::systems::FormInstallOptions install;
    ale.configureInstallOptions(install);
    EXPECT_EQ(install.compiler_options.geometry_sensitivity.mode,
              svmp::FE::forms::GeometrySensitivityMode::GeometryConstant);
    EXPECT_TRUE(install.extra_trial_fields.empty());

    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_EQ(sys.dofHandler().getNumDofs(), 4);
    EXPECT_EQ(sys.fieldDofHandler(ale.mesh_velocity_field).getNumDofs(), 8);
    EXPECT_EQ(sys.fieldMap().numFields(), 1u);
}

TEST(FESystem, ALEBindingCoupledDisplacementCreatesDerivedVelocityAndInstallMetadata)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto temperature = sys.addField(
        FieldSpec{.name = "temperature", .space = scalar_space, .components = 1});
    const auto displacement =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);

    const auto ale = resolveALEBinding(
        sys,
        ALEBindingOptions{
            .enabled = true,
            .dimension = 2,
            .mesh_velocity_source = ALEMeshVelocitySource::CoupledDisplacement,
            .mesh_velocity_field_name = "mesh_velocity",
            .mesh_displacement_field_name = "mesh_displacement",
            .mesh_velocity_space = vector_space,
            .mesh_displacement_space = vector_space,
        });

    ASSERT_TRUE(ale.enabled);
    EXPECT_TRUE(ale.coupled());
    EXPECT_EQ(ale.mesh_displacement_field, displacement);
    ASSERT_NE(ale.mesh_velocity_field, svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Displacement), displacement);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Velocity),
              ale.mesh_velocity_field);

    const auto& velocity_record = sys.fieldRecord(ale.mesh_velocity_field);
    EXPECT_EQ(velocity_record.source_kind, FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(velocity_record.derived.source_field, displacement);
    EXPECT_EQ(velocity_record.derived.role, DerivedFieldRole::TimeDerivative);
    EXPECT_FALSE(sys.fieldParticipatesInUnknownVector(ale.mesh_velocity_field));
    EXPECT_TRUE(sys.geometricNonlinearityEnabled());

    svmp::FE::systems::FormInstallOptions install;
    ale.configureInstallOptions(install);
    EXPECT_EQ(install.compiler_options.geometry_sensitivity.mode,
              svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns);
    EXPECT_EQ(install.compiler_options.geometry_sensitivity.mesh_motion_field,
              displacement);
    ASSERT_EQ(install.extra_trial_fields.size(), 1u);
    EXPECT_EQ(install.extra_trial_fields.front(), displacement);

    sys.addOperator("mass");
    sys.addCellKernel("mass", temperature, std::make_shared<MassKernel>(1.0));
    sys.addCellKernel("mass", displacement, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_EQ(sys.dofHandler().getNumDofs(), 12);
    EXPECT_EQ(sys.fieldDofHandler(ale.mesh_velocity_field).getNumDofs(), 8);
    EXPECT_EQ(sys.fieldMap().numFields(), 2u);
}

TEST(FESystem, MeshDisplacementBindingRegistersUnknownAndBindsRole)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto binding = resolveMeshDisplacementBinding(
        sys,
        MeshDisplacementBindingOptions{
            .enabled = true,
            .dimension = 2,
            .field_name = "mesh_displacement",
            .space = vector_space,
            .auto_register_field = true,
            .bind_as_mesh_displacement = true,
        });

    ASSERT_TRUE(binding.enabled);
    ASSERT_NE(binding.displacement_field, svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Displacement),
              binding.displacement_field);
    EXPECT_EQ(sys.fieldRecord(binding.displacement_field).source_kind,
              FieldSourceKind::Unknown);
    EXPECT_TRUE(sys.fieldParticipatesInUnknownVector(binding.displacement_field));

    sys.addOperator("mass");
    sys.addCellKernel("mass", binding.displacement_field, std::make_shared<MassKernel>(1.0));
    sys.setup();

    EXPECT_EQ(sys.dofHandler().getNumDofs(), 8);
    EXPECT_EQ(sys.fieldMap().numFields(), 1u);
}

TEST(FESystem, MeshDisplacementBindingRejectsPrescribedDataField)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto prescribed =
        sys.addMeshMotionDataField("mesh_displacement", vector_space, /*components=*/2);
    ASSERT_EQ(sys.fieldRecord(prescribed).source_kind, FieldSourceKind::PrescribedData);

    EXPECT_THROW(
        (void)resolveMeshDisplacementBinding(
            sys,
            MeshDisplacementBindingOptions{
                .enabled = true,
                .dimension = 2,
                .field_name = "mesh_displacement",
                .space = vector_space,
                .auto_register_field = false,
                .bind_as_mesh_displacement = true,
            }),
        std::invalid_argument);
}

TEST(FESystem, ALEBindingRejectsMismatchedBoundAndNamedDisplacementFields)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto bound =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space, /*components=*/2);
    const auto other =
        sys.addField(FieldSpec{.name = "other_mesh_displacement",
                               .space = vector_space,
                               .components = 2});
    ASSERT_NE(bound, other);
    ASSERT_EQ(sys.meshMotionField(MeshMotionFieldRole::Displacement), bound);

    EXPECT_THROW(
        (void)resolveALEBinding(
            sys,
            ALEBindingOptions{
                .enabled = true,
                .dimension = 2,
                .mesh_velocity_source = ALEMeshVelocitySource::CoupledDisplacement,
                .mesh_velocity_field_name = "mesh_velocity",
                .mesh_displacement_field_name = "other_mesh_displacement",
                .mesh_velocity_space = vector_space,
                .mesh_displacement_space = vector_space,
            }),
        std::invalid_argument);
}

TEST(FESystem, StandardMeshMotionFieldsSyncFromMeshStorageToPrescribedBuffers)
{
    auto mesh = build_single_quad_mesh();
    const auto handles = svmp::motion::attach_motion_fields(*mesh, 2);
    auto* velocity_data = svmp::MeshFields::field_data_as<svmp::real_t>(
        mesh->local_mesh(), handles.velocity);
    ASSERT_NE(velocity_data, nullptr);
    const auto ncomp = svmp::MeshFields::field_components(mesh->local_mesh(), handles.velocity);
    ASSERT_EQ(ncomp, 2u);
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        velocity_data[v * ncomp + 0] = 3.0 + static_cast<Real>(v);
        velocity_data[v * ncomp + 1] = -4.0 - static_cast<Real>(v);
    }

    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto temperature = sys.addField(
        FieldSpec{.name = "temperature", .space = scalar_space, .components = 1});
    const auto velocity = sys.addMeshMotionDataField("mesh_velocity", vector_space);
    EXPECT_EQ(sys.bindStandardMeshMotionFieldsByName(), 1u);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Velocity), velocity);
    sys.addOperator("mass");
    sys.addCellKernel("mass", temperature, std::make_shared<MassKernel>(1.0));
    sys.setup();

    const std::size_t written = sys.syncBoundMeshMotionFieldsToPrescribedBuffers();
    EXPECT_EQ(written, mesh->n_vertices() * 2u);
    const auto coeffs = sys.prescribedFieldCoefficients(velocity);
    ASSERT_EQ(coeffs.size(), mesh->n_vertices() * 2u);
    const auto* entity_map = sys.fieldDofHandler(velocity).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<GlobalIndex>(v));
        ASSERT_EQ(vdofs.size(), 2u);
        EXPECT_NEAR(coeffs[static_cast<std::size_t>(vdofs[0])],
                    velocity_data[v * ncomp + 0],
                    1e-12);
        EXPECT_NEAR(coeffs[static_cast<std::size_t>(vdofs[1])],
                    velocity_data[v * ncomp + 1],
                    1e-12);
    }
}

TEST(FESystem, StandardMeshMotionFieldsSyncFromMeshStorageToFEState)
{
    auto mesh = build_single_quad_mesh();
    const auto handles = svmp::motion::attach_motion_fields(*mesh, 2);

    auto fill_field = [&](svmp::FieldHandle h, Real x_base, Real y_base) {
        auto* data = svmp::MeshFields::field_data_as<svmp::real_t>(mesh->local_mesh(), h);
        ASSERT_NE(data, nullptr);
        const auto ncomp = svmp::MeshFields::field_components(mesh->local_mesh(), h);
        ASSERT_EQ(ncomp, 2u);
        for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
            data[v * ncomp + 0] = x_base + static_cast<Real>(v);
            data[v * ncomp + 1] = y_base - static_cast<Real>(v);
        }
    };

    fill_field(handles.displacement, 10.0, 20.0);
    fill_field(handles.velocity, 30.0, 40.0);
    fill_field(handles.acceleration, 50.0, 60.0);
    fill_field(handles.previous_coordinates, 70.0, 80.0);
    fill_field(handles.previous_displacement, 90.0, 100.0);
    fill_field(handles.previous_velocity, 110.0, 120.0);

    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto displacement = sys.addField(
        FieldSpec{.name = "mesh_displacement", .space = vector_space, .components = 2});
    const auto velocity = sys.addField(
        FieldSpec{.name = "mesh_velocity", .space = vector_space, .components = 2});
    const auto acceleration = sys.addField(
        FieldSpec{.name = "mesh_acceleration", .space = vector_space, .components = 2});
    const auto previous_coordinates = sys.addField(
        FieldSpec{.name = "previous_coordinates", .space = vector_space, .components = 2});
    const auto previous_displacement = sys.addField(
        FieldSpec{.name = "previous_mesh_displacement", .space = vector_space, .components = 2});
    const auto previous_velocity = sys.addField(
        FieldSpec{.name = "previous_mesh_velocity", .space = vector_space, .components = 2});

    EXPECT_EQ(sys.bindStandardMeshMotionFieldsByName(), 6u);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Displacement), displacement);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Velocity), velocity);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::Acceleration), acceleration);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::PreviousCoordinates), previous_coordinates);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::PreviousDisplacement), previous_displacement);
    EXPECT_EQ(sys.meshMotionField(MeshMotionFieldRole::PreviousVelocity), previous_velocity);

    sys.addOperator("mass");
    sys.addCellKernel("mass", displacement, std::make_shared<MassKernel>(1.0));
    sys.setup();

    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    const std::size_t written = sys.syncBoundMeshMotionFieldsToState(state);
    EXPECT_EQ(written, 6u * mesh->n_vertices() * 2u);

    auto expect_vertex_values = [&](svmp::FE::FieldId field,
                                    Real x_base,
                                    Real y_base) {
        std::vector<double> values(mesh->n_vertices() * 2u, 0.0);
        SystemStateView view;
        view.u = state;
        ASSERT_TRUE(sys.evaluateFieldAtVertices(field,
                                                view,
                                                static_cast<GlobalIndex>(mesh->n_vertices()),
                                                values));
        for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
            EXPECT_NEAR(values[v * 2u + 0], x_base + static_cast<Real>(v), 1e-12);
            EXPECT_NEAR(values[v * 2u + 1], y_base - static_cast<Real>(v), 1e-12);
        }
    };

    expect_vertex_values(displacement, 10.0, 20.0);
    expect_vertex_values(velocity, 30.0, 40.0);
    expect_vertex_values(acceleration, 50.0, 60.0);
    expect_vertex_values(previous_coordinates, 70.0, 80.0);
    expect_vertex_values(previous_displacement, 90.0, 100.0);
    expect_vertex_values(previous_velocity, 110.0, 120.0);
}

TEST(FESystem, MeshDisplacementUnknownUpdatesCurrentCoordinatesWithRollbackAndCommit)
{
    auto mesh = build_single_quad_mesh();
    const auto handles = svmp::motion::attach_motion_fields(*mesh, 2);
    auto* disp = svmp::MeshFields::field_data_as<svmp::real_t>(
        mesh->local_mesh(), handles.displacement);
    ASSERT_NE(disp, nullptr);
    const auto ncomp = svmp::MeshFields::field_components(
        mesh->local_mesh(), handles.displacement);
    ASSERT_EQ(ncomp, 2u);
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        disp[v * ncomp + 0] = 0.1 + 0.01 * static_cast<Real>(v);
        disp[v * ncomp + 1] = -0.2 + 0.02 * static_cast<Real>(v);
    }

    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto displacement =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    ASSERT_EQ(sys.meshMotionField(MeshMotionFieldRole::Displacement), displacement);
    sys.addOperator("mass");
    sys.addCellKernel("mass", displacement, std::make_shared<MassKernel>(1.0));
    sys.setup();

    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    ASSERT_EQ(sys.syncBoundMeshMotionFieldsToState(state), mesh->n_vertices() * 2u);

    SystemStateView view;
    view.u = state;
    const auto result = sys.updateCurrentCoordinatesFromMeshDisplacement(view);
    EXPECT_TRUE(sys.meshCoordinateTransactionActive());
    EXPECT_EQ(result.vertices_updated, mesh->n_vertices());
    EXPECT_EQ(result.components_updated, mesh->n_vertices() * 2u);
    EXPECT_EQ(result.stage, MeshCoordinateUpdateStage::TrialNonlinearIterate);
    ASSERT_TRUE(mesh->has_current_coords());

    const auto& X_ref = mesh->local_mesh().X_ref();
    const auto& X_cur = mesh->local_mesh().X_cur();
    ASSERT_EQ(X_cur.size(), X_ref.size());
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        EXPECT_NEAR(X_cur[v * 2u + 0], X_ref[v * 2u + 0] + disp[v * ncomp + 0], 1e-12);
        EXPECT_NEAR(X_cur[v * 2u + 1], X_ref[v * 2u + 1] + disp[v * ncomp + 1], 1e-12);
    }

    sys.rollbackMeshCoordinateTransaction();
    EXPECT_FALSE(sys.meshCoordinateTransactionActive());
    EXPECT_FALSE(mesh->has_current_coords());

    MeshCoordinateUpdateOptions options;
    options.stage = MeshCoordinateUpdateStage::AcceptedTimeStep;
    options.mode = MeshCoordinateUpdateMode::AbsoluteFromReference;
    (void)sys.updateCurrentCoordinatesFromMeshDisplacement(view, options);
    EXPECT_FALSE(sys.meshCoordinateTransactionActive());
    ASSERT_TRUE(mesh->has_current_coords());
    sys.commitMeshCoordinateTransaction();
    sys.rollbackMeshCoordinateTransaction();
    EXPECT_TRUE(mesh->has_current_coords());
}

TEST(FESystem, MeshMotionFieldBindingRejectsMissingScalarAndDimensionMismatchedFields)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto vector_3d_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/3);

    FESystem sys(mesh);
    const auto scalar = sys.addField(
        FieldSpec{.name = "temperature", .space = scalar_space, .components = 1});
    const auto wrong_dim = sys.addField(
        FieldSpec{.name = "mesh_displacement_3d", .space = vector_3d_space, .components = 3});

    EXPECT_THROW(sys.bindMeshMotionField("displacement", "missing_field"),
                 svmp::FE::InvalidArgumentException);
    EXPECT_THROW(sys.bindMeshMotionField(MeshMotionFieldRole::Displacement, scalar),
                 svmp::FE::InvalidArgumentException);
    EXPECT_THROW(sys.bindMeshMotionField(MeshMotionFieldRole::Displacement, wrong_dim),
                 svmp::FE::InvalidArgumentException);
}

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

TEST(FESystem, LayoutRevisionDomainsTrackDefinitionAndSetup)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto initial = sys.feLayoutRevisionState();
    const auto initial_system_revision = sys.systemLayoutRevision();

    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    EXPECT_GT(sys.spaceRevision(), initial.space);
    EXPECT_GT(sys.blockLayoutRevision(), initial.block_layout);
    EXPECT_EQ(sys.dofLayoutRevision(), initial.dof_layout);

    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    const auto before_setup = sys.feLayoutRevisionState();
    sys.setup();
    EXPECT_GT(sys.dofLayoutRevision(), before_setup.dof_layout);
    EXPECT_GT(sys.blockLayoutRevision(), before_setup.block_layout);
    EXPECT_GT(sys.constraintLayoutRevision(), before_setup.constraint_layout);
    EXPECT_NE(sys.systemLayoutRevision(), initial_system_revision);

    const auto after_setup = sys.feLayoutRevisionState();
    sys.addConstraint(std::make_unique<TwoMasterConstraint>(0, 1, 2));
    EXPECT_GT(sys.constraintLayoutRevision(), after_setup.constraint_layout);
    EXPECT_EQ(sys.dofLayoutRevision(), after_setup.dof_layout);
    EXPECT_EQ(sys.spaceRevision(), after_setup.space);
}

TEST(FESystem, ConstraintRefreshTracksDeclaredGeometryDependencies)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    auto updates = std::make_shared<int>(0);
    sys.addConstraint(std::make_unique<GeometryValueConstraint>(updates));
    sys.setup();

    EXPECT_FALSE(sys.constraintStateStaleForCurrentRevisions());
    auto no_change = sys.refreshConstraintStateForCurrentRevisions();
    EXPECT_FALSE(no_change.dependency_changed);
    EXPECT_EQ(*updates, 0);

    auto cur = mesh->local_mesh().X_ref();
    cur[0] += 0.25;
    mesh->local_mesh().set_current_coords(cur);

    EXPECT_TRUE(sys.constraintStateStaleForCurrentRevisions());
    auto refreshed = sys.refreshConstraintStateForCurrentRevisions(/*time=*/0.0, /*dt=*/0.0);
    EXPECT_TRUE(refreshed.dependency_changed);
    EXPECT_TRUE(refreshed.value_update);
    EXPECT_FALSE(refreshed.structural_rebuild);
    EXPECT_EQ(*updates, 1);

    auto c = sys.constraints().getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 2.0);
    EXPECT_FALSE(sys.constraintStateStaleForCurrentRevisions());
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
