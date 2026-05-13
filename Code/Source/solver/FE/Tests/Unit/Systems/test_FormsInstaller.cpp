/**
 * @file test_FormsInstaller.cpp
 * @brief Unit tests for Systems FormsInstaller helpers
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FormsInstallerDetail.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Auxiliary/AuxiliaryOperatorBuilder.h"
#include "Auxiliary/AuxiliaryStateManager.h"

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/CutIntegrationContext.h"

#include "Elements/Element.h"

#include "Geometry/CutQuadrature.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Forms/WeakForm.h"

#include "Quadrature/QuadratureFactory.h"

#include "Spaces/H1Space.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"

#include "Systems/AuxiliaryQuadratureLayout.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::INVALID_FIELD_ID;
using svmp::FE::Real;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
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

bool exprContainsType(const svmp::FE::forms::FormExprNode& node,
                      svmp::FE::forms::FormExprType type)
{
    if (node.type() == type) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && exprContainsType(*child, type)) {
            return true;
        }
    }
    return false;
}

std::optional<std::uint32_t> firstAuxiliaryOutputRefIndex(
    const svmp::FE::forms::FormExprNode& node)
{
    if (node.type() == svmp::FE::forms::FormExprType::AuxiliaryOutputRef) {
        return node.slotIndex();
    }
    for (const auto& child : node.childrenShared()) {
        if (!child) {
            continue;
        }
        if (const auto found = firstAuxiliaryOutputRefIndex(*child)) {
            return found;
        }
    }
    return std::nullopt;
}

const svmp::FE::analysis::FormTerminalMetadata* findBridgeTerminal(
    const std::vector<svmp::FE::analysis::FormTerminalMetadata>& terminals,
    svmp::FE::analysis::FormTerminalKind kind,
    FieldId field_id = INVALID_FIELD_ID)
{
    const auto it = std::find_if(
        terminals.begin(),
        terminals.end(),
        [kind, field_id](const auto& terminal) {
            return terminal.kind == kind &&
                   (field_id == INVALID_FIELD_ID ||
                    terminal.field_id == field_id);
        });
    return it == terminals.end() ? nullptr : &*it;
}

std::shared_ptr<svmp::FE::systems::AuxiliaryStateModel> makeScalarOutputModel(
    const std::string& name)
{
    return svmp::FE::systems::AuxiliaryModelBuilder(name)
        .state("x")
        .ode("x", svmp::FE::forms::FormExpr::constant(0.0))
        .output("P_out", svmp::FE::systems::modelState("x"))
        .build();
}

class QuadratureOverrideElement final : public svmp::FE::elements::Element {
public:
    QuadratureOverrideElement(
        std::shared_ptr<const svmp::FE::elements::Element> prototype,
        std::shared_ptr<const svmp::FE::quadrature::QuadratureRule> quadrature)
        : prototype_(std::move(prototype))
        , quadrature_(std::move(quadrature))
    {
    }

    [[nodiscard]] svmp::FE::elements::ElementInfo info() const noexcept override
    {
        return prototype_->info();
    }

    [[nodiscard]] int dimension() const noexcept override
    {
        return prototype_->dimension();
    }

    [[nodiscard]] std::size_t num_dofs() const noexcept override
    {
        return prototype_->num_dofs();
    }

    [[nodiscard]] std::size_t num_nodes() const noexcept override
    {
        return prototype_->num_nodes();
    }

    [[nodiscard]] const svmp::FE::basis::BasisFunction& basis() const noexcept override
    {
        return prototype_->basis();
    }

    [[nodiscard]] std::shared_ptr<const svmp::FE::basis::BasisFunction> basis_ptr() const noexcept override
    {
        return prototype_->basis_ptr();
    }

    [[nodiscard]] std::shared_ptr<const svmp::FE::quadrature::QuadratureRule> quadrature() const noexcept override
    {
        return quadrature_;
    }

private:
    std::shared_ptr<const svmp::FE::elements::Element> prototype_{};
    std::shared_ptr<const svmp::FE::quadrature::QuadratureRule> quadrature_{};
};

class VariableQuadratureTetraSpace final : public svmp::FE::spaces::FunctionSpace {
public:
    VariableQuadratureTetraSpace()
        : base_(std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/2))
        , default_element_(base_->element_ptr())
        , reduced_element_(std::make_shared<QuadratureOverrideElement>(
              default_element_,
              svmp::FE::quadrature::QuadratureFactory::create(ElementType::Tetra4, /*order=*/1)))
    {
    }

    [[nodiscard]] svmp::FE::spaces::SpaceType space_type() const noexcept override
    {
        return base_->space_type();
    }
    [[nodiscard]] svmp::FE::FieldType field_type() const noexcept override
    {
        return base_->field_type();
    }
    [[nodiscard]] svmp::FE::Continuity continuity() const noexcept override
    {
        return base_->continuity();
    }
    [[nodiscard]] int value_dimension() const noexcept override { return base_->value_dimension(); }
    [[nodiscard]] int topological_dimension() const noexcept override
    {
        return base_->topological_dimension();
    }
    [[nodiscard]] int polynomial_order() const noexcept override
    {
        return base_->polynomial_order();
    }
    [[nodiscard]] ElementType element_type() const noexcept override
    {
        return base_->element_type();
    }
    [[nodiscard]] const svmp::FE::elements::Element& element() const noexcept override
    {
        return *default_element_;
    }
    [[nodiscard]] const svmp::FE::elements::Element& getElement(
        ElementType /*cell_type*/,
        GlobalIndex cell_id) const noexcept override
    {
        return (cell_id == 1) ? *reduced_element_ : *default_element_;
    }
    [[nodiscard]] std::shared_ptr<const svmp::FE::elements::Element> element_ptr() const noexcept override
    {
        return default_element_;
    }

private:
    std::shared_ptr<svmp::FE::spaces::H1Space> base_{};
    std::shared_ptr<const svmp::FE::elements::Element> default_element_{};
    std::shared_ptr<const svmp::FE::elements::Element> reduced_element_{};
};

class TwoCellTetraMeshAccess final : public svmp::FE::assembly::IMeshAccess {
public:
    TwoCellTetraMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0},  // 4
        };
        cells_ = {
            {0, 1, 2, 3},
            {1, 2, 3, 4},
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(4);
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] svmp::FE::LocalIndex getLocalFaceIndex(
        GlobalIndex /*face_id*/,
        GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(
        GlobalIndex /*face_id*/) const override
    {
        return {0, 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(/*face_id=*/0, /*cell_minus=*/0, /*cell_plus=*/1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

class TwoCellOwnedSubsetTetraMeshAccess final : public svmp::FE::assembly::IMeshAccess {
public:
    TwoCellOwnedSubsetTetraMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
        };
        cells_ = {
            {0, 1, 2, 3},
            {1, 2, 3, 4},
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_id == 0;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override
    {
        return static_cast<int>(cell_id);
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(4);
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] svmp::FE::LocalIndex getLocalFaceIndex(
        GlobalIndex /*face_id*/,
        GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(
        GlobalIndex /*face_id*/) const override
    {
        return {0, 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(/*face_id=*/0, /*cell_minus=*/0, /*cell_plus=*/1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

svmp::FE::dofs::MeshTopologyInfo twoCellTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 1, 2, 3, 4};
    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

svmp::FE::assembly::DenseMatrixView assembleBilinear(
    const svmp::FE::forms::FormExpr& form,
    const svmp::FE::spaces::FunctionSpace& test_space,
    const svmp::FE::spaces::FunctionSpace& trial_space,
    const svmp::FE::assembly::IMeshAccess& mesh)
{
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileBilinear(form);
    svmp::FE::forms::FormKernel kernel(std::move(ir));

    auto dof_map = svmp::FE::forms::test::createSingleTetraDofMap();
    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    svmp::FE::assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, test_space, trial_space, kernel, mat);
    return mat;
}

svmp::FE::assembly::DenseVectorView assembleLinear(
    const svmp::FE::forms::FormExpr& form,
    const svmp::FE::spaces::FunctionSpace& test_space,
    const svmp::FE::assembly::IMeshAccess& mesh)
{
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileLinear(form);
    svmp::FE::forms::FormKernel kernel(std::move(ir));

    auto dof_map = svmp::FE::forms::test::createSingleTetraDofMap();
    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    svmp::FE::assembly::DenseVectorView vec(4);
    vec.zero();
    (void)assembler.assembleVector(mesh, test_space, kernel, vec);
    return vec;
}

const svmp::FE::forms::NonlinearFormKernel* unwrapNonlinearKernel(
    const std::shared_ptr<svmp::FE::assembly::AssemblyKernel>& kernel)
{
    if (!kernel) {
        return nullptr;
    }
    if (const auto* jit = dynamic_cast<const svmp::FE::forms::jit::JITKernelWrapper*>(kernel.get())) {
        return dynamic_cast<const svmp::FE::forms::NonlinearFormKernel*>(&jit->fallbackKernel());
    }
    return dynamic_cast<const svmp::FE::forms::NonlinearFormKernel*>(kernel.get());
}

struct MovingMeshAssemblySnapshot {
    std::vector<Real> matrix;
    std::vector<Real> vector;
    GlobalIndex n_dofs{0};
    std::string kernel_name;
};

MovingMeshAssemblySnapshot assembleMovingMeshResidualWithPath(
    svmp::FE::forms::GeometryTangentPath path,
    bool enable_jit)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    auto vector_space = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, /*components=*/3);

    svmp::FE::systems::FESystem sys(mesh);
    const auto displacement =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    sys.addOperator("mesh");

    const auto u =
        svmp::FE::forms::FormExpr::trialFunction(*vector_space, "mesh_displacement");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*vector_space, "v");
    const auto residual =
        (inner(svmp::FE::forms::currentCoordinate(), v) +
         svmp::FE::forms::currentMeasure() * inner(u, v) +
         trace(svmp::FE::forms::currentJacobian()) * v.component(0)).dx();

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.geometry_sensitivity.mode =
        svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
    opts.compiler_options.geometry_sensitivity.mesh_motion_field = displacement;
    opts.compiler_options.geometry_tangent_path = path;
    opts.compiler_options.use_symbolic_tangent =
        path != svmp::FE::forms::GeometryTangentPath::ADReference;
    opts.compiler_options.jit = svmp::FE::forms::test::makeUnitTestJITOptions();
    opts.compiler_options.jit.enable = enable_jit;

    const auto installed = svmp::FE::systems::installResidualForm(
        sys, "mesh", displacement, displacement, residual, opts);
    EXPECT_NE(installed, nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    EXPECT_EQ(n_dofs, 12);

    std::vector<Real> solution(static_cast<std::size_t>(n_dofs), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.025) + Real(0.003) * static_cast<Real>(i));
    }

    svmp::FE::systems::SystemStateView state;
    state.u = solution;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "mesh";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    const auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    MovingMeshAssemblySnapshot snapshot;
    snapshot.n_dofs = n_dofs;
    snapshot.kernel_name = installed ? installed->name() : std::string{};
    snapshot.matrix.resize(static_cast<std::size_t>(n_dofs * n_dofs), 0.0);
    snapshot.vector.resize(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        snapshot.vector[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            snapshot.matrix[static_cast<std::size_t>(i * n_dofs + j)] =
                out.getMatrixEntry(i, j);
        }
    }
    return snapshot;
}

} // namespace

TEST(FormsInstaller, FormsInstaller_InstallFormulation_RegistersKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    auto installed = svmp::FE::systems::installFormulation(sys, "op", {u_field}, residual_form);
    ASSERT_FALSE(installed.residual.empty());
    ASSERT_NE(installed.residual[0], nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    // Use trialFunction for the verification helper (assembleBilinear requires it).
    const auto u_trial = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto mass = assembleBilinear((u_trial * v).dx(), *space, *space, *mesh);

    // Matrix matches mass.
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), mass.getMatrixEntry(i, j), 1e-12);
        }
    }

    // Vector matches mass * U.
    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += mass.getMatrixEntry(i, j) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(out.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormsInstaller, FormsInstaller_RegistersCutVolumeKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const int marker = 77;
    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form =
        (u * v).dCutVolume(marker, svmp::FE::forms::CutVolumeSide::Negative);

    auto installed = svmp::FE::systems::installFormulation(
        sys, "op", {u_field}, residual_form);
    ASSERT_FALSE(installed.residual.empty());
    ASSERT_NE(installed.residual[0], nullptr);

    const auto& def = sys.operatorDefinition("op");
    EXPECT_TRUE(def.cells.empty());
    ASSERT_EQ(def.cut_volumes.size(), 1u);
    EXPECT_EQ(def.cut_volumes.front().marker, marker);
    EXPECT_EQ(def.cut_volumes.front().side,
              svmp::FE::geometry::CutIntegrationSide::Negative);
    EXPECT_EQ(def.cut_volumes.front().test_field, u_field);
    EXPECT_EQ(def.cut_volumes.front().trial_field, u_field);
    EXPECT_NE(def.cut_volumes.front().kernel, nullptr);
}

TEST(FormsInstaller, FormsInstaller_AssemblesCutVolumeKernelThroughSystem)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    constexpr int marker = 78;
    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form =
        (u * v).dCutVolume(marker, svmp::FE::forms::CutVolumeSide::Negative);

    auto installed = svmp::FE::systems::installFormulation(
        sys, "op", {u_field}, residual_form);
    ASSERT_FALSE(installed.residual.empty());
    ASSERT_NE(installed.residual[0], nullptr);

    auto cut_context = std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
    svmp::FE::geometry::CutQuadratureRule rule;
    rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
    rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
    rule.parent_measure = Real{1.0} / Real{6.0};
    rule.measure = Real{1.0} / Real{12.0};
    rule.volume_fraction = Real{0.5};
    rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
    rule.provenance.parent_entity = 0;
    rule.provenance.marker = marker;
    rule.points.push_back(svmp::FE::geometry::CutQuadraturePoint{
        .point = {{Real{0.25}, Real{0.25}, Real{0.25}}},
        .weight = rule.measure,
    });

    svmp::FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.side = svmp::FE::geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = rule.volume_fraction;
    cut_context->addGeneratedVolumeRule(marker, metadata, rule);
    sys.setCutIntegrationContext(cut_context);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    std::vector<Real> U = {Real{1.0}, Real{1.0}, Real{1.0}, Real{1.0}};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();
    const auto result = sys.assemble(req, state, &out, &out);
    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.elements_assembled, 1);

    constexpr Real expected_vector = Real{1.0} / Real{48.0};
    constexpr Real expected_matrix = Real{1.0} / Real{192.0};
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(out.getVectorEntry(i), expected_vector, 1.0e-12);
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), expected_matrix, 1.0e-12);
        }
    }
}

TEST(FormsInstaller, InstallFormulationWithMetadata_CapturesInstalledRecord)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto a_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "a", .space = space, .components = 1});
    const auto b_field = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "b", .space = space, .components = 1});
    sys.addOperator("op");

    const auto a = svmp::FE::forms::FormExpr::stateField(a_field, *space, "a");
    const auto wa = svmp::FE::forms::FormExpr::testFunction(a_field, *space, "wa");
    const auto wb = svmp::FE::forms::FormExpr::testFunction(b_field, *space, "wb");
    const auto residual = (a * wa).dx() + (a * wb).ds(5);

    svmp::FE::analysis::FormAnalysisBridgeOptions metadata_options;
    metadata_options.contribution_name = "generic_boundary_coupling";
    metadata_options.origin = "FormsInstallerTest";
    metadata_options.system_name = "shared_system";

    const auto installed = svmp::FE::systems::installFormulationWithMetadata(
        sys,
        "op",
        {a_field, b_field},
        residual,
        svmp::FE::systems::FormInstallOptions{},
        metadata_options);

    ASSERT_NE(installed.kernels.mixed_plan, nullptr);

    const auto& metadata = installed.analysis;
    EXPECT_EQ(metadata.contribution_name, "generic_boundary_coupling");
    EXPECT_TRUE(metadata.contribution_name_explicit);
    EXPECT_EQ(metadata.origin, "FormsInstallerTest");
    EXPECT_EQ(metadata.system_name, "shared_system");
    EXPECT_EQ(metadata.operator_tag, "op");
    ASSERT_EQ(metadata.installed_fields.size(), 2u);
    EXPECT_EQ(metadata.installed_fields[0], a_field);
    EXPECT_EQ(metadata.installed_fields[1], b_field);

    const auto* a_state = findBridgeTerminal(
        metadata.terminals,
        svmp::FE::analysis::FormTerminalKind::StateField,
        a_field);
    ASSERT_NE(a_state, nullptr);
    ASSERT_TRUE(a_state->graph_variable.has_value());
    EXPECT_EQ(*a_state->graph_variable,
              svmp::FE::analysis::VariableKey::field(a_field));

    const auto* b_test = findBridgeTerminal(
        metadata.terminals,
        svmp::FE::analysis::FormTerminalKind::TestField,
        b_field);
    ASSERT_NE(b_test, nullptr);
    EXPECT_EQ(b_test->domain, svmp::FE::analysis::DomainKind::Boundary);
    EXPECT_EQ(b_test->boundary_marker, 5);

    const auto block_it = std::find_if(
        metadata.installed_blocks.begin(),
        metadata.installed_blocks.end(),
        [a_field, b_field](const auto& block) {
            return block.residual_row ==
                       svmp::FE::analysis::VariableKey::field(b_field) &&
                   block.dependency ==
                       svmp::FE::analysis::VariableKey::field(a_field);
        });
    ASSERT_NE(block_it, metadata.installed_blocks.end());
    EXPECT_TRUE(block_it->has_matrix);
    EXPECT_TRUE(block_it->has_vector);
    ASSERT_FALSE(block_it->domains.empty());
    EXPECT_EQ(block_it->domains[0], svmp::FE::analysis::DomainKind::Boundary);

    EXPECT_TRUE(svmp::FE::analysis::bridgeFeatureAvailable(
        metadata, svmp::FE::analysis::FormBridgeFeature::ContributionIdentity));
    EXPECT_TRUE(svmp::FE::analysis::bridgeFeatureAvailable(
        metadata, svmp::FE::analysis::FormBridgeFeature::OwningSystem));
    EXPECT_TRUE(svmp::FE::analysis::bridgeFeatureAvailable(
        metadata, svmp::FE::analysis::FormBridgeFeature::InstalledBlocks));
}

TEST(FormsInstaller, FormsInstaller_InstallFormulation_AffineWithRHS)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto one = svmp::FE::forms::FormExpr::constant(1.0);

    // Residual: ∫ u v dx - ∫ 1 * v dx
    const auto residual_form = (u * v - one * v).dx();

    auto installed = svmp::FE::systems::installFormulation(sys, "op", {u_field}, residual_form);
    ASSERT_FALSE(installed.residual.empty());
    ASSERT_NE(installed.residual[0], nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    // Use trialFunction for the verification helper (assembleBilinear requires it).
    const auto u_trial = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto mass = assembleBilinear((u_trial * v).dx(), *space, *space, *mesh);
    const auto rhs = assembleLinear((-one * v).dx(), *space, *mesh);

    for (GlobalIndex i = 0; i < 4; ++i) {
        // Matrix matches mass.
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), mass.getMatrixEntry(i, j), 1e-12);
        }

        // Vector matches mass*U + rhs.
        Real expected = rhs.getVectorEntry(i);
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += mass.getMatrixEntry(i, j) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(out.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormsInstaller, OperatorMatrixStateIndependenceDetectedFromInstalledForms)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem affine_sys(mesh);
    const auto affine_u =
        affine_sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    affine_sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(affine_u, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto affine_residual = inner(grad(u), grad(v)).dx();
    (void)svmp::FE::systems::installFormulation(affine_sys, "op", {affine_u}, affine_residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    affine_sys.setup({}, inputs);
    EXPECT_TRUE(affine_sys.operatorMatrixStateIndependent("op"));

    svmp::FE::systems::FESystem nonlinear_sys(mesh);
    const auto nonlinear_u =
        nonlinear_sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    nonlinear_sys.addOperator("op");

    const auto u_nl = svmp::FE::forms::FormExpr::stateField(nonlinear_u, *space, "u");
    const auto v_nl = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto nonlinear_residual = (u_nl * u_nl * v_nl).dx();
    (void)svmp::FE::systems::installFormulation(nonlinear_sys, "op", {nonlinear_u}, nonlinear_residual);
    nonlinear_sys.setup({}, inputs);
    EXPECT_FALSE(nonlinear_sys.operatorMatrixStateIndependent("op"));
}

TEST(FormsInstaller, FormsInstaller_InstallFormulation_MultiOpInstallsConstraintsOnce)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    const auto bc = svmp::FE::forms::bc::strongDirichlet(u_field, marker, svmp::FE::forms::FormExpr::constant(2.5), "u");
    svmp::FE::systems::installStrongDirichlet(sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    (void)svmp::FE::systems::installFormulation(sys, "op", {u_field}, residual_form);
    (void)svmp::FE::systems::installFormulation(sys, "op2", {u_field}, residual_form);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));

    for (svmp::FE::GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-15);
    }

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView mat(sys.dofHandler().getNumDofs());
    mat.zero();
    (void)sys.assemble(req, state, &mat, nullptr);

    for (svmp::FE::GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(mat.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (svmp::FE::GlobalIndex j = 0; j < mat.numCols(); ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mat.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_ADModeForward)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * u * v).dx();

    auto installed = svmp::FE::systems::installResidualForm(
        sys, "op", u_field, u_field, residual_form,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Forward});
    ASSERT_NE(installed, nullptr);
    EXPECT_NE(unwrapNonlinearKernel(installed), nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_ADModeReverse)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * u * v).dx();

    auto installed = svmp::FE::systems::installResidualForm(
        sys, "op", u_field, u_field, residual_form,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Reverse});
    ASSERT_NE(installed, nullptr);
    EXPECT_NE(unwrapNonlinearKernel(installed), nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(FormsInstaller, MovingMeshSymbolicWithADCheckInstallsJITPrimaryAndPassesRuntimeReference)
{
    requireLLVMJITOrSkip();

    const auto snapshot = assembleMovingMeshResidualWithPath(
        svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck,
        /*enable_jit=*/true);

    EXPECT_NE(snapshot.kernel_name.find("SymbolicADReferenceCheckKernel"), std::string::npos);
    EXPECT_NE(snapshot.kernel_name.find("JITKernelWrapper"), std::string::npos);
    EXPECT_NE(snapshot.kernel_name.find("SymbolicNonlinearFormKernel"), std::string::npos);
}

TEST(FormsInstaller, MovingMeshSymbolicJITAssemblyMatchesADReferenceAssembly)
{
    requireLLVMJITOrSkip();

    const auto ad = assembleMovingMeshResidualWithPath(
        svmp::FE::forms::GeometryTangentPath::ADReference,
        /*enable_jit=*/false);
    const auto symbolic_jit = assembleMovingMeshResidualWithPath(
        svmp::FE::forms::GeometryTangentPath::SymbolicRequired,
        /*enable_jit=*/true);

    ASSERT_EQ(symbolic_jit.n_dofs, ad.n_dofs);
    ASSERT_EQ(symbolic_jit.vector.size(), ad.vector.size());
    ASSERT_EQ(symbolic_jit.matrix.size(), ad.matrix.size());
    EXPECT_NE(symbolic_jit.kernel_name.find("JITKernelWrapper"), std::string::npos);
    EXPECT_NE(symbolic_jit.kernel_name.find("SymbolicNonlinearFormKernel"), std::string::npos);

    for (std::size_t i = 0; i < symbolic_jit.vector.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "vector i=" << i);
        EXPECT_NEAR(symbolic_jit.vector[i], ad.vector[i], 1.0e-12);
    }
    for (std::size_t i = 0; i < symbolic_jit.matrix.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "matrix flat i=" << i);
        EXPECT_NEAR(symbolic_jit.matrix[i], ad.matrix[i], 2.0e-10);
    }
}

TEST(FormsInstaller, MovingGeometrySensitivityRequiresMeshTrialField)
{
    constexpr int marker = 12;
    auto mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(
            marker);
    auto scalar_space =
        std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    auto vector_space =
        std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, /*components=*/3);

    svmp::FE::systems::FESystem sys(mesh);
    const auto fluid = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "fluid", .space = scalar_space, .components = 1});
    const auto displacement = sys.addField(
        svmp::FE::systems::FieldSpec{.name = "mesh_displacement",
                                     .space = vector_space,
                                     .components = 3});
    sys.bindMeshMotionField(svmp::FE::systems::MeshMotionFieldRole::Displacement,
                            displacement);
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(fluid, *scalar_space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(fluid, *scalar_space, "v");
    const auto residual = (u * v * svmp::FE::forms::currentMeasure()).ds(marker);

    svmp::FE::systems::FormInstallOptions install;
    install.compiler_options.geometry_sensitivity.mode =
        svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
    install.compiler_options.geometry_sensitivity.mesh_motion_field = displacement;
    install.compiler_options.geometry_tangent_path =
        svmp::FE::forms::GeometryTangentPath::SymbolicRequired;

    try {
        (void)svmp::FE::systems::installFormulation(sys, "op", {fluid}, residual, install);
        FAIL() << "Expected missing moving-geometry tangent path diagnostic";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("moving-geometry tangent path"), std::string::npos);
        EXPECT_NE(message.find("FormInstallOptions::extra_trial_fields"),
                  std::string::npos);
        EXPECT_NE(message.find(std::to_string(displacement)), std::string::npos);
    }

    install.extra_trial_fields.push_back(displacement);
    const auto kernels =
        svmp::FE::systems::installFormulation(sys, "op", {fluid}, residual, install);
    ASSERT_EQ(kernels.jacobian_blocks.size(), 1u);
    ASSERT_EQ(kernels.jacobian_blocks.front().size(), 2u);
    EXPECT_NE(kernels.jacobian_blocks.front()[1], nullptr);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_InvalidFieldId_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    EXPECT_THROW((void)svmp::FE::systems::installResidualForm(sys, "op", INVALID_FIELD_ID, u_field, residual_form),
                 svmp::FE::InvalidArgumentException);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualBlocks_MultipleBlocksRegistered)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    const auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, (u * v).dx());
    blocks.setBlock(1, 1, (p * q).dx());

    const std::array<FieldId, 2> fields = {u_field, p_field};
    const auto kernels = svmp::FE::systems::installResidualBlocks(sys, "op", fields, fields, blocks);
    ASSERT_EQ(kernels.size(), 2u);
    ASSERT_EQ(kernels[0].size(), 2u);

    EXPECT_NE(kernels[0][0], nullptr);
    EXPECT_EQ(kernels[0][1], nullptr);
    EXPECT_EQ(kernels[1][0], nullptr);
    EXPECT_NE(kernels[1][1], nullptr);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualBlocks_EmptyBlocksSkipped)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, (u * v).dx());

    const std::array<FieldId, 2> fields = {u_field, p_field};
    const auto kernels = svmp::FE::systems::installResidualBlocks(sys, "op", fields, fields, blocks);
    EXPECT_NE(kernels[0][0], nullptr);
    EXPECT_EQ(kernels[0][1], nullptr);
    EXPECT_EQ(kernels[1][0], nullptr);
    EXPECT_EQ(kernels[1][1], nullptr);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualBlocks_InitializerListOverload)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, (u * v).dx());

    // Span overload.
    svmp::FE::systems::FESystem sys_span(mesh);
    const auto u0 = sys_span.addField(svmp::FE::systems::FieldSpec{.name = "u0", .space = space, .components = 1});
    const auto u1 = sys_span.addField(svmp::FE::systems::FieldSpec{.name = "u1", .space = space, .components = 1});
    sys_span.addOperator("op");
    const std::array<FieldId, 2> fields_span = {u0, u1};
    const auto kernels_span = svmp::FE::systems::installResidualBlocks(sys_span, "op", fields_span, fields_span, blocks);

    // Initializer-list overload.
    svmp::FE::systems::FESystem sys_list(mesh);
    const auto v0 = sys_list.addField(svmp::FE::systems::FieldSpec{.name = "v0", .space = space, .components = 1});
    const auto v1 = sys_list.addField(svmp::FE::systems::FieldSpec{.name = "v1", .space = space, .components = 1});
    sys_list.addOperator("op");
    const auto kernels_list = svmp::FE::systems::installResidualBlocks(sys_list, "op", {v0, v1}, {v0, v1}, blocks);

    ASSERT_EQ(kernels_span.size(), kernels_list.size());
    for (std::size_t i = 0; i < kernels_span.size(); ++i) {
        ASSERT_EQ(kernels_span[i].size(), kernels_list[i].size());
        for (std::size_t j = 0; j < kernels_span[i].size(); ++j) {
            EXPECT_EQ(static_cast<bool>(kernels_span[i][j]), static_cast<bool>(kernels_list[i][j]));
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallFormulation_CoupledSeparatesVectorAndMatrix)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(u_field, *space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(p_field, *space, "q");

    const auto residual =
        (u_state * v + p_state * v).dx() +  // depends on u and p
        (q * u_state).dx();                  // depends on u only

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    const auto installed =
        svmp::FE::systems::installFormulation(sys, "op", {u_field, p_field}, residual, opts);

    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_TRUE(installed.mixed_plan->usesMonolithicCellKernel());
    ASSERT_EQ(installed.mixed_plan->blocks.size(), 3u);

    std::size_t matrix_blocks = 0;
    std::size_t vector_blocks = 0;
    for (const auto& block : installed.mixed_plan->blocks) {
        if (block.want_matrix) {
            ++matrix_blocks;
        }
        if (block.want_vector) {
            ++vector_blocks;
        }
    }
    EXPECT_EQ(matrix_blocks, 3u);
    EXPECT_EQ(vector_blocks, 2u);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 8);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.1) * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    svmp::FE::systems::AssemblyRequest req_mat;
    req_mat.op = "op";
    req_mat.want_matrix = true;

    svmp::FE::systems::AssemblyRequest req_both;
    req_both.op = "op";
    req_both.want_matrix = true;
    req_both.want_vector = true;

    svmp::FE::assembly::DenseVectorView R_vec(n_dofs);
    svmp::FE::assembly::DenseMatrixView J_mat(n_dofs);
    svmp::FE::assembly::DenseSystemView both(n_dofs);
    R_vec.zero();
    J_mat.zero();
    both.zero();

    (void)sys.assemble(req_vec, state, nullptr, &R_vec);
    (void)sys.assemble(req_mat, state, &J_mat, nullptr);
    (void)sys.assemble(req_both, state, &both, &both);

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        EXPECT_NEAR(both.getVectorEntry(i), R_vec.getVectorEntry(i), 1e-12);
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            EXPECT_NEAR(both.getMatrixEntry(i, j), J_mat.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(FormsInstaller, FormsInstaller_MixedResidualNegativeBoundaryTermCompiles)
{
    constexpr int marker = 19;
    auto mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field =
        sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field =
        sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(u_field, *space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(p_field, *space, "q");

    const auto residual =
        (u_state * v).dx() +
        (p_state * q).dx() -
        (u_state * v).ds(marker);

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(sys, "op", {u_field, p_field}, residual, opts));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    EXPECT_NO_THROW(sys.setup({}, inputs));

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 8);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseMatrixView J(n_dofs);
    svmp::FE::assembly::DenseVectorView R(n_dofs);
    J.zero();
    R.zero();

    const auto result = sys.assemble(req, state, &J, &R);
    EXPECT_TRUE(result.success);
}

TEST(FormsInstaller, FormsInstaller_InstallCoupledResidual_StateFieldsTracked)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(u_field, *space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(p_field, *space, "q");

    svmp::FE::forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(0, (u_state * v + p_state * v).dx()); // depends on u and p
    residual.setBlock(1, (q * u_state).dx());               // depends on u only

    const std::array<FieldId, 2> fields = {u_field, p_field};
    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    const auto installed =
        svmp::FE::systems::installCoupledResidual(sys, "op", fields, fields, residual, opts);

    ASSERT_EQ(installed.jacobian_blocks.size(), 2u);
    ASSERT_EQ(installed.jacobian_blocks[0].size(), 2u);
    ASSERT_EQ(installed.jacobian_blocks[1].size(), 2u);

    EXPECT_NE(installed.jacobian_blocks[0][0], nullptr);
    EXPECT_NE(installed.jacobian_blocks[0][1], nullptr);
    EXPECT_NE(installed.jacobian_blocks[1][0], nullptr);
    EXPECT_EQ(installed.jacobian_blocks[1][1], nullptr); // dR_p / dp == 0

    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_TRUE(installed.mixed_plan->usesMonolithicCellKernel());
    ASSERT_EQ(installed.mixed_plan->blocks.size(), 3u);

    std::size_t vector_blocks = 0;
    for (const auto& block : installed.mixed_plan->blocks) {
        if (block.want_vector) {
            ++vector_blocks;
        }
    }
    EXPECT_EQ(vector_blocks, 2u);
}

TEST(FormsInstaller, FormsInstaller_InstallFormulation_FormWithoutDx_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    try {
        (void)svmp::FE::systems::installFormulation(sys, "op", {u_field}, (u * v));
        FAIL() << "Expected installFormulation to throw for residual missing dx()/ds()/dS()";
    } catch (const svmp::FE::InvalidArgumentException&) {
        SUCCEED();
    } catch (const std::invalid_argument&) {
        SUCCEED();
    }
}

TEST(FormsInstaller, FormsInstaller_InstanceQualifiedAuxiliaryOutput)
{
    // Deploy two auxiliary models with same output name, then install a
    // form referencing AuxiliaryOutput(instance, name) to verify the
    // FormsInstaller resolves the instance-qualified path correctly.
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Two models both with output named "P_out".
    auto model_a = AuxiliaryModelBuilder("model_a")
        .state("x")
        .ode("x", -modelState("x"))
        .output("P_out", modelState("x") * forms::FormExpr::constant(2.0))
        .build();

    auto model_b = AuxiliaryModelBuilder("model_b")
        .state("y")
        .ode("y", -modelState("y"))
        .output("P_out", modelState("y") * forms::FormExpr::constant(3.0))
        .build();

    sys.deployAuxiliaryModel(
        use(model_a)
            .name("inst_a")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .initialize({1.0}));

    sys.deployAuxiliaryModel(
        use(model_b)
            .name("inst_b")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .initialize({1.0}));

    sys.finalizeAuxiliaryLayout();

    // Install a form referencing instance-qualified AuxiliaryOutput.
    // The form: (AuxiliaryOutput("inst_a", "P_out") * v).dx()
    const auto v = forms::FormExpr::testFunction(*space, "v");
    auto aux_a = forms::AuxiliaryOutput("inst_a", "P_out");
    const auto residual_form = (aux_a * v).dx();

    // This should NOT throw — the FormsInstaller should resolve
    // "inst_a/P_out" to a valid slot via auxiliaryOutputSlotOf("inst_a", "P_out").
    auto installed = installFormulation(sys, "op", {u_field}, residual_form);
    EXPECT_FALSE(installed.residual.empty());

    // Verify slot resolution correctness: inst_a and inst_b should get
    // different slots, and the form should reference inst_a's slot.
    auto slot_a = sys.auxiliaryOutputSlotOf("inst_a", "P_out");
    auto slot_b = sys.auxiliaryOutputSlotOf("inst_b", "P_out");
    EXPECT_NE(slot_a, slot_b);

    // Prepare auxiliary state: x=5 → inst_a output = 5*2 = 10,
    //                          y=7 → inst_b output = 7*3 = 21.
    sys.prepareAuxiliaryForAssembly({});

    auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), std::max(slot_a, slot_b));
    // x was initialized to 1.0, so P_out_a = 1.0*2 = 2.0
    // y was initialized to 1.0, so P_out_b = 1.0*3 = 3.0.
    EXPECT_DOUBLE_EQ(outputs[slot_a], 2.0);
    EXPECT_DOUBLE_EQ(outputs[slot_b], 3.0);
}

TEST(FormsInstaller, FormsInstaller_AuxiliaryInput_AutoResolution)
{
    // Deploy an auxiliary model that consumes an input, register the input,
    // then install a form referencing AuxiliaryInput("Q") to verify the
    // FormsInstaller auto-resolves the input symbol to a slot ref.
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Register an auxiliary input "Q".
    auto& reg = sys.auxiliaryInputRegistry();
    reg.registerInput({.name = "Q", .size = 1},
                      [](Real, Real, std::span<Real> out) { out[0] = 42.0; });

    // Deploy a model that uses input "Q".
    auto model = AuxiliaryModelBuilder("rcr")
        .state("P").input("Q")
        .ode("P", -modelState("P"))
        .output("P_out", modelState("P"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("rcr_inst")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .bind("Q", "Q")
            .initialize({0.0}));

    sys.finalizeAuxiliaryLayout();

    // Install a form referencing AuxiliaryInput("Q").
    const auto v = forms::FormExpr::testFunction(*space, "v");
    auto q = forms::AuxiliaryInput("Q");
    const auto residual_form = (q * v).dx();

    // This should NOT throw — "Q" is registered and should resolve.
    auto installed = installFormulation(sys, "op", {u_field}, residual_form);
    EXPECT_FALSE(installed.residual.empty());

    // Verify the slot was resolved: "Q" should have slot 0.
    auto input_slot = reg.slotOf("Q");
    EXPECT_EQ(input_slot, 0u);
}

TEST(FormsInstaller, RegisterSampledFieldInput_ReadsFromSolution)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Install a simple form so setup() can build DOF maps.
    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Register a sampled field input for "u" with 4 vertices.
    sys.registerSampledFieldInput("u_sampled", "u", 4);

    // Deploy an auxiliary model that uses the sampled input.
    auto model = AuxiliaryModelBuilder("sampler")
        .state("x")
        .input("Q")
        .ode("x", -modelState("x"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("samp_inst")
            .scope(AuxiliaryStateScope::Node)
            .entityCount(4)
            .bind("Q", "u_sampled")
            .initialize({0.0}));

    sys.finalizeAuxiliaryLayout();

    // Set solution vector and prepare assembly.
    std::vector<Real> U = {10.0, 20.0, 30.0, 40.0};
    SystemStateView state;
    state.u = U; state.time = 0.0; state.dt = 0.1;
    sys.prepareAuxiliaryForAssembly(state);

    // The sampled input should now contain the field values at each vertex.
    auto& reg = sys.auxiliaryInputRegistry();
    ASSERT_TRUE(reg.hasInput("u_sampled"));
    for (std::size_t i = 0; i < 4; ++i) {
        auto vals = reg.valuesOf("u_sampled", i);
        ASSERT_FALSE(vals.empty());
        EXPECT_NEAR(vals[0], U[i], 1e-12)
            << "Vertex " << i << " should have field value " << U[i];
    }
}

TEST(FormsInstaller, RegisterBoundaryNodalSumInput_SumsCorrectNodes)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    // SingleTetraOneBoundaryFaceMeshAccess: 1 Tet4, boundary face 0 with marker 42.
    // Tet4 face 0 local vertices = {1,2,3}.
    // Global nodes: 0,1,2,3.  Face nodes = {1,2,3}.
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(42);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *space, "p");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Register boundary nodal sum for boundary marker 42.
    sys.registerBoundaryNodalSumInput("p_bnd", "p", 42);

    // Deploy aux model using the boundary sum.
    auto model = AuxiliaryModelBuilder("bnd_test")
        .state("x")
        .input("Q")
        .ode("x", -modelState("x"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("bnd_inst")
            .bind("Q", "p_bnd")
            .initialize({0.0}));
    sys.finalizeAuxiliaryLayout();

    // Solution: p = {100, 200, 300, 400} at nodes 0,1,2,3.
    // Boundary face nodes = {1,2,3} → sum = 200 + 300 + 400 = 900.
    std::vector<Real> U = {100.0, 200.0, 300.0, 400.0};
    SystemStateView state;
    state.u = U; state.time = 0.0; state.dt = 0.1;
    sys.prepareAuxiliaryForAssembly(state);

    auto& reg = sys.auxiliaryInputRegistry();
    ASSERT_TRUE(reg.hasInput("p_bnd"));
    auto vals = reg.valuesOf("p_bnd");
    ASSERT_FALSE(vals.empty());
    // Node 0 is NOT on the boundary face, so only 200+300+400 = 900.
    EXPECT_NEAR(vals[0], 900.0, 1e-10);
}

TEST(FormsInstaller, DynamicAuxiliaryOutputLoweringPreservesMetadataOutputRef)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    const int marker = 42;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");

    auto Q = sys.boundaryIntegral(u, marker);
    auto model = AuxiliaryModelBuilder("rcr_like")
        .input("Q")
        .state("X")
        .param("Rp")
        .param("Rd")
        .param("C")
        .param("Pd")
        .ode("X",
             (modelInput("Q") - (modelState("X") - modelParam("Pd")) / modelParam("Rd")) /
                 modelParam("C"))
        .output("P_out", modelState("X") + modelParam("Rp") * modelInput("Q"))
        .build();

    auto inst = sys.deploy(
        use(model).name("rcr_inst")
            .boundary(marker)
            .monolithic()
            .params({{"Rp", 10.0}, {"Rd", 100.0}, {"C", 0.001}, {"Pd", 50.0}})
            .bind("Q", Q)
            .initialState({{"X", 50.0}}));

    installFormulation(sys, "op", {u_field}, (inst.output("P_out") * v).dx());

    const auto lowered = sys.loweredAuxiliaryOutputExpr("rcr_inst/P_out");
    ASSERT_TRUE(lowered.has_value());
    ASSERT_TRUE(lowered->isValid());
    ASSERT_NE(lowered->node(), nullptr);
    EXPECT_TRUE(exprContainsType(*lowered->node(), forms::FormExprType::AuxiliaryStateRef));
    EXPECT_TRUE(exprContainsType(*lowered->node(), forms::FormExprType::AuxiliaryInputRef));
    EXPECT_FALSE(exprContainsType(*lowered->node(), forms::FormExprType::AuxiliaryOutputRef));
    EXPECT_FALSE(exprContainsType(*lowered->node(), forms::FormExprType::ParameterRef));

    const auto& recs = sys.formulationRecords();
    ASSERT_EQ(recs.size(), 1u);
    ASSERT_EQ(recs[0].block_residual_exprs.size(), 1u);
    ASSERT_NE(recs[0].block_residual_exprs[0].second, nullptr);
    EXPECT_TRUE(exprContainsType(
        *recs[0].block_residual_exprs[0].second,
        forms::FormExprType::AuxiliaryOutputRef));

    const auto output_id = sys.auxiliaryOutputIdOf("rcr_inst", "P_out");
    ASSERT_NE(output_id, static_cast<std::size_t>(-1));

    const auto ref_id = firstAuxiliaryOutputRefIndex(*recs[0].block_residual_exprs[0].second);
    ASSERT_TRUE(ref_id.has_value());
    EXPECT_EQ(*ref_id, output_id);

    const auto consumers = sys.consumersOfAuxiliaryOutput(output_id);
    ASSERT_EQ(consumers.size(), 1u);
    EXPECT_EQ(consumers[0].qualified_output_name, "rcr_inst/P_out");
    EXPECT_EQ(consumers[0].operator_tag, "op");
}

TEST(FormsInstaller, BoundaryConditionManagerPreservesAuxiliaryOutputMetadataUntilInstall)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    const int marker = 55;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    const auto Q = sys.boundaryIntegral(u, marker);

    auto model = AuxiliaryModelBuilder("resistive_bc")
                     .input("Q")
                     .state("P", AuxiliaryVariableKind::Algebraic)
                     .param("Rp")
                     .algebraic("P", modelState("P") - modelParam("Rp") * modelInput("Q"))
                     .output("P_out", modelState("P"))
                     .build();

    auto inst = sys.deploy(
        use(model).name("resistive_bc_inst")
            .boundary(marker)
            .monolithic()
            .param("Rp", 110.0)
            .bind("Q", Q)
            .initialState({{"P", 0.0}}));

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<forms::bc::NaturalBC>(marker, inst.output("P_out")));

    auto residual = inner(grad(u), grad(v)).dx();
    bc_manager.applyAll(sys, residual, u, v, u_field);
    (void)installFormulation(sys, "op", {u_field}, residual);

    const auto output_id = sys.auxiliaryOutputIdOf("resistive_bc_inst", "P_out");
    ASSERT_NE(output_id, static_cast<std::size_t>(-1));

    const auto& recs = sys.formulationRecords();
    ASSERT_EQ(recs.size(), 1u);
    ASSERT_EQ(recs[0].block_residual_exprs.size(), 1u);
    ASSERT_NE(recs[0].block_residual_exprs[0].second, nullptr);
    EXPECT_TRUE(exprContainsType(
        *recs[0].block_residual_exprs[0].second,
        forms::FormExprType::AuxiliaryOutputRef));

    const auto ref_id = firstAuxiliaryOutputRefIndex(*recs[0].block_residual_exprs[0].second);
    ASSERT_TRUE(ref_id.has_value());
    EXPECT_EQ(*ref_id, output_id);

    const auto consumers = sys.consumersOfAuxiliaryOutput(output_id);
    ASSERT_EQ(consumers.size(), 1u);
    EXPECT_EQ(consumers[0].qualified_output_name, "resistive_bc_inst/P_out");
    EXPECT_EQ(consumers[0].operator_tag, "op");
}

TEST(FormsInstaller, QualifiedAuxiliaryOutputIdAndConsumersExistBeforeLayoutFinalization)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_identity"))
            .name("qp_identity_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto output_id = sys.auxiliaryOutputIdOf("qp_identity_inst", "P_out");
    ASSERT_NE(output_id, static_cast<std::size_t>(-1));
    const auto* desc = sys.auxiliaryOutputDescriptor(output_id);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->instance_name, "qp_identity_inst");
    EXPECT_EQ(desc->output_name, "P_out");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    const auto consumers = sys.consumersOfAuxiliaryOutput(output_id);
    ASSERT_EQ(consumers.size(), 1u);
    EXPECT_EQ(consumers[0].qualified_output_name, "qp_identity_inst/P_out");
    EXPECT_EQ(consumers[0].operator_tag, "op");
    EXPECT_EQ(consumers[0].reference_field, u_field);
}

TEST(FormsInstaller, CellScopedAuxiliaryOutputResolvesPerCurrentCell)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<TwoCellTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto model = AuxiliaryModelBuilder("cell_output")
        .state("x")
        .ode("x", forms::FormExpr::constant(0.0))
        .output("P_out", modelState("x"))
        .build();

    auto inst = sys.deploy(
        use(model).name("cell_output_inst")
            .scope(AuxiliaryStateScope::Cell)
            .partitioned("BackwardEuler")
            .entityCount(2)
            .initialize({0.0}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx() - (inst.output("P_out") * v).dx();
    (void)installFormulation(sys, "op", {u_field}, residual_form);

    SetupInputs inputs;
    inputs.topology_override = twoCellTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();
    sys.beginTimeStep();

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state;
    state.u = sol;
    state.time = 0.0;
    state.dt = 0.1;

    auto assembleResidual = [&](std::vector<Real> cell_values) {
        auto& blk = sys.auxiliaryStateManager().getBlock("cell_output_inst");
        blk.initialize(cell_values);
        sys.prepareAuxiliaryForAssembly(state, /*is_nonlinear_iteration=*/false);

        svmp::FE::assembly::DenseVectorView out(
            static_cast<svmp::FE::GlobalIndex>(sol.size()));
        out.zero();

        AssemblyRequest req;
        req.op = "op";
        req.want_vector = true;
        req.want_matrix = false;

        const auto result = sys.assemble(req, state, nullptr, &out);
        EXPECT_TRUE(result.success);

        std::vector<Real> values(sol.size(), 0.0);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = out.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        }
        return values;
    };

    const auto cell0_only = assembleResidual({1.0, 0.0});
    ASSERT_EQ(cell0_only.size(), 2u);
    EXPECT_GT(std::abs(cell0_only[0]), 1e-12);
    EXPECT_NEAR(cell0_only[1], 0.0, 1e-12);

    const auto cell1_only = assembleResidual({0.0, 1.0});
    ASSERT_EQ(cell1_only.size(), 2u);
    EXPECT_NEAR(cell1_only[0], 0.0, 1e-12);
    EXPECT_GT(std::abs(cell1_only[1]), 1e-12);
}

TEST(FormsInstaller, QuadraturePointAutoLayoutSingleConsumerDerivesCount)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_auto"))
            .name("qp_auto_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_auto_inst"));
    auto& block = sys.auxiliaryStateManager().getBlock("qp_auto_inst");
    EXPECT_EQ(block.entityCount(), 4u);
}

TEST(FormsInstaller, QuadraturePointRegionRestrictedAutoLayoutUsesCoveredCellsOnly)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<TwoCellTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    AuxiliaryDeploymentRegion region;
    region.explicit_entities = {1};

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_region"))
            .name("qp_region_inst")
            .quadraturePoint()
            .region(region)
            .monolithic()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = twoCellTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_region_inst"));
    auto& block = sys.auxiliaryStateManager().getBlock("qp_region_inst");
    EXPECT_EQ(block.entityCount(), 4u);
}

TEST(FormsInstaller, DormantUnusedQuadraturePointDeploymentDoesNotMaterialize)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    sys.deploy(
        use(makeScalarOutputModel("qp_dormant"))
            .name("qp_dormant_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    EXPECT_FALSE(sys.auxiliaryStateManager().hasBlock("qp_dormant_inst"));
}

TEST(FormsInstaller, ForcedActiveQuadraturePointWithoutConsumerThrows)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    sys.deploy(
        use(makeScalarOutputModel("qp_forced"))
            .name("qp_forced_inst")
            .quadraturePoint()
            .monolithic()
            .alwaysActive()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::systems::InvalidStateException);
}

TEST(FormsInstaller, UnselectedQuadraturePointVariantStaysDormant)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto active = sys.deploy(
        use(makeScalarOutputModel("qp_variant_active"))
            .name("qp_variant_a")
            .quadraturePoint()
            .variant("ep_model", "a")
            .monolithic()
            .initialize({0.25}));
    (void)sys.deploy(
        use(makeScalarOutputModel("qp_variant_inactive"))
            .name("qp_variant_b")
            .quadraturePoint()
            .variant("ep_model", "b")
            .monolithic()
            .initialize({0.5}));
    sys.selectAuxiliaryVariant("ep_model", "a");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (active.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    EXPECT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_variant_a"));
    EXPECT_FALSE(sys.auxiliaryStateManager().hasBlock("qp_variant_b"));
}

TEST(FormsInstaller, ReferencingUnselectedQuadraturePointVariantThrows)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    (void)sys.deploy(
        use(makeScalarOutputModel("qp_variant_selected"))
            .name("qp_variant_selected")
            .quadraturePoint()
            .variant("ep_model", "a")
            .monolithic()
            .initialize({0.25}));
    auto inactive = sys.deploy(
        use(makeScalarOutputModel("qp_variant_unselected"))
            .name("qp_variant_unselected")
            .quadraturePoint()
            .variant("ep_model", "b")
            .monolithic()
            .initialize({0.5}));
    sys.selectAuxiliaryVariant("ep_model", "a");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inactive.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::systems::InvalidStateException);
}

TEST(FormsInstaller, VariantSelectionIsFrozenAfterSetup)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    (void)sys.deploy(
        use(makeScalarOutputModel("qp_variant_freeze"))
            .name("qp_variant_freeze_a")
            .quadraturePoint()
            .variant("ep_model", "a")
            .monolithic()
            .initialize({0.25}));
    sys.selectAuxiliaryVariant("ep_model", "a");

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(sys.selectAuxiliaryVariant("ep_model", "b"),
                 svmp::FE::systems::InvalidStateException);
}

TEST(FormsInstaller, QuadraturePointBoundaryConsumerIsRejected)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    const int marker = 42;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_boundary"))
            .name("qp_boundary_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).ds(marker));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(FormsInstaller, QuadraturePointQuadratureFromOperatorSupportsForcedExternalLayout)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    (void)sys.deploy(
        use(makeScalarOutputModel("qp_hint_operator"))
            .name("qp_hint_operator_inst")
            .quadraturePoint()
            .quadratureFromOperator("op")
            .monolithic()
            .alwaysActive()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    const auto expected_qp =
        numAuxiliaryCellQuadraturePoints(*mesh, *space, /*cell_id=*/0);
    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_hint_operator_inst"));
    EXPECT_EQ(sys.auxiliaryStateManager().getBlock("qp_hint_operator_inst").entityCount(),
              expected_qp);
}

TEST(FormsInstaller, QuadraturePointQuadratureLikeFieldSupportsConstraintOnlyActivation)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    const int marker = 42;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    (void)sys.deploy(
        use(makeScalarOutputModel("qp_hint_constraint"))
            .name("qp_hint_constraint_inst")
            .quadraturePoint()
            .quadratureLike(u_field)
            .monolithic()
            .initialize({0.25})
            .drivesStrongDirichlet(u_field, marker, "P_out"));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    const auto expected_qp =
        numAuxiliaryCellQuadraturePoints(*mesh, *space, /*cell_id=*/0);
    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_hint_constraint_inst"));
    EXPECT_EQ(sys.auxiliaryStateManager().getBlock("qp_hint_constraint_inst").entityCount(),
              expected_qp);
}

TEST(FormsInstaller, QuadraturePointExplicitOffsetsMatchingConsumerAreAccepted)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_offsets_ok"))
            .name("qp_offsets_ok_inst")
            .quadraturePoint()
            .monolithic()
            .qpOffsets({0, 4})
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_offsets_ok_inst"));
    EXPECT_EQ(sys.auxiliaryStateManager().getBlock("qp_offsets_ok_inst").entityCount(), 4u);
}

TEST(FormsInstaller, QuadraturePointExplicitOffsetsMismatchConsumerRejected)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_offsets_bad"))
            .name("qp_offsets_bad_inst")
            .quadraturePoint()
            .monolithic()
            .qpOffsets({0, 2})
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(FormsInstaller, QuadraturePointOwnedCellInferenceUsesOwnedSubset)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<TwoCellOwnedSubsetTetraMeshAccess>();
    auto space = std::make_shared<spaces::L2Space>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_owned_subset"))
            .name("qp_owned_subset_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = twoCellTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_owned_subset_inst"));
    EXPECT_EQ(sys.auxiliaryStateManager().getBlock("qp_owned_subset_inst").entityCount(), 4u);
}

TEST(FormsInstaller, QuadraturePointAutoLayoutSupportsVariablePerCellQuadratureCounts)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<TwoCellTetraMeshAccess>();
    auto space = std::make_shared<VariableQuadratureTetraSpace>();

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_variable_q"))
            .name("qp_variable_q_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, (u * v).dx() - (inst.output("P_out") * v).dx());

    SetupInputs inputs;
    inputs.topology_override = twoCellTetraTopology();
    sys.setup({}, inputs);
    sys.finalizeAuxiliaryLayout();

    const auto expected_cell0 =
        numAuxiliaryCellQuadraturePoints(*mesh, *space, /*cell_id=*/0);
    const auto expected_cell1 =
        numAuxiliaryCellQuadraturePoints(*mesh, *space, /*cell_id=*/1);

    ASSERT_TRUE(sys.auxiliaryStateManager().hasBlock("qp_variable_q_inst"));
    EXPECT_EQ(sys.auxiliaryStateManager().getBlock("qp_variable_q_inst").entityCount(),
              expected_cell0 + expected_cell1);
    EXPECT_NE(expected_cell0, expected_cell1);
}

TEST(FormsInstaller, QuadraturePointConflictingConsumerSpacesAreRejected)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto low_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    auto high_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/2);

    FESystem sys(mesh);
    const auto u_low = sys.addField(FieldSpec{.name = "u_low", .space = low_space, .components = 1});
    const auto u_high = sys.addField(FieldSpec{.name = "u_high", .space = high_space, .components = 1});
    sys.addOperator("op_low");
    sys.addOperator("op_high");

    auto inst = sys.deploy(
        use(makeScalarOutputModel("qp_conflict"))
            .name("qp_conflict_inst")
            .quadraturePoint()
            .monolithic()
            .initialize({0.25}));

    const auto ul = forms::FormExpr::stateField(u_low, *low_space, "u_low");
    const auto vl = forms::FormExpr::testFunction(*low_space, "v_low");
    (void)installFormulation(sys, "op_low", {u_low}, (ul * vl).dx() - (inst.output("P_out") * vl).dx());

    const auto uh = forms::FormExpr::stateField(u_high, *high_space, "u_high");
    const auto vh = forms::FormExpr::testFunction(*high_space, "v_high");
    (void)installFormulation(sys, "op_high", {u_high}, (uh * vh).dx() - (inst.output("P_out") * vh).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

// ---------------------------------------------------------------------------
//  Guardrail tests for FE-coupled helpers
// ---------------------------------------------------------------------------

TEST(FormsInstaller, RegisterSampledFieldInput_ThrowsBeforeSetup)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    // Do NOT call setup().

    EXPECT_THROW(
        sys.registerSampledFieldInput("u_sampled", "u", 4),
        InvalidStateException);
}

TEST(FormsInstaller, RegisterBoundaryNodalSumInput_ThrowsBeforeSetup)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(1);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "p", .space = space, .components = 1});

    EXPECT_THROW(
        sys.registerBoundaryNodalSumInput("p_bnd", "p", 1),
        InvalidStateException);
}

TEST(FormsInstaller, RegisterSampledFieldInput_RejectsNonVertexSpace)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    // HCurl space has edge DOFs, no vertex DOFs.
    auto hcurl = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto e_field = sys.addField(FieldSpec{.name = "E", .space = hcurl, .components = 3});
    sys.addOperator("op");

    // Install a dummy form so setup() builds DOF handlers.
    const auto u = forms::FormExpr::stateField(e_field, *hcurl, "E");
    const auto v = forms::FormExpr::testFunction(*hcurl, "v");
    installFormulation(sys, "op", {e_field}, forms::inner(u, v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // HCurl space should have no vertex DOFs → rejection.
    EXPECT_THROW(
        sys.registerSampledFieldInput("E_sampled", "E", 4),
        InvalidArgumentException);
}

TEST(FormsInstaller, RegisterBoundaryNodalSumInput_RejectsNonVertexSpace)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(1);
    auto hcurl = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto e_field = sys.addField(FieldSpec{.name = "E", .space = hcurl, .components = 3});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(e_field, *hcurl, "E");
    const auto v = forms::FormExpr::testFunction(*hcurl, "v");
    installFormulation(sys, "op", {e_field}, forms::inner(u, v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(
        sys.registerBoundaryNodalSumInput("E_bnd", "E", 1),
        InvalidArgumentException);
}

TEST(FormsInstaller, FieldToFieldOperator_Rejected)
{
    using namespace svmp::FE::systems;

    // Different field names.
    EXPECT_THROW(
        AuxiliaryOperatorBuilder("bad_op")
            .source("field:velocity")
            .target("field:pressure")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build(),
        std::invalid_argument);

    // Same field name — also rejected (not misclassified as AuxSelf).
    EXPECT_THROW(
        AuxiliaryOperatorBuilder("bad_self")
            .source("field:u")
            .target("field:u")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build(),
        std::invalid_argument);
}

TEST(FormsInstaller, FormsInstaller_MismatchedFieldSpaces_Behavior)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_field = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    auto space_form = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space_field, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space_form, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space_form, "v");
    const auto residual_form = (u * v).dx();

    EXPECT_THROW((void)svmp::FE::systems::installResidualForm(sys, "op", u_field, u_field, residual_form),
                 svmp::FE::InvalidArgumentException);
}
