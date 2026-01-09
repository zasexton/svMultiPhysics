/**
 * @file test_SystemAssembly.cpp
 * @brief Unit tests for Systems assembleOperator()
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/SystemAssembly.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <cmath>
#include <memory>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::DenseSystemView;
using svmp::FE::assembly::DenseVectorView;
using svmp::FE::assembly::PoissonKernel;
using svmp::FE::assembly::SourceKernel;

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

class GlobalVectorKernel final : public svmp::FE::systems::GlobalKernel {
public:
    svmp::FE::assembly::AssemblyResult assemble(const FESystem& system,
                                                const AssemblyRequest& request,
                                                const SystemStateView&,
                                                svmp::FE::assembly::GlobalSystemView*,
                                                svmp::FE::assembly::GlobalSystemView* vector_out) override
    {
        svmp::FE::assembly::AssemblyResult r;
        if (!request.want_vector || vector_out == nullptr) {
            return r;
        }
        const auto n = system.dofHandler().getNumDofs();
        for (GlobalIndex i = 0; i < n; ++i) {
            vector_out->addVectorEntry(i, 1.0);
        }
        return r;
    }
};

} // namespace

TEST(SystemAssembly, SystemAssembly_AssembleOperator_MatrixOnly)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<PoissonKernel>(/*constant_source=*/1.0));
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    A.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    SystemStateView state;
    auto result = svmp::FE::systems::assembleOperator(sys, req, state, &A, nullptr);
    EXPECT_TRUE(result.success);

    // Non-trivial: stiffness matrix is not all zeros.
    Real sum = 0.0;
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            sum += std::abs(A.getMatrixEntry(i, j));
        }
    }
    EXPECT_GT(sum, 0.0);
}

TEST(SystemAssembly, SystemAssembly_AssembleOperator_VectorOnly)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<SourceKernel>(/*constant_source=*/2.0));
    sys.setup();

    DenseVectorView b(sys.dofHandler().getNumDofs());
    b.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_vector = true;

    SystemStateView state;
    auto result = svmp::FE::systems::assembleOperator(sys, req, state, nullptr, &b);
    EXPECT_TRUE(result.success);

    Real sum = 0.0;
    for (GlobalIndex i = 0; i < b.numRows(); ++i) {
        sum += std::abs(b.getVectorEntry(i));
    }
    EXPECT_GT(sum, 0.0);
}

TEST(SystemAssembly, SystemAssembly_AssembleOperator_BothMatrixAndVector)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<PoissonKernel>(/*constant_source=*/1.0));
    sys.setup();

    const auto n = sys.dofHandler().getNumDofs();
    DenseSystemView out(n);
    out.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    SystemStateView state;
    auto result = svmp::FE::systems::assembleOperator(sys, req, state, &out, &out);
    EXPECT_TRUE(result.success);

    Real sum_m = 0.0;
    Real sum_v = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        sum_v += std::abs(out.getVectorEntry(i));
        for (GlobalIndex j = 0; j < n; ++j) {
            sum_m += std::abs(out.getMatrixEntry(i, j));
        }
    }
    EXPECT_GT(sum_m, 0.0);
    EXPECT_GT(sum_v, 0.0);
}

TEST(SystemAssembly, SystemAssembly_AssembleOperator_ZeroOutputsTrue_ClearsOutputs)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<PoissonKernel>(/*constant_source=*/1.0));
    sys.setup();

    const auto n = sys.dofHandler().getNumDofs();
    DenseSystemView ref(n);
    ref.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.zero_outputs = true;

    SystemStateView state;
    (void)svmp::FE::systems::assembleOperator(sys, req, state, &ref, &ref);

    DenseSystemView out(n);
    for (auto& v : out.matrixDataMutable()) v = 123.0;
    for (auto& v : out.vectorDataMutable()) v = 456.0;

    (void)svmp::FE::systems::assembleOperator(sys, req, state, &out, &out);

    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(out.getVectorEntry(i), ref.getVectorEntry(i), 1e-12);
        for (GlobalIndex j = 0; j < n; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), ref.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(SystemAssembly, SystemAssembly_AssembleOperator_ZeroOutputsFalse_Accumulates)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<PoissonKernel>(/*constant_source=*/1.0));
    sys.setup();

    const auto n = sys.dofHandler().getNumDofs();
    DenseSystemView out(n);
    out.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.zero_outputs = true;

    SystemStateView state;
    (void)svmp::FE::systems::assembleOperator(sys, req, state, &out, &out);

    DenseSystemView once = out;

    req.zero_outputs = false;
    (void)svmp::FE::systems::assembleOperator(sys, req, state, &out, &out);

    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(out.getVectorEntry(i), 2.0 * once.getVectorEntry(i), 1e-12);
        for (GlobalIndex j = 0; j < n; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), 2.0 * once.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(SystemAssembly, SystemAssembly_NullMatrixWithWantMatrix_Behavior)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<PoissonKernel>(1.0));
    sys.setup();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    SystemStateView state;
    EXPECT_THROW((void)svmp::FE::systems::assembleOperator(sys, req, state, nullptr, nullptr), svmp::FE::InvalidArgumentException);
}

TEST(SystemAssembly, SystemAssembly_NullVectorWithWantVector_Behavior)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, u, std::make_shared<SourceKernel>(1.0));
    sys.setup();

    AssemblyRequest req;
    req.op = "op";
    req.want_vector = true;

    SystemStateView state;
    EXPECT_THROW((void)svmp::FE::systems::assembleOperator(sys, req, state, nullptr, nullptr), svmp::FE::InvalidArgumentException);
}

TEST(SystemAssembly, SystemAssembly_EmptyOperator_ReturnsSuccess)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("empty");
    sys.setup();

    const auto n = sys.dofHandler().getNumDofs();
    DenseVectorView b(n);
    b.zero();

    AssemblyRequest req;
    req.op = "empty";
    req.want_vector = true;

    SystemStateView state;
    auto result = svmp::FE::systems::assembleOperator(sys, req, state, nullptr, &b);
    EXPECT_TRUE(result.success);

    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(b.getVectorEntry(i), 0.0, 1e-12);
    }
}

TEST(SystemAssembly, SystemAssembly_GlobalKernelsOnly_AssemblesCorrectly)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("glob");
    sys.addGlobalKernel("glob", std::make_shared<GlobalVectorKernel>());
    sys.setup();

    const auto n = sys.dofHandler().getNumDofs();
    DenseVectorView b(n);
    b.zero();

    AssemblyRequest req;
    req.op = "glob";
    req.want_vector = true;

    SystemStateView state;
    auto result = svmp::FE::systems::assembleOperator(sys, req, state, nullptr, &b);
    EXPECT_TRUE(result.success);

    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(b.getVectorEntry(i), 1.0, 1e-12);
    }
}
