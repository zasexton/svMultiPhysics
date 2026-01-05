/**
 * @file test_ParameterRegistry.cpp
 * @brief Systems parameter contract validation + defaults
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

namespace svmp::FE::systems::test {
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

class ParamScaledMassKernel final : public assembly::BilinearFormKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::IntegrationWeights | assembly::RequiredData::BasisValues;
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        params::Spec s;
        s.key = "alpha";
        s.type = params::ValueType::Real;
        s.required = true;
        s.default_value = params::Value{Real(2.0)};
        s.doc = "Scalar coefficient for mass scaling.";
        return {s};
    }

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& out) override
    {
        const auto n = ctx.numTestDofs();
        out.reserve(n, ctx.numTrialDofs(), /*need_matrix=*/true, /*need_vector=*/false);
        out.clear();

        Real alpha = 0.0;
        if (const auto* get = ctx.realParameterGetter(); get != nullptr && static_cast<bool>(*get)) {
            alpha = (*get)("alpha").value_or(0.0);
        }

        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            const Real w = ctx.integrationWeight(q);
            for (LocalIndex i = 0; i < n; ++i) {
                const Real phi_i = ctx.basisValue(i, q);
                for (LocalIndex j = 0; j < n; ++j) {
                    const Real phi_j = ctx.basisValue(j, q);
                    out.matrixEntry(i, j) += alpha * w * phi_i * phi_j;
                }
            }
        }
    }

    [[nodiscard]] std::string name() const override { return "ParamScaledMassKernel"; }
};

} // namespace

TEST(ParameterRegistry, ProvidesDefaultsWhenStateHasNoGetters)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, std::make_shared<ParamScaledMassKernel>());
    sys.setup();

    const auto ndofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> uvec(ndofs, 0.0);

    SystemStateView state_default;
    state_default.u = uvec;

    SystemStateView state_explicit;
    state_explicit.u = uvec;
    state_explicit.getRealParam = [](std::string_view key) -> std::optional<Real> {
        if (key == "alpha") return 2.0;
        return std::nullopt;
    };

    assembly::DenseMatrixView A_default(static_cast<GlobalIndex>(ndofs));
    assembly::DenseMatrixView A_explicit(static_cast<GlobalIndex>(ndofs));
    A_default.zero();
    A_explicit.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    (void)sys.assemble(req, state_default, &A_default, nullptr);
    (void)sys.assemble(req, state_explicit, &A_explicit, nullptr);

    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(ndofs); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(ndofs); ++j) {
            EXPECT_NEAR(A_default.getMatrixEntry(i, j), A_explicit.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(ParameterRegistry, ThrowsOnTypeMismatch)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addCellKernel("op", u, std::make_shared<ParamScaledMassKernel>());
    sys.setup();

    const auto ndofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> uvec(ndofs, 0.0);

    SystemStateView bad;
    bad.u = uvec;
    bad.getParam = [](std::string_view key) -> std::optional<params::Value> {
        if (key == "alpha") return params::Value{true};
        return std::nullopt;
    };

    assembly::DenseMatrixView A(static_cast<GlobalIndex>(ndofs));
    A.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    EXPECT_THROW((void)sys.assemble(req, bad, &A, nullptr), InvalidArgumentException);
}

} // namespace svmp::FE::systems::test
