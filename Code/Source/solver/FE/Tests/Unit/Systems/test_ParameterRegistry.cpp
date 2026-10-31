/**
 * @file test_ParameterRegistry.cpp
 * @brief Systems parameter contract validation + defaults
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeSeries.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Forms/JIT/JITValidation.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
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

// ---- Time series and time-function parameters ----

TEST(TimeSeries, InterpolatesLinearlyAndClampsOrRepeatsOutside)
{
    const TimeSeries clamped({0.0, 1.0, 3.0}, {2.0, 4.0, -2.0}, TimeSeriesEnd::Clamp);
    EXPECT_EQ(clamped(-1.0), 2.0);
    EXPECT_EQ(clamped(0.0), 2.0);
    EXPECT_EQ(clamped(0.5), 3.0);
    EXPECT_EQ(clamped(2.5), -0.5);
    EXPECT_EQ(clamped(3.0), -2.0);
    EXPECT_EQ(clamped(5.0), -2.0);

    const TimeSeries periodic({0.0, 1.0, 3.0}, {2.0, 4.0, 2.0}, TimeSeriesEnd::Periodic);
    EXPECT_EQ(periodic(3.5), periodic(0.5));
    EXPECT_EQ(periodic(-0.5), periodic(2.5));
    EXPECT_EQ(periodic(2.5), 2.5);

    const TimeSeries two({0.0, 2.0}, {1.0, 10.0, 3.0, 20.0}, TimeSeriesEnd::Clamp, /*components=*/2);
    EXPECT_EQ(two.value(1.0, 0), 2.0);
    EXPECT_EQ(two.value(1.0, 1), 15.0);
    EXPECT_THROW((void)two.value(1.0, 2), InvalidArgumentException);
}

TEST(TimeSeries, RejectsMalformedTables)
{
    EXPECT_THROW(TimeSeries({0.0}, {1.0}, TimeSeriesEnd::Clamp), InvalidArgumentException);
    EXPECT_THROW(TimeSeries({0.0, 0.0}, {1.0, 2.0}, TimeSeriesEnd::Clamp), InvalidArgumentException);
    EXPECT_THROW(TimeSeries({1.0, 0.0}, {1.0, 2.0}, TimeSeriesEnd::Clamp), InvalidArgumentException);
    EXPECT_THROW(TimeSeries({0.0, 1.0}, {1.0}, TimeSeriesEnd::Clamp), InvalidArgumentException);
    EXPECT_THROW(TimeSeries({0.0, 1.0}, {1.0, std::numeric_limits<Real>::quiet_NaN()}, TimeSeriesEnd::Clamp),
                 InvalidArgumentException);
}

TEST(TimeSeries, ReadsTheTemporalValuesFormat)
{
    const auto path = std::filesystem::temp_directory_path() / "svmp_fe_time_series.dat";
    {
        std::ofstream out(path);
        out << "3 1\n0.0 1.0\n0.5 2.0\n1.0 0.0\n";
    }
    const auto series = readTimeSeries(path.string(), TimeSeriesEnd::Periodic);
    EXPECT_EQ(series.times().size(), 3u);
    EXPECT_EQ(series(0.25), 1.5);
    EXPECT_EQ(series(1.25), 1.5);
    {
        std::ofstream out(path);
        out << "3 1\n0.0 1.0\n0.5\n";
    }
    EXPECT_THROW((void)readTimeSeries(path.string(), TimeSeriesEnd::Clamp), InvalidArgumentException);
    std::filesystem::remove(path);
    EXPECT_THROW((void)readTimeSeries(path.string(), TimeSeriesEnd::Clamp), InvalidArgumentException);
}

TEST(ParameterRegistry, TimeFunctionGivesItsValueAtTheStateTime)
{
    ParameterRegistry registry;
    registry.addTimeFunction("Q", [](Real t) { return 3.0 * t + 1.0; });
    ASSERT_NE(registry.find("Q"), nullptr);
    EXPECT_EQ(registry.find("Q")->type, params::ValueType::Real);

    SystemStateView state;
    state.time = 0.25;
    EXPECT_NO_THROW(registry.validate(state));
    EXPECT_EQ(registry.makeRealGetter(state)("Q"), 1.75);
    EXPECT_EQ(params::get<Real>(*registry.makeParamGetter(state)("Q")), 1.75);
    ASSERT_EQ(registry.slotCount(), 1u);
    EXPECT_EQ(registry.evaluateRealSlots(state), std::vector<Real>{1.75});

    // A value the state supplies comes first.
    state.getRealParam = [](std::string_view key) -> std::optional<Real> {
        return key == "Q" ? std::optional<Real>(-2.0) : std::nullopt;
    };
    EXPECT_EQ(registry.makeRealGetter(state)("Q"), -2.0);

    registry.clear();
    EXPECT_EQ(registry.find("Q"), nullptr);
}

TEST(TimeFunction, DeclaresAParameterThatSetupRegistersAndKernelsReadAsASlot)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    const TimeSeries flow({0.0, 0.5, 1.0}, {0.0, 2.5, -1.0}, TimeSeriesEnd::Periodic);
    const forms::ScalarCoefficient w = [](Real x, Real y, Real) { return 1.0 + 0.5 * x + 0.25 * y; };

    // R = (u v - Q(t) w v) dx, with Q a time function or inside one callback.
    const auto assembled = [&](bool parameter, Real time) {
        FESystem sys(mesh);
        const auto u_id = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("op");
        const auto u = forms::StateField(u_id, *space, "u");
        const auto v = forms::TestField(u_id, *space, "v");
        const auto Qw = parameter
                            ? timeFunction(sys, "Q", flow) * forms::FormExpr::coefficient("w", w)
                            : forms::FormExpr::coefficient(
                                  "Qw", forms::TimeScalarCoefficient([&](Real x, Real y, Real z, Real t) {
                                      return flow(t) * w(x, y, z);
                                  }));
        (void)installFormulation(sys, "op", {u_id}, (u * v - Qw * v).dx());
        sys.setup();
        if (parameter) {
            EXPECT_TRUE(sys.parameterRegistry().slotOf("Q").has_value());
        }
        const auto n = sys.dofHandler().getNumDofs();
        std::vector<Real> U(static_cast<std::size_t>(n), 0.5);
        SystemStateView state;
        state.u = U;
        state.time = time;
        assembly::DenseVectorView R(n);
        R.zero();
        AssemblyRequest request;
        request.op = "op";
        request.want_vector = true;
        (void)sys.assemble(request, state, nullptr, &R);
        std::vector<Real> out;
        for (GlobalIndex i = 0; i < n; ++i) {
            out.push_back(R.getVectorEntry(i));
        }
        return out;
    };
    // The kernels differ (a slot and a product against one callback, which
    // also decides the affine split), so the vectors agree to round-off.
    for (const Real time : {0.0, 0.3, 0.75, 1.4}) {
        const auto parameter = assembled(true, time);
        const auto callback = assembled(false, time);
        ASSERT_EQ(parameter.size(), callback.size());
        for (std::size_t i = 0; i < parameter.size(); ++i) {
            EXPECT_NEAR(parameter[i], callback[i], 1e-15) << "t=" << time;
        }
    }

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto Q = timeFunction(sys, "Q", flow);
    ASSERT_TRUE(Q.isValid());
    EXPECT_EQ(Q.node()->type(), forms::FormExprType::ParameterSymbol);
    ASSERT_NE(findTimeFunction(sys, "Q"), nullptr);
    EXPECT_EQ((*findTimeFunction(sys, "Q"))(0.25), flow(0.25));
    EXPECT_EQ(findTimeFunction(sys, "P"), nullptr);
    EXPECT_THROW((void)timeFunction(sys, "Q", flow), InvalidArgumentException);
    EXPECT_THROW((void)timeFunction(sys, "", flow), InvalidArgumentException);
    EXPECT_THROW((void)timeFunction(sys, "P", std::function<Real(Real)>{}), InvalidArgumentException);
}

TEST(TimeFunction, ResolvedParameterIsJITStrictWhereTheCallbackIsNot)
{
    const TimeSeries flow({0.0, 1.0}, {0.0, 1.0}, TimeSeriesEnd::Clamp);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    const auto v = forms::FormExpr::testFunction(*space, "v");
    const auto x = forms::FormExpr::coordinate();

    // Kernels resolve the parameter to its registry slot (resolveParameterSlots).
    ParameterRegistry registry;
    registry.addTimeFunction("Q", flow);
    const auto resolved = (forms::FormExpr::parameter("Q") * (1.0 + x.component(0)) * v)
                              .transformNodes([&](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
                                  if (node.type() != forms::FormExprType::ParameterSymbol) {
                                      return std::nullopt;
                                  }
                                  return forms::FormExpr::parameterRef(*registry.slotOf(*node.symbolName()));
                              });
    const auto strict = forms::jit::canCompile(resolved);
    EXPECT_TRUE(strict.ok);
    EXPECT_TRUE(strict.cacheable);

    const auto callback = forms::FormExpr::coefficient(
        "Q", forms::TimeScalarCoefficient([&](Real, Real, Real, Real t) { return flow(t); }));
    EXPECT_FALSE(forms::jit::canCompile(callback * (1.0 + x.component(0)) * v).ok);
}

} // namespace svmp::FE::systems::test
