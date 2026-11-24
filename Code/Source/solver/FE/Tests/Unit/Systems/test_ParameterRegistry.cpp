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
#include "Spaces/SpaceFactory.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
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

namespace {

/// One cell of each shape, its corners moved off an affine map where the shape
/// allows it (a tapered quadrilateral, a warped hexahedron and wedge).
struct VertexDataCell {
    const char* name;
    ElementType type;
    CellFamily family;
    int dim;
    std::vector<svmp::real_t> coordinates;
};

[[nodiscard]] std::vector<VertexDataCell> vertexDataCells()
{
    return {
        {"Triangle3", ElementType::Triangle3, CellFamily::Triangle, 2, {0.1, 0.0, 1.3, 0.2, 0.4, 0.9}},
        {"Quad4", ElementType::Quad4, CellFamily::Quad, 2, {0.0, 0.0, 2.0, 0.0, 1.5, 1.0, 0.2, 1.3}},
        {"Tetra4", ElementType::Tetra4, CellFamily::Tetra, 3,
         {0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.2, 0.9, 0.1, 0.1, 0.3, 1.1}},
        {"Hexa8", ElementType::Hex8, CellFamily::Hex, 3,
         {0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 1.2, 1.1, 0.0, 0.0, 1.0, 0.2,
          0.1, 0.0, 1.0, 1.1, 0.2, 1.3, 0.9, 1.0, 1.0, 0.0, 1.2, 0.9}},
        {"Wedge6", ElementType::Wedge6, CellFamily::Wedge, 3,
         {0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.1, 1.0, 1.3, 0.0, 1.2, 0.2, 1.1, 0.9}},
        {"Pyramid5", ElementType::Pyramid5, CellFamily::Pyramid, 3,
         {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.4, 0.6, 1.0}},
    };
}

[[nodiscard]] std::shared_ptr<Mesh> buildOneCellMesh(const VertexDataCell& cell)
{
    auto base = std::make_shared<MeshBase>();
    const auto n_vertices = cell.coordinates.size() / static_cast<std::size_t>(cell.dim);
    std::vector<svmp::index_t> connectivity(n_vertices);
    for (std::size_t i = 0; i < n_vertices; ++i) {
        connectivity[i] = static_cast<svmp::index_t>(i);
    }
    CellShape shape{};
    shape.family = cell.family;
    shape.num_corners = static_cast<int>(n_vertices);
    shape.order = 1;
    base->build_from_arrays(cell.dim, cell.coordinates, {0, static_cast<svmp::offset_t>(n_vertices)}, connectivity,
                            {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

} // namespace

// Data linear in space, given at the mesh vertices, are reproduced exactly at
// every quadrature point of every cell shape: the residual of
// (u - f_h) . v against the vertex data matches that against the same linear
// function written with the coordinate, at two times (the second assembly
// refills the field).  The quadrature points of the default rule are not
// vertices, and the quadrilateral, hexahedron and wedge are not affine.
TEST(MeshVertexDataField, ReproducesDataLinearInSpaceAtEveryQuadraturePoint)
{
    for (const auto& cell : vertexDataCells()) {
        SCOPED_TRACE(cell.name);
        const auto mesh = buildOneCellMesh(cell);
        const int dim = cell.dim;
        const auto space = spaces::VectorSpace(spaces::SpaceType::H1, cell.type, 1, dim);
        // f_c(x, t) = (1 + t) (1 + c + sum_d (0.5 + c - 0.25 d) x_d)
        const auto exact = [dim](int c, const Real* x, Real t) {
            Real value = 1.0 + static_cast<Real>(c);
            for (int d = 0; d < dim; ++d) {
                value += (0.5 + static_cast<Real>(c) - 0.25 * static_cast<Real>(d)) * x[d];
            }
            return (1.0 + t) * value;
        };
        const auto assembled = [&](bool vertex_data, Real time) {
            FESystem sys(mesh);
            const auto u_id = sys.addField(FieldSpec{.name = "u", .space = space, .components = dim});
            sys.addOperator("op");
            const auto u = forms::StateField(u_id, *space, "u");
            const auto v = forms::TestField(u_id, *space, "v");
            forms::FormExpr f;
            if (vertex_data) {
                const auto& X = mesh->local_mesh().X_ref();
                f = meshVertexDataField(
                    sys, "f",
                    MeshVertexData{.space = space,
                                   .components = dim,
                                   .values = [&X, dim, exact](Real t, std::span<Real> out) {
                                       const auto n = out.size() / static_cast<std::size_t>(dim);
                                       for (std::size_t vertex = 0; vertex < n; ++vertex) {
                                           for (int c = 0; c < dim; ++c) {
                                               out[vertex * static_cast<std::size_t>(dim) + static_cast<std::size_t>(c)] =
                                                   exact(c, &X[vertex * static_cast<std::size_t>(dim)], t);
                                           }
                                       }
                                   }});
            } else {
                const auto x = forms::FormExpr::coordinate();
                std::vector<forms::FormExpr> components;
                for (int c = 0; c < dim; ++c) {
                    auto value = forms::FormExpr::constant(1.0 + static_cast<Real>(c));
                    for (int d = 0; d < dim; ++d) {
                        value = value + forms::FormExpr::constant(0.5 + static_cast<Real>(c) - 0.25 * static_cast<Real>(d)) *
                                            x.component(d);
                    }
                    components.push_back((1.0 + forms::FormExpr::time()) * value);
                }
                f = forms::FormExpr::asVector(std::move(components));
            }
            (void)installFormulation(sys, "op", {u_id}, (forms::inner(u - f, v)).dx());
            sys.setup();
            const auto n = sys.dofHandler().getNumDofs();
            std::vector<Real> U(static_cast<std::size_t>(n), 0.25);
            std::vector<std::vector<Real>> out;
            for (const Real t : {time, time + 0.5}) {
                SystemStateView state;
                state.u = U;
                state.time = t;
                assembly::DenseVectorView R(n);
                R.zero();
                AssemblyRequest request;
                request.op = "op";
                request.want_vector = true;
                EXPECT_TRUE(sys.assemble(request, state, nullptr, &R).success);
                out.emplace_back();
                for (GlobalIndex i = 0; i < n; ++i) {
                    out.back().push_back(R.getVectorEntry(i));
                }
            }
            return out;
        };
        const auto data = assembled(true, 0.3);
        const auto expression = assembled(false, 0.3);
        ASSERT_EQ(data.size(), expression.size());
        for (std::size_t k = 0; k < data.size(); ++k) {
            ASSERT_EQ(data[k].size(), expression[k].size());
            Real scale = 0.0;
            for (const auto value : expression[k]) {
                scale = std::max(scale, std::abs(value));
            }
            ASSERT_GT(scale, 0.0);
            for (std::size_t i = 0; i < data[k].size(); ++i) {
                EXPECT_NEAR(data[k][i], expression[k][i], 1e-14 * scale) << "time " << k << " row " << i;
            }
        }
    }
}

// The data must give every node of the field a value: a vertex the data leave
// out, or a space with nodes that are not mesh vertices, fails at the first
// assembly, naming the field.
TEST(MeshVertexDataField, RefusesNodesWithoutData)
{
    const auto mesh = build_single_quad_mesh();
    const auto assemble = [&](std::shared_ptr<const spaces::FunctionSpace> data_space, bool skip_vertex) {
        FESystem sys(mesh);
        auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
        const auto u_id = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("op");
        const auto f = meshVertexDataField(
            sys, "f",
            MeshVertexData{.space = data_space,
                           .components = 1,
                           .values = [skip_vertex](Real, std::span<Real> out) {
                               for (std::size_t i = skip_vertex ? 1u : 0u; i < out.size(); ++i) {
                                   out[i] = 1.0;
                               }
                           }});
        EXPECT_THROW((void)meshVertexDataField(sys, "f", MeshVertexData{.space = data_space,
                                                                         .values = [](Real, std::span<Real>) {}}),
                     InvalidArgumentException);
        const auto u = forms::StateField(u_id, *space, "u");
        const auto v = forms::TestField(u_id, *space, "v");
        (void)installFormulation(sys, "op", {u_id}, ((u - f) * v).dx());
        sys.setup();
        const auto n = sys.dofHandler().getNumDofs();
        std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
        SystemStateView state;
        state.u = U;
        assembly::DenseVectorView R(n);
        R.zero();
        AssemblyRequest request;
        request.op = "op";
        request.want_vector = true;
        try {
            return sys.assemble(request, state, nullptr, &R).success;
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("'f'"), std::string::npos) << error.what();
            return false;
        }
    };
    EXPECT_TRUE(assemble(std::make_shared<spaces::H1Space>(ElementType::Quad4, 1), false));
    EXPECT_FALSE(assemble(std::make_shared<spaces::H1Space>(ElementType::Quad4, 1), true));
    EXPECT_FALSE(assemble(std::make_shared<spaces::H1Space>(ElementType::Quad4, 2), false));
}


} // namespace svmp::FE::systems::test
