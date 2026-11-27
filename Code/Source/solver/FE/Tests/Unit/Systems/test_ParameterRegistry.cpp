/**
 * @file test_ParameterRegistry.cpp
 * @brief Systems parameter contract validation + defaults
 */

#include <gtest/gtest.h>

#include "Analysis/FormExprScanner.h"
#include "Backends/Eigen/EigenFactory.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/TimeSeries.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"

#include "Forms/CutCellForms.h"
#include "Forms/FormCompiler.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/Vocabulary.h"
#include "LevelSet/LevelSetGeometryService.h"
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
#include <numeric>
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
            EXPECT_EQ(sys.parameterRegistry().slotOf("Q"), 0u);
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

TEST(TimeFunction, MonolithicTangentAndResidualRebindNamedParameters)
{
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    const auto u = forms::FormExpr::trialFunction(*space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    const auto load = forms::FormExpr::parameter("Q") + forms::FormExpr::parameterRef(5);
    forms::FormCompiler compiler;
    std::vector<forms::MonolithicCellKernel::BlockSpec> blocks(1);
    auto& block = blocks.front();
    block.tangent_ir = compiler.compileBilinear((load * u * v).dx());
    block.residual_ir = compiler.compileResidual((load * (1.0 + u) * v).dx());
    forms::MonolithicCellKernel kernel(std::move(blocks), nullptr, {});
    ParameterRegistry registry;
    registry.addTimeFunction("Q", [](Real) { return 5.0; });
    const auto check = [&] {
        const auto slot = registry.kernelSlotOf("Q");
        kernel.resolveParameterSlots([&](std::string_view key) { return registry.kernelSlotOf(key); });
        const auto& resolved = kernel.blockSpec(0);
        for (const auto* ir : {&*resolved.tangent_ir, &*resolved.residual_ir}) {
            unsigned named = 0, positional = 0;
            for (const auto& term : ir->terms()) {
                (void)term.integrand.transformNodes([&](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
                    if (node.type() == forms::FormExprType::ParameterRef) {
                        if (const auto name = node.symbolName()) {
                            EXPECT_EQ(*name, "Q");
                            EXPECT_EQ(node.slotIndex(), slot);
                            ++named;
                        } else {
                            EXPECT_EQ(node.slotIndex(), 5u);
                            ++positional;
                        }
                    }
                    return std::nullopt;
                });
            }
            EXPECT_EQ(named, 1u);
            EXPECT_EQ(positional, 1u);
        }
    };
    check();
    registry.addTimeFunction("A", [](Real) { return 7.0; });
    check();
}

TEST(TimeFunction, OrdinaryInteriorFacetRetainsDensePositionalParameters)
{
    auto base = std::make_shared<MeshBase>();
    const CellShape quad{.family = CellFamily::Quad, .num_corners = 4, .order = 1};
    base->build_from_arrays(2, {0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1},
                            {0, 4, 8}, {0, 1, 4, 3, 1, 2, 5, 4}, {quad, quad});
    base->finalize();
    FESystem system(svmp::create_mesh(std::move(base)));
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    const auto field = system.addField({.name = "u", .space = space});
    auto u = forms::StateField(field, *space, "u");
    auto v = forms::TestField(field, *space, "v");
    auto load = timeFunction(system, "load", [](Real t) { return 2.0 + 3.0 * t; });
    system.addOperator("facet");
    (void)installFormulation(system, "facet", {field},
        ((load + forms::FormExpr::parameterRef(0)) * (1.0 + forms::avg(u)) * forms::avg(v)).dS());
    system.setup();

    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> zero(n, 0.0);
    const auto check = [&] {
        EXPECT_EQ(system.parameterRegistry().slotCount(), 1u);
        EXPECT_EQ(system.parameterRegistry().slotOf("load"), 0u);
        SystemStateView state;
        state.u = zero;
        state.time = 0.5;
        AssemblyRequest request;
        request.op = "facet";
        request.want_vector = true;
        assembly::DenseVectorView result(n);
        const auto assembled = system.assemble(request, state, nullptr, &result);
        ASSERT_TRUE(assembled.success) << assembled.error_message;
        // One unit-length interior facet. Both slots carry the load at t = 1/2.
        EXPECT_NEAR(std::accumulate(result.data().begin(), result.data().end(), 0.0), 7.0, 1e-13);
    };
    check();
    system.setCutIntegrationContext(std::make_shared<assembly::CutIntegrationContext>());
    check();
    system.setup();
    check();
    system.setCutIntegrationContext(nullptr);
    system.setup();
    check();
}

TEST(TimeFunction, NamedValueAndCutMetadataReachTheSameVolumeKernel)
{
    auto mesh = build_single_quad_mesh();
    FESystem system(mesh);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    const auto phi = system.addField({.name = "phi", .space = space, .source_kind = FieldSourceKind::PrescribedData});
    const auto field = system.addField({.name = "u", .space = space});
    auto geometry = level_set::cutGeometry(system, *mesh, {.level_set_field_name = "phi", .domain_id = "quarter"});
    const auto side = geometry.side(forms::CutVolumeSide::Negative);
    const auto u = forms::StateField(field, *space, "u");
    const auto v = forms::TestField(field, *space, "v");
    const auto load = timeFunction(system, "load", [](Real t) { return 2.0 + 3.0 * t; });
    for (int i = 1; i <= 6; ++i) {
        (void)timeFunction(system, "other" + std::to_string(i), [i](Real) { return 10.0 + i; });
    }
    const auto metadata = 2.0 * forms::cutVolumeFraction() + forms::cutSideIndicator()
                        + forms::cutMeasureSensitivity().component(0);
    const auto slot_zero = forms::FormExpr::parameterRef(0);
    const auto both_measures = slot_zero.dx() + slot_zero.dx(side.volume());
    const auto usages = analysis::scanFormExpr(*both_measures.node()).parameter_usages;
    std::vector<int> usage_markers;
    for (const auto& usage : usages) {
        EXPECT_EQ(usage.slot, 0u);
        usage_markers.push_back(usage.interface_marker);
    }
    std::sort(usage_markers.begin(), usage_markers.end());
    EXPECT_EQ(usage_markers, (std::vector<int>{-1, geometry.interface().marker()}));
    const auto residual = ((1.0 + u) * (load + metadata) * v).dx(side.volume());
    system.addOperator("cut");
    (void)installFormulation(system, "cut", {field}, residual);
    system.setup();
    const std::array<Real, 4> nodal_phi{-0.25, 0.75, 0.75, -0.25}; // x - 1/4 on the unit square
    std::vector<Real> coefficients(system.fieldDofHandler(phi).getNumDofs());
    (void)system.projectMeshVertexValuesToFieldCoefficients(phi, nodal_phi, 1, coefficients);
    system.setPrescribedFieldCoefficients(phi, coefficients);
    const auto cut_context = geometry.refresh().installed_context.lock();
    ASSERT_NE(cut_context, nullptr);

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    backends::EigenFactory factory;
    auto matrix = factory.createMatrix(system.sparsity("cut"));
    std::vector<Real> initial(system.dofHandler().getNumDofs(), 0.0);
    SystemStateView measured;
    measured.time = 0.5;
    measured.u = initial;
    auto view = matrix->createAssemblyView();
    AssemblyRequest jacobian;
    jacobian.op = "cut";
    jacobian.want_matrix = true;
    ASSERT_TRUE(system.assemble(jacobian, measured, view.get(), nullptr).success);
    matrix->finalizeAssembly();
    ASSERT_TRUE(system.updateAnalysisSummariesFromAssembledOperator(*matrix, "cut", &measured));
    const auto& properties = system.latestAnalysisSummaries()->coefficient_properties;
    const auto property = [&](std::string_view label) {
        return std::find_if(properties.begin(), properties.end(),
            [&](const auto& entry) { return entry.coefficient == label; });
    };
    const auto fraction = property("parameter_slot:0");
    ASSERT_NE(fraction, properties.end());
    EXPECT_FALSE(fraction->local_spectrum_coverage_present);
    const auto named = property("parameter:load");
    ASSERT_NE(named, properties.end());
    EXPECT_DOUBLE_EQ(named->min_eigenvalue, 3.5);
#endif

    // A second system acquires just the snapshot for a post-setup functional.
    // Its compact named parameter is evaluated through the functional's getter.
    FESystem snapshot_reader(mesh);
    const auto snapshot_field = snapshot_reader.addField({.name = "u", .space = space});
    const auto snapshot_load = timeFunction(snapshot_reader, "load", [](Real t) { return 2.0 + 3.0 * t; });
    for (int i = 1; i <= 6; ++i) {
        (void)timeFunction(snapshot_reader, "other" + std::to_string(i), [i](Real) { return 10.0 + i; });
    }
    const auto reader_u = forms::StateField(snapshot_field, *space, "u");
    const auto reader_v = forms::TestField(snapshot_field, *space, "v");
    snapshot_reader.addOperator("whole");
    (void)installFormulation(snapshot_reader, "whole", {snapshot_field},
                             ((1.0 + reader_u) * snapshot_load * reader_v).dx());
    snapshot_reader.setup();
    ASSERT_EQ(snapshot_reader.parameterRegistry().slotOf("load"), 0u);
    const std::array snapshot_forms{
        (snapshot_load + metadata).dx(side.volume()),
        (forms::FormExpr::boundParameter("load", 99) + metadata).dx(side.volume()),
        forms::FormExpr::parameterRef(6).dx()};
    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> zero(n, 0.0);
    for (Real t : {0.0, 0.5}) {
        SystemStateView state;
        state.time = t;
        state.u = zero;
        const auto& registry = system.parameterRegistry();
        const auto slot = registry.slotOf("load");
        ASSERT_TRUE(slot.has_value());
        EXPECT_EQ(registry.keyForSlot(*slot), "load");
        ASSERT_EQ(*slot, 0u);
        // Without a measure, slot zero can also be the rule's cut fraction.
        EXPECT_FALSE(registry.keyForKernelSlot(*slot).has_value());
        const auto compiled_slot = registry.kernelSlotOf("load");
        ASSERT_TRUE(compiled_slot);
        EXPECT_GE(*compiled_slot, forms::cutCellParameterCount());
        EXPECT_EQ(registry.keyForKernelSlot(*compiled_slot), "load");
        const auto values = registry.evaluateRealSlots(state);
        ASSERT_EQ(values.size(), registry.slotCount());
        ASSERT_LT(*slot, values.size());
        EXPECT_EQ(values[*slot], 2.0 + 3.0 * t);
        AssemblyRequest request;
        request.op = "cut";
        request.want_vector = true;
        request.want_matrix = true;
        assembly::DenseVectorView result(n);
        assembly::DenseMatrixView tangent(n);
        const auto assembled = system.assemble(request, state, &tangent, &result);
        ASSERT_TRUE(assembled.success) << assembled.error_message;
        // Test with 1.  The cut has area and volume fraction 1/4, and side indicator -1.
        const double expected = 0.25 * (2.0 + 3.0 * t + 0.5 - 1.0);
        EXPECT_NEAR(std::accumulate(result.data().begin(), result.data().end(), 0.0), expected, 1e-13);
        EXPECT_NEAR(std::accumulate(tangent.data().begin(), tangent.data().end(), 0.0), expected, 1e-13);
        const auto measured = evaluateFunctionalForms(snapshot_reader, snapshot_forms, state, {},
                                                      system.cutIntegrationContext());
        ASSERT_EQ(measured.size(), 3u);
        EXPECT_NEAR(measured[0].value, expected, 1e-13);
        EXPECT_NEAR(measured[1].value, expected, 1e-13);
        EXPECT_NEAR(measured[2].value, 16.0, 1e-13);
    }

    const auto check_whole = [&] {
        SystemStateView state;
        state.u = zero;
        state.time = 0.5;
        AssemblyRequest request;
        request.op = "whole";
        request.want_vector = true;
        assembly::DenseVectorView result(n);
        const auto assembled = snapshot_reader.assemble(request, state, nullptr, &result);
        ASSERT_TRUE(assembled.success) << assembled.error_message;
        EXPECT_NEAR(std::accumulate(result.data().begin(), result.data().end(), 0.0), 3.5, 1e-13);
    };
    check_whole();
    snapshot_reader.setCutIntegrationContext(cut_context);
    snapshot_reader.setup();
    check_whole();
    snapshot_reader.setCutIntegrationContext(nullptr);
    snapshot_reader.setup();
    check_whole();
    // Add a cut term and a lexically earlier key after the first setup. The
    // existing whole-cell kernel must still evaluate the same named load.
    snapshot_reader.addOperator("late_cut");
    const auto earlier = timeFunction(snapshot_reader, "another", [](Real) { return 7.0; });
    for (int i = 7; i <= 13; ++i) {
        (void)timeFunction(snapshot_reader, "other" + std::to_string(i), [i](Real) { return 10.0 + i; });
    }
    (void)installFormulation(snapshot_reader, "late_cut", {snapshot_field},
        ((earlier + reader_u) * reader_v).dx(side.volume()));
    snapshot_reader.setCutIntegrationContext(cut_context);
    snapshot_reader.setup();
    EXPECT_EQ(snapshot_reader.parameterRegistry().slotCount(), 15u);
    check_whole();
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
