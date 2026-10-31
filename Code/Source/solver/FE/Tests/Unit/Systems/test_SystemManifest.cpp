/**
 * @file test_SystemManifest.cpp
 * @brief Ordered kernel and constraint manifests of an FESystem
 */

#include <gtest/gtest.h>

#include "Constraints/StrongDirichletConstraint.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/LLVMJITBuildInfo.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemManifest.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::systems::test {
namespace {

using forms::FormExpr;

struct ManifestSystem {
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh{
        std::make_shared<forms::test::SingleTetraMeshAccess>()};
    std::shared_ptr<spaces::H1Space> space{
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1)};
    std::unique_ptr<FESystem> system{};
    FieldId u_id{INVALID_FIELD_ID};
};

/// Installs `residual(u, v)` on operator "op" of a fresh one-field system.
ManifestSystem buildSystem(
    const std::function<FormExpr(const FormExpr&, const FormExpr&)>& residual)
{
    ManifestSystem out;
    out.system = std::make_unique<FESystem>(out.mesh);
    out.u_id = out.system->addField(FieldSpec{.name = "u", .space = out.space, .components = 1});
    out.system->addOperator("op");
    const auto u = forms::StateField(out.u_id, *out.space, "u");
    const auto v = forms::TestField(out.u_id, *out.space, "v");
    (void)installFormulation(*out.system, "op", {out.u_id}, residual(u, v));
    return out;
}

[[nodiscard]] std::vector<std::string> linesContaining(const std::vector<std::string>& lines,
                                                       std::string_view token)
{
    std::vector<std::string> out;
    for (const auto& line : lines) {
        if (line.find(token) != std::string::npos) {
            out.push_back(line);
        }
    }
    return out;
}

/// Value of `key=` in a manifest line ("" when absent).
[[nodiscard]] std::string valueOf(const std::string& line, std::string_view key)
{
    const std::string needle = " " + std::string(key) + "=";
    const auto at = line.find(needle);
    if (at == std::string::npos) {
        return {};
    }
    const auto begin = at + needle.size();
    const auto end = line.find(' ', begin);
    return line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
}

/// Term lines of the residual forms (the trees as written; tangent trees
/// come from differentiation).  Residuals affine in the state are installed
/// through the affine split (bilinear and linear forms) and have none, so
/// the tests below use residuals that are nonlinear in u.
[[nodiscard]] std::vector<std::string> residualTerms(const FESystem& system)
{
    std::vector<std::string> out;
    for (const auto& line : kernelManifestLines(system)) {
        if (line.find(" ir=residual ") != std::string::npos && line.find(" term=") != std::string::npos) {
            out.push_back(line);
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::string> fieldOf(const std::vector<std::string>& lines,
                                               std::string_view key)
{
    std::vector<std::string> out;
    for (const auto& line : lines) {
        out.push_back(valueOf(line, key));
    }
    return out;
}

TEST(SystemManifest, IdenticalFormulationsGiveIdenticalManifests)
{
    const auto residual = [](const FormExpr& u, const FormExpr& v) {
        return (inner(grad(u), grad(v)) + FormExpr::constant(0.25) * u * v).dx() +
               (FormExpr::constant(2.0) * u * v).ds(3);
    };
    const auto a = buildSystem(residual);
    const auto b = buildSystem(residual);
    const auto lines_a = kernelManifestLines(*a.system);
    const auto lines_b = kernelManifestLines(*b.system);
    ASSERT_FALSE(lines_a.empty());
    EXPECT_EQ(lines_a, lines_b);
    EXPECT_FALSE(linesContaining(lines_a, " term=").empty());
    EXPECT_FALSE(linesContaining(lines_a, " group=").empty());
    EXPECT_EQ(lines_a.front().rfind("operator op ", 0), 0u);
}

/// Residual term lines of two one-field systems, for the operand-order tests.
struct TermPair {
    std::vector<std::string> a;
    std::vector<std::string> b;
};

using Residual = std::function<FormExpr(const FormExpr&, const FormExpr&)>;

[[nodiscard]] TermPair residualTermsOf(const Residual& residual_a, const Residual& residual_b)
{
    const auto a = buildSystem(residual_a);
    const auto b = buildSystem(residual_b);
    return TermPair{residualTerms(*a.system), residualTerms(*b.system)};
}

TEST(SystemManifest, CommutedOperandsChangeTheTreeButNotTheKernelIR)
{
    // The KernelIR hash orders the two operands of the operations the
    // KernelIR treats as commutative: +, inner, double contraction, min, max,
    // == and != (KernelIR.cpp, isCommutative).
    const auto expect_same_kir = [](const TermPair& terms) {
        ASSERT_FALSE(terms.a.empty());
        ASSERT_EQ(terms.a.size(), terms.b.size());
        EXPECT_NE(fieldOf(terms.a, "tree"), fieldOf(terms.b, "tree"));
        EXPECT_EQ(fieldOf(terms.a, "kir"), fieldOf(terms.b, "kir"));
        for (const auto& kir : fieldOf(terms.a, "kir")) {
            EXPECT_NE(kir, "unsupported");
        }
    };
    // inner(a, b) against inner(b, a).
    expect_same_kir(residualTermsOf(
        [](const FormExpr& u, const FormExpr& v) { return inner(u * grad(u), grad(v)).dx(); },
        [](const FormExpr& u, const FormExpr& v) { return inner(grad(v), u * grad(u)).dx(); }));
    // a + b against b + a, under a product (a sum directly under dx() is
    // split into separate integral terms).
    expect_same_kir(residualTermsOf(
        [](const FormExpr& u, const FormExpr& v) { return ((u * u + u) * v).dx(); },
        [](const FormExpr& u, const FormExpr& v) { return ((u + u * u) * v).dx(); }));
}

TEST(SystemManifest, CommutedProductOperandsChangeTheKernelIR)
{
    // Multiply is not commutative in the KernelIR (a product may be a
    // tensor product such as a matrix times a vector), so commuting the
    // factors of a product shows in kir as well as in tree.
    const auto terms = residualTermsOf(
        [](const FormExpr& u, const FormExpr& v) { return ((FormExpr::constant(3.0) * u) * u * v).dx(); },
        [](const FormExpr& u, const FormExpr& v) { return (v * ((FormExpr::constant(3.0) * u) * u)).dx(); });
    ASSERT_FALSE(terms.a.empty());
    ASSERT_EQ(terms.a.size(), terms.b.size());
    EXPECT_NE(fieldOf(terms.a, "tree"), fieldOf(terms.b, "tree"));
    EXPECT_NE(fieldOf(terms.a, "kir"), fieldOf(terms.b, "kir"));
}

TEST(SystemManifest, ReassociationChangesTheKernelIR)
{
    const auto a = buildSystem([](const FormExpr& u, const FormExpr& v) {
        return (((FormExpr::constant(3.0) * u) * u) * v).dx();
    });
    const auto b = buildSystem([](const FormExpr& u, const FormExpr& v) {
        return ((FormExpr::constant(3.0) * (u * u)) * v).dx();
    });
    const auto terms_a = residualTerms(*a.system);
    const auto terms_b = residualTerms(*b.system);
    ASSERT_FALSE(terms_a.empty());
    EXPECT_NE(fieldOf(terms_a, "kir"), fieldOf(terms_b, "kir"));
}

TEST(SystemManifest, ConstantBitsEnterTheTreeHash)
{
    // 0.1 and 0.1 + 2^-56 print alike at six digits but are different trees.
    const auto a = buildSystem([](const FormExpr& u, const FormExpr& v) {
        return (FormExpr::constant(0.1) * u * u * v).dx();
    });
    const auto b = buildSystem([](const FormExpr& u, const FormExpr& v) {
        return (FormExpr::constant(0.1 + 0x1p-56) * u * u * v).dx();
    });
    const auto terms_a = residualTerms(*a.system);
    const auto terms_b = residualTerms(*b.system);
    ASSERT_FALSE(terms_a.empty());
    EXPECT_NE(fieldOf(terms_a, "tree"), fieldOf(terms_b, "tree"));
}

TEST(SystemManifest, MeasuresAppearInTermAndGroupLines)
{
    const auto a = buildSystem([](const FormExpr& u, const FormExpr& v) {
        return (u * v).dx() + (u * v).ds(7);
    });
    const auto terms = linesContaining(kernelManifestLines(*a.system), " term=");
    const auto has = [&](std::string_view domain, std::string_view marker_key, std::string_view marker) {
        return std::any_of(terms.begin(), terms.end(), [&](const std::string& line) {
            return valueOf(line, "domain") == domain && valueOf(line, marker_key) == marker;
        });
    };
    EXPECT_TRUE(has("Cell", "bm", "-1"));
    EXPECT_TRUE(has("Boundary", "bm", "7"));
}

TEST(SystemManifest, ConstraintsAreListedInRegistrationOrder)
{
    auto s = buildSystem([](const FormExpr& u, const FormExpr& v) { return (u * v).dx(); });
    s.system->addSystemConstraint(std::make_unique<constraints::VertexDirichletConstraint>(
        s.u_id,
        std::vector<constraints::VertexDirichletValue>{{0, 1.5}, {2, -0.5}}));
    s.system->addSystemConstraint(std::make_unique<constraints::StrongDirichletConstraint>(
        s.u_id, /*boundary_marker=*/4, FormExpr::constant(0.0)));
    s.system->registerGeneratedEmbeddedInterfaceMarker(42);
    s.system->registerGeneratedEmbeddedInterfaceMarker(17);

    const auto lines = constraintManifestLines(*s.system);
    const auto system_lines = linesContaining(lines, "constraint system#");
    ASSERT_EQ(system_lines.size(), 2u);
    EXPECT_NE(system_lines[0].find("VertexDirichletConstraint"), std::string::npos);
    EXPECT_NE(system_lines[0].find("count=2"), std::string::npos);
    EXPECT_NE(system_lines[1].find("StrongDirichletConstraint"), std::string::npos);
    EXPECT_NE(system_lines[1].find("marker=4"), std::string::npos);
    const auto markers = linesContaining(lines, "generated_embedded_interface_markers");
    ASSERT_EQ(markers.size(), 1u);
    EXPECT_NE(markers[0].find("markers=17,42"), std::string::npos);

    // Same constraints in the other order give a different manifest.
    auto t = buildSystem([](const FormExpr& u, const FormExpr& v) { return (u * v).dx(); });
    t.system->addSystemConstraint(std::make_unique<constraints::StrongDirichletConstraint>(
        t.u_id, /*boundary_marker=*/4, FormExpr::constant(0.0)));
    t.system->addSystemConstraint(std::make_unique<constraints::VertexDirichletConstraint>(
        t.u_id,
        std::vector<constraints::VertexDirichletValue>{{0, 1.5}, {2, -0.5}}));
    EXPECT_NE(linesContaining(constraintManifestLines(*t.system), "constraint system#"),
              system_lines);
}

TEST(SystemManifest, BoundaryDataFactoriesGiveTheManifestsOfTheSpelledOutConditions)
{
    namespace bc = forms::bc;
    using Conditions = std::vector<std::unique_ptr<bc::BoundaryCondition>>;
    struct Data {
        bc::BoundaryData<bc::ScalarValue> flux{};
        bc::BoundaryData<bc::RobinValue> robin{};
        bc::BoundaryData<bc::ScalarValue> dirichlet{};
    };
    const auto spelled = [](const Data& d) {
        Conditions out;
        out.push_back(std::make_unique<bc::NaturalBC>(
            3, bc::toScalarExpr(d.flux.value, bc::markerValueName("heat_neumann", 3))));
        out.push_back(std::make_unique<bc::RobinBC>(
            5,
            bc::toScalarExpr(d.robin.value.alpha, bc::markerValueName("heat_robin_alpha", 5)),
            bc::toScalarExpr(d.robin.value.rhs, bc::markerValueName("heat_robin_rhs", 5))));
        out.push_back(std::make_unique<bc::EssentialBC>(
            4, bc::toScalarExpr(d.dirichlet.value, bc::markerValueName("heat_dirichlet", 4)), "u"));
        return out;
    };
    const auto typed = [](const Data& d) {
        Conditions out;
        out.push_back(bc::naturalBC(d.flux, "heat_neumann"));
        out.push_back(bc::robinBC(d.robin, "heat_robin_alpha", "heat_robin_rhs"));
        out.push_back(bc::essentialBC(d.dirichlet, "heat_dirichlet", "u"));
        return out;
    };

    // The same nonlinear residual plus the conditions, with JIT kernel keys
    // when the LLVM JIT is built.
    FormInstallOptions options;
    options.compiler_options.jit.enable = forms::jit::llvmJITEnabled();
    const auto manifests = [&](Conditions conditions) {
        ManifestSystem out;
        out.system = std::make_unique<FESystem>(out.mesh);
        out.u_id = out.system->addField(FieldSpec{.name = "u", .space = out.space, .components = 1});
        out.system->addOperator("op");
        const auto u = forms::StateField(out.u_id, *out.space, "u");
        const auto v = forms::TestField(out.u_id, *out.space, "v");
        auto residual = (inner(grad(u), grad(v)) + u * u * v).dx();
        BoundaryConditionManager bcs;
        for (auto& condition : conditions) {
            bcs.add(std::move(condition));
        }
        bcs.applyAll(*out.system, residual, u, v, out.u_id);
        (void)installFormulation(*out.system, "op", {out.u_id}, residual, options);
        return std::pair{kernelManifestLines(*out.system), constraintManifestLines(*out.system)};
    };

    // Literal and expression data: equal manifests, JIT keys included.
    const Data literal{.flux = {.boundary_marker = 3, .value = 1.5},
                       .robin = {.boundary_marker = 5,
                                 .value = {.alpha = 2.5, .rhs = FormExpr::constant(0.5) * forms::t()}},
                       .dirichlet = {.boundary_marker = 4, .value = -0.75}};
    const auto [spelled_kernels, spelled_constraints] = manifests(spelled(literal));
    const auto [typed_kernels, typed_constraints] = manifests(typed(literal));
    EXPECT_EQ(typed_kernels, spelled_kernels);
    EXPECT_EQ(typed_constraints, spelled_constraints);
    EXPECT_FALSE(linesContaining(typed_kernels, " term=").empty());
    if (forms::jit::llvmJITEnabled()) {
        bool keyed = false;
        for (const auto& line : typed_kernels) {
            const auto key = valueOf(line, "key");
            keyed = keyed || (!key.empty() && key != "-");
        }
        EXPECT_TRUE(keyed);
    }

    // Callback data names each coefficient after the prefix and the marker.
    // The KernelIR keys a callback by its node's address (never cacheable),
    // so the trees, strings and constraints are compared, not kir or keys.
    const forms::ScalarCoefficient g = [](Real x, Real y, Real z) { return 1.0 + x + 2.0 * y + 3.0 * z; };
    const forms::TimeScalarCoefficient h = [](Real x, Real, Real, Real t) { return x * t; };
    const Data callbacks{.flux = {.boundary_marker = 3, .value = h},
                         .robin = {.boundary_marker = 5, .value = {.alpha = 2.5, .rhs = g}},
                         .dirichlet = {.boundary_marker = 4, .value = g}};
    const auto withoutAddressKeys = [](std::vector<std::string> lines) {
        for (auto& line : lines) {
            for (const std::string_view field : {" kir=", " ir_hash=", " key="}) {
                const auto at = line.find(field);
                if (at != std::string::npos) {
                    line.erase(at, line.find(' ', at + 1) == std::string::npos ? std::string::npos
                                                                               : line.find(' ', at + 1) - at);
                }
            }
        }
        return lines;
    };
    const auto [spelled_callback_kernels, spelled_callback_constraints] = manifests(spelled(callbacks));
    const auto [typed_callback_kernels, typed_callback_constraints] = manifests(typed(callbacks));
    EXPECT_EQ(withoutAddressKeys(typed_callback_kernels), withoutAddressKeys(spelled_callback_kernels));
    EXPECT_EQ(typed_callback_constraints, spelled_callback_constraints);
    EXPECT_FALSE(linesContaining(typed_callback_constraints, "heat_dirichlet_4").empty());
}

} // namespace
} // namespace svmp::FE::systems::test
