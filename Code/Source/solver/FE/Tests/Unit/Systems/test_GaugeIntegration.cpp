/**
 * @file test_GaugeIntegration.cpp
 * @brief End-to-end tests for gauge/nullspace detection through FESystem setup
 *
 * Tests the full pipeline: installFormulation → FormContributionLowerer →
 * GaugeRegistry → SystemSetup resolution → automatic constraint creation.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/BoundaryConditionManager.h"

#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryModelDSL.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Forms/FormExpr.h"
#include "Forms/StandardBCs.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Constraints/GaugeRegistry.h"
#include "Constraints/DirichletBC.h"

#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/SpaceFactory.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <iostream>

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;
using namespace svmp::FE::gauge;

namespace {

dofs::MeshTopologyInfo singleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
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

dofs::MeshTopologyInfo quadStripTopology(int n_cells)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = 2 * (n_cells + 1);
    topo.dim = 2;
    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u);
    topo.cell2vertex_data.reserve(static_cast<std::size_t>(4 * n_cells));
    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices));
    topo.cell_gids.resize(static_cast<std::size_t>(n_cells));
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), 0);

    for (int v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = v;
    }
    for (int c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] =
            static_cast<dofs::gid_t>(topo.cell2vertex_data.size());
        const GlobalIndex n0 = 2 * c;
        const GlobalIndex n1 = 2 * c + 1;
        const GlobalIndex n2 = 2 * (c + 1) + 1;
        const GlobalIndex n3 = 2 * (c + 1);
        topo.cell2vertex_data.push_back(n0);
        topo.cell2vertex_data.push_back(n1);
        topo.cell2vertex_data.push_back(n2);
        topo.cell2vertex_data.push_back(n3);
        topo.cell_gids[static_cast<std::size_t>(c)] = c;
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] =
        static_cast<dofs::gid_t>(topo.cell2vertex_data.size());
    return topo;
}

/// Two disconnected tetrahedra: vertices 0-3 form tet A, vertices 4-7 form tet B.
/// No shared vertices → two connected components.
dofs::MeshTopologyInfo twoDisconnectedTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 8;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

/// Mesh access for two disconnected tetrahedra
class TwoDisconnectedTetraMeshAccess final : public assembly::IMeshAccess {
public:
    TwoDisconnectedTetraMeshAccess()
    {
        // Tet A: vertices 0-3 at origin
        nodes_ = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
            // Tet B: vertices 4-7 offset by (5,0,0)
            {5.0, 0.0, 0.0}, {6.0, 0.0, 0.0}, {5.0, 1.0, 0.0}, {5.0, 0.0, 1.0}
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override {
        std::vector<GlobalIndex> nds;
        getCellNodes(cell_id, nds);
        coords.resize(nds.size());
        for (std::size_t i = 0; i < nds.size(); ++i)
            coords[i] = nodes_.at(static_cast<std::size_t>(nds[i]));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0, 0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> nodes_;
};

class StripQuadMeshAccess final : public assembly::IMeshAccess {
public:
    explicit StripQuadMeshAccess(int n_cells)
        : n_cells_(n_cells)
    {
        for (int i = 0; i <= n_cells_; ++i) {
            nodes_.push_back({static_cast<Real>(i), 0.0, 0.0});
            nodes_.push_back({static_cast<Real>(i), 1.0, 0.0});
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return n_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return n_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const int c = static_cast<int>(cell_id);
        nodes = {
            static_cast<GlobalIndex>(2 * c),
            static_cast<GlobalIndex>(2 * c + 1),
            static_cast<GlobalIndex>(2 * (c + 1) + 1),
            static_cast<GlobalIndex>(2 * (c + 1))
        };
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        std::vector<GlobalIndex> nds;
        getCellNodes(cell_id, nds);
        coords.resize(nds.size());
        for (std::size_t i = 0; i < nds.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(nds[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
    {
        if (face_id == 0 && cell_id == 0) {
            return 3;
        }
        if (face_id == 1 && cell_id == static_cast<GlobalIndex>(n_cells_ - 1)) {
            return 1;
        }
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        if (face_id == 0) {
            return 11;
        }
        if (face_id == 1) {
            return 12;
        }
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> cb) const override
    {
        for (int c = 0; c < n_cells_; ++c) {
            cb(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override
    {
        for (int c = 0; c < n_cells_; ++c) {
            cb(c);
        }
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> cb) const override
    {
        if (marker < 0 || marker == 11) {
            cb(0, 0);
        }
        if (marker < 0 || marker == 12) {
            cb(1, static_cast<GlobalIndex>(n_cells_ - 1));
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    int n_cells_{0};
    std::vector<std::array<Real, 3>> nodes_{};
};

struct GaugeSummary {
    bool has_registry{false};
    std::size_t candidates{0};
    std::size_t resolved{0};
    std::size_t constraints{0};
    bool pressure_mode{false};
    bool pressure_anchored{false};
    bool pressure_mean_zero{false};
};

struct AnalysisSummary {
    std::size_t total_claims{0};
    std::size_t mixed_claims{0};
    std::size_t nullspace_claims{0};
    std::size_t pressure_mixed_claims{0};
    std::size_t pressure_nullspace_claims{0};
    std::size_t pressure_nullspace_claims_with_family{0};
    std::size_t pressure_nullspace_claims_from_mixed_analyzer{0};
};

GaugeSummary summarizeGauge(const systems::FESystem& sys, FieldId pressure_field)
{
    GaugeSummary out;
    out.has_registry = sys.hasGaugeRegistry();
    out.constraints = sys.constraints().numConstraints();
    if (!out.has_registry) {
        return out;
    }

    const auto* reg = sys.gaugeRegistryIfPresent();
    if (!reg) {
        return out;
    }

    out.candidates = reg->candidates().size();
    out.resolved = reg->resolvedModes().size();
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == pressure_field) {
            out.pressure_mode = true;
            if (mode.status == GaugeStatus::Anchored) {
                out.pressure_anchored = true;
            }
            if (mode.policy == EnforcementPolicy::MeanZeroElimination) {
                out.pressure_mean_zero = true;
            }
        }
    }
    return out;
}

AnalysisSummary summarizeAnalysis(const systems::FESystem& sys, FieldId pressure_field)
{
    AnalysisSummary out;
    const auto& report = sys.analysisReport();
    out.total_claims = report.claims.size();
    out.mixed_claims = report.countByKind(PropertyKind::MixedSaddlePoint);
    out.nullspace_claims = report.countByKind(PropertyKind::Nullspace);
    for (const auto& claim : report.claims) {
        if (claim.kind == PropertyKind::MixedSaddlePoint) {
            const bool touches_pressure =
                std::any_of(claim.variables.begin(),
                            claim.variables.end(),
                            [pressure_field](const VariableKey& var) {
                                return var.kind == VariableKind::FieldComponent &&
                                       var.field_id == pressure_field;
                            });
            if (touches_pressure || claim.field == pressure_field) {
                ++out.pressure_mixed_claims;
            }
        }
        if (claim.kind == PropertyKind::Nullspace &&
            (claim.field == pressure_field ||
             std::any_of(claim.variables.begin(),
                         claim.variables.end(),
                         [pressure_field](const VariableKey& var) {
                             return var.kind == VariableKind::FieldComponent &&
                                    var.field_id == pressure_field;
                         }))) {
            ++out.pressure_nullspace_claims;
            if (claim.nullspace_family.has_value()) {
                ++out.pressure_nullspace_claims_with_family;
            }
            if (claim.claim_origin == "MixedOperatorAnalyzer") {
                ++out.pressure_nullspace_claims_from_mixed_analyzer;
            }
        }
    }
    return out;
}

struct ExactAssemblyProbeOptions {
    bool transient{true};
    bool linear_reaction{true};
    bool diffusion{true};
    bool nonlinear_reaction{true};
    bool monolithic_outlets{true};
    bool constrain_velocity_boundaries{false};
};

std::unique_ptr<systems::FESystem> buildQuadStripMixedSystem(int n_cells,
                                                             bool transient,
                                                             bool monolithic_outlets,
                                                             FieldId& velocity_field,
                                                             FieldId& pressure_field)
{
    auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells);
    auto u_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                       ElementType::Quad4,
                                       /*order=*/1,
                                       /*components=*/2);
    auto p_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    auto sys = std::make_unique<systems::FESystem>(mesh);
    velocity_field =
        sys->addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
    pressure_field =
        sys->addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys->addOperator("op");

    const auto u = FormExpr::stateField(velocity_field, *u_space, "u");
    const auto p = FormExpr::stateField(pressure_field, *p_space, "p");
    const auto v = FormExpr::testFunction(velocity_field, *u_space, "v");
    const auto q = FormExpr::testFunction(pressure_field, *p_space, "q");
    const auto n = FormExpr::normal();

    FormExpr residual =
        inner(grad(u), grad(v)).dx() -
        (p * div(v)).dx() +
        (q * div(u)).dx();

    if (transient) {
        residual = inner(u.dt(1), v).dx() + residual;
    }

    if (monolithic_outlets) {
        const auto u_disc = FormExpr::discreteField(velocity_field, *u_space, "u_disc");
        const auto Q_left = sys->boundaryIntegral(inner(u_disc, n), 11);
        const auto Q_right = sys->boundaryIntegral(inner(u_disc, n), 12);

        auto model = systems::aux::model("resistive_direct_only_probe", [](systems::ModelFacade& m) {
            auto Q = m.input("Q");
            auto P = m.state("P", systems::AuxiliaryVariableKind::Algebraic);
            auto Rsum = m.param("Rsum");
            auto Pd = m.param("Pd");

            m.initialGuess("P", 0.0);
            m << systems::alg(P) == P - (Pd + Rsum * Q);
            m << systems::out("P_out") == P;
        });

        auto left = sys->deploy(
            systems::use(model).name("left_outlet").boundary(11).monolithic()
                .bind("Q", Q_left)
                .param("Rsum", 80.0)
                .param("Pd", 20.0)
                .initialState({{"P", 20.0}}));
        auto right = sys->deploy(
            systems::use(model).name("right_outlet").boundary(12).monolithic()
                .bind("Q", Q_right)
                .param("Rsum", 110.0)
                .param("Pd", 25.0)
                .initialState({{"P", 25.0}}));

        residual = residual -
            (left.output("P_out") * inner(v, n)).ds(11) +
            (right.output("P_out") * inner(v, n)).ds(12);
    }

    systems::installFormulation(*sys, "op", {velocity_field, pressure_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = quadStripTopology(n_cells);
    sys->setup({}, inputs);
    if (monolithic_outlets) {
        sys->finalizeAuxiliaryLayout();
    }
    return sys;
}

std::unique_ptr<systems::FESystem> buildQuadStripMixedSystemExactAssemblyProbe(int n_cells,
                                                                               ExactAssemblyProbeOptions options,
                                                                               FieldId& velocity_field,
                                                                               FieldId& pressure_field)
{
    auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells);
    auto u_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                       ElementType::Quad4,
                                       /*order=*/1,
                                       /*components=*/2);
    auto p_space = spaces::Space(spaces::SpaceType::H1,
                                 ElementType::Quad4,
                                 /*order=*/1,
                                 /*components=*/1);

    auto sys = std::make_unique<systems::FESystem>(mesh);
    velocity_field =
        sys->addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
    pressure_field =
        sys->addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys->addOperator("op");

    const auto u = FormExpr::stateField(velocity_field, *u_space, "u");
    const auto p = FormExpr::stateField(pressure_field, *p_space, "p");
    const auto u_disc = FormExpr::discreteField(velocity_field, *u_space, "u_disc");
    const auto v = FormExpr::testFunction(velocity_field, *u_space, "v");
    const auto q = FormExpr::testFunction(pressure_field, *p_space, "q");
    const auto n = FormExpr::normal();

    const auto one = FormExpr::constant(Real(1.0));
    const auto lambda = FormExpr::constant(Real(0.75));
    const auto nu = FormExpr::constant(Real(0.05));
    const auto eps = FormExpr::constant(Real(0.20));
    const auto kappa = FormExpr::constant(Real(0.0));
    FormExpr momentum = - (p * div(v));
    if (options.transient) {
        momentum = inner(u.dt(1), v) + momentum;
    }
    if (options.linear_reaction) {
        momentum = lambda * inner(u, v) + momentum;
    }
    if (options.diffusion) {
        momentum = nu * inner(grad(u), grad(v)) + momentum;
    }
    if (options.nonlinear_reaction) {
        momentum = eps * (one + inner(u, u)) * inner(u, v) + momentum;
    }

    FormExpr residual = momentum.dx() +
                        (q * div(u) + kappa * p * q).dx();

    if (options.monolithic_outlets) {
        const auto Q_left = sys->boundaryIntegral(inner(u_disc, n), 11);
        const auto Q_right = sys->boundaryIntegral(inner(u_disc, n), 12);

        auto model =
            systems::aux::model("resistive_direct_only_exact_probe", [](systems::ModelFacade& m) {
                auto Q = m.input("Q");
                auto P = m.state("P", systems::AuxiliaryVariableKind::Algebraic);
                auto Rsum = m.param("Rsum");
                auto Pd = m.param("Pd");

                m.initialGuess("P", 0.0);
                m << systems::alg(P) == P - (Pd + Rsum * Q);
                m << systems::out("P_out") == P;
            });

        auto left = sys->deploy(
            systems::use(model).name("left_outlet").boundary(11).monolithic()
                .bind("Q", Q_left)
                .param("Rsum", 80.0)
                .param("Pd", 20.0)
                .initialState({{"P", 20.0}}));
        auto right = sys->deploy(
            systems::use(model).name("right_outlet").boundary(12).monolithic()
                .bind("Q", Q_right)
                .param("Rsum", 110.0)
                .param("Pd", 25.0)
                .initialState({{"P", 25.0}}));

        residual = residual -
            (left.output("P_out") * inner(v, n)).ds(11) +
            (right.output("P_out") * inner(v, n)).ds(12);
    }

    systems::installFormulation(*sys, "op", {velocity_field, pressure_field}, residual);

    if (options.constrain_velocity_boundaries) {
        auto left_bc = bc::strongDirichlet(velocity_field,
                                           11,
                                           FormExpr::constant(0.0),
                                           "u");
        auto right_bc = bc::strongDirichlet(velocity_field,
                                            12,
                                            FormExpr::constant(0.0),
                                            "u");
        systems::installStrongDirichlet(
            *sys,
            std::span<const bc::StrongDirichlet>(&left_bc, 1));
        systems::installStrongDirichlet(
            *sys,
            std::span<const bc::StrongDirichlet>(&right_bc, 1));
    }

    systems::SetupInputs inputs;
    inputs.topology_override = quadStripTopology(n_cells);
    sys->setup({}, inputs);
    if (options.monolithic_outlets) {
        sys->finalizeAuxiliaryLayout();
    }
    return sys;
}

void logGaugeSummary(const char* label, const GaugeSummary& s)
{
    std::cout << "[gauge-strip-probe] case=" << label
              << " has_gauge=" << (s.has_registry ? 1 : 0)
              << " candidates=" << s.candidates
              << " resolved=" << s.resolved
              << " constraints=" << s.constraints
              << " pressure_mode=" << (s.pressure_mode ? 1 : 0)
              << " pressure_anchored=" << (s.pressure_anchored ? 1 : 0)
              << " pressure_mean_zero=" << (s.pressure_mean_zero ? 1 : 0)
              << std::endl;
}

void logAnalysisSummary(const char* label, const AnalysisSummary& s)
{
    std::cerr << "[gauge-analysis] " << label
              << " total_claims=" << s.total_claims
              << " mixed_claims=" << s.mixed_claims
              << " nullspace_claims=" << s.nullspace_claims
              << " pressure_mixed_claims=" << s.pressure_mixed_claims
              << " pressure_nullspace_claims=" << s.pressure_nullspace_claims
              << " pressure_nullspace_with_family=" << s.pressure_nullspace_claims_with_family
              << " pressure_nullspace_from_mixed=" << s.pressure_nullspace_claims_from_mixed_analyzer
              << std::endl;
}

/// Two disconnected tetrahedra with one boundary face on tet A (marker 1).
/// Face 0 is on cell 0 (tet A), so Robin BC on marker 1 only touches region A.
class TwoTetraOneBoundaryMeshAccess final : public assembly::IMeshAccess {
public:
    TwoTetraOneBoundaryMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
            {5.0, 0.0, 0.0}, {6.0, 0.0, 0.0}, {5.0, 1.0, 0.0}, {5.0, 0.0, 1.0}
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override {
        std::vector<GlobalIndex> nds;
        getCellNodes(cell_id, nds);
        coords.resize(nds.size());
        for (std::size_t i = 0; i < nds.size(); ++i)
            coords[i] = nodes_.at(static_cast<std::size_t>(nds[i]));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override {
        return (face_id == 0) ? 1 : -1;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0, 0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> cb) const override {
        // Face 0: marker 1, belongs to cell 0 (tet A)
        if (marker < 0 || marker == 1) {
            cb(0, 0);
        }
    }
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> nodes_;
};

} // namespace

// ============================================================================
// Scalar Poisson with pure Neumann → auto gauge constraint
// ============================================================================

TEST(GaugeIntegration, ScalarPoisson_PureNeumann_AutoZeroMean)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // No Dirichlet BCs → pure Neumann-like
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Check that the gauge registry was populated and resolved
    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_NE(reg, nullptr);

    EXPECT_FALSE(reg->candidates().empty());
    EXPECT_TRUE(reg->isResolved());

    // Should have created an ExactNullspace mode with MeanZeroElimination
    bool found_exact = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::ExactNullspace &&
            mode.policy == EnforcementPolicy::MeanZeroElimination) {
            found_exact = true;
        }
    }
    EXPECT_TRUE(found_exact);

    // The AffineConstraints should have at least one constrained DOF
    // (from the auto-created GlobalConstraint::zeroMean)
    const auto& ac = sys.constraints();
    bool any_constrained = false;
    for (GlobalIndex d = 0; d < 4; ++d) {
        if (ac.isConstrained(d)) {
            any_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(any_constrained);
}

// ============================================================================
// Scalar Poisson with Dirichlet → anchored, no auto constraint
// ============================================================================

TEST(GaugeIntegration, ScalarPoisson_WithDirichlet_NoAutoConstraint)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Add a Dirichlet constraint on DOF 0
    auto dirichlet = std::make_unique<constraints::DirichletBC>(0, 0.0);
    sys.addConstraint(std::move(dirichlet));

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // The gauge registry should detect the Dirichlet anchoring
    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    bool found_anchored = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) {
            found_anchored = true;
        }
    }
    EXPECT_TRUE(found_anchored);
}

// ============================================================================
// Reaction-diffusion: mass term anchors, no auto constraint
// ============================================================================

TEST(GaugeIntegration, ScalarReactionDiffusion_NoNullspace)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx() + (u * v).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Mass term anchors the mode → no candidates should be produced
    if (sys.hasGaugeRegistry()) {
        EXPECT_TRUE(sys.gaugeRegistryIfPresent()->candidates().empty());
    }
}

// ============================================================================
// BC anchoring verdicts — StandardBCs
// ============================================================================

TEST(GaugeIntegration, EssentialBC_DescriptorAnchorsConstantAndTranslation)
{
    // EssentialBC descriptor has anchors_constant_mode=true and
    // anchors_rigid_body_translation=true. descriptorToVerdict returns
    // Anchored for ScalarConstant/ComponentwiseConstant, PartiallyAnchored
    // for KernelOfSymGrad (translation anchored, rotation not).
    bc::EssentialBC bc(1, FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad), AnchoringVerdict::PartiallyAnchored);
}

TEST(GaugeIntegration, NaturalBC_Preserves_AllFamilies)
{
    bc::NaturalBC bc(1, FormExpr::constant(1.0));
    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant), AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant), AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad), AnchoringVerdict::Preserved);
}

TEST(GaugeIntegration, RobinBC_Anchors_ConstantModes_PartiallyAnchors_KernelOfSymGrad)
{
    bc::RobinBC bc(1, FormExpr::constant(1.0), FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad), AnchoringVerdict::PartiallyAnchored);
}

TEST(GaugeIntegration, ReservedBC_EmptyDescriptors)
{
    // ReservedBC returns empty descriptors — no mathematical constraint.
    bc::ReservedBC bc(1);
    auto descs = bc.analysisMetadata(0, nullptr);
    EXPECT_TRUE(descs.empty());
}

// ============================================================================
// Vector field — componentwise constant detection through FESystem
// ============================================================================

TEST(GaugeIntegration, VectorField_GradOnly_ComponentwiseConstantCandidate)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto space = std::make_shared<spaces::ProductSpace>(base, 3);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 3});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Gauge candidates are populated during setup() from NullspaceHints.
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto& candidates = sys.gaugeRegistryIfPresent()->candidates();
    ASSERT_FALSE(candidates.empty());
    bool found_cw = false;
    for (const auto& c : candidates) {
        if (c.family == NullspaceModeFamily::ComponentwiseConstant ||
            c.family == NullspaceModeFamily::ScalarConstant) {
            found_cw = true;
        }
    }
    EXPECT_TRUE(found_cw) << "Expected ComponentwiseConstant or per-component ScalarConstant candidate";
}

// ============================================================================
// Vector field — rigid-body mode detection through FESystem
// ============================================================================

TEST(GaugeIntegration, VectorField_SymGrad_KernelOfSymGradCandidate)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto space = std::make_shared<spaces::ProductSpace>(base, 3);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 3});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Gauge candidates are populated during setup() from NullspaceHints.
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto& candidates = sys.gaugeRegistryIfPresent()->candidates();
    ASSERT_FALSE(candidates.empty());
    bool found_rbm = false;
    for (const auto& c : candidates) {
        if (c.family == NullspaceModeFamily::KernelOfSymGrad) {
            found_rbm = true;
        }
    }
    EXPECT_TRUE(found_rbm) << "Expected KernelOfSymGrad candidate";
}

// ============================================================================
// Phase 3: Non-Forms kernel analysisContributions() hooks
// ============================================================================

namespace {

/// Minimal AssemblyKernel that declares a gauge candidate via analysisContributions()
class GaugeDeclaringKernel final : public assembly::AssemblyKernel {
public:
    GaugeDeclaringKernel(FieldId field, NullspaceModeFamily family)
        : field_(field), family_(family)
    {
    }

    [[nodiscard]] assembly::RequiredData getRequiredData() const override {
        return assembly::RequiredData::None;
    }

    void computeCell(const assembly::AssemblyContext& /*ctx*/,
                     assembly::KernelOutput& out) override
    {
        out.reserve(0, 0, true, false);
    }

    [[nodiscard]] std::vector<analysis::ContributionDescriptor> analysisContributions() const override
    {
        analysis::ContributionDescriptor cd;
        cd.operator_tag = "gauge_declaring_kernel";
        cd.origin = "GaugeDeclaringKernel";
        cd.domain = analysis::DomainKind::Cell;
        cd.role = analysis::ContributionRole::DiagonalBlock;
        cd.test_variables = {analysis::VariableKey::field(field_)};
        cd.trial_variables = {analysis::VariableKey::field(field_)};
        cd.confidence = analysis::AnalysisConfidence::High;

        analysis::NullspaceHint hint;
        hint.field = field_;
        hint.component = -1;
        hint.confidence = analysis::AnalysisConfidence::High;
        hint.reason = "Explicitly declared by GaugeDeclaringKernel";
        if (family_ == NullspaceModeFamily::ScalarConstant) {
            hint.family = analysis::NullspaceFamily::ScalarConstant;
        } else if (family_ == NullspaceModeFamily::KernelOfSymGrad) {
            hint.family = analysis::NullspaceFamily::KernelOfSymGrad;
        } else {
            hint.family = analysis::NullspaceFamily::ComponentwiseConstant;
        }
        cd.nullspace_hints.push_back(std::move(hint));
        return {cd};
    }

private:
    FieldId field_;
    NullspaceModeFamily family_;
};

/// Minimal GlobalKernel that declares a gauge candidate via analysisContributions()
class GaugeDeclaringGlobalKernel final : public systems::GlobalKernel {
public:
    GaugeDeclaringGlobalKernel(FieldId field, NullspaceModeFamily family)
        : field_(field), family_(family)
    {
    }

    [[nodiscard]] assembly::AssemblyResult assemble(
        const systems::FESystem& /*system*/,
        const systems::AssemblyRequest& /*request*/,
        const systems::SystemStateView& /*state*/,
        assembly::GlobalSystemView* /*matrix_out*/,
        assembly::GlobalSystemView* /*vector_out*/) override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::ContributionDescriptor> analysisContributions() const override
    {
        analysis::ContributionDescriptor cd;
        cd.operator_tag = "gauge_declaring_global_kernel";
        cd.origin = "GaugeDeclaringGlobalKernel";
        cd.domain = analysis::DomainKind::Cell;
        cd.role = analysis::ContributionRole::DiagonalBlock;
        cd.test_variables = {analysis::VariableKey::field(field_)};
        cd.trial_variables = {analysis::VariableKey::field(field_)};
        cd.confidence = analysis::AnalysisConfidence::High;

        analysis::NullspaceHint hint;
        hint.field = field_;
        hint.component = -1;
        hint.confidence = analysis::AnalysisConfidence::High;
        hint.reason = "Explicitly declared by GaugeDeclaringGlobalKernel";
        if (family_ == NullspaceModeFamily::ScalarConstant) {
            hint.family = analysis::NullspaceFamily::ScalarConstant;
        } else if (family_ == NullspaceModeFamily::KernelOfSymGrad) {
            hint.family = analysis::NullspaceFamily::KernelOfSymGrad;
        } else {
            hint.family = analysis::NullspaceFamily::ComponentwiseConstant;
        }
        cd.nullspace_hints.push_back(std::move(hint));
        return {cd};
    }

private:
    FieldId field_;
    NullspaceModeFamily family_;
};

} // namespace

TEST(GaugeIntegration, CellKernel_AnalysisContributions_CollectedDuringSetup)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Add a hand-written kernel that declares ScalarConstant nullspace
    auto kernel = std::make_shared<GaugeDeclaringKernel>(u_field, NullspaceModeFamily::ScalarConstant);
    sys.addCellKernel("op", u_field, kernel);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // The gauge registry should have collected the candidate from the kernel's
    // analysisContributions() NullspaceHint
    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_NE(reg, nullptr);

    bool found_explicit = false;
    for (const auto& c : reg->candidates()) {
        if (c.field == u_field &&
            c.family == NullspaceModeFamily::ScalarConstant &&
            c.source == gauge::CandidateSource::ExplicitDeclaration) {
            found_explicit = true;
        }
    }
    EXPECT_TRUE(found_explicit);

    // Without Dirichlet BC, it should resolve to ExactNullspace
    bool found_exact = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == u_field &&
            mode.status == gauge::GaugeStatus::ExactNullspace) {
            found_exact = true;
        }
    }
    EXPECT_TRUE(found_exact);
}

TEST(GaugeIntegration, GlobalKernel_AnalysisContributions_CollectedDuringSetup)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Add a global kernel that declares ScalarConstant nullspace
    auto gk = std::make_shared<GaugeDeclaringGlobalKernel>(u_field, NullspaceModeFamily::ScalarConstant);
    sys.addGlobalKernel("op", gk);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    bool found_explicit = false;
    for (const auto& c : reg->candidates()) {
        if (c.field == u_field &&
            c.source == gauge::CandidateSource::ExplicitDeclaration) {
            found_explicit = true;
        }
    }
    EXPECT_TRUE(found_explicit);
}

TEST(GaugeIntegration, CellKernel_AnalysisContributions_AnchoredByDirichlet)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto kernel = std::make_shared<GaugeDeclaringKernel>(u_field, NullspaceModeFamily::ScalarConstant);
    sys.addCellKernel("op", u_field, kernel);

    // Add Dirichlet to anchor the mode
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(0, 0.0));

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    bool found_anchored = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == u_field &&
            mode.status == gauge::GaugeStatus::Anchored) {
            found_anchored = true;
        }
    }
    EXPECT_TRUE(found_anchored);
}

TEST(GaugeIntegration, DefaultKernel_EmptyAnalysisContributions)
{
    // Verify that the base class default returns empty
    class NoMetadataKernel final : public assembly::AssemblyKernel {
    public:
        [[nodiscard]] assembly::RequiredData getRequiredData() const override {
            return assembly::RequiredData::None;
        }
        void computeCell(const assembly::AssemblyContext&, assembly::KernelOutput& out) override {
            out.reserve(0, 0, true, false);
        }
    };

    NoMetadataKernel nk;
    EXPECT_TRUE(nk.analysisContributions().empty());
}

TEST(GaugeIntegration, DefaultGlobalKernel_EmptyAnalysisContributions)
{
    class NoMetadataGlobalKernel final : public systems::GlobalKernel {
    public:
        [[nodiscard]] assembly::AssemblyResult assemble(
            const systems::FESystem&, const systems::AssemblyRequest&,
            const systems::SystemStateView&,
            assembly::GlobalSystemView*, assembly::GlobalSystemView*) override
        {
            return {};
        }
    };

    NoMetadataGlobalKernel nk;
    EXPECT_TRUE(nk.analysisContributions().empty());
}

// ============================================================================
// Phase 4: LinearSolver nullspace interface
// ============================================================================

TEST(GaugeIntegration, LinearSolver_DefaultSupportsNullspace_IsFalse)
{
    // Create a minimal solver to check the default
    class MinimalSolver final : public backends::LinearSolver {
    public:
        [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
            return backends::BackendKind::Eigen;
        }
        void setOptions(const backends::SolverOptions& o) override { opts_ = o; }
        [[nodiscard]] const backends::SolverOptions& getOptions() const noexcept override { return opts_; }
        [[nodiscard]] backends::SolverReport solve(
            const backends::GenericMatrix&, backends::GenericVector&,
            const backends::GenericVector&) override { return {}; }
    private:
        backends::SolverOptions opts_{};
    };

    MinimalSolver solver;
    EXPECT_FALSE(solver.supportsNullspace());

    // setNullspaceBasis should be a no-op (shouldn't crash)
    std::vector<std::vector<double>> basis = {{1.0, 0.0}, {0.0, 1.0}};
    solver.setNullspaceBasis(basis);  // no-op
}

// ============================================================================
// Phase 4: GaugeRegistry — resolver always picks algebraic enforcement
// ============================================================================

TEST(GaugeIntegration, Resolve_AlwaysAlgebraic_ScalarConstant)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };

    // Resolver always picks algebraic enforcement (MeanZero for scalar constant)
    reg.resolve(dof_provider);

    ASSERT_EQ(reg.resolvedModes().size(), 1u);
    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::MeanZeroElimination);
}

// ============================================================================
// Phase 4: buildNullspaceBasis — used for solver-side supplemental projection
// ============================================================================

TEST(GaugeIntegration, BuildNullspaceBasis_Empty_WhenAlgebraicEnforcement)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    // buildNullspaceBasis only emits for SolverNullspace policy.
    // Algebraic enforcement (MeanZero) modifies the constraint system,
    // so the original nullspace vector is NOT the nullspace of the
    // constrained operator — projecting against it would be incorrect.
    auto basis = reg.buildNullspaceBasis(/*n_total_dofs=*/4, dof_provider);
    EXPECT_TRUE(basis.empty());
}

TEST(GaugeIntegration, BuildNullspaceBasis_Empty_WhenAnchored)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "Dirichlet"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    auto basis = reg.buildNullspaceBasis(4, dof_provider);
    EXPECT_TRUE(basis.empty());
}

TEST(GaugeIntegration, ApplyEnforcement_AlgebraicMeanZero_CreatesConstraint)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);

    // Algebraic MeanZero enforcement creates 1 constraint
    EXPECT_EQ(n, 1);
    EXPECT_TRUE(ac.isConstrained(0));
}

// ============================================================================
// Resolver always uses algebraic — solver-side projection supplements in Newton
// ============================================================================

TEST(GaugeIntegration, Resolve_AlwaysAlgebraic_SetupWithNoPreference)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    // Resolver always picks algebraic enforcement (MeanZero)
    bool found_mean_zero = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.policy == EnforcementPolicy::MeanZeroElimination) {
            found_mean_zero = true;
        }
    }
    EXPECT_TRUE(found_mean_zero);

    // Algebraic constraint should have been created
    const auto& ac = sys.constraints();
    bool any_constrained = false;
    for (GlobalIndex d = 0; d < 4; ++d) {
        if (ac.isConstrained(d)) {
            any_constrained = true;
        }
    }
    EXPECT_TRUE(any_constrained);
}

TEST(GaugeIntegration, MixedSaddlePointPressureFieldGetsGauge)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector_space = std::make_shared<spaces::ProductSpace>(scalar_space, 3);

    systems::FESystem sys(mesh);
    const auto velocity_field =
        sys.addField(systems::FieldSpec{.name = "u", .space = vector_space, .components = 3});
    const auto pressure_field =
        sys.addField(systems::FieldSpec{.name = "p", .space = scalar_space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(velocity_field, *vector_space, "u");
    auto p = FormExpr::stateField(pressure_field, *scalar_space, "p");
    auto v = FormExpr::testFunction(velocity_field, *vector_space, "v");
    auto q = FormExpr::testFunction(pressure_field, *scalar_space, "q");

    auto residual =
        inner(grad(u), grad(v)).dx() -
        (p * div(v)).dx() +
        (q * div(u)).dx();

    systems::installFormulation(sys, "op", {velocity_field, pressure_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_FALSE(reg->candidates().empty());
    EXPECT_TRUE(reg->isResolved());

    bool found_pressure_mode = false;
    bool found_mean_zero = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == pressure_field) {
            found_pressure_mode = true;
            if (mode.policy == EnforcementPolicy::MeanZeroElimination) {
                found_mean_zero = true;
            }
        }
    }

    EXPECT_TRUE(found_pressure_mode);
    EXPECT_TRUE(found_mean_zero);
    EXPECT_GT(sys.constraints().numConstraints(), 0u);
}

TEST(GaugeIntegration, MixedSaddlePointPressureFieldGetsGauge_VectorSpace)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector_space =
        spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Tetra4, /*order=*/1, /*components=*/3);

    systems::FESystem sys(mesh);
    const auto velocity_field =
        sys.addField(systems::FieldSpec{.name = "u", .space = vector_space, .components = 3});
    const auto pressure_field =
        sys.addField(systems::FieldSpec{.name = "p", .space = scalar_space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(velocity_field, *vector_space, "u");
    auto p = FormExpr::stateField(pressure_field, *scalar_space, "p");
    auto v = FormExpr::testFunction(velocity_field, *vector_space, "v");
    auto q = FormExpr::testFunction(pressure_field, *scalar_space, "q");

    auto residual =
        inner(grad(u), grad(v)).dx() -
        (p * div(v)).dx() +
        (q * div(u)).dx();

    systems::installFormulation(sys, "op", {velocity_field, pressure_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_FALSE(reg->candidates().empty());
    EXPECT_TRUE(reg->isResolved());

    bool found_pressure_mode = false;
    bool found_mean_zero = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == pressure_field) {
            found_pressure_mode = true;
            if (mode.policy == EnforcementPolicy::MeanZeroElimination) {
                found_mean_zero = true;
            }
        }
    }

    EXPECT_TRUE(found_pressure_mode);
    EXPECT_TRUE(found_mean_zero);
    EXPECT_GT(sys.constraints().numConstraints(), 0u);
}

TEST(GaugeIntegration, MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyReactionFallback)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    ExactAssemblyProbeOptions options;
    options.constrain_velocity_boundaries = true;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           options,
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());

    const auto gauge = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);

    EXPECT_TRUE(gauge.has_registry);
    EXPECT_EQ(gauge.candidates, 1u);
    EXPECT_EQ(gauge.resolved, 1u);
    EXPECT_TRUE(gauge.pressure_mode);
    EXPECT_TRUE(gauge.pressure_mean_zero);
    EXPECT_GT(gauge.constraints, 0u);

    EXPECT_EQ(analysis.pressure_mixed_claims, 1u);
    EXPECT_EQ(analysis.pressure_nullspace_claims, 1u);
    EXPECT_EQ(analysis.pressure_nullspace_claims_with_family, 1u);
    EXPECT_EQ(analysis.pressure_nullspace_claims_from_mixed_analyzer, 1u);
}

TEST(GaugeIntegration, MixedSaddlePointPressureFieldAnchoredByNaturalVelocityBoundary_QuadStripExactAssembly)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           ExactAssemblyProbeOptions{},
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());

    const auto gauge = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);

    EXPECT_TRUE(gauge.has_registry);
    EXPECT_EQ(gauge.candidates, 1u);
    EXPECT_EQ(gauge.resolved, 1u);
    EXPECT_TRUE(gauge.pressure_mode);
    EXPECT_TRUE(gauge.pressure_anchored);
    EXPECT_FALSE(gauge.pressure_mean_zero);
    EXPECT_EQ(gauge.constraints, 0u);

    EXPECT_EQ(analysis.pressure_mixed_claims, 1u);
    EXPECT_EQ(analysis.pressure_nullspace_claims, 1u);
    EXPECT_EQ(analysis.pressure_nullspace_claims_with_family, 1u);
    EXPECT_EQ(analysis.pressure_nullspace_claims_from_mixed_analyzer, 1u);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripSteadyProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    auto sys = buildQuadStripMixedSystem(/*n_cells=*/4,
                                         /*transient=*/false,
                                         /*monolithic_outlets=*/false,
                                         velocity_field,
                                         pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    logGaugeSummary("quad_strip_steady", summary);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripTransientProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    auto sys = buildQuadStripMixedSystem(/*n_cells=*/4,
                                         /*transient=*/true,
                                         /*monolithic_outlets=*/false,
                                         velocity_field,
                                         pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    logGaugeSummary("quad_strip_transient", summary);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripTransientOutletProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    auto sys = buildQuadStripMixedSystem(/*n_cells=*/4,
                                         /*transient=*/true,
                                         /*monolithic_outlets=*/true,
                                         velocity_field,
                                         pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    logGaugeSummary("quad_strip_transient_outlets", summary);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           ExactAssemblyProbeOptions{},
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);
    logGaugeSummary("quad_strip_exact_assembly_probe", summary);
    logAnalysisSummary("quad_strip_exact_assembly_probe", analysis);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyNoNonlinearProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    ExactAssemblyProbeOptions options;
    options.nonlinear_reaction = false;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           options,
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);
    logGaugeSummary("quad_strip_exact_no_nonlinear", summary);
    logAnalysisSummary("quad_strip_exact_no_nonlinear", analysis);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyNoLinearProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    ExactAssemblyProbeOptions options;
    options.linear_reaction = false;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           options,
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);
    logGaugeSummary("quad_strip_exact_no_linear", summary);
    logAnalysisSummary("quad_strip_exact_no_linear", analysis);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyNoReactionProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    ExactAssemblyProbeOptions options;
    options.linear_reaction = false;
    options.nonlinear_reaction = false;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           options,
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);
    logGaugeSummary("quad_strip_exact_no_reaction", summary);
    logAnalysisSummary("quad_strip_exact_no_reaction", analysis);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyNoOutletProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    ExactAssemblyProbeOptions options;
    options.monolithic_outlets = false;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           options,
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);
    logGaugeSummary("quad_strip_exact_no_outlet", summary);
    logAnalysisSummary("quad_strip_exact_no_outlet", analysis);
}

TEST(GaugeIntegration, DISABLED_MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyTransientDiffusionOutletProbe)
{
    FieldId velocity_field = INVALID_FIELD_ID;
    FieldId pressure_field = INVALID_FIELD_ID;
    ExactAssemblyProbeOptions options;
    options.linear_reaction = false;
    options.nonlinear_reaction = false;
    auto sys = buildQuadStripMixedSystemExactAssemblyProbe(/*n_cells=*/4,
                                                           options,
                                                           velocity_field,
                                                           pressure_field);
    ASSERT_TRUE(sys);
    ASSERT_TRUE(sys->isSetup());
    const auto summary = summarizeGauge(*sys, pressure_field);
    const auto analysis = summarizeAnalysis(*sys, pressure_field);
    logGaugeSummary("quad_strip_exact_transient_diffusion_outlet", summary);
    logAnalysisSummary("quad_strip_exact_transient_diffusion_outlet", analysis);
}

// ============================================================================
// Connected-component scoping: two disconnected regions, Dirichlet on one
// ============================================================================

TEST(GaugeIntegration, TwoRegions_DirichletOnRegionA_RegionBGetsGauge)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Add Dirichlet on DOF 0 (vertex 0, in region A).
    // Region B (vertices 4-7) has no Dirichlet → should get auto-gauge.
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(0, 0.0));

    systems::SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_TRUE(reg->isResolved());

    // The two-tet mesh has 2 connected components.  Region expansion should
    // produce 2 candidate modes (one per region).  The Dirichlet on DOF 0
    // (region A) should anchor only region A; region B should be ExactNullspace.
    ASSERT_EQ(reg->resolvedModes().size(), 2u)
        << "Expected 2 resolved modes (one per disconnected region)";

    int n_anchored = 0;
    int n_exact_nullspace = 0;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) ++n_anchored;
        if (mode.status == GaugeStatus::ExactNullspace) ++n_exact_nullspace;
    }

    EXPECT_EQ(n_anchored, 1) << "Exactly one region should be anchored (has Dirichlet)";
    EXPECT_EQ(n_exact_nullspace, 1) << "Exactly one region should be ExactNullspace (no Dirichlet)";

    // The unanchored region should have gotten a gauge constraint
    const auto& ac = sys.constraints();
    bool region_b_constrained = false;
    // Region B DOFs are 4-7; check if any of them got constrained by gauge
    for (GlobalIndex d = 4; d < 8; ++d) {
        if (ac.isConstrained(d)) {
            region_b_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(region_b_constrained)
        << "Region B (no Dirichlet) should have an auto-gauge constraint";
}

// ============================================================================
// Connected-component scoping: Robin BC on one region, end-to-end
// ============================================================================

TEST(GaugeIntegration, TwoRegions_RobinOnRegionA_RegionBGetsGauge)
{
    // End-to-end test: Robin BC on boundary marker 1 (only touches tet A / region A).
    // Region A is anchored by the Robin BC.  Region B has pure Neumann → should
    // get auto-gauge enforcement.  This exercises:
    //  - BC manager setting boundary_marker on evidence
    //  - retagEvidenceRegions() converting marker 1 → region A
    //  - Resolver correctly scoping the anchor to region A only
    auto mesh = std::make_shared<TwoTetraOneBoundaryMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    // Robin BC on marker 1 (only on tet A): u + penalty*u = g
    // descriptorToVerdict() returns Anchored for ScalarConstant.
    systems::BoundaryConditionManager bc_mgr;
    bc_mgr.add(std::make_unique<bc::RobinBC>(1, FormExpr::constant(1.0), FormExpr::constant(0.0)));
    bc_mgr.apply(sys, residual, u, v, u_field);

    systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_TRUE(reg->isResolved());

    // Expect 2 resolved modes (one per region).
    // Region A: anchored by Robin BC (via retagged evidence).
    // Region B: ExactNullspace (no anchor) → gets gauge enforcement.
    ASSERT_EQ(reg->resolvedModes().size(), 2u)
        << "Expected 2 resolved modes (one per disconnected region)";

    int n_anchored = 0;
    int n_exact_nullspace = 0;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) ++n_anchored;
        if (mode.status == GaugeStatus::ExactNullspace) ++n_exact_nullspace;
    }

    EXPECT_EQ(n_anchored, 1)
        << "Exactly one region should be anchored (has Robin BC)";
    EXPECT_EQ(n_exact_nullspace, 1)
        << "Exactly one region should be ExactNullspace (no BC)";

    // Verify that region B DOFs actually got a gauge constraint.
    // Region B vertices are 4-7 → DOFs 4-7 for a scalar field.
    const auto& ac = sys.constraints();
    bool region_b_constrained = false;
    for (GlobalIndex d = 4; d < 8; ++d) {
        if (ac.isConstrained(d)) {
            region_b_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(region_b_constrained)
        << "Region B (no Robin BC) should have an auto-gauge constraint";
}

TEST(GaugeIntegration, TwoRegions_VectorFieldFirst_RegionScopingUsesMeshTopology)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector_space = std::make_shared<spaces::ProductSpace>(scalar_space, 3);

    systems::FESystem sys(mesh);
    const auto velocity_field = sys.addField(
        systems::FieldSpec{.name = "u", .space = vector_space, .components = 3});
    const auto pressure_field = sys.addField(
        systems::FieldSpec{.name = "p", .space = scalar_space, .components = 1});
    (void)velocity_field;
    sys.addOperator("op");

    auto p = FormExpr::stateField(pressure_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto residual = inner(grad(p), grad(q)).dx();

    systems::installFormulation(sys, "op", {pressure_field}, residual);

    // Anchor only region A of the pressure field. Region B should still be
    // detected as an exact nullspace mode and receive gauge elimination.
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(24, 0.0));

    systems::SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_TRUE(reg->isResolved());
    ASSERT_EQ(reg->resolvedModes().size(), 2u)
        << "Expected pressure nullspace expansion per disconnected region";

    int n_anchored = 0;
    int n_exact_nullspace = 0;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) ++n_anchored;
        if (mode.status == GaugeStatus::ExactNullspace) ++n_exact_nullspace;
    }

    EXPECT_EQ(n_anchored, 1);
    EXPECT_EQ(n_exact_nullspace, 1);

    const auto& ac = sys.constraints();
    bool region_b_pressure_constrained = false;
    for (GlobalIndex d = 28; d < 32; ++d) {
        if (ac.isConstrained(d)) {
            region_b_pressure_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(region_b_pressure_constrained)
        << "Region B pressure DOFs should receive automatic gauge elimination";
}

// ============================================================================
// Global anchor without boundary_marker on connected mesh → anchors correctly
// ============================================================================

TEST(GaugeIntegration, GlobalAnchor_NoBoundaryMarker_ConnectedMesh_AnchorsCorrectly)
{
    // Regression: a global Anchored evidence without boundary_marker (e.g.,
    // from the NS module's unlabeled-face detection) should anchor the pressure
    // on a connected mesh.  The resolver allows ev.region=-1 to match
    // candidate.region=-1 (both global).  Without this, a naturally-anchored
    // pressure would get an unnecessary MeanZeroElimination constraint.
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    c.source = CandidateSource::ExplicitDeclaration;
    c.reason = "Pressure nullspace";
    reg.addCandidate(c);

    // Global anchor without boundary_marker (mimics NS unlabeled-face path)
    gauge::AnchoringEvidence ev;
    ev.field = 0;
    ev.family = NullspaceModeFamily::ScalarConstant;
    ev.verdict = AnchoringVerdict::Anchored;
    ev.source = "Velocity natural BC (unlabeled faces)";
    // ev.boundary_marker stays -1 (no marker)
    reg.addAnchoring(ev);

    // No region provider → connected mesh → no region expansion
    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    reg.resolve(dof_provider);

    // Single global candidate should be Anchored
    ASSERT_EQ(reg.resolvedModes().size(), 1u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::Anchored);
    EXPECT_EQ(reg.resolvedModes()[0].policy, EnforcementPolicy::None);

    // No constraints should be created
    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 0);
}

TEST(GaugeIntegration, GlobalAnchor_NoBoundaryMarker_DisconnectedMesh_KnownLimitation)
{
    // Known limitation: on a disconnected mesh, global Anchored evidence
    // without boundary_marker is blocked from per-region candidates.  Both
    // regions get gauge enforcement even if one is genuinely anchored.
    // This can over-constrain the anchored region (MeanZeroElimination may
    // change the pressure level).  Production meshes should label all
    // boundary faces to enable correct per-region scoping via
    // retagEvidenceRegions().
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    gauge::AnchoringEvidence ev;
    ev.field = 0;
    ev.family = NullspaceModeFamily::ScalarConstant;
    ev.verdict = AnchoringVerdict::Anchored;
    ev.source = "Velocity natural BC (unlabeled faces)";
    reg.addAnchoring(ev);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    // Two regions expanded.  Global anchor blocked → both ExactNullspace.
    // This is the known-limitation behavior, not the ideal outcome.
    ASSERT_EQ(reg.resolvedModes().size(), 2u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);

    // Both regions get gauge enforcement (may over-constrain)
    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 2);
}
