/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ThreadedAssembly.cpp
 * @brief Threaded StandardAssembler loops give the serial system bitwise.
 *
 * The threaded path (FE/Docs/ThreadedAssembly.md) computes items on several
 * threads and inserts them in item order. These tests assemble the same
 * system with 1 and with several threads on a structured tetrahedral mesh
 * whose vertices are shared by many cells, so most global entries receive
 * many contributions and any change of summation order would show up, and
 * compare the matrix and vector bit for bit. They also cover the serial
 * continuation after lazy work deferred from a thread, constrained insertion,
 * and the thread pool itself.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyThreadPool.h"
#include "Assembly/ConcurrentCompute.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Constraints/AffineConstraints.h"
#include "Core/DeterministicParallel.h"
#include "Dofs/DofMap.h"
#include "Elements/ReferenceElement.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/MonolithicCellKernel.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

// Assembly threads are used only with one OpenMP thread (with more, the
// serial loops would split cell batches across OpenMP threads), so the tests
// run with one OpenMP thread whatever OMP_NUM_THREADS says.
class SingleOpenMPThreadEnvironment final : public ::testing::Environment {
public:
    void SetUp() override
    {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
    }
};

[[maybe_unused]] ::testing::Environment* const kSingleOpenMPThread =
    ::testing::AddGlobalTestEnvironment(new SingleOpenMPThreadEnvironment);

// Structured tetrahedral mesh of the unit cube: n^3 cubes, six tetrahedra per
// cube (Freudenthal split along the main diagonal, conforming), with
// deterministically perturbed interior vertices so that no two cells are
// congruent.
class StructuredTetMesh final : public IMeshAccess {
public:
    explicit StructuredTetMesh(int n) : n_(n)
    {
        const int np = n + 1;
        const Real h = Real{1.0} / static_cast<Real>(n);
        coords_.resize(static_cast<std::size_t>(np * np * np));
        for (int k = 0; k < np; ++k) {
            for (int j = 0; j < np; ++j) {
                for (int i = 0; i < np; ++i) {
                    std::array<Real, 3> x{h * i, h * j, h * k};
                    const bool interior = i > 0 && j > 0 && k > 0 && i < n && j < n && k < n;
                    if (interior) {
                        x[0] += Real{0.11} * h * std::sin(Real{1.3} * i + Real{0.7} * j + Real{0.3} * k);
                        x[1] += Real{0.09} * h * std::cos(Real{0.4} * i + Real{1.1} * j + Real{0.5} * k);
                        x[2] += Real{0.07} * h * std::sin(Real{0.9} * i + Real{0.2} * j + Real{1.7} * k);
                    }
                    coords_[static_cast<std::size_t>(node(i, j, k))] = x;
                }
            }
        }
        static constexpr std::array<std::array<int, 3>, 6> perms{{
            {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}}, {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
        }};
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (const auto& p : perms) {
                        std::array<int, 3> c{i, j, k};
                        std::array<GlobalIndex, 4> tet{};
                        tet[0] = node(c[0], c[1], c[2]);
                        for (int s = 0; s < 3; ++s) {
                            c[static_cast<std::size_t>(p[static_cast<std::size_t>(s)])] += 1;
                            tet[static_cast<std::size_t>(s + 1)] = node(c[0], c[1], c[2]);
                        }
                        if (signedVolume(tet) < 0.0) {
                            std::swap(tet[1], tet[2]);
                        }
                        cells_.push_back(tet);
                    }
                }
            }
        }
        buildFaces();
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<GlobalIndex>(boundary_faces_.size());
    }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return static_cast<GlobalIndex>(interior_faces_.size());
    }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(c.begin(), c.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return coords_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        coords.clear();
        for (const auto v : c) {
            coords.push_back(coords_[static_cast<std::size_t>(v)]);
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
    {
        const auto& f = faces_.at(static_cast<std::size_t>(face_id));
        for (std::size_t s = 0; s < f.cells.size(); ++s) {
            if (f.cells[s] == cell_id) {
                return f.local[s];
            }
        }
        throw std::invalid_argument("StructuredTetMesh: cell is not adjacent to face");
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return 1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        const auto& f = faces_.at(static_cast<std::size_t>(face_id));
        return {f.cells[0], f.cells.size() > 1 ? f.cells[1] : GlobalIndex{-1}};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        for (const auto f : boundary_faces_) {
            callback(f, faces_[static_cast<std::size_t>(f)].cells[0]);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        for (const auto f : interior_faces_) {
            const auto& face = faces_[static_cast<std::size_t>(f)];
            callback(f, face.cells[0], face.cells[1]);
        }
    }

    [[nodiscard]] GlobalIndex numNodes() const { return static_cast<GlobalIndex>(coords_.size()); }
    [[nodiscard]] const std::array<GlobalIndex, 4>& cell(GlobalIndex c) const
    {
        return cells_[static_cast<std::size_t>(c)];
    }

private:
    struct Face {
        std::vector<GlobalIndex> cells;
        std::vector<LocalIndex> local;
    };

    [[nodiscard]] GlobalIndex node(int i, int j, int k) const
    {
        const int np = n_ + 1;
        return static_cast<GlobalIndex>(i + np * (j + np * k));
    }

    [[nodiscard]] Real signedVolume(const std::array<GlobalIndex, 4>& t) const
    {
        const auto& a = coords_[static_cast<std::size_t>(t[0])];
        std::array<std::array<Real, 3>, 3> e{};
        for (int r = 0; r < 3; ++r) {
            const auto& b = coords_[static_cast<std::size_t>(t[static_cast<std::size_t>(r + 1)])];
            for (int d = 0; d < 3; ++d) {
                e[static_cast<std::size_t>(r)][static_cast<std::size_t>(d)] =
                    b[static_cast<std::size_t>(d)] - a[static_cast<std::size_t>(d)];
            }
        }
        return e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
               e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
               e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    }

    void buildFaces()
    {
        const auto& ref = elements::ReferenceElement::shared(ElementType::Tetra4);
        std::map<std::array<GlobalIndex, 3>, std::size_t> index;
        for (std::size_t c = 0; c < cells_.size(); ++c) {
            for (std::size_t lf = 0; lf < 4u; ++lf) {
                const auto& fn = ref.face_nodes(lf);
                std::array<GlobalIndex, 3> key{
                    cells_[c][static_cast<std::size_t>(fn[0])],
                    cells_[c][static_cast<std::size_t>(fn[1])],
                    cells_[c][static_cast<std::size_t>(fn[2])]};
                std::sort(key.begin(), key.end());
                auto [it, inserted] = index.emplace(key, faces_.size());
                if (inserted) {
                    faces_.emplace_back();
                }
                auto& face = faces_[it->second];
                face.cells.push_back(static_cast<GlobalIndex>(c));
                face.local.push_back(static_cast<LocalIndex>(lf));
            }
        }
        for (std::size_t f = 0; f < faces_.size(); ++f) {
            (faces_[f].cells.size() == 2u ? interior_faces_ : boundary_faces_)
                .push_back(static_cast<GlobalIndex>(f));
        }
    }

    int n_;
    std::vector<std::array<Real, 3>> coords_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<Face> faces_{};
    std::vector<GlobalIndex> interior_faces_{};
    std::vector<GlobalIndex> boundary_faces_{};
};

// DOF map with `components` components per vertex: component c of vertex v is
// global DOF offset + c * n_nodes + v, element DOFs ordered component-major.
dofs::DofMap makeVertexDofMap(const StructuredTetMesh& mesh, int components)
{
    const auto n_nodes = mesh.numNodes();
    const auto per_cell = static_cast<LocalIndex>(4 * components);
    dofs::DofMap dof_map(mesh.numCells(), n_nodes * components, per_cell);
    std::vector<GlobalIndex> dofs(static_cast<std::size_t>(per_cell));
    for (GlobalIndex c = 0; c < mesh.numCells(); ++c) {
        const auto& t = mesh.cell(c);
        for (int comp = 0; comp < components; ++comp) {
            for (int v = 0; v < 4; ++v) {
                dofs[static_cast<std::size_t>(comp * 4 + v)] =
                    static_cast<GlobalIndex>(comp) * n_nodes + t[static_cast<std::size_t>(v)];
            }
        }
        dof_map.setCellDofs(c, dofs);
    }
    dof_map.setNumDofs(n_nodes * components);
    dof_map.setNumLocalDofs(n_nodes * components);
    dof_map.finalize();
    return dof_map;
}

// One partial cut-volume rule per cell, with points and weights that vary
// from cell to cell.
CutIntegrationContext makeCutContext(const StructuredTetMesh& mesh, int marker)
{
    CutIntegrationContext context;
    for (GlobalIndex c = 0; c < mesh.numCells(); ++c) {
        const Real s = Real{0.01} * static_cast<Real>(c % 7);
        const std::array<std::array<Real, 3>, 5> points{{
            {{Real{0.10} + s, Real{0.20}, Real{0.15}}},
            {{Real{0.30}, Real{0.10} + s, Real{0.20}}},
            {{Real{0.15}, Real{0.35}, Real{0.05} + s}},
            {{Real{0.20}, Real{0.20}, Real{0.30}}},
            {{Real{0.05} + s, Real{0.10}, Real{0.10}}},
        }};
        geometry::CutQuadratureRule rule;
        rule.kind = geometry::CutQuadratureKind::Volume;
        rule.side = geometry::CutIntegrationSide::Negative;
        for (std::size_t i = 0; i < points.size(); ++i) {
            geometry::CutQuadraturePoint qp;
            qp.point = points[i];
            qp.parent_coordinate = qp.point;
            qp.normal = {{Real{0.0}, Real{0.6}, Real{0.8}}};
            qp.weight = Real{0.01} + Real{0.002} * static_cast<Real>(i) +
                        Real{0.0001} * static_cast<Real>(c % 11);
            rule.measure += qp.weight;
            rule.points.push_back(qp);
        }
        rule.parent_measure = Real{1.0} / Real{6.0};
        rule.volume_fraction = rule.measure / rule.parent_measure;
        rule.exact_polynomial_order = 2;
        rule.provenance.parent_entity = static_cast<MeshIndex>(c);
        rule.provenance.marker = marker;
        rule.provenance.cut_topology_revision = 71u;
        rule.provenance.predicate_policy_key = 59u;
        rule.provenance.source_value_revision = 61u;

        CutCellAssemblyMetadata metadata;
        metadata.cell = static_cast<MeshIndex>(c);
        metadata.parent_entity = static_cast<MeshIndex>(c);
        metadata.volume_fraction = rule.volume_fraction;
        metadata.side = rule.side;
        metadata.provenance_id = "unit-threaded-assembly";
        metadata.cut_topology_id = "unit-threaded-assembly";
        metadata.revision_key = rule.provenance.cut_topology_revision;
        metadata.cut_topology_revision = rule.provenance.cut_topology_revision;
        metadata.quadrature_policy_key = rule.provenance.predicate_policy_key;
        metadata.source_value_revision = rule.provenance.source_value_revision;
        context.addGeneratedVolumeRule(marker, std::move(metadata), std::move(rule));
    }
    return context;
}

// Uses values, gradients and Hessians of test and trial bases, the solution
// coefficients and entity measures; `weight` makes the four blocks differ.
// With `defer_cell` >= 0 the kernel asks for serial work on that cell when it
// runs on an assembly thread (as a JIT compile would).
class ProbeKernel final : public AssemblyKernel {
public:
    explicit ProbeKernel(Real weight, GlobalIndex defer_cell = -1, GlobalIndex throw_cell = -1)
        : weight_(weight), defer_cell_(defer_cell), throw_cell_(throw_cell)
    {
    }

    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override
    {
        if (defer_cell_ >= 0 && ctx.cellId() == defer_cell_) {
            deferral_checks_.fetch_add(1, std::memory_order_relaxed);
            requireSerial("unit-test deferred cell");
        }
        if (defer_cell_ >= 0 && concurrentComputeActive() && ctx.cellId() > defer_cell_ + 128) {
            threaded_after_deferral_.fetch_add(1, std::memory_order_relaxed);
        }
        if (throw_cell_ >= 0 && ctx.cellId() == throw_cell_) {
            throw std::runtime_error("unit-test kernel error");
        }
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        const auto n_qpts = ctx.numQuadraturePoints();
        output.reserve(n_test, n_trial, /*need_matrix=*/true, /*need_vector=*/true);
        const Real h = ctx.cellDiameter();
        const auto coeffs = ctx.solutionCoefficients();
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q) * weight_;
            Real u = 0.0;
            for (LocalIndex j = 0; j < n_trial && static_cast<std::size_t>(j) < coeffs.size(); ++j) {
                u += coeffs[static_cast<std::size_t>(j)] * ctx.trialBasisValue(j, q);
            }
            for (LocalIndex i = 0; i < n_test; ++i) {
                const Real phi = ctx.basisValue(i, q);
                const auto g = ctx.physicalGradient(i, q);
                const auto H = ctx.physicalHessian(i, q);
                output.local_vector[static_cast<std::size_t>(i)] +=
                    w * (phi * (Real{1.0} + u * u + H[0][0]) +
                         Real{0.2} * (g[0] - g[1] + g[2]) + Real{0.01} * h * phi);
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    const Real psi = ctx.trialBasisValue(j, q);
                    const auto tg = ctx.trialPhysicalGradient(j, q);
                    Real gg = 0.0;
                    for (std::size_t r = 0; r < 3u; ++r) {
                        gg += g[r] * tg[r];
                    }
                    output.local_matrix[static_cast<std::size_t>(i * n_trial + j)] +=
                        w * (phi * psi * (Real{1.0} + u) + Real{0.3} * gg);
                }
            }
        }
    }

    [[nodiscard]] RequiredData getRequiredData() const override
    {
        return RequiredData::BasisValues | RequiredData::PhysicalGradients |
               RequiredData::BasisHessians | RequiredData::IntegrationWeights |
               RequiredData::EntityMeasures | RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] int deferralChecks() const { return deferral_checks_.load(); }
    /// Cells well after defer_cell computed on an assembly thread.
    [[nodiscard]] int threadedAfterDeferral() const { return threaded_after_deferral_.load(); }

private:
    Real weight_;
    GlobalIndex defer_cell_;
    GlobalIndex throw_cell_;
    std::atomic<int> deferral_checks_{0};
    std::atomic<int> threaded_after_deferral_{0};
};

// Interior-face kernel coupling both sides (all four blocks and both vectors).
class FaceProbeKernel final : public AssemblyKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override
    {
        return RequiredData::BasisValues | RequiredData::IntegrationWeights |
               RequiredData::Normals | RequiredData::PhysicalPoints |
               RequiredData::SolutionCoefficients;
    }
    void computeCell(const AssemblyContext&, KernelOutput&) override {}
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }

    void computeInteriorFace(const AssemblyContext& m,
                             const AssemblyContext& p,
                             KernelOutput& out_m,
                             KernelOutput& out_p,
                             KernelOutput& mp,
                             KernelOutput& pm) override
    {
        const auto nm = m.numTestDofs();
        const auto np = p.numTestDofs();
        out_m.reserve(nm, nm, true, true);
        out_p.reserve(np, np, true, true);
        mp.reserve(nm, np, true, false);
        pm.reserve(np, nm, true, false);
        const auto cm = m.solutionCoefficients();
        const auto cp = p.solutionCoefficients();
        for (LocalIndex q = 0; q < m.numQuadraturePoints(); ++q) {
            const Real w = m.integrationWeight(q);
            const auto n = m.normal(q);
            const auto x = m.physicalPoint(q);
            Real um = 0.0;
            Real up = 0.0;
            for (LocalIndex j = 0; j < nm && static_cast<std::size_t>(j) < cm.size(); ++j) {
                um += cm[static_cast<std::size_t>(j)] * m.basisValue(j, q);
            }
            for (LocalIndex j = 0; j < np && static_cast<std::size_t>(j) < cp.size(); ++j) {
                up += cp[static_cast<std::size_t>(j)] * p.basisValue(j, q);
            }
            const Real jump = um - up;
            const Real scale = w * (Real{1.0} + Real{0.1} * (n[0] + Real{2.0} * n[1]) + x[2]);
            for (LocalIndex i = 0; i < nm; ++i) {
                const Real phi = m.basisValue(i, q);
                out_m.local_vector[static_cast<std::size_t>(i)] += scale * jump * phi;
                for (LocalIndex j = 0; j < nm; ++j) {
                    out_m.local_matrix[static_cast<std::size_t>(i * nm + j)] +=
                        scale * phi * m.basisValue(j, q);
                }
                for (LocalIndex j = 0; j < np; ++j) {
                    mp.local_matrix[static_cast<std::size_t>(i * np + j)] -=
                        scale * phi * p.basisValue(j, q);
                }
            }
            for (LocalIndex i = 0; i < np; ++i) {
                const Real phi = p.basisValue(i, q);
                out_p.local_vector[static_cast<std::size_t>(i)] -= scale * jump * phi;
                for (LocalIndex j = 0; j < np; ++j) {
                    out_p.local_matrix[static_cast<std::size_t>(i * np + j)] +=
                        scale * phi * p.basisValue(j, q);
                }
                for (LocalIndex j = 0; j < nm; ++j) {
                    pm.local_matrix[static_cast<std::size_t>(i * nm + j)] -=
                        scale * phi * m.basisValue(j, q);
                }
            }
        }
    }
};

std::vector<Real> makeSolution(GlobalIndex n)
{
    std::vector<Real> u(static_cast<std::size_t>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        u[static_cast<std::size_t>(i)] = Real{0.3} * std::sin(Real{0.37} * static_cast<Real>(i)) + Real{0.1};
    }
    return u;
}

AssemblyOptions threadOptions(int threads)
{
    AssemblyOptions options;
    options.num_threads = threads;
    return options;
}

void expectBitwiseEqual(const DenseSystemView& actual,
                        const DenseSystemView& expected,
                        const std::string& label)
{
    const auto am = actual.matrixData();
    const auto em = expected.matrixData();
    const auto av = actual.vectorData();
    const auto ev = expected.vectorData();
    ASSERT_EQ(am.size(), em.size()) << label;
    ASSERT_EQ(av.size(), ev.size()) << label;
    std::size_t matrix_differences = 0;
    std::size_t vector_differences = 0;
    bool any_nonzero = false;
    for (std::size_t i = 0; i < am.size(); ++i) {
        matrix_differences += std::memcmp(&am[i], &em[i], sizeof(Real)) != 0 ? 1u : 0u;
        any_nonzero = any_nonzero || am[i] != Real{0.0};
    }
    for (std::size_t i = 0; i < av.size(); ++i) {
        vector_differences += std::memcmp(&av[i], &ev[i], sizeof(Real)) != 0 ? 1u : 0u;
    }
    EXPECT_EQ(matrix_differences, 0u) << label;
    EXPECT_EQ(vector_differences, 0u) << label;
    EXPECT_TRUE(any_nonzero) << label;
}

struct FusedSetup {
    StructuredTetMesh mesh{5};
    std::shared_ptr<spaces::H1Space> scalar{std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1)};
    spaces::ProductSpace velocity{scalar, 3};
    spaces::H1Space pressure{ElementType::Tetra4, 1};
    dofs::DofMap u_map{makeVertexDofMap(mesh, 3)};
    dofs::DofMap p_map{makeVertexDofMap(mesh, 1)};
    GlobalIndex n_u{u_map.getNumDofs()};
    GlobalIndex n_total{u_map.getNumDofs() + p_map.getNumDofs()};
    std::vector<Real> solution{makeSolution(n_total)};
    static constexpr int marker = 417;
    CutIntegrationContext cut_context{makeCutContext(mesh, marker)};
};

std::vector<FusedCellTerm> makeFusedTerms(FusedSetup& s,
                                          std::array<ProbeKernel*, 4> kernels,
                                          DenseSystemView& system)
{
    const auto term = [&](const spaces::FunctionSpace& test,
                          const spaces::FunctionSpace& trial,
                          ProbeKernel* kernel,
                          const dofs::DofMap& row_map,
                          GlobalIndex row_offset,
                          const dofs::DofMap& col_map,
                          GlobalIndex col_offset) {
        FusedCellTerm t;
        t.test_space = &test;
        t.trial_space = &trial;
        t.kernel = kernel;
        t.row_dof_map = &row_map;
        t.col_dof_map = &col_map;
        t.row_dof_offset = row_offset;
        t.col_dof_offset = col_offset;
        t.matrix_view = &system;
        t.vector_view = &system;
        t.assemble_matrix = true;
        t.assemble_vector = true;
        return t;
    };
    std::vector<FusedCellTerm> terms;
    terms.push_back(term(s.velocity, s.velocity, kernels[0], s.u_map, 0, s.u_map, 0));
    terms.push_back(term(s.velocity, s.pressure, kernels[1], s.u_map, 0, s.p_map, s.n_u));
    terms.push_back(term(s.pressure, s.velocity, kernels[2], s.p_map, s.n_u, s.u_map, 0));
    terms.push_back(term(s.pressure, s.pressure, kernels[3], s.p_map, s.n_u, s.p_map, s.n_u));
    return terms;
}

// Assembles the fused cut-volume system twice (second call: warm caches).
void assembleFused(FusedSetup& s, int threads, std::array<ProbeKernel*, 4> kernels,
                   DenseSystemView& first, DenseSystemView& second,
                   const constraints::AffineConstraints* constraints = nullptr)
{
    StandardAssembler assembler(threadOptions(threads));
    assembler.setDofMap(s.u_map);
    if (constraints != nullptr) {
        assembler.setConstraints(constraints);
    }
    assembler.setCurrentSolution(s.solution);
    auto first_terms = makeFusedTerms(s, kernels, first);
    auto result = assembler.assembleCutVolumesFused(
        s.mesh, s.cut_context, FusedSetup::marker, geometry::CutIntegrationSide::Negative, first_terms);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, s.mesh.numCells());
    auto second_terms = makeFusedTerms(s, kernels, second);
    result = assembler.assembleCutVolumesFused(
        s.mesh, s.cut_context, FusedSetup::marker, geometry::CutIntegrationSide::Negative, second_terms);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, s.mesh.numCells());
}

} // namespace

TEST(AssemblyThreadPool, RunsEveryParticipantOnceAndRethrows)
{
    auto& pool = AssemblyThreadPool::global();
    for (int n : {1, 2, 3, 5, 8}) {
        std::vector<int> hits(static_cast<std::size_t>(n), 0);
        pool.run(n, [&](int p) { hits[static_cast<std::size_t>(p)] += 1; });
        for (int p = 0; p < n; ++p) {
            EXPECT_EQ(hits[static_cast<std::size_t>(p)], 1) << "n=" << n << " p=" << p;
        }
    }
    EXPECT_THROW(pool.run(4, [&](int p) {
                     if (p == 2) {
                         throw std::runtime_error("participant 2");
                     }
                 }),
                 std::runtime_error);
    // A nested run executes its participants serially on the calling thread.
    std::atomic<int> nested{0};
    pool.run(3, [&](int) {
        EXPECT_TRUE(AssemblyThreadPool::insideParallelRegion());
        pool.run(2, [&](int) { nested.fetch_add(1); });
    });
    EXPECT_EQ(nested.load(), 6);
    EXPECT_FALSE(AssemblyThreadPool::insideParallelRegion());
}

TEST(AssemblyThreadPool, SharesItsWorkersWithTheGeometryLoops)
{
    // The assembly pool and deterministicParallelFor use one team: workers
    // started by one are reused by the other, and a geometry loop inside an
    // assembly participant (or the reverse) runs serially.
    auto& pool = AssemblyThreadPool::global();
    ASSERT_TRUE(pool.reserveWorkers(3));
    EXPECT_GE(parallelTeamWorkerCount(), 3);
    EXPECT_EQ(pool.workerCount(), parallelTeamWorkerCount());
    const int before = parallelTeamWorkerCount();
    std::vector<int> owner(256, -1);
    deterministicParallelFor(
        owner.size(), 4,
        [&](std::size_t item, int participant) {
            EXPECT_TRUE(AssemblyThreadPool::insideParallelRegion());
            owner[item] = participant;
        },
        16u, 64u);
    EXPECT_EQ(parallelTeamWorkerCount(), before);
    for (std::size_t item = 0; item < owner.size(); ++item) {
        EXPECT_EQ(owner[item], static_cast<int>((item / 16u) % 4u)) << item;
    }
    std::atomic<int> nested_parallel{0};
    pool.run(3, [&](int) {
        deterministicParallelFor(
            128u, 4,
            [&](std::size_t, int participant) {
                if (participant != 0) {
                    nested_parallel.fetch_add(1);
                }
            },
            16u, 64u);
    });
    EXPECT_EQ(nested_parallel.load(), 0);
}

TEST(ConcurrentCompute, RequireSerialThrowsOnlyInsideScope)
{
    EXPECT_FALSE(concurrentComputeActive());
    EXPECT_NO_THROW(requireSerial("outside"));
    {
        ConcurrentComputeScope scope;
        EXPECT_TRUE(concurrentComputeActive());
        EXPECT_THROW(requireSerial("inside"), DeferredSerialWork);
    }
    EXPECT_FALSE(concurrentComputeActive());
}

TEST(ThreadedAssembly, FusedCutVolumesBitwiseEqualForAnyThreadCount)
{
    FusedSetup s;
    ProbeKernel uu(Real{1.0}), up(Real{0.7}), pu(Real{-0.4}), pp(Real{1.3});
    const std::array<ProbeKernel*, 4> kernels{&uu, &up, &pu, &pp};
    DenseSystemView ref_first(s.n_total), ref_second(s.n_total);
    assembleFused(s, 1, kernels, ref_first, ref_second);
    expectBitwiseEqual(ref_second, ref_first, "serial warm vs cold");
    for (int threads : {2, 3, 4, 8}) {
        DenseSystemView first(s.n_total), second(s.n_total);
        assembleFused(s, threads, kernels, first, second);
        expectBitwiseEqual(first, ref_first, "threads=" + std::to_string(threads) + " cold");
        expectBitwiseEqual(second, ref_first, "threads=" + std::to_string(threads) + " warm");
    }
}

TEST(ThreadedAssembly, ReversedItemOrderWouldChangeBits)
{
    // Guards the sensitivity of the bitwise tests: the mesh shares vertices
    // between many cells, so summing the same contributions in another order
    // changes some entries.
    FusedSetup s;
    ProbeKernel kernel(Real{1.0});
    StandardAssembler assembler;
    assembler.setDofMap(s.p_map);
    assembler.setCurrentSolution(s.solution);
    DenseSystemView forward(s.p_map.getNumDofs());
    auto result = assembler.assembleCutVolumes(
        s.mesh, s.cut_context, FusedSetup::marker, geometry::CutIntegrationSide::Negative,
        s.pressure, s.pressure, kernel, &forward, &forward, true, true);
    ASSERT_TRUE(result.success);

    // Same per-cell contributions inserted in reverse cell order.
    DenseSystemView reverse(s.p_map.getNumDofs());
    for (GlobalIndex c = s.mesh.numCells() - 1; c >= 0; --c) {
        DenseSystemView single(s.p_map.getNumDofs());
        CutIntegrationContext one;
        const auto& rule = s.cut_context.volumeRules()[static_cast<std::size_t>(c)];
        const auto& meta = s.cut_context.metadata()[static_cast<std::size_t>(c)];
        auto rule_copy = rule;
        auto meta_copy = meta;
        one.addGeneratedVolumeRule(FusedSetup::marker, std::move(meta_copy), std::move(rule_copy));
        StandardAssembler a;
        a.setDofMap(s.p_map);
        a.setCurrentSolution(s.solution);
        (void)a.assembleCutVolumes(s.mesh, one, FusedSetup::marker,
                                   geometry::CutIntegrationSide::Negative, s.pressure,
                                   s.pressure, kernel, &single, &single, true, true);
        const auto sm = single.matrixData();
        const auto sv = single.vectorData();
        const auto dofs = s.p_map.getCellDofs(c);
        for (const auto r : dofs) {
            reverse.addVectorEntry(r, sv[static_cast<std::size_t>(r)]);
            for (const auto col : dofs) {
                reverse.addMatrixEntry(
                    r, col,
                    sm[static_cast<std::size_t>(r) * static_cast<std::size_t>(s.p_map.getNumDofs()) +
                       static_cast<std::size_t>(col)]);
            }
        }
    }
    std::size_t differences = 0;
    const auto fm = forward.matrixData();
    const auto rm = reverse.matrixData();
    for (std::size_t i = 0; i < fm.size(); ++i) {
        differences += std::memcmp(&fm[i], &rm[i], sizeof(Real)) != 0 ? 1u : 0u;
    }
    EXPECT_GT(differences, 0u);
}

TEST(ThreadedAssembly, DeferredLazyWorkContinuesSeriallyWithSameBits)
{
    FusedSetup s;
    ProbeKernel uu(Real{1.0}), up(Real{0.7}), pp(Real{1.3});
    // Defer on a cell in a late block, so earlier blocks are inserted by the
    // threaded path and the rest by the serial continuation.
    const GlobalIndex defer_cell = s.mesh.numCells() - 40;
    ProbeKernel pu_deferring(Real{-0.4}, defer_cell);
    ProbeKernel pu(Real{-0.4});
    DenseSystemView ref_first(s.n_total), ref_second(s.n_total);
    assembleFused(s, 1, {&uu, &up, &pu, &pp}, ref_first, ref_second);
    EXPECT_EQ(pu_deferring.deferralChecks(), 0);
    DenseSystemView first(s.n_total), second(s.n_total);
    assembleFused(s, 4, {&uu, &up, &pu_deferring, &pp}, first, second);
    EXPECT_GT(pu_deferring.deferralChecks(), 0);
    expectBitwiseEqual(first, ref_first, "deferred cold");
    expectBitwiseEqual(second, ref_first, "deferred warm");
}

TEST(ThreadedAssembly, ThreadsResumeAfterADeferredBlock)
{
    // Lazy work on an early cell: its block runs serially, then the threads
    // resume, so cells well after it are computed on threads again.
    FusedSetup s;
    ProbeKernel uu(Real{1.0}), up(Real{0.7}), pp(Real{1.3});
    const GlobalIndex defer_cell = 40;
    ProbeKernel pu_deferring(Real{-0.4}, defer_cell);
    ProbeKernel pu(Real{-0.4});
    DenseSystemView ref_first(s.n_total), ref_second(s.n_total);
    assembleFused(s, 1, {&uu, &up, &pu, &pp}, ref_first, ref_second);
    DenseSystemView first(s.n_total), second(s.n_total);
    assembleFused(s, 4, {&uu, &up, &pu_deferring, &pp}, first, second);
    EXPECT_GT(pu_deferring.deferralChecks(), 0);
    EXPECT_GT(pu_deferring.threadedAfterDeferral(), 0);
    expectBitwiseEqual(first, ref_first, "resumed cold");
    expectBitwiseEqual(second, ref_first, "resumed warm");
}

TEST(ThreadedAssembly, KernelErrorIsRaisedByTheSerialContinuation)
{
    // An error on a thread stops the threaded part; the serial loop then
    // meets the same error at the same cell and raises it.
    FusedSetup s;
    ProbeKernel uu(Real{1.0}), up(Real{0.7}), pp(Real{1.3});
    ProbeKernel pu_throwing(Real{-0.4}, /*defer_cell=*/-1, /*throw_cell=*/s.mesh.numCells() / 2);
    for (int threads : {1, 4}) {
        StandardAssembler assembler(threadOptions(threads));
        assembler.setDofMap(s.u_map);
        assembler.setCurrentSolution(s.solution);
        DenseSystemView system(s.n_total);
        auto terms = makeFusedTerms(s, {&uu, &up, &pu_throwing, &pp}, system);
        EXPECT_THROW((void)assembler.assembleCutVolumesFused(
                         s.mesh, s.cut_context, FusedSetup::marker,
                         geometry::CutIntegrationSide::Negative, terms),
                     std::runtime_error)
            << "threads=" << threads;
    }
}

TEST(ThreadedAssembly, RepeatedAssembliesStayBitwiseEqual)
{
    // Many calls on one assembler: worker caches warm up and the insertion
    // ring wraps around many times.
    FusedSetup s;
    ProbeKernel uu(Real{1.0}), up(Real{0.7}), pu(Real{-0.4}), pp(Real{1.3});
    const std::array<ProbeKernel*, 4> kernels{&uu, &up, &pu, &pp};
    DenseSystemView reference(s.n_total), unused(s.n_total);
    assembleFused(s, 1, kernels, reference, unused);
    StandardAssembler assembler(threadOptions(3));
    assembler.setDofMap(s.u_map);
    assembler.setCurrentSolution(s.solution);
    for (int repeat = 0; repeat < 6; ++repeat) {
        DenseSystemView system(s.n_total);
        auto terms = makeFusedTerms(s, kernels, system);
        auto result = assembler.assembleCutVolumesFused(
            s.mesh, s.cut_context, FusedSetup::marker, geometry::CutIntegrationSide::Negative, terms);
        ASSERT_TRUE(result.success);
        expectBitwiseEqual(system, reference, "repeat " + std::to_string(repeat));
    }
}

TEST(ThreadedAssembly, ConstrainedInsertionBitwiseEqual)
{
    FusedSetup s;
    // Dirichlet values on the first vertices of every field and an affine
    // tie of one pressure DOF to two others: constrained cells go through
    // insertLocalConstrained during replay.
    constraints::AffineConstraints constraints;
    for (GlobalIndex d = 0; d < 6; ++d) {
        constraints.addDirichlet(d, Real{0.1} * static_cast<Real>(d + 1));
    }
    const GlobalIndex tied = s.n_u + 20;
    constraints.addLine(tied);
    constraints.addEntry(tied, s.n_u + 21, Real{0.5});
    constraints.addEntry(tied, s.n_u + 22, Real{0.5});
    constraints.close();

    ProbeKernel uu(Real{1.0}), up(Real{0.7}), pu(Real{-0.4}), pp(Real{1.3});
    const std::array<ProbeKernel*, 4> kernels{&uu, &up, &pu, &pp};
    DenseSystemView ref_first(s.n_total), ref_second(s.n_total);
    assembleFused(s, 1, kernels, ref_first, ref_second, &constraints);
    for (int threads : {2, 4}) {
        DenseSystemView first(s.n_total), second(s.n_total);
        assembleFused(s, threads, kernels, first, second, &constraints);
        expectBitwiseEqual(first, ref_first, "constrained threads=" + std::to_string(threads));
        expectBitwiseEqual(second, ref_first, "constrained warm threads=" + std::to_string(threads));
    }
}

TEST(ThreadedAssembly, InteriorFacesBitwiseEqualForAnyThreadCount)
{
    StructuredTetMesh mesh(5);
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = makeVertexDofMap(mesh, 1);
    const auto n = dof_map.getNumDofs();
    const auto solution = makeSolution(n);
    FaceProbeKernel kernel;
    const auto assemble = [&](int threads, DenseSystemView& system) {
        StandardAssembler assembler(threadOptions(threads));
        assembler.setDofMap(dof_map);
        assembler.setCurrentSolution(solution);
        auto result = assembler.assembleInteriorFaces(mesh, space, space, kernel, system, &system);
        ASSERT_TRUE(result.success);
        EXPECT_EQ(result.interior_faces_assembled, mesh.numInteriorFaces());
    };
    DenseSystemView reference(n);
    assemble(1, reference);
    for (int threads : {2, 3, 4, 8}) {
        DenseSystemView system(n);
        assemble(threads, system);
        expectBitwiseEqual(system, reference, "interior faces threads=" + std::to_string(threads));
    }
}

TEST(ThreadedAssembly, MonolithicCellLoopsBitwiseEqualForAnyThreadCount)
{
    // Two scalar P1 fields coupled by a monolithic cell kernel of three
    // blocks: the batched path (matrix and vector) and the residual-only path
    // of assembleCellsFused. Constraints keep the per-block insertion, which
    // is the path the threads take.
    StructuredTetMesh mesh(5);
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto field_map = makeVertexDofMap(mesh, 1);
    const auto n = field_map.getNumDofs();
    const auto solution = makeSolution(2 * n);
    constraints::AffineConstraints constraints;
    for (GlobalIndex d = 0; d < 6; ++d) {
        constraints.addDirichlet(d, Real{0.2});
        constraints.addDirichlet(n + d, Real{-0.1});
    }
    constraints.close();

    auto k00 = std::make_shared<ProbeKernel>(Real{1.0});
    auto k11 = std::make_shared<ProbeKernel>(Real{0.6});
    auto k01 = std::make_shared<ProbeKernel>(Real{-0.3});
    const auto block = [&](std::shared_ptr<AssemblyKernel> kernel, int test_field, int trial_field) {
        return forms::MonolithicCellKernel::BlockSpec{
            .test_field = static_cast<FieldId>(test_field),
            .trial_field = static_cast<FieldId>(trial_field),
            .want_matrix = true,
            .want_vector = true,
            .fallback_kernel = std::move(kernel),
            .test_space = &space,
            .trial_space = &space,
            .row_dof_map = &field_map,
            .col_dof_map = &field_map,
            .row_dof_offset = test_field == 0 ? GlobalIndex{0} : n,
            .col_dof_offset = trial_field == 0 ? GlobalIndex{0} : n,
        };
    };
    const auto assemble = [&](int threads, bool want_matrix, DenseSystemView& system) {
        std::vector<forms::MonolithicCellKernel::BlockSpec> blocks;
        blocks.push_back(block(k00, 0, 0));
        blocks.push_back(block(k11, 1, 1));
        blocks.push_back(block(k01, 0, 1));
        forms::MonolithicCellKernel kernel(
            std::move(blocks), std::shared_ptr<forms::jit::JITCompiler>{}, forms::JITOptions{});
        kernel.setResolved();
        StandardAssembler assembler(threadOptions(threads));
        assembler.setDofMap(field_map);
        assembler.setConstraints(&constraints);
        assembler.setCurrentSolution(solution);
        // The first call warms the worker caches; the second one is compared.
        DenseSystemView warm_up(system.numRows());
        for (auto* target : {&warm_up, &system}) {
            const FusedCellTerm term{
                .test_space = &space,
                .trial_space = &space,
                .kernel = &kernel,
                .row_dof_map = &field_map,
                .col_dof_map = &field_map,
                .row_dof_offset = 0,
                .col_dof_offset = 0,
                .matrix_view = want_matrix ? target : nullptr,
                .vector_view = target,
                .assemble_matrix = want_matrix,
                .assemble_vector = true,
            };
            auto result = assembler.assembleCellsFused(mesh, std::span<const FusedCellTerm>(&term, 1));
            ASSERT_TRUE(result.success);
        }
    };
    for (bool want_matrix : {true, false}) {
        DenseSystemView reference(2 * n);
        assemble(1, want_matrix, reference);
        for (int threads : {2, 3, 8}) {
            DenseSystemView system(2 * n);
            assemble(threads, want_matrix, system);
            const auto label = std::string(want_matrix ? "matrix+vector" : "vector only") +
                               " threads=" + std::to_string(threads);
            if (want_matrix) {
                expectBitwiseEqual(system, reference, label);
            } else {
                const auto v = system.vectorData();
                const auto rv = reference.vectorData();
                std::size_t differences = 0;
                for (std::size_t i = 0; i < v.size(); ++i) {
                    differences += std::memcmp(&v[i], &rv[i], sizeof(Real)) != 0 ? 1u : 0u;
                }
                EXPECT_EQ(differences, 0u) << label;
            }
        }
    }
}

#if defined(SVMP_FE_ENABLE_LLVM_JIT) && SVMP_FE_ENABLE_LLVM_JIT
TEST(ThreadedAssembly, JITKernelDefersCompilesAndMatchesSerialBitwise)
{
    // A JIT-compiled form kernel shared by the threads: the first threaded
    // call meets the compile on a thread, defers it and continues serially
    // (compiles happen in the serial order); later calls run threaded.
    FusedSetup s;
    const auto make_kernel = [&]() {
        forms::SymbolicOptions sym_opts;
        sym_opts.jit.enable = true;
        forms::FormCompiler compiler(sym_opts);
        const auto u = forms::FormExpr::trialFunction(s.pressure, "u");
        const auto v = forms::FormExpr::testFunction(s.pressure, "v");
        const auto form = (u * v + forms::inner(forms::grad(u), forms::grad(v))).dx();
        auto fallback = std::make_shared<forms::FormKernel>(compiler.compileBilinear(form));
        forms::JITOptions jit_opts;
        jit_opts.enable = true;
        return std::make_unique<forms::jit::JITKernelWrapper>(fallback, jit_opts);
    };
    const auto assemble = [&](int threads, forms::jit::JITKernelWrapper& kernel,
                              DenseSystemView& system) {
        StandardAssembler assembler(threadOptions(threads));
        assembler.setDofMap(s.p_map);
        auto result = assembler.assembleCutVolumes(
            s.mesh, s.cut_context, FusedSetup::marker, geometry::CutIntegrationSide::Negative,
            s.pressure, s.pressure, kernel, &system, nullptr, true, false);
        ASSERT_TRUE(result.success);
    };
    const auto n = s.p_map.getNumDofs();
    auto serial_kernel = make_kernel();
    DenseSystemView reference(n);
    assemble(1, *serial_kernel, reference);

    auto threaded_kernel = make_kernel();
    DenseSystemView first(n), second(n);
    assemble(4, *threaded_kernel, first);
    assemble(4, *threaded_kernel, second);
    const auto rm = reference.matrixData();
    for (auto* system : {&first, &second}) {
        const auto m = system->matrixData();
        ASSERT_EQ(m.size(), rm.size());
        std::size_t differences = 0;
        for (std::size_t i = 0; i < m.size(); ++i) {
            differences += std::memcmp(&m[i], &rm[i], sizeof(Real)) != 0 ? 1u : 0u;
        }
        EXPECT_EQ(differences, 0u);
    }
}
#endif

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
