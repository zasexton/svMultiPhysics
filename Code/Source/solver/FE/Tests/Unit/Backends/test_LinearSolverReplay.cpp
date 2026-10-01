/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Replays FSILS system snapshots written with SVMP_FSILS_DUMP_SYSTEM_PREFIX
// through the available backend solvers and prints one line per solve:
// iterations, wall time, true residual and distance to the recorded solution.
//
// Skipped unless SVMP_LINEAR_REPLAY_FILES lists snapshot files (separated by
// commas, spaces or newlines) in solve order.  SVMP_LINEAR_REPLAY_CONFIGS
// selects configurations by name (comma separated); the default runs all.
// One solver object per configuration persists over the whole sequence, so
// configurations with reuse see the same solve history as the application.

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsShared.h"
#include "Backends/FSILS/FsilsSystemDump.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/Utils/BackendOptions.h"
#include "Sparsity/SparsityPattern.h"

#if defined(FE_HAS_EIGEN)
#include "Backends/Eigen/EigenFactory.h"
#include "Backends/Eigen/EigenMatrix.h"
#include "Backends/Eigen/EigenVector.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace svmp::FE::backends {
namespace {

std::vector<std::string> splitList(const char* text)
{
    std::vector<std::string> out;
    if (text == nullptr) {
        return out;
    }
    std::string cur;
    for (const char* p = text; *p != '\0'; ++p) {
        const char ch = *p;
        if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t' || ch == ';') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

sparsity::SparsityPattern scalarPattern(const FsilsSystemSnapshot& s)
{
    const GlobalIndex n = static_cast<GlobalIndex>(s.n_nodes) * s.dof;
    sparsity::SparsityPattern pattern(n, n);
    std::vector<GlobalIndex> row_cols;
    for (int node = 0; node < s.n_nodes; ++node) {
        row_cols.clear();
        for (auto p = s.row_ptr[static_cast<std::size_t>(node)];
             p < s.row_ptr[static_cast<std::size_t>(node) + 1u]; ++p) {
            const GlobalIndex c = s.cols[static_cast<std::size_t>(p)];
            for (int k = 0; k < s.dof; ++k) {
                row_cols.push_back(c * s.dof + k);
            }
        }
        for (int r = 0; r < s.dof; ++r) {
            pattern.addEntries(static_cast<GlobalIndex>(node) * s.dof + r, row_cols);
        }
    }
    pattern.finalize();
    return pattern;
}

/// y = A x on the snapshot (backend numbering).
std::vector<double> snapshotMatVec(const FsilsSystemSnapshot& s, const std::vector<double>& x)
{
    const int d = s.dof;
    std::vector<double> y(x.size(), 0.0);
    for (int i = 0; i < s.n_nodes; ++i) {
        for (auto p = s.row_ptr[static_cast<std::size_t>(i)]; p < s.row_ptr[static_cast<std::size_t>(i) + 1u];
             ++p) {
            const int j = s.cols[static_cast<std::size_t>(p)];
            const double* blk = s.values.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(d * d);
            for (int r = 0; r < d; ++r) {
                double acc = 0.0;
                for (int c = 0; c < d; ++c) {
                    acc += blk[r * d + c] * x[static_cast<std::size_t>(j) * d + c];
                }
                y[static_cast<std::size_t>(i) * d + r] += acc;
            }
        }
    }
    return y;
}

struct ResidualStats {
    double rel_all{0.0};
    double rel_free{0.0};
    double err_vs_recorded{0.0};
};

ResidualStats residualStats(const FsilsSystemSnapshot& s, const std::vector<double>& x)
{
    const auto ax = snapshotMatVec(s, x);
    std::vector<char> is_dir(x.size(), 0);
    for (const auto d : s.dirichlet_dofs) {
        if (d >= 0 && static_cast<std::size_t>(d) < is_dir.size()) {
            is_dir[static_cast<std::size_t>(d)] = 1;
        }
    }
    double r_all = 0.0;
    double b_all = 0.0;
    double r_free = 0.0;
    double b_free = 0.0;
    double e = 0.0;
    double xr = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double ri = s.rhs[i] - ax[i];
        r_all += ri * ri;
        b_all += s.rhs[i] * s.rhs[i];
        if (!is_dir[i]) {
            r_free += ri * ri;
            b_free += s.rhs[i] * s.rhs[i];
        }
        const double di = x[i] - s.solution[i];
        e += di * di;
        xr += s.solution[i] * s.solution[i];
    }
    ResidualStats st;
    st.rel_all = std::sqrt(r_all) / std::max(std::sqrt(b_all), 1e-300);
    st.rel_free = std::sqrt(r_free) / std::max(std::sqrt(b_free), 1e-300);
    st.err_vs_recorded = std::sqrt(e) / std::max(std::sqrt(xr), 1e-300);
    return st;
}

/// Pressure-last grouping used by the FSILS BlockSchur route: every component
/// before the scalar constraint forms the computational primary block.
BlockLayout groupedSaddleLayout(const FsilsSystemSnapshot& s)
{
    BlockLayout layout;
    int pressure = -1;
    for (const auto& b : s.blocks) {
        if (b.name == "Pressure" && b.n_components == 1) {
            pressure = b.start_component;
        }
    }
    if (pressure != s.dof - 1) {
        return layout;
    }
    layout.blocks.push_back({"Primary", 0, s.dof - 1, BlockRole::PrimaryField});
    layout.blocks.push_back({"Pressure", s.dof - 1, 1, BlockRole::ConstraintField});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    return layout;
}

BlockLayout snapshotLayout(const FsilsSystemSnapshot& s)
{
    BlockLayout layout;
    layout.blocks = s.blocks;
    for (std::size_t i = 0; i < layout.blocks.size(); ++i) {
        if (layout.blocks[i].name == "Velocity") {
            layout.blocks[i].role = BlockRole::PrimaryField;
            layout.momentum_block = static_cast<int>(i);
        } else if (layout.blocks[i].name == "Pressure") {
            layout.blocks[i].role = BlockRole::ConstraintField;
            layout.constraint_block = static_cast<int>(i);
        }
    }
    return layout;
}

struct ReplayConfig {
    std::string name;
    std::string backend;  // "fsils" or "eigen"
    std::function<SolverOptions(const FsilsSystemSnapshot&)> options;
};

SolverOptions baseFsilsOptions(const FsilsSystemSnapshot& s)
{
    SolverOptions o;
    o.method = SolverMethod::GMRES;
    o.preconditioner = PreconditionerType::RowColumnScaling;
    o.fsils_use_rcs = true;
    o.rel_tol = s.rel_tol;
    o.abs_tol = s.abs_tol;
    o.max_iter = s.max_iter;
    o.krylov_dim = s.krylov_dim;
    o.block_layout = snapshotLayout(s);
    o.fsils_residual_check_policy = FsilsResidualCheckPolicy::Always;
    return o;
}

std::vector<ReplayConfig> allConfigs()
{
    std::vector<ReplayConfig> configs;
    configs.push_back({"fsils-gmres-rcs", "fsils", [](const FsilsSystemSnapshot& s) {
                           return baseFsilsOptions(s);
                       }});
    auto right_pc = [](RightPreconditionerType pc, bool reuse) {
        return [pc, reuse](const FsilsSystemSnapshot& s) {
            auto o = baseFsilsOptions(s);
            o.right_preconditioner = pc;
            o.reuse_preconditioner = reuse;
            o.right_preconditioner_constraint_block = "Pressure";
            return o;
        };
    };
    configs.push_back({"fsils-gmres-rcs-bilu0", "fsils", right_pc(RightPreconditionerType::BlockILU0, false)});
    configs.push_back({"fsils-gmres-rcs-bilu0-reuse", "fsils", right_pc(RightPreconditionerType::BlockILU0, true)});
    configs.push_back({"fsils-gmres-rcs-simple", "fsils", right_pc(RightPreconditionerType::Simple, false)});
    configs.push_back({"fsils-gmres-rcs-simple-reuse", "fsils", right_pc(RightPreconditionerType::Simple, true)});
    configs.push_back({"fsils-gmres-diag", "fsils", [](const FsilsSystemSnapshot& s) {
                           auto o = baseFsilsOptions(s);
                           o.preconditioner = PreconditionerType::Diagonal;
                           o.fsils_use_rcs = false;
                           return o;
                       }});
    configs.push_back({"fsils-bicgs-rcs", "fsils", [](const FsilsSystemSnapshot& s) {
                           auto o = baseFsilsOptions(s);
                           o.method = SolverMethod::BiCGSTAB;
                           o.max_iter = 1000;
                           return o;
                       }});
    configs.push_back({"fsils-ns", "fsils", [](const FsilsSystemSnapshot& s) {
                           auto o = baseFsilsOptions(s);
                           o.method = SolverMethod::BlockSchur;
                           o.max_iter = 100;
                           o.block_layout = groupedSaddleLayout(s);
                           o.fsils_blockschur_gm_max_iter = 1000;
                           o.fsils_blockschur_cg_max_iter = 1000;
                           o.fsils_blockschur_gm_rel_tol = 1e-2;
                           o.fsils_blockschur_cg_rel_tol = 1e-2;
                           o.fsils_blockschur_schur_preconditioner =
                               FsilsBlockSchurSchurPreconditioner::AlgebraicSchur;
                           o.fsils_blockschur_momentum_approximation =
                               FsilsBlockSchurMomentumApproximation::ILUK;
                           return o;
                       }});
#if defined(FE_HAS_EIGEN)
    auto eigen = [](SolverMethod method, PreconditionerType pc) {
        return [method, pc](const FsilsSystemSnapshot& s) {
            SolverOptions o;
            o.method = method;
            o.preconditioner = pc;
            o.rel_tol = s.rel_tol;
            o.abs_tol = s.abs_tol;
            o.max_iter = 100;
            o.krylov_dim = s.krylov_dim;
            return o;
        };
    };
    configs.push_back({"eigen-gmres-ilut", "eigen", eigen(SolverMethod::GMRES, PreconditionerType::ILU)});
    configs.push_back({"eigen-bicgstab-ilut", "eigen", eigen(SolverMethod::BiCGSTAB, PreconditionerType::ILU)});
    configs.push_back({"eigen-gmres-diag", "eigen", eigen(SolverMethod::GMRES, PreconditionerType::Diagonal)});
    configs.push_back({"eigen-direct", "eigen", eigen(SolverMethod::Direct, PreconditionerType::None)});
    auto eigen_reuse = [eigen](SolverMethod method, PreconditionerType pc) {
        return [f = eigen(method, pc)](const FsilsSystemSnapshot& s) {
            auto o = f(s);
            o.reuse_preconditioner = true;
            return o;
        };
    };
    configs.push_back({"eigen-gmres-ilut-reuse", "eigen", eigen_reuse(SolverMethod::GMRES, PreconditionerType::ILU)});
    configs.push_back({"eigen-direct-reuse", "eigen", eigen_reuse(SolverMethod::Direct, PreconditionerType::None)});
#endif
    return configs;
}

struct FsilsReplaySystem {
    std::unique_ptr<GenericMatrix> A;
    std::unique_ptr<GenericVector> b;
    std::unique_ptr<GenericVector> x;
};

FsilsReplaySystem buildFsils(const FsilsSystemSnapshot& s, const sparsity::SparsityPattern& pattern)
{
    FsilsFactory factory(s.dof);
    FsilsReplaySystem sys;
    sys.A = factory.createMatrix(pattern);
    auto* A = dynamic_cast<FsilsMatrix*>(sys.A.get());
    const auto shared = A->shared();
    const int d = s.dof;
    for (int i = 0; i < s.n_nodes; ++i) {
        const int ri = shared->globalNodeToInternal(i);
        for (auto p = s.row_ptr[static_cast<std::size_t>(i)]; p < s.row_ptr[static_cast<std::size_t>(i) + 1u];
             ++p) {
            const int ci = shared->globalNodeToInternal(s.cols[static_cast<std::size_t>(p)]);
            A->addBlock(ri, ci, s.values.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(d * d), d,
                        assembly::AddMode::Insert);
        }
    }
    A->finalizeAssembly();
    sys.b = factory.createVector(static_cast<GlobalIndex>(s.n_nodes) * d);
    sys.x = factory.createVector(static_cast<GlobalIndex>(s.n_nodes) * d);
    auto* b = dynamic_cast<FsilsVector*>(sys.b.get());
    auto& bd = b->data();
    for (int g = 0; g < s.n_nodes; ++g) {
        const int old = shared->globalNodeToOld(g);
        for (int c = 0; c < d; ++c) {
            bd[static_cast<std::size_t>(old) * d + c] = s.rhs[static_cast<std::size_t>(g) * d + c];
        }
    }
    return sys;
}

std::vector<double> fsilsSolution(const FsilsSystemSnapshot& s, const GenericMatrix& Ag, const GenericVector& xg)
{
    const auto* A = dynamic_cast<const FsilsMatrix*>(&Ag);
    const auto* x = dynamic_cast<const FsilsVector*>(&xg);
    const auto shared = A->shared();
    std::vector<double> out(static_cast<std::size_t>(s.n_nodes) * s.dof, 0.0);
    const auto& xd = x->data();
    for (int g = 0; g < s.n_nodes; ++g) {
        const int old = shared->globalNodeToOld(g);
        for (int c = 0; c < s.dof; ++c) {
            out[static_cast<std::size_t>(g) * s.dof + c] = xd[static_cast<std::size_t>(old) * s.dof + c];
        }
    }
    return out;
}

TEST(LinearSolverReplay, ReplaySnapshots)
{
    const auto files = splitList(std::getenv("SVMP_LINEAR_REPLAY_FILES"));
    if (files.empty()) {
        GTEST_SKIP() << "SVMP_LINEAR_REPLAY_FILES not set";
    }
    const auto selected = splitList(std::getenv("SVMP_LINEAR_REPLAY_CONFIGS"));
    std::vector<ReplayConfig> configs;
    for (auto& c : allConfigs()) {
        if (selected.empty() ||
            std::find(selected.begin(), selected.end(), c.name) != selected.end()) {
            configs.push_back(std::move(c));
        }
    }
    ASSERT_FALSE(configs.empty());

    for (const auto& config : configs) {
        std::unique_ptr<LinearSolver> solver;
        long long total_iterations = 0;
        double total_seconds = 0.0;
        int failures = 0;
        for (std::size_t fi = 0; fi < files.size(); ++fi) {
            const auto snap = readFsilsSystemSnapshot(files[fi]);
            const auto pattern = scalarPattern(snap);
            const auto options = config.options(snap);
            SolverReport report;
            std::vector<double> x;
            std::string error;
            double seconds = 0.0;
            try {
                if (config.backend == "fsils") {
                    auto sys = buildFsils(snap, pattern);
                    if (!solver) {
                        solver = std::make_unique<FsilsLinearSolver>(options);
                    } else {
                        solver->setOptions(options);
                    }
                    std::vector<GlobalIndex> dir(snap.dirichlet_dofs.begin(), snap.dirichlet_dofs.end());
                    solver->setDirichletDofs(dir);
                    const auto t0 = std::chrono::steady_clock::now();
                    report = solver->solve(*sys.A, *sys.x, *sys.b);
                    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    x = fsilsSolution(snap, *sys.A, *sys.x);
                }
#if defined(FE_HAS_EIGEN)
                else {
                    EigenFactory factory;
                    auto Ag = factory.createMatrix(pattern);
                    auto* A = dynamic_cast<EigenMatrix*>(Ag.get());
                    const int d = snap.dof;
                    for (int i = 0; i < snap.n_nodes; ++i) {
                        for (auto p = snap.row_ptr[static_cast<std::size_t>(i)];
                             p < snap.row_ptr[static_cast<std::size_t>(i) + 1u]; ++p) {
                            const int j = snap.cols[static_cast<std::size_t>(p)];
                            const double* blk =
                                snap.values.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(d * d);
                            for (int r = 0; r < d; ++r) {
                                for (int c = 0; c < d; ++c) {
                                    A->addValue(static_cast<GlobalIndex>(i) * d + r,
                                                static_cast<GlobalIndex>(j) * d + c, blk[r * d + c],
                                                assembly::AddMode::Insert);
                                }
                            }
                        }
                    }
                    A->finalizeAssembly();
                    auto bg = factory.createVector(A->numRows());
                    auto xg = factory.createVector(A->numRows());
                    auto* b = dynamic_cast<EigenVector*>(bg.get());
                    for (std::size_t i = 0; i < snap.rhs.size(); ++i) {
                        b->eigen()(static_cast<Eigen::Index>(i)) = snap.rhs[i];
                    }
                    if (!solver) {
                        solver = factory.createLinearSolver(options);
                    } else {
                        solver->setOptions(options);
                    }
                    const auto t0 = std::chrono::steady_clock::now();
                    report = solver->solve(*Ag, *xg, *bg);
                    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    const auto* xe = dynamic_cast<EigenVector*>(xg.get());
                    x.assign(xe->eigen().data(), xe->eigen().data() + xe->eigen().size());
                }
#endif
            } catch (const std::exception& e) {
                error = e.what();
            }
            if (!error.empty()) {
                ++failures;
                std::printf("REPLAY config=%s file=%zu error=\"%s\"\n", config.name.c_str(), fi, error.c_str());
                continue;
            }
            const auto st = residualStats(snap, x);
            total_iterations += report.iterations;
            total_seconds += seconds;
            if (!report.converged) {
                ++failures;
            }
            std::printf("REPLAY config=%s file=%zu nodes=%d dof=%d iters=%d converged=%d rel_all=%.3e "
                        "rel_free=%.3e err_vs_recorded=%.3e seconds=%.4f recorded_iters=%d "
                        "recorded_seconds=%.4f\n",
                        config.name.c_str(), fi, snap.n_nodes, snap.dof, report.iterations,
                        report.converged ? 1 : 0, st.rel_all, st.rel_free, st.err_vs_recorded, seconds,
                        snap.iterations, snap.solve_seconds);
            std::fflush(stdout);
            if (config.name == "fsils-gmres-rcs") {
                EXPECT_EQ(report.iterations, snap.iterations) << files[fi];
            }
        }
        std::printf("REPLAY_SUMMARY config=%s solves=%zu total_iters=%lld total_seconds=%.4f failures=%d\n",
                    config.name.c_str(), files.size(), total_iterations, total_seconds, failures);
        std::fflush(stdout);
    }
}

} // namespace
} // namespace svmp::FE::backends
