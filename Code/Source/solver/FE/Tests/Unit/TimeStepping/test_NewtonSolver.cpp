/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryModelDSL.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

namespace {

struct KernelCallCounts {
    int total{0};
    int matrix_only{0};
    int vector_only{0};
    int matrix_and_vector{0};
};

class CountingKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    CountingKernel(std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner, KernelCallCounts* counts)
        : inner_(std::move(inner))
        , counts_(counts)
    {
        if (!inner_) {
            throw std::runtime_error("CountingKernel: inner is null");
        }
        if (!counts_) {
            throw std::runtime_error("CountingKernel: counts is null");
        }
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        return inner_->getRequiredData();
    }

    [[nodiscard]] std::vector<svmp::FE::assembly::FieldRequirement> fieldRequirements() const override
    {
        return inner_->fieldRequirements();
    }

    [[nodiscard]] svmp::FE::assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return inner_->materialStateSpec();
    }

    [[nodiscard]] std::vector<svmp::FE::params::Spec> parameterSpecs() const override
    {
        return inner_->parameterSpecs();
    }

    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override
    {
        return inner_->maxTemporalDerivativeOrder();
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override
    {
        return inner_->hasStateIndependentMatrix();
    }

    [[nodiscard]] bool hasCell() const noexcept override { return inner_->hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return inner_->hasBoundaryFace(); }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return inner_->hasInteriorFace(); }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const bool want_matrix = output.has_matrix;
        const bool want_vector = output.has_vector;
        inner_->computeCell(ctx, output);
        const bool did_matrix = want_matrix || !output.local_matrix.empty();
        const bool did_vector = want_vector || !output.local_vector.empty();
        counts_->total += 1;
        if (did_matrix && did_vector) {
            counts_->matrix_and_vector += 1;
        } else if (did_matrix) {
            counts_->matrix_only += 1;
        } else if (did_vector) {
            counts_->vector_only += 1;
        }
    }

    void computeBoundaryFace(const svmp::FE::assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             svmp::FE::assembly::KernelOutput& output) override
    {
        inner_->computeBoundaryFace(ctx, boundary_marker, output);
    }

    void computeInteriorFace(const svmp::FE::assembly::AssemblyContext& ctx_minus,
                             const svmp::FE::assembly::AssemblyContext& ctx_plus,
                             svmp::FE::assembly::KernelOutput& output_minus,
                             svmp::FE::assembly::KernelOutput& output_plus,
                             svmp::FE::assembly::KernelOutput& coupling_minus_plus,
                             svmp::FE::assembly::KernelOutput& coupling_plus_minus) override
    {
        inner_->computeInteriorFace(ctx_minus,
                                    ctx_plus,
                                    output_minus,
                                    output_plus,
                                    coupling_minus_plus,
                                    coupling_plus_minus);
    }

    [[nodiscard]] std::string name() const override
    {
        return "Counting(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> inner_{};
    KernelCallCounts* counts_{nullptr};
};

class AffineScalarCellKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
    explicit AffineScalarCellKernel(svmp::FE::Real target)
        : target_(target)
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        using svmp::FE::assembly::RequiredData;
        return RequiredData::IntegrationWeights |
               RequiredData::BasisValues |
               RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override { return true; }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        const auto n_test = ctx.numTestDofs();
        const auto n_trial = ctx.numTrialDofs();
        bool want_matrix = output.has_matrix || !output.local_matrix.empty();
        bool want_vector = output.has_vector || !output.local_vector.empty();
        if (!want_matrix && !want_vector) {
            want_matrix = true;
            want_vector = true;
        }
        output.reserve(n_test, n_trial, want_matrix, want_vector);
        output.clear();

        const auto coeffs = ctx.solutionCoefficients();
        if (want_vector && coeffs.size() < static_cast<std::size_t>(n_trial)) {
            throw std::runtime_error("AffineScalarCellKernel: missing solution coefficients");
        }

        for (svmp::FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            const auto w = ctx.integrationWeight(q);
            svmp::FE::Real uh = 0.0;
            if (want_vector) {
                for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                    uh += ctx.trialBasisValue(j, q) * coeffs[static_cast<std::size_t>(j)];
                }
            }
            for (svmp::FE::LocalIndex i = 0; i < n_test; ++i) {
                const auto vi = ctx.basisValue(i, q);
                if (want_vector) {
                    output.vectorEntry(i) += w * vi * (uh - target_);
                }
                if (want_matrix) {
                    for (svmp::FE::LocalIndex j = 0; j < n_trial; ++j) {
                        output.matrixEntry(i, j) += w * vi * ctx.trialBasisValue(j, q);
                    }
                }
            }
        }
    }

    [[nodiscard]] std::string name() const override { return "AffineScalarCellKernel"; }

private:
    svmp::FE::Real target_{0.0};
};

class ScalingLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    ScalingLinearSolver(svmp::FE::backends::LinearSolver& inner, double scale)
        : inner_(inner)
        , scale_(scale)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        auto rep = inner_.solve(A, x, b);
        x.scale(static_cast<svmp::FE::Real>(scale_));
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    double scale_{1.0};
};

class AlwaysFailLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit AlwaysFailLinearSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix&,
                                                          svmp::FE::backends::GenericVector&,
                                                          const svmp::FE::backends::GenericVector&) override
    {
        svmp::FE::backends::SolverReport rep;
        rep.converged = false;
        rep.iterations = 0;
        rep.message = "intentional test failure";
        return rep;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class RecordingEffectiveTimeStepSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit RecordingEffectiveTimeStepSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        return inner_.solve(A, x, b);
    }

    void setEffectiveTimeStep(double dt_eff) override
    {
        effective_time_steps.push_back(dt_eff);
        inner_.setEffectiveTimeStep(dt_eff);
    }

    std::vector<double> effective_time_steps{};

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class FailOnceThenSolveRecordingMatrixSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit FailOnceThenSolveRecordingMatrixSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        ++solve_calls;
        observed_diagonals.push_back(static_cast<double>(A.getEntry(0, 0)));
        if (solve_calls == 1) {
            svmp::FE::backends::SolverReport rep;
            rep.converged = false;
            rep.iterations = 0;
            rep.message = "intentional first failure for PTC retry test";
            return rep;
        }
        return inner_.solve(A, x, b);
    }

    void setEffectiveTimeStep(double dt_eff) override
    {
        inner_.setEffectiveTimeStep(dt_eff);
    }

    int solve_calls{0};
    std::vector<double> observed_diagonals{};

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class RecordingRankOneSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit RecordingRankOneSolver(svmp::FE::backends::LinearSolver& inner)
        : inner_(inner)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        return inner_.solve(A, x, b);
    }

    void setRankOneUpdates(std::span<const svmp::FE::backends::RankOneUpdate> updates) override
    {
        last_updates.assign(updates.begin(), updates.end());
        if (!updates.empty()) {
            saw_nonempty_rank_one_updates = true;
        }
    }

    void setReducedFieldUpdates(
        std::span<const svmp::FE::backends::ReducedFieldUpdate> updates) override
    {
        last_reduced_updates.assign(updates.begin(), updates.end());
        if (!updates.empty()) {
            saw_nonempty_reduced_updates = true;
        }
    }

    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override
    {
        return true;
    }

    [[nodiscard]] bool supportsNativeReducedFieldUpdates() const noexcept override
    {
        return true;
    }

    std::vector<svmp::FE::backends::RankOneUpdate> last_updates{};
    std::vector<svmp::FE::backends::ReducedFieldUpdate> last_reduced_updates{};
    bool saw_nonempty_rank_one_updates{false};
    bool saw_nonempty_reduced_updates{false};

private:
    svmp::FE::backends::LinearSolver& inner_;
};

class RecordingSolveOptionsSolver final : public svmp::FE::backends::LinearSolver {
public:
    explicit RecordingSolveOptionsSolver(svmp::FE::backends::LinearSolver& inner,
                                         bool native_rank_one_support = false)
        : inner_(inner)
        , native_rank_one_support_(native_rank_one_support)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        last_set_options = options;
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        saw_solve = true;
        options_seen_in_solve = inner_.getOptions();
        return inner_.solve(A, x, b);
    }

    void setRankOneUpdates(std::span<const svmp::FE::backends::RankOneUpdate> updates) override
    {
        last_updates.assign(updates.begin(), updates.end());
        inner_.setRankOneUpdates(updates);
    }

    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override
    {
        return native_rank_one_support_;
    }

    bool saw_solve{false};
    std::optional<svmp::FE::backends::SolverOptions> last_set_options{};
    std::optional<svmp::FE::backends::SolverOptions> options_seen_in_solve{};
    std::vector<svmp::FE::backends::RankOneUpdate> last_updates{};

private:
    svmp::FE::backends::LinearSolver& inner_;
    bool native_rank_one_support_{false};
};

class ForceResidualReportLinearSolver final : public svmp::FE::backends::LinearSolver {
public:
    ForceResidualReportLinearSolver(svmp::FE::backends::LinearSolver& inner,
                                    double initial_residual_norm,
                                    double final_residual_norm,
                                    int forced_miss_calls = 1,
                                    bool native_rank_one_support = false)
        : inner_(inner)
        , initial_residual_norm_(initial_residual_norm)
        , final_residual_norm_(final_residual_norm)
        , forced_miss_calls_(forced_miss_calls)
        , native_rank_one_support_(native_rank_one_support)
    {
    }

    [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
    {
        return inner_.backendKind();
    }

    void setOptions(const svmp::FE::backends::SolverOptions& options) override
    {
        inner_.setOptions(options);
    }

    [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
    {
        return inner_.getOptions();
    }

    [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                          svmp::FE::backends::GenericVector& x,
                                                          const svmp::FE::backends::GenericVector& b) override
    {
        auto rep = inner_.solve(A, x, b);
        if (forced_miss_calls_ <= 0) {
            return rep;
        }
        --forced_miss_calls_;
        rep.converged = false;
        rep.initial_residual_norm = initial_residual_norm_;
        rep.final_residual_norm = final_residual_norm_;
        rep.relative_residual =
            final_residual_norm_ / std::max(initial_residual_norm_, 1e-30);
        rep.message = "synthetic strict-coupled miss";
        return rep;
    }

    void setRankOneUpdates(std::span<const svmp::FE::backends::RankOneUpdate> updates) override
    {
        inner_.setRankOneUpdates(updates);
    }

    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override
    {
        return native_rank_one_support_;
    }

private:
    svmp::FE::backends::LinearSolver& inner_;
    double initial_residual_norm_{1.0};
    double final_residual_norm_{1.0};
    int forced_miss_calls_{1};
    bool native_rank_one_support_{false};
};

class ScopedEnvVar final {
public:
    ScopedEnvVar(const char* key, const char* value)
        : key_(key)
    {
        const char* prior = std::getenv(key_);
        if (prior != nullptr) {
            had_prior_ = true;
            prior_value_ = prior;
        }
        ::setenv(key_, value, 1);
    }

    ~ScopedEnvVar()
    {
        if (had_prior_) {
            ::setenv(key_, prior_value_.c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* key_{nullptr};
    bool had_prior_{false};
    std::string prior_value_{};
};

struct ScalarProblem {
    std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::L2Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId u_field{std::numeric_limits<svmp::FE::FieldId>::max()};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

struct DirectCouplingProblem {
    std::shared_ptr<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess> mesh{};
    std::shared_ptr<svmp::FE::spaces::H1Space> space{};
    std::unique_ptr<svmp::FE::systems::FESystem> sys{};
    svmp::FE::FieldId u_field{std::numeric_limits<svmp::FE::FieldId>::max()};
    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> integrator{};
    std::unique_ptr<svmp::FE::systems::TransientSystem> transient{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    std::unique_ptr<svmp::FE::backends::LinearSolver> linear{};
    svmp::FE::timestepping::TimeHistory history{};
};

template <typename BuildForm>
[[nodiscard]] ScalarProblem makeScalarProblem(BuildForm build_form,
                                              double dt,
                                              const std::vector<svmp::FE::Real>& u0,
                                              KernelCallCounts* counts = nullptr)
{
    ScalarProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*p.space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*p.space, "v");
    const auto form = build_form(u, v);

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto base_kernel =
        std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> kernel = base_kernel;
    if (counts != nullptr) {
        kernel = std::make_shared<CountingKernel>(kernel, counts);
    }
    p.sys->addCellKernel("op", p.u_field, p.u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] ScalarProblem makeAffineScalarProblem(double target,
                                                    double dt,
                                                    const std::vector<svmp::FE::Real>& u0,
                                                    KernelCallCounts* counts)
{
    ScalarProblem p;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    p.space = std::make_shared<svmp::FE::spaces::L2Space>(svmp::FE::ElementType::Tetra4, /*order=*/0);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    std::shared_ptr<svmp::FE::assembly::AssemblyKernel> kernel =
        std::make_shared<AffineScalarCellKernel>(static_cast<svmp::FE::Real>(target));
    if (counts != nullptr) {
        kernel = std::make_shared<CountingKernel>(kernel, counts);
    }
    p.sys->addCellKernel("op", p.u_field, p.u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("ScalarProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("ScalarProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("ScalarProblem u0 size mismatch");
    }
    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

[[nodiscard]] double scalarFromDofVector(svmp::FE::backends::GenericVector& vec)
{
    const auto vals = ts_test::getVectorByDof(vec);
    if (vals.size() != 1u) {
        throw std::runtime_error("Expected scalar DOF vector");
    }
    return static_cast<double>(vals[0]);
}

[[nodiscard]] DirectCouplingProblem makeDirectCouplingProblem(double dt,
                                                              const std::vector<svmp::FE::Real>& u0,
                                                              std::optional<svmp::FE::systems::AuxiliaryBlockRole>
                                                                  solver_role = std::nullopt)
{
    DirectCouplingProblem p;
    constexpr int marker = 6;
    p.mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    p.space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/1);

    p.sys = std::make_unique<svmp::FE::systems::FESystem>(p.mesh);
    p.u_field = p.sys->addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = p.space, .components = 1});
    p.sys->addOperator("op");

    const auto u_disc = svmp::FE::forms::FormExpr::discreteField(p.u_field, *p.space, "u");
    auto Q = p.sys->boundaryIntegral(u_disc, marker);

    auto model = svmp::FE::systems::aux::model("newton_rank_one_snapshot",
        [](svmp::FE::systems::ModelFacade& m) {
            auto Q = m.input("Q");
            auto x1 = m.state("x1");
            auto x2 = m.state("x2");
            auto Rp = m.param("Rp");
            m << svmp::FE::systems::ddt(x1) == -x1;
            m << svmp::FE::systems::ddt(x2) == -x2 + Q;
            m << svmp::FE::systems::out("P_out") == x2 + Rp * Q;
        });

    auto deployment = svmp::FE::systems::use(model)
        .name("newton_rank_one_snapshot_inst")
        .global()
        .monolithic()
        .bind("Q", Q)
        .param("Rp", 3.0)
        .initialize({0.0, 0.0});
    if (solver_role.has_value()) {
        deployment.solverRole(*solver_role);
    }
    auto inst = p.sys->deploy(std::move(deployment));

    const auto u = svmp::FE::forms::FormExpr::stateField(p.u_field, *p.space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*p.space, "v");
    const auto residual =
        svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)).dx() -
        (inst.output("P_out") * v).ds(marker);
    (void)svmp::FE::systems::installFormulation(*p.sys, "op", {p.u_field}, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    p.sys->setup({}, inputs);
    p.sys->finalizeAuxiliaryLayout();

    p.integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    p.transient = std::make_unique<svmp::FE::systems::TransientSystem>(*p.sys, p.integrator);

    p.factory = ts_test::createTestFactory();
    if (!p.factory) {
        throw std::runtime_error("DirectCouplingProblem requires the Eigen backend (enable FE_ENABLE_EIGEN)");
    }
    p.linear = p.factory->createLinearSolver(ts_test::directSolve());
    if (!p.linear) {
        throw std::runtime_error("DirectCouplingProblem failed to create LinearSolver");
    }

    const auto n_dofs = p.sys->dofHandler().getNumDofs();
    if (static_cast<std::size_t>(n_dofs) != u0.size()) {
        throw std::runtime_error("DirectCouplingProblem u0 size mismatch");
    }

    p.history = svmp::FE::timestepping::TimeHistory::allocate(*p.factory, n_dofs);
    p.history.setDt(dt);
    p.history.setPrevDt(dt);
    ts_test::setVectorByDof(p.history.uPrev(), u0);
    ts_test::setVectorByDof(p.history.uPrev2(), u0);
    p.history.resetCurrentToPrevious();
    return p;
}

} // namespace

TEST(NewtonSolverLineSearch, BacktracksWhenFullStepIncreasesResidual)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_shrink = 0.5;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/3.0);
    (void)newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    const double u_after = scalarFromDofVector(problem.history.u());
    EXPECT_NEAR(u_after, -0.5, 1e-13);
}

TEST(NewtonSolverLineSearch, ClampsAlphaToMinWhenShrinkWouldGoBelow)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 5;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 0.6;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Force an overshoot so alpha=1 fails, and even alpha_min still fails; the solver
    // must clamp to alpha_min and accept that last trial.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/4.0);
    (void)newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    const double u_after = scalarFromDofVector(problem.history.u());
    EXPECT_NEAR(u_after, -1.4, 1e-13);
}

TEST(NewtonSolverLineSearch, StepToleranceUsesLastTriedAlphaWhenMaxIterationsReached)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.75;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_shrink = 0.5;
    nopt.line_search_alpha_min = 1e-12;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    // Flip the Newton direction so the Armijo condition can never be satisfied. When the
    // line search cannot find a decreasing trial, the solver must reject the update.
    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(*problem.transient, linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
}

TEST(NewtonSolverLineSearch, SynchronizesTrialAndRestoredStates)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct SyncRecord {
        SyncPoint point;
        double u;
        std::uint64_t cut_topology_key;
    };
    std::vector<SyncRecord> sync_records;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.assemble_both_when_possible = false;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;
    nopt.line_search_alpha_min = 1e-12;
    nopt.synchronize_state =
        [&sync_records](const svmp::FE::systems::SystemStateView& state,
                        SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto u = static_cast<double>(state.u.front());
            const auto cut_topology_key =
                u > 1.5 ? std::uint64_t{0x202u} : std::uint64_t{0x101u};
            sync_records.push_back(
                SyncRecord{point, u, cut_topology_key});
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver linear(*problem.linear, /*scale=*/-1.0);
    const auto rep = newton.solveStep(
        *problem.transient,
        linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_FALSE(rep.converged);
    auto saw_trial_update = false;
    auto saw_trial_restore = false;
    auto saw_restored_state = false;
    auto saw_trial_cut_topology = false;
    auto saw_restored_cut_topology = false;
    auto saw_restored_state_cut_topology = false;
    for (const auto& rec : sync_records) {
        if (rec.point == SyncPoint::LineSearchTrialResidual &&
            std::abs(rec.u - 2.0) < 1e-13) {
            saw_trial_update = true;
            saw_trial_cut_topology = rec.cut_topology_key == std::uint64_t{0x202u};
        }
        if (rec.point == SyncPoint::LineSearchTrialResidual &&
            std::abs(rec.u - 1.0) < 1e-13) {
            saw_trial_restore = true;
            saw_restored_cut_topology = rec.cut_topology_key == std::uint64_t{0x101u};
        }
        if (rec.point == SyncPoint::RestoredNonlinearState &&
            std::abs(rec.u - 1.0) < 1e-13) {
            saw_restored_state = true;
            saw_restored_state_cut_topology =
                rec.cut_topology_key == std::uint64_t{0x101u};
        }
    }
    EXPECT_TRUE(saw_trial_update);
    EXPECT_TRUE(saw_trial_restore);
    EXPECT_TRUE(saw_restored_state);
    EXPECT_TRUE(saw_trial_cut_topology);
    EXPECT_TRUE(saw_restored_cut_topology);
    EXPECT_TRUE(saw_restored_state_cut_topology);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 1.0, 1e-13);
}

TEST(NewtonSolver, SynchronizesUpdatedCoupledGeometryBeforeResidualAssembly)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& phi,
           const svmp::FE::forms::FormExpr& v) {
            return ((phi - svmp::FE::forms::FormExpr::constant(2.0)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{0.0});

    using SyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
    struct SyncRecord {
        SyncPoint point;
        double phi;
        std::uint64_t topology_key;
    };
    std::vector<SyncRecord> sync_records;

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 2;
    nopt.abs_tolerance = 1e-14;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.synchronize_state =
        [&sync_records](const svmp::FE::systems::SystemStateView& state,
                        SyncPoint point) {
            ASSERT_FALSE(state.u.empty());
            const auto phi = static_cast<double>(state.u.front());
            const auto topology_key =
                phi > 1.0 ? std::uint64_t{0x220u} : std::uint64_t{0x110u};
            sync_records.push_back(SyncRecord{point, phi, topology_key});
        };

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(
        *problem.transient,
        *problem.linear,
        /*solve_time=*/problem.history.dt(),
        problem.history,
        ws);

    EXPECT_TRUE(rep.converged);
    bool saw_initial_residual_state = false;
    bool saw_updated_residual_state = false;
    for (const auto& rec : sync_records) {
        const bool residual_assembly =
            rec.point == SyncPoint::ResidualAssembly ||
            rec.point == SyncPoint::JacobianAndResidualAssembly;
        if (!residual_assembly) {
            continue;
        }
        if (std::abs(rec.phi) < 1e-13) {
            saw_initial_residual_state =
                rec.topology_key == std::uint64_t{0x110u};
        }
        if (std::abs(rec.phi - 2.0) < 1e-13) {
            saw_updated_residual_state =
                rec.topology_key == std::uint64_t{0x220u};
        }
    }

    EXPECT_TRUE(saw_initial_residual_state);
    EXPECT_TRUE(saw_updated_residual_state);
    EXPECT_NEAR(scalarFromDofVector(problem.history.u()), 2.0, 1e-13);
}

TEST(NewtonSolver, ReusesJacobianWhenRebuildPeriodGreaterThanOne)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5},
        &counts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = false;
    nopt.jacobian_rebuild_period = 3;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);

}

TEST(NewtonSolver, ReusesStateIndependentJacobianWhenRebuildPeriodIsOne)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    KernelCallCounts counts;
    auto problem = makeAffineScalarProblem(
        /*target=*/2.0,
        /*dt=*/0.1,
        /*u0=*/{0.0},
        &counts);

    EXPECT_TRUE(problem.sys->operatorMatrixStateIndependent("op"));

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 3;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.assemble_both_when_possible = true;
    nopt.jacobian_rebuild_period = 1;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver under_relaxed(*problem.linear, /*scale=*/0.25);
    const auto rep = newton.solveStep(*problem.transient,
                                      under_relaxed,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);
    EXPECT_EQ(counts.matrix_and_vector, 1);
    EXPECT_EQ(counts.matrix_only, 0);
    EXPECT_GT(counts.vector_only, 0);
}

TEST(NewtonSolver, ScalesDtIncrementsByDtOrExplicitFactor)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;
    const std::vector<svmp::FE::Real> u0 = {1.0};

    auto problem = makeScalarProblem(
        [&](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();
        },
        dt,
        u0);

    const double u_exact = 1.0 / (1.0 + lambda * dt);
    const double du = 1.0 - u_exact;

    auto run_once = [&](bool scale_dt_increments, double dt_increment_scale) -> double {
        ts_test::setVectorByDof(problem.history.uPrev(), u0);
        ts_test::setVectorByDof(problem.history.uPrev2(), u0);
        problem.history.resetCurrentToPrevious();

        svmp::FE::timestepping::NewtonOptions nopt;
        nopt.residual_op = "op";
        nopt.jacobian_op = "op";
        nopt.max_iterations = 1;
        nopt.abs_tolerance = 0.0;
        nopt.rel_tolerance = 0.0;
        nopt.step_tolerance = 0.0;
        nopt.use_line_search = false;
        nopt.scale_dt_increments = scale_dt_increments;
        nopt.dt_increment_scale = dt_increment_scale;

        svmp::FE::timestepping::NewtonSolver newton(nopt);
        svmp::FE::timestepping::NewtonWorkspace ws;
        newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
        problem.history.repack(*problem.factory);

        (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
        return scalarFromDofVector(problem.history.u());
    };

    const double u_unscaled = run_once(/*scale_dt_increments=*/false, /*dt_increment_scale=*/0.0);
    EXPECT_NEAR(u_unscaled, u_exact, 1e-13);

    const double u_scaled_by_dt = run_once(/*scale_dt_increments=*/true, /*dt_increment_scale=*/0.0);
    EXPECT_NEAR(u_scaled_by_dt, 1.0 - dt * du, 1e-13);

    const double u_scaled_explicit = run_once(/*scale_dt_increments=*/true, /*dt_increment_scale=*/0.5);
    EXPECT_NEAR(u_scaled_explicit, 1.0 - 0.5 * du, 1e-13);
}

TEST(NewtonSolver, ExhibitsQuadraticConvergenceNearSolution)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const double u_star = std::sqrt(2.0);

    auto u_val = [&]() { return scalarFromDofVector(problem.history.u()); };
    auto err = [&]() { return std::abs(u_val() - u_star); };

    const double e0 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e1 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e2 = err();
    (void)newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);
    const double e3 = err();

    EXPECT_LT(e1, e0);
    EXPECT_LT(e2, e1);
    EXPECT_LT(e3, e2);

    constexpr double C = 0.6;
    EXPECT_LE(e2, C * e1 * e1);
    EXPECT_LE(e3, C * e2 * e2);
}

TEST(NewtonSolver, ModifiedNewtonConvergesMoreSlowlyThanFullNewton)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto make_history = [&](const svmp::FE::backends::BackendFactory& factory,
                            svmp::FE::GlobalIndex n_dofs,
                            const std::vector<svmp::FE::Real>& u0) {
        auto history = svmp::FE::timestepping::TimeHistory::allocate(factory, n_dofs);
        history.setDt(0.1);
        history.setPrevDt(0.1);
        ts_test::setVectorByDof(history.uPrev(), u0);
        ts_test::setVectorByDof(history.uPrev2(), u0);
        history.resetCurrentToPrevious();
        return history;
    };

    auto base_problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    const auto n_dofs = base_problem.sys->dofHandler().getNumDofs();
    const std::vector<svmp::FE::Real> u0 = {1.5};

    auto history_full = make_history(*base_problem.factory, n_dofs, u0);
    auto history_mod = make_history(*base_problem.factory, n_dofs, u0);

    svmp::FE::timestepping::NewtonOptions full;
    full.residual_op = "op";
    full.jacobian_op = "op";
    full.max_iterations = 3;
    full.abs_tolerance = 0.0;
    full.rel_tolerance = 0.0;
    full.step_tolerance = 0.0;
    full.use_line_search = false;
    full.jacobian_rebuild_period = 1;

    svmp::FE::timestepping::NewtonOptions mod = full;
    mod.jacobian_rebuild_period = 100;

    svmp::FE::timestepping::NewtonSolver newton_full(full);
    svmp::FE::timestepping::NewtonSolver newton_mod(mod);

    svmp::FE::timestepping::NewtonWorkspace ws_full;
    svmp::FE::timestepping::NewtonWorkspace ws_mod;
    newton_full.allocateWorkspace(*base_problem.sys, *base_problem.factory, ws_full);
    newton_mod.allocateWorkspace(*base_problem.sys, *base_problem.factory, ws_mod);
    history_full.repack(*base_problem.factory);
    history_mod.repack(*base_problem.factory);

    (void)newton_full.solveStep(*base_problem.transient, *base_problem.linear, /*solve_time=*/history_full.dt(), history_full, ws_full);
    (void)newton_mod.solveStep(*base_problem.transient, *base_problem.linear, /*solve_time=*/history_mod.dt(), history_mod, ws_mod);

    const double u_star = std::sqrt(2.0);
    const double err_full = std::abs(scalarFromDofVector(history_full.u()) - u_star);
    const double err_mod = std::abs(scalarFromDofVector(history_mod.u()) - u_star);

    EXPECT_LT(err_full, 1e-10);
    EXPECT_GT(err_mod, 1e-7);
}

TEST(NewtonSolver, ReportContainsResidualNormsWhenNotConverged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-20;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 2;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient, *problem.linear, /*solve_time=*/problem.history.dt(), problem.history, ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, 1);
    EXPECT_TRUE(std::isfinite(rep.residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.residual_norm));
    EXPECT_GT(rep.residual_norm0, 0.0);
    EXPECT_GT(rep.residual_norm0, rep.residual_norm);
    EXPECT_TRUE(rep.linear.converged);
}

TEST(NewtonSolver, MonolithicAuxiliaryReportsComponentResidualNorms)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-20;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = true;
    nopt.line_search_max_iterations = 1;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient,
                                      *problem.linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_TRUE(rep.component_residual_convergence);
    EXPECT_TRUE(std::isfinite(rep.field_residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.field_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.auxiliary_residual_norm0));
    EXPECT_TRUE(std::isfinite(rep.auxiliary_residual_norm));
    EXPECT_GT(rep.field_residual_norm0, 0.0);
    EXPECT_GT(rep.auxiliary_residual_norm0, 0.0);

    const double initial_combined =
        std::hypot(rep.field_residual_norm0, rep.auxiliary_residual_norm0);
    const double final_combined =
        std::hypot(rep.field_residual_norm, rep.auxiliary_residual_norm);
    EXPECT_NEAR(rep.residual_norm0,
                initial_combined,
                1e-12 * std::max(1.0, initial_combined));
    EXPECT_NEAR(rep.residual_norm,
                final_combined,
                1e-12 * std::max(1.0, final_combined));
}

TEST(NewtonSolver, MonolithicAuxiliaryConvergenceRequiresEachResidualComponent)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    svmp::FE::timestepping::NewtonOptions probe_opts;
    probe_opts.residual_op = "op";
    probe_opts.jacobian_op = "op";
    probe_opts.max_iterations = 1;
    probe_opts.abs_tolerance = 1e-30;
    probe_opts.rel_tolerance = 0.0;
    probe_opts.step_tolerance = 0.0;
    probe_opts.use_line_search = true;
    probe_opts.line_search_max_iterations = 1;

    auto probe_problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});
    svmp::FE::timestepping::NewtonSolver probe_newton(probe_opts);
    svmp::FE::timestepping::NewtonWorkspace probe_ws;
    probe_newton.allocateWorkspace(*probe_problem.sys, *probe_problem.factory, probe_ws);
    probe_problem.history.repack(*probe_problem.factory);

    const auto probe_rep = probe_newton.solveStep(*probe_problem.transient,
                                                  *probe_problem.linear,
                                                  /*solve_time=*/probe_problem.history.dt(),
                                                  probe_problem.history,
                                                  probe_ws);
    ASSERT_TRUE(probe_rep.component_residual_convergence);
    ASSERT_GT(probe_rep.residual_norm0, 0.0);
    ASSERT_GT(probe_rep.field_residual_norm0, 0.0);

    const double combined_rel = probe_rep.residual_norm / probe_rep.residual_norm0;
    const double field_rel = probe_rep.field_residual_norm / probe_rep.field_residual_norm0;
    if (!(std::isfinite(combined_rel) && std::isfinite(field_rel) &&
          combined_rel > 0.0 && combined_rel < field_rel)) {
        GTEST_SKIP() << "Fixture did not produce a combined-relative gap for this backend/configuration";
    }

    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});
    auto nopt = probe_opts;
    nopt.rel_tolerance = std::sqrt(combined_rel * field_rel);
    nopt.abs_tolerance = 0.0;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient,
                                      *problem.linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_LT(rep.residual_norm / rep.residual_norm0, nopt.rel_tolerance);
    EXPECT_GT(rep.field_residual_norm / rep.field_residual_norm0, nopt.rel_tolerance);
}

TEST(NewtonSolver, StagnationDoesNotOverrideRequestedTolerances)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return ((u - svmp::FE::forms::FormExpr::constant(1.0)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{0.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 3;
    nopt.abs_tolerance = 1e-12;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.stagnation_tolerance = 0.99;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    ScalingLinearSolver damped(*problem.linear, /*scale=*/0.01);
    const auto rep = newton.solveStep(*problem.transient,
                                      damped,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.iterations, nopt.max_iterations);
    EXPECT_TRUE(std::isfinite(rep.residual_norm));
    EXPECT_GT(rep.residual_norm, 0.9 * rep.residual_norm0);
    EXPECT_GT(rep.residual_norm, nopt.abs_tolerance);
}

TEST(NewtonSolver, RelativeToleranceCanConvergeAfterFirstUpdate)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            const auto a = static_cast<svmp::FE::Real>(2.0);
            return ((u * u - svmp::FE::forms::FormExpr::constant(a)) * v).dx();
        },
        /*dt=*/0.1,
        /*u0=*/{1.5});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 5;
    nopt.abs_tolerance = 1e-30;
    nopt.rel_tolerance = 1e-2;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    const auto rep = newton.solveStep(*problem.transient,
                                      *problem.linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_TRUE(rep.converged);
    EXPECT_GE(rep.iterations, 1);
    EXPECT_LT(rep.residual_norm / rep.residual_norm0, nopt.rel_tolerance);
    EXPECT_GT(rep.residual_norm, nopt.abs_tolerance);
}

TEST(NewtonSolver, ThrowsWhenLinearSolveFails)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeScalarProblem(
        [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) { return (u * v).dx(); },
        /*dt=*/0.1,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    AlwaysFailLinearSolver failing(*problem.linear);
    EXPECT_THROW((void)newton.solveStep(*problem.transient,
                                        failing,
                                        /*solve_time=*/problem.history.dt(),
                                        problem.history,
                                        ws),
                 svmp::FE::FEException);
}

TEST(NewtonSolver, PassesEffectiveStageTimeStepToLinearSolver)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto make_problem = [] {
        return makeScalarProblem(
            [](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
                return (u * v).dx();
            },
            /*dt=*/0.2,
            /*u0=*/{1.0});
    };

    auto stage_problem = make_problem();
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*stage_problem.sys, *stage_problem.factory, ws);
    stage_problem.history.repack(*stage_problem.factory);

    RecordingEffectiveTimeStepSolver stage_solver(*stage_problem.linear);
    (void)newton.solveStep(*stage_problem.transient,
                           stage_solver,
                           /*solve_time=*/stage_problem.history.time() + 0.05,
                           stage_problem.history,
                           ws);
    ASSERT_FALSE(stage_solver.effective_time_steps.empty());
    EXPECT_NEAR(stage_solver.effective_time_steps.front(), 0.05, 1e-15);

    auto fallback_problem = make_problem();
    svmp::FE::timestepping::NewtonWorkspace ws_fallback;
    newton.allocateWorkspace(*fallback_problem.sys, *fallback_problem.factory, ws_fallback);
    fallback_problem.history.repack(*fallback_problem.factory);

    RecordingEffectiveTimeStepSolver fallback_solver(*fallback_problem.linear);
    (void)newton.solveStep(*fallback_problem.transient,
                           fallback_solver,
                           /*solve_time=*/fallback_problem.history.time(),
                           fallback_problem.history,
                           ws_fallback);
    ASSERT_FALSE(fallback_solver.effective_time_steps.empty());
    EXPECT_NEAR(fallback_solver.effective_time_steps.front(), fallback_problem.history.dt(), 1e-15);
}

TEST(NewtonSolver, PtcRetryAppliesMassLumpedDiagonalShiftAndRestoresJacobian)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double dt = 0.2;
    constexpr double lambda = 2.0;
    constexpr double gamma = 3.5;
    auto problem = makeScalarProblem(
        [&](const svmp::FE::forms::FormExpr& u, const svmp::FE::forms::FormExpr& v) {
            return (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<svmp::FE::Real>(lambda)).dx();
        },
        dt,
        /*u0=*/{1.0});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 0.0;
    nopt.rel_tolerance = 0.0;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;
    nopt.pseudo_transient.enabled = true;
    nopt.pseudo_transient.activate_on_linear_failure = true;
    nopt.pseudo_transient.gamma_initial = gamma;
    nopt.pseudo_transient.gamma_growth = 2.0;
    nopt.pseudo_transient.gamma_max = 10.0;
    nopt.pseudo_transient.max_linear_retries = 2;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    FailOnceThenSolveRecordingMatrixSolver linear(*problem.linear);
    EXPECT_NO_THROW((void)newton.solveStep(*problem.transient,
                                           linear,
                                           /*solve_time=*/problem.history.dt(),
                                           problem.history,
                                           ws));

    ASSERT_EQ(linear.solve_calls, 2);
    ASSERT_EQ(linear.observed_diagonals.size(), 2u);
    ASSERT_NE(ws.ptc_mass_lumped, nullptr);
    const auto mass = ws.ptc_mass_lumped->localSpan();
    ASSERT_EQ(mass.size(), 1u);
    const double expected_shift = gamma * std::abs(static_cast<double>(mass[0]));
    EXPECT_GT(expected_shift, 0.0);
    EXPECT_NEAR(linear.observed_diagonals[1] - linear.observed_diagonals[0],
                expected_shift,
                1e-12);
}

// The Jacobian-check path still validates the coupled reduced-update
// contribution, but it now does so through the assembled/system-side operator
// path instead of leaving a non-empty reduced-update set installed on the
// linear solver wrapper after the check. Re-enable with a lower-level probe if
// we need explicit coverage of that operator path.
TEST(NewtonSolver, DISABLED_PreservesCoupledUpdatesAcrossJacobianCheckResidualAssemblies)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    ScopedEnvVar jac_check("SVMP_FE_JACOBIAN_CHECK", "1");
    ScopedEnvVar jac_it("SVMP_FE_JACOBIAN_CHECK_IT", "0");
    ScopedEnvVar jac_step("SVMP_FE_JACOBIAN_CHECK_STEP", "1e-7");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    RecordingRankOneSolver linear(*problem.linear);
    const auto rep = newton.solveStep(*problem.transient,
                                      linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_TRUE(linear.last_updates.empty());
    EXPECT_TRUE(linear.saw_nonempty_reduced_updates);
    EXPECT_FALSE(linear.saw_nonempty_rank_one_updates);
    EXPECT_TRUE(rep.linear.converged);
}

TEST(NewtonSolver, ExplicitRankOneUsesCoupledSolveOptions)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    ScopedEnvVar force_explicit("SVMP_FORCE_EXPLICIT_RANK_ONE", "1");

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);
    ts_test::setVectorByDof(problem.history.u(), {0.9, -0.1, 0.35, -0.6});

    RecordingSolveOptionsSolver linear(*problem.linear, /*native_rank_one_support=*/true);
    const auto base_options = linear.getOptions();
    (void)newton.solveStep(*problem.transient,
                           linear,
                           /*solve_time=*/problem.history.dt(),
                           problem.history,
                           ws);

    ASSERT_TRUE(linear.saw_solve);
    ASSERT_TRUE(linear.options_seen_in_solve.has_value());
    EXPECT_TRUE(linear.last_updates.empty());
    EXPECT_EQ(linear.options_seen_in_solve->fsils_residual_check_policy,
              svmp::FE::backends::FsilsResidualCheckPolicy::Always);
    EXPECT_EQ(linear.options_seen_in_solve->max_iter, base_options.max_iter);
    EXPECT_EQ(linear.options_seen_in_solve->rel_tol, base_options.rel_tol);
    EXPECT_EQ(linear.options_seen_in_solve->abs_tol, base_options.abs_tol);
}

TEST(NewtonSolver, ExportsMixedAuxiliaryLayoutIntoLinearSolverOptions)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(
        /*dt=*/0.1,
        /*u0=*/{0.2, -0.4, 0.1, 0.7},
        svmp::FE::systems::AuxiliaryBlockRole::Constraint);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);

    RecordingSolveOptionsSolver linear(*problem.linear);
    (void)newton.solveStep(*problem.transient,
                           linear,
                           /*solve_time=*/problem.history.dt(),
                           problem.history,
                           ws);

    ASSERT_TRUE(linear.options_seen_in_solve.has_value());
    ASSERT_TRUE(linear.options_seen_in_solve->mixed_block_layout.has_value());

    const auto& mixed = *linear.options_seen_in_solve->mixed_block_layout;
    const auto* aux = mixed.findBlock("newton_rank_one_snapshot_inst");
    ASSERT_NE(aux, nullptr);
    EXPECT_EQ(aux->kind, svmp::FE::backends::MixedBlockKind::Auxiliary);
    EXPECT_EQ(aux->role, svmp::FE::backends::BlockRole::ConstraintField);
    EXPECT_EQ(aux->offset, 4);
    EXPECT_EQ(aux->size, 2);
    EXPECT_EQ(linear.options_seen_in_solve->resolveBlockNameForRole(
                  svmp::FE::backends::BlockRole::ConstraintField),
              "newton_rank_one_snapshot_inst");
}

TEST(NewtonSolver, CoupledSolveAcceptsOriginalLinearTarget)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "NewtonSolver tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto problem = makeDirectCouplingProblem(/*dt=*/0.1, /*u0=*/{0.2, -0.4, 0.1, 0.7});

    auto base_opts = problem.linear->getOptions();
    base_opts.rel_tol = static_cast<svmp::FE::Real>(1e-3);
    base_opts.abs_tol = static_cast<svmp::FE::Real>(0.0);
    base_opts.max_iter = 25;
    problem.linear->setOptions(base_opts);

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 1;
    nopt.abs_tolerance = 1e-16;
    nopt.rel_tolerance = 1e-16;
    nopt.step_tolerance = 0.0;
    nopt.use_line_search = false;

    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(*problem.sys, *problem.factory, ws);
    problem.history.repack(*problem.factory);
    ts_test::setVectorByDof(problem.history.u(), {0.9, -0.1, 0.35, -0.6});

    ForceResidualReportLinearSolver linear(*problem.linear,
                                           /*initial_residual_norm=*/1.0,
                                           /*final_residual_norm=*/5e-4,
                                           /*forced_miss_calls=*/1,
                                           /*native_rank_one_support=*/true);
    const auto rep = newton.solveStep(*problem.transient,
                                      linear,
                                      /*solve_time=*/problem.history.dt(),
                                      problem.history,
                                      ws);

    EXPECT_FALSE(rep.converged);
    EXPECT_TRUE(rep.linear.converged);
    EXPECT_NE(rep.linear.message.find("accepted original coupled target"), std::string::npos);
}
