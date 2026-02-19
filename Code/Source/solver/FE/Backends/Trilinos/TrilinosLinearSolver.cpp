/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Trilinos/TrilinosLinearSolver.h"

#if defined(FE_HAS_TRILINOS)

#include "Backends/Trilinos/TrilinosMatrix.h"
#include "Backends/Trilinos/TrilinosVector.h"
#include "Backends/Trilinos/TrilinosUtils.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>
#include <Ifpack2_Factory.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#if defined(__has_include)
  #if __has_include(<MueLu_CreateTpetraPreconditioner.hpp>)
    #define SVMP_FE_TRILINOS_HAS_MUELU 1
    #include <MueLu_CreateTpetraPreconditioner.hpp>
  #else
    #define SVMP_FE_TRILINOS_HAS_MUELU 0
  #endif
#else
  #define SVMP_FE_TRILINOS_HAS_MUELU 0
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <sstream>

namespace svmp {
namespace FE {
namespace backends {

TrilinosLinearSolver::TrilinosLinearSolver(const SolverOptions& options)
{
    setOptions(options);
}

void TrilinosLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "TrilinosLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "TrilinosLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "TrilinosLinearSolver: abs_tol must be >= 0");
    options_ = options;
}

namespace {

using MV = Tpetra::MultiVector<trilinos::Scalar, trilinos::LO, trilinos::GO, trilinos::Node>;
using OP = Tpetra::Operator<trilinos::Scalar, trilinos::LO, trilinos::GO, trilinos::Node>;
using RowMatrix = Tpetra::RowMatrix<trilinos::Scalar, trilinos::LO, trilinos::GO, trilinos::Node>;

[[nodiscard]] bool oopTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

[[nodiscard]] std::string belosSolverName(SolverMethod method)
{
    switch (method) {
        case SolverMethod::GMRES:
        case SolverMethod::PGMRES:
        case SolverMethod::FGMRES: return "PseudoBlockGmres";
        case SolverMethod::CG: return "CG";
        case SolverMethod::BiCGSTAB: return "BiCGStab";
        case SolverMethod::BlockSchur:
            FE_THROW(NotImplementedException, "TrilinosLinearSolver: BlockSchur solves not implemented (use PETSc for now)");
        case SolverMethod::Direct:
        default:
            FE_THROW(NotImplementedException, "TrilinosLinearSolver: direct solves not implemented (use PETSc for now)");
    }
}

[[nodiscard]] Teuchos::RCP<Teuchos::ParameterList> buildBelosParams(const SolverOptions& options)
{
    auto params = Teuchos::parameterList();
    params->set("Maximum Iterations", options.max_iter);
    params->set("Convergence Tolerance", options.rel_tol);

    for (const auto& [k, v] : options.passthrough) {
        if (k.empty()) continue;
        params->set(k, v);
    }

    if (!options.trilinos_xml_file.empty()) {
        Teuchos::updateParametersFromXmlFile(options.trilinos_xml_file, params.ptr());
    }

    return params;
}

[[nodiscard]] Teuchos::RCP<Ifpack2::Preconditioner<trilinos::Scalar, trilinos::LO, trilinos::GO, trilinos::Node>>
buildIfpackPreconditioner(const SolverOptions& options,
                          const Teuchos::RCP<const RowMatrix>& A)
{
    if (options.preconditioner == PreconditionerType::None) {
        return Teuchos::null;
    }

    Ifpack2::Factory factory;
    std::string prec_name = "RELAXATION";
    switch (options.preconditioner) {
        case PreconditionerType::Diagonal:
        case PreconditionerType::RowColumnScaling:
            prec_name = "RELAXATION";
            break;
        case PreconditionerType::ILU:
            // Prefer a simple ILU-style preconditioner when available.
            // (Exact support depends on the Trilinos build; this may throw at runtime.)
            prec_name = "ILUT";
            break;
        case PreconditionerType::AMG:
            return Teuchos::null; // Handled separately (MueLu) when available.
        case PreconditionerType::FieldSplit:
            FE_THROW(NotImplementedException, "TrilinosLinearSolver: field-split preconditioning not implemented");
        case PreconditionerType::None:
            return Teuchos::null;
    }

    Teuchos::RCP<Ifpack2::Preconditioner<trilinos::Scalar, trilinos::LO, trilinos::GO, trilinos::Node>> prec;
    try {
        prec = factory.create(prec_name, A);
    } catch (const std::exception& e) {
        FE_THROW(FEException, std::string("TrilinosLinearSolver: failed to create Ifpack2 preconditioner '") +
                                  prec_name + "': " + e.what());
    }

    Teuchos::ParameterList pl;
    if (prec_name == "RELAXATION") {
        pl.set("relaxation: type", "Jacobi");
        pl.set("relaxation: sweeps", 1);
        pl.set("relaxation: damping factor", 1.0);
    }

    prec->setParameters(pl);
    prec->initialize();
    prec->compute();
    return prec;
}

} // namespace

SolverReport TrilinosLinearSolver::solve(const GenericMatrix& A_in,
                                         GenericVector& x_in,
                                         const GenericVector& b_in)
{
    const auto* A = dynamic_cast<const TrilinosMatrix*>(&A_in);
    auto* x = dynamic_cast<TrilinosVector*>(&x_in);
    const auto* b = dynamic_cast<const TrilinosVector*>(&b_in);

    FE_THROW_IF(!A || !x || !b, InvalidArgumentException, "TrilinosLinearSolver::solve: backend mismatch");
    FE_THROW_IF(options_.preconditioner == PreconditionerType::FieldSplit, NotImplementedException,
                "TrilinosLinearSolver::solve: field-split preconditioning not implemented");
    FE_THROW_IF(A->numRows() != A->numCols(), NotImplementedException,
                "TrilinosLinearSolver::solve: rectangular systems not implemented");
    FE_THROW_IF(b->size() != A->numRows() || x->size() != b->size(), InvalidArgumentException,
                "TrilinosLinearSolver::solve: size mismatch");

    if (!options_.use_initial_guess) {
        x->zero();
    }

    // Compute initial residual.
    SolverReport rep;
    {
        trilinos::Vector r(x->map());
        r.update(1.0, *b->tpetra(), 0.0);
        trilinos::Vector Ax(x->map());
        A->tpetra()->apply(*x->tpetra(), Ax);
        r.update(-1.0, Ax, 1.0);
        rep.initial_residual_norm = static_cast<Real>(r.norm2());
    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "TrilinosLinearSolver::solve: n=" << A->numRows()
            << " method=" << solverMethodToString(options_.method)
            << " pc=" << preconditionerToString(options_.preconditioner)
            << " rel_tol=" << options_.rel_tol
            << " abs_tol=" << options_.abs_tol
            << " max_iter=" << options_.max_iter
            << " use_initial_guess=" << (options_.use_initial_guess ? 1 : 0);
        if (!options_.trilinos_xml_file.empty()) {
            oss << " xml='" << options_.trilinos_xml_file << "'";
        }
        oss << " r0=" << rep.initial_residual_norm;
        traceLog(oss.str());
    }

    const auto belos_params = buildBelosParams(options_);
    Belos::SolverFactory<trilinos::Scalar, MV, OP> factory;
    const auto solver_name = belosSolverName(options_.method);
    auto solver = factory.create(solver_name, belos_params);

    auto problem = Teuchos::rcp(new Belos::LinearProblem<trilinos::Scalar, MV, OP>(A->tpetra(), x->tpetra(), b->tpetra()));

    if (options_.preconditioner == PreconditionerType::AMG) {
#if SVMP_FE_TRILINOS_HAS_MUELU
        auto rowMat = Teuchos::rcp_dynamic_cast<RowMatrix>(A->tpetra());
        FE_THROW_IF(rowMat.is_null(), FEException, "TrilinosLinearSolver: AMG requested but matrix is not a RowMatrix");
        Teuchos::ParameterList muelu_params;
        auto prec = MueLu::CreateTpetraPreconditioner<trilinos::Scalar, trilinos::LO, trilinos::GO, trilinos::Node>(rowMat, muelu_params);
        problem->setRightPrec(prec);
#else
        FE_THROW(NotImplementedException, "TrilinosLinearSolver: AMG requested but MueLu is not available in this Trilinos build");
#endif
    } else {
        const auto rowMat = Teuchos::rcp_dynamic_cast<const RowMatrix>(A->tpetra());
        const auto prec = buildIfpackPreconditioner(options_, rowMat);
        if (!prec.is_null()) {
            problem->setRightPrec(Teuchos::rcp_dynamic_cast<const OP>(prec));
        }
    }

    const bool ok = problem->setProblem();
    FE_THROW_IF(!ok, FEException, "TrilinosLinearSolver: failed to set up Belos problem");
    solver->setProblem(problem);

    const auto ret = solver->solve();
    rep.converged = (ret == Belos::Converged);
    rep.iterations = static_cast<int>(solver->getNumIters());
    x->invalidateLocalCache();

    // Final residual.
    {
        trilinos::Vector r(x->map());
        r.update(1.0, *b->tpetra(), 0.0);
        trilinos::Vector Ax(x->map());
        A->tpetra()->apply(*x->tpetra(), Ax);
        r.update(-1.0, Ax, 1.0);
        rep.final_residual_norm = static_cast<Real>(r.norm2());
    }
    const Real denom = std::max<Real>(rep.initial_residual_norm, 1e-30);
    rep.relative_residual = rep.final_residual_norm / denom;
    rep.message = "trilinos/belos";

    // Belos uses rel tol; apply abs tol if provided.
    if (options_.abs_tol > 0.0) {
        rep.converged = rep.converged || (rep.final_residual_norm <= options_.abs_tol);
    }

    if (oopTraceEnabled()) {
        std::ostringstream oss;
        oss << "TrilinosLinearSolver::solve: converged=" << (rep.converged ? 1 : 0)
            << " iters=" << rep.iterations
            << " rn=" << rep.final_residual_norm
            << " rel=" << rep.relative_residual
            << " msg='" << rep.message << "'";
        traceLog(oss.str());
    }

    return rep;
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS
