#include "Application/Core/ApplicationDriver.h"

#include "Application/Core/OopMpiLog.h"
#include "Application/Core/SimulationBuilder.h"

#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/MeshComm.h"
#include "Parameters.h"
#include "tinyxml2.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool parse_bool_relaxed(const std::string& raw)
{
  const auto v = lower_copy(trim_copy(raw));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

std::string step_reject_reason_to_string(svmp::FE::timestepping::StepRejectReason r)
{
  using svmp::FE::timestepping::StepRejectReason;
  switch (r) {
    case StepRejectReason::NonlinearSolveFailed:
      return "NonlinearSolveFailed";
    case StepRejectReason::ErrorTooLarge:
      return "ErrorTooLarge";
  }
  return std::to_string(static_cast<int>(r));
}

const EquationParameters* first_equation(const Parameters& params)
{
  for (const auto* e : params.equation_parameters) {
    if (e) {
      return e;
    }
  }
  return nullptr;
}

class ZeroTimeDerivativeIntegrator final : public svmp::FE::systems::TimeIntegrator {
public:
  [[nodiscard]] std::string name() const override { return "ZeroTimeDerivative"; }
  [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

  [[nodiscard]] svmp::FE::assembly::TimeIntegrationContext buildContext(
      int max_time_derivative_order, const svmp::FE::systems::SystemStateView& /*state*/) const override
  {
    svmp::FE::assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = name();
    if (max_time_derivative_order <= 0) {
      return ctx;
    }

    if (max_time_derivative_order >= 1) {
      svmp::FE::assembly::TimeDerivativeStencil s;
      s.order = 1;
      s.a.assign(1, static_cast<svmp::FE::Real>(0.0));
      ctx.dt1 = std::move(s);
    }
    if (max_time_derivative_order >= 2) {
      svmp::FE::assembly::TimeDerivativeStencil s;
      s.order = 2;
      s.a.assign(1, static_cast<svmp::FE::Real>(0.0));
      ctx.dt2 = std::move(s);
    }

    return ctx;
  }
};

} // namespace

namespace application {
namespace core {

struct VtkTimeSeriesCollection {
  struct Entry {
    double time{};
    std::string file{};
  };

  std::filesystem::path pvd_path{};
  std::vector<Entry> entries{};
};

namespace {

void write_pvd_collection(const VtkTimeSeriesCollection& pvd)
{
  if (pvd.pvd_path.empty() || pvd.entries.empty()) {
    return;
  }

  std::ofstream out(pvd.pvd_path);
  if (!out.is_open()) {
    throw std::runtime_error("[svMultiPhysics::Application] Failed to open PVD file '" + pvd.pvd_path.string() + "'.");
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <Collection>\n";

  out << std::setprecision(16) << std::fixed;
  for (const auto& e : pvd.entries) {
    out << "    <DataSet timestep=\"" << e.time << "\" group=\"\" part=\"0\" file=\"" << e.file << "\"/>\n";
  }

  out << "  </Collection>\n";
  out << "</VTKFile>\n";
}

} // namespace

bool ApplicationDriver::shouldUseNewSolver(const std::string& xml_file)
{
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(xml_file.c_str()) != tinyxml2::XML_SUCCESS) {
    return false;
  }

  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (!root) {
    return false;
  }

  auto* general = root->FirstChildElement("GeneralSimulationParameters");
  if (!general) {
    return false;
  }

  auto* flag_elem = general->FirstChildElement("Use_new_OOP_solver");
  if (!flag_elem || !flag_elem->GetText()) {
    return false;
  }

  return parse_bool_relaxed(flag_elem->GetText());
}

void ApplicationDriver::run(const std::string& xml_file)
{
  oopCout() << "[svMultiPhysics::Application] ApplicationDriver::run(xml_file='" << xml_file << "')" << std::endl;
  if (!xml_file.empty()) {
    std::error_code ec;
    const auto abs = std::filesystem::absolute(xml_file, ec);
    if (!ec) {
      oopCout() << "[svMultiPhysics::Application] XML path: " << abs.string() << std::endl;
    }
  }

  Parameters params;
  params.read_xml(xml_file);
  runWithParameters(params);
}

void ApplicationDriver::runWithParameters(const Parameters& params)
{
  const auto comm = svmp::MeshComm::world();
  if (comm.is_parallel() && comm.rank() == 0 && !oopTraceEnabled()) {
    oopCout() << "[svMultiPhysics::Application] MPI ranks=" << comm.size()
              << "; suppressing non-root log output (set SVMP_OOP_SOLVER_TRACE=1 for per-rank logs)." << std::endl;
  }

  oopCout() << "[svMultiPhysics::Application] <Use_new_OOP_solver>=true; running new OOP solver path." << std::endl;
  oopCout()
      << "[svMultiPhysics::Application] Supported (initial): equation types heatS/heatF (Poisson), fluid, stokes; "
         "single <Add_mesh>; steady constant BCs; transient time loop (Generalized-α)."
      << std::endl;
  oopCout() << "[svMultiPhysics::Application] Not supported yet: Domain_file_path, multiple domains, "
               "spatial/temporal BC files, profiles/imposed flux, restart/continuation, FSI/etc. "
               "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver."
            << std::endl;

  const auto mesh_blocks =
      static_cast<int>(std::count_if(params.mesh_parameters.begin(), params.mesh_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  const auto equation_blocks =
      static_cast<int>(std::count_if(params.equation_parameters.begin(), params.equation_parameters.end(),
                                     [](const auto* p) { return p != nullptr; }));
  oopCout() << "[svMultiPhysics::Application] Input summary: meshes=" << mesh_blocks
            << " equations=" << equation_blocks << std::endl;

  SimulationBuilder builder(params);
  auto sim = builder.build();

  if (sim.primary_mesh) {
    const auto global_verts = sim.primary_mesh->global_n_vertices();
    const auto global_cells = sim.primary_mesh->global_n_cells();
    const auto global_faces = sim.primary_mesh->global_n_faces();

    oopCout() << "[svMultiPhysics::Application] Loaded mesh '" << sim.primary_mesh_name << "' (global): "
              << global_verts << " vertices, " << global_cells << " cells, " << global_faces << " faces."
              << std::endl;
  } else {
    oopCout() << "[svMultiPhysics::Application] No meshes were loaded from <Add_mesh>." << std::endl;
  }

  oopCout() << "[svMultiPhysics::Application] Components: fe_system=" << (sim.fe_system ? "yes" : "no")
            << " physics_modules=" << static_cast<int>(sim.physics_modules.size())
            << " backend=" << (sim.backend ? svmp::FE::backends::backendKindToString(sim.backend->backendKind()) : "none")
            << " linear_solver=" << (sim.linear_solver ? svmp::FE::backends::backendKindToString(sim.linear_solver->backendKind()) : "none")
            << " time_history=" << (sim.time_history ? "yes" : "no") << std::endl;
  if (sim.fe_system) {
    oopCout() << "[svMultiPhysics::Application] FE system: ndofs=" << sim.fe_system->dofHandler().getNumDofs()
              << " constraints=" << sim.fe_system->constraints().numConstraints() << std::endl;
  }
  if (sim.time_history) {
    oopCout() << "[svMultiPhysics::Application] TimeHistory: time=" << sim.time_history->time()
              << " dt=" << sim.time_history->dt() << " step=" << sim.time_history->stepIndex()
              << " depth=" << sim.time_history->historyDepth() << std::endl;
  }

  const int num_steps = params.general_simulation_parameters.number_of_time_steps.value();
  const double dt = params.general_simulation_parameters.time_step_size.value();
  oopCout() << "[svMultiPhysics::Application] Time stepping: Number_of_time_steps=" << num_steps
            << " Time_step_size=" << dt << std::endl;

  VtkTimeSeriesCollection pvd{};
  VtkTimeSeriesCollection* pvd_ptr = nullptr;
  if (params.general_simulation_parameters.combine_time_series.defined() &&
      params.general_simulation_parameters.combine_time_series.value()) {
    const bool vtk_enabled = params.general_simulation_parameters.save_results_to_vtk_format.defined() &&
                             params.general_simulation_parameters.save_results_to_vtk_format.value();
    if (!vtk_enabled) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] <Combine_time_series> is only implemented for VTK output "
          "(.pvd collection file). Other output formats are not implemented.");
    }

    std::filesystem::path out_dir = ".";
    if (params.general_simulation_parameters.save_results_in_folder.defined() &&
        !params.general_simulation_parameters.save_results_in_folder.value().empty()) {
      out_dir = params.general_simulation_parameters.save_results_in_folder.value();
    }

    std::string prefix = "result";
    if (params.general_simulation_parameters.name_prefix_of_saved_vtk_files.defined() &&
        !params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value().empty()) {
      prefix = params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value();
    }

    pvd.pvd_path = out_dir / (prefix + ".pvd");
    pvd_ptr = &pvd;
  }

  if (num_steps <= 1) {
    oopCout() << "[svMultiPhysics::Application] Starting steady-state solve." << std::endl;
    runSteadyState(sim, params, pvd_ptr);
  } else {
    oopCout() << "[svMultiPhysics::Application] Starting transient solve." << std::endl;
    runTransient(sim, params, pvd_ptr);
  }
}

void ApplicationDriver::runSteadyState(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd)
{
  if (!sim.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] Steady solve requires an FE system.");
  }
  if (!sim.backend || !sim.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Steady solve requires a backend + linear solver.");
  }
  if (!sim.time_history) {
    throw std::runtime_error("[svMultiPhysics::Application] Steady solve requires a TimeHistory.");
  }

  if (params.general_simulation_parameters.continue_previous_simulation.defined() &&
      params.general_simulation_parameters.continue_previous_simulation.value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Continue_previous_simulation> is not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  svmp::FE::timestepping::NewtonOptions newton_opts{};
  if (const auto* eq = first_equation(params)) {
    if (eq->max_iterations.defined()) {
      newton_opts.max_iterations = eq->max_iterations.value();
    }
    if (eq->tolerance.defined()) {
      const double tol = eq->tolerance.value();
      if (tol > 0.0) {
        // Legacy semantics: <Add_equation><Tolerance> is a *relative* tolerance (see eqType::tol).
        //
        // The FE Newton driver checks both absolute and relative tolerances, so use a very loose
        // absolute tolerance here and let the relative criterion control convergence.
        newton_opts.rel_tolerance = tol;
        newton_opts.abs_tolerance = 1.0e30;
      }
    }
  }

  // Use the unified "equations" operator tag (same as transient).
  newton_opts.residual_op = "equations";
  newton_opts.jacobian_op = "equations";
  newton_opts.use_line_search = false;

  const auto integrator = std::make_shared<const ZeroTimeDerivativeIntegrator>();
  svmp::FE::systems::TransientSystem transient(*sim.fe_system, integrator);

  svmp::FE::timestepping::NewtonSolver newton(newton_opts);
  svmp::FE::timestepping::NewtonWorkspace workspace;

  oopCout() << "[svMultiPhysics::Application] Steady: allocating Newton workspace." << std::endl;
  newton.allocateWorkspace(transient.system(), *sim.backend, workspace);
  oopCout() << "[svMultiPhysics::Application] Steady: Newton workspace allocated." << std::endl;

  // Ensure time-history vectors use the same backend layout as the Newton workspace.
  oopCout() << "[svMultiPhysics::Application] Steady: repacking TimeHistory for backend layout." << std::endl;
  sim.time_history->repack(*sim.backend);
  oopCout() << "[svMultiPhysics::Application] Steady: TimeHistory repacked." << std::endl;

  const double solve_time = sim.time_history->time();
  oopCout() << "[svMultiPhysics::Application] Steady solve: time=" << solve_time
            << " newton(max_it=" << newton_opts.max_iterations << ", abs_tol=" << newton_opts.abs_tolerance
            << ", rel_tol=" << newton_opts.rel_tolerance << ")" << std::endl;

  const auto report = newton.solveStep(transient, *sim.linear_solver, solve_time, *sim.time_history, workspace);
  oopCout() << "[svMultiPhysics::Application] Steady Newton: converged=" << report.converged
            << " iterations=" << report.iterations << " residual_norm=" << report.residual_norm << std::endl;

  if (!report.converged) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Steady solve did not converge (Newton reached max iterations). "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  sim.fe_system->commitTimeStep();
  outputResults(sim, params, /*step=*/1, solve_time, pvd);

  const auto comm = svmp::MeshComm::world();
  if (pvd && comm.rank() == 0) {
    write_pvd_collection(*pvd);
    if (!pvd->entries.empty()) {
      oopCout() << "[svMultiPhysics::Application] Wrote PVD: " << pvd->pvd_path.string() << std::endl;
    }
  }
}

void ApplicationDriver::runTransient(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd)
{
  if (!sim.fe_system) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires an FE system.");
  }
  if (!sim.backend || !sim.linear_solver) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires a backend + linear solver.");
  }
  if (!sim.time_history) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires a TimeHistory.");
  }

  if (params.general_simulation_parameters.continue_previous_simulation.defined() &&
      params.general_simulation_parameters.continue_previous_simulation.value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] <Continue_previous_simulation> is not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const int num_steps = params.general_simulation_parameters.number_of_time_steps.value();
  if (num_steps <= 1) {
    throw std::runtime_error("[svMultiPhysics::Application] runTransient() requires Number_of_time_steps > 1.");
  }

  const double dt = sim.time_history->dt();
  if (!(dt > 0.0)) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve requires Time_step_size > 0.");
  }

  svmp::FE::timestepping::TimeLoopOptions opts{};
  opts.t0 = 0.0;
  opts.dt = dt;
  opts.t_end = static_cast<double>(num_steps) * dt;
  opts.max_steps = num_steps;
  opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
  if (params.general_simulation_parameters.spectral_radius_of_infinite_time_step.defined()) {
    opts.generalized_alpha_rho_inf = params.general_simulation_parameters.spectral_radius_of_infinite_time_step.value();
  }

  if (const auto* eq = first_equation(params)) {
    if (eq->max_iterations.defined()) {
      opts.newton.max_iterations = eq->max_iterations.value();
    }
    if (eq->tolerance.defined()) {
      const double tol = eq->tolerance.value();
      if (tol > 0.0) {
        // Legacy semantics: <Add_equation><Tolerance> is a *relative* tolerance (see eqType::tol).
        //
        // The FE Newton driver checks both absolute and relative tolerances, so use a very loose
        // absolute tolerance here and let the relative criterion control convergence.
        opts.newton.rel_tolerance = tol;
        opts.newton.abs_tolerance = 1.0e30;
      }
    }
  }
  // NOTE: Newton update scaling for dt(·) fields is available in the FE library
  // (NewtonOptions::scale_dt_increments), but enabling it globally can severely
  // slow convergence for linear problems (e.g., Stokes). Keep it off by default.
  opts.newton.scale_dt_increments = false;

  // Use the unified "equations" operator tag installed by the NS module (and
  // Poisson, etc.). Setting both tags equal enables same_op=true in NewtonSolver,
  // which uses the combined assembleJacobianAndResidual() path and reduces the
  // number of mesh traversals per Newton iteration.
  opts.newton.residual_op = "equations";
  opts.newton.jacobian_op = "equations";

  // Disable backtracking line search to match legacy solver behavior.  The legacy
  // solver applies a full Newton update without line search.  Disabling it avoids
  // 2 extra residual assembly passes per Newton iteration.
  opts.newton.use_line_search = false;

  oopCout() << "[svMultiPhysics::Application] Transient solve: t0=" << opts.t0 << " dt=" << opts.dt
            << " t_end=" << opts.t_end << " max_steps=" << opts.max_steps
            << " scheme=GeneralizedAlpha rho_inf=" << opts.generalized_alpha_rho_inf
            << " newton(max_it=" << opts.newton.max_iterations << ", abs_tol=" << opts.newton.abs_tolerance
            << ", rel_tol=" << opts.newton.rel_tolerance << ")" << std::endl;

  // Ensure time-history vectors use the same backend layout as the solver workspace.
  oopCout() << "[svMultiPhysics::Application] Transient: repacking TimeHistory for backend layout." << std::endl;
  sim.time_history->repack(*sim.backend);
  oopCout() << "[svMultiPhysics::Application] Transient: TimeHistory repacked." << std::endl;

  auto bdf1 = std::make_shared<const svmp::FE::systems::BDFIntegrator>(1);
  svmp::FE::systems::TransientSystem transient(*sim.fe_system, std::move(bdf1));

  svmp::FE::timestepping::TimeLoopCallbacks callbacks{};
  callbacks.on_step_start = [&](const svmp::FE::timestepping::TimeHistory& h) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_start step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
  };
  callbacks.on_nonlinear_done = [&](const svmp::FE::timestepping::TimeHistory& h,
                                   const svmp::FE::timestepping::NewtonReport& nr) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: nonlinear_done step=" << h.stepIndex()
              << " time=" << h.time() << " converged=" << nr.converged
              << " iters=" << nr.iterations << " ||r||=" << nr.residual_norm
              << " (linear: converged=" << nr.linear.converged
              << " iters=" << nr.linear.iterations
              << " rel=" << nr.linear.relative_residual << ")" << std::endl;
  };
  callbacks.on_step_accepted = [&](const svmp::FE::timestepping::TimeHistory& h) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_accepted step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << std::endl;
    outputResults(sim, params, h.stepIndex(), h.time(), pvd);
  };
  callbacks.on_step_rejected = [&](const svmp::FE::timestepping::TimeHistory& h,
                                  svmp::FE::timestepping::StepRejectReason reason,
                                  const svmp::FE::timestepping::NewtonReport& nr) {
    oopCout() << "[svMultiPhysics::Application] TimeLoop: step_rejected step=" << h.stepIndex()
              << " time=" << h.time() << " dt=" << h.dt() << " reason=" << step_reject_reason_to_string(reason)
              << " (newton: converged=" << nr.converged << " iters=" << nr.iterations
              << " ||r||=" << nr.residual_norm << ")" << std::endl;
  };
  callbacks.on_dt_updated = [&](double old_dt, double new_dt, int step_index, int attempt_index) {
    if (!oopTraceEnabled()) {
      return;
    }
    oopCout() << "[svMultiPhysics::Application] TimeLoop: dt_updated step=" << step_index
              << " attempt=" << attempt_index << " old_dt=" << old_dt << " new_dt=" << new_dt << std::endl;
  };

  svmp::FE::timestepping::TimeLoop loop(opts);
  oopCout() << "[svMultiPhysics::Application] TimeLoop: entering loop.run()" << std::endl;
  const auto rep = loop.run(transient, *sim.backend, *sim.linear_solver, *sim.time_history, callbacks);
  oopCout() << "[svMultiPhysics::Application] TimeLoop: loop.run() returned success=" << rep.success
            << " steps_taken=" << rep.steps_taken << " final_time=" << rep.final_time
            << " message='" << rep.message << "'" << std::endl;

  if (!rep.success) {
    throw std::runtime_error("[svMultiPhysics::Application] Transient solve failed: " + rep.message +
                             ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto comm = svmp::MeshComm::world();
  if (pvd && comm.rank() == 0) {
    write_pvd_collection(*pvd);
    if (!pvd->entries.empty()) {
      oopCout() << "[svMultiPhysics::Application] Wrote PVD: " << pvd->pvd_path.string() << std::endl;
    }
  }
}

void ApplicationDriver::outputResults(const SimulationComponents& sim, const Parameters& params, int step, double time,
                                      VtkTimeSeriesCollection* pvd)
{
  if (!params.general_simulation_parameters.save_results_to_vtk_format.defined() ||
      !params.general_simulation_parameters.save_results_to_vtk_format.value()) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: disabled (<Save_results_to_VTK_format>=false)." << std::endl;
    }
    return;
  }

  if (!sim.primary_mesh || !sim.fe_system || !sim.time_history) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: missing mesh/system/history; skipping." << std::endl;
    }
    return;
  }

  const int save_incr = std::max(1, params.general_simulation_parameters.increment_in_saving_vtk_files.value());
  const int save_ats = std::max(0, params.general_simulation_parameters.start_saving_after_time_step.value());
  if (step < save_ats || (step % save_incr) != 0) {
    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: skipping step=" << step
                << " (start_after=" << save_ats << " increment=" << save_incr << ")" << std::endl;
    }
    return;
  }

  const auto comm = svmp::MeshComm::world();
  const bool is_root = (comm.rank() == 0);
  const bool mpi_parallel = (comm.size() > 1);

  std::filesystem::path out_dir = ".";
  if (params.general_simulation_parameters.save_results_in_folder.defined() &&
      !params.general_simulation_parameters.save_results_in_folder.value().empty()) {
    out_dir = params.general_simulation_parameters.save_results_in_folder.value();
  }
  std::filesystem::create_directories(out_dir);

  std::string prefix = "result";
  if (params.general_simulation_parameters.name_prefix_of_saved_vtk_files.defined() &&
      !params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value().empty()) {
    prefix = params.general_simulation_parameters.name_prefix_of_saved_vtk_files.value();
  }

  std::ostringstream fname;
  fname << prefix << "_" << std::setw(3) << std::setfill('0') << step << (mpi_parallel ? ".pvtu" : ".vtu");
  const auto out_path = out_dir / fname.str();
  if (oopTraceEnabled() && is_root) {
    oopCout() << "[svMultiPhysics::Application] VTK output: begin step=" << step << " time=" << time
              << " file='" << out_path.string() << "'" << std::endl;
  }

  svmp::FE::systems::SystemStateView state;
  state.time = time;
  state.dt = sim.time_history->dt();
  state.dt_prev = sim.time_history->dtPrev();
  state.u = sim.time_history->uSpan();
  state.u_prev = sim.time_history->uPrevSpan();
  state.u_prev2 = sim.time_history->uPrev2Span();
  state.u_vector = &sim.time_history->u();
  state.u_prev_vector = &sim.time_history->uPrev();
  state.u_prev2_vector = &sim.time_history->uPrev2();
  state.u_history = sim.time_history->uHistorySpans();
  state.dt_history = sim.time_history->dtHistory();

  auto& mesh = *sim.primary_mesh;
  const int mesh_dim = mesh.dim();
  const auto& coords = mesh.X_ref();

  const auto ensure_point_field = [&](const std::string& name, std::size_t components) -> svmp::FieldHandle {
    if (mesh.has_field(svmp::EntityKind::Vertex, name)) {
      auto h = mesh.field_handle(svmp::EntityKind::Vertex, name);
      if (mesh.field_type(h) == svmp::FieldScalarType::Float64 && mesh.field_components(h) == components) {
        return h;
      }
      mesh.remove_field(h);
    }
    return mesh.attach_field(svmp::EntityKind::Vertex, name, svmp::FieldScalarType::Float64, components);
  };

  const auto n_fields = sim.fe_system->fieldMap().numFields();
  for (std::size_t i = 0; i < n_fields; ++i) {
    const auto field_id = static_cast<svmp::FE::FieldId>(i);
    const auto& rec = sim.fe_system->fieldRecord(field_id);
    const auto ncomp = static_cast<std::size_t>(std::max(1, rec.components));

    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: evaluating field '" << rec.name
                << "' components=" << ncomp << std::endl;
    }

    auto h = ensure_point_field(rec.name, ncomp);
    auto* data = static_cast<double*>(mesh.field_data(h));
    if (!data) {
      throw std::runtime_error("[svMultiPhysics::Application] Failed to allocate VTK field '" + rec.name + "'.");
    }

    for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
      std::array<svmp::FE::Real, 3> p{0.0, 0.0, 0.0};
      for (int d = 0; d < mesh_dim; ++d) {
        p[static_cast<std::size_t>(d)] = static_cast<svmp::FE::Real>(coords[v * static_cast<std::size_t>(mesh_dim) +
                                                                           static_cast<std::size_t>(d)]);
      }

      const auto val = sim.fe_system->evaluateFieldAtPoint(field_id, state, p);
      if (!val) {
        throw std::runtime_error("[svMultiPhysics::Application] Failed to evaluate field '" + rec.name +
                                 "' at a mesh vertex for VTK output.");
      }

      for (std::size_t c = 0; c < ncomp; ++c) {
        data[v * ncomp + c] = static_cast<double>((*val)[c]);
      }
    }

    if (oopTraceEnabled()) {
      oopCout() << "[svMultiPhysics::Application] VTK output: field '" << rec.name << "' done." << std::endl;
    }
  }

  svmp::MeshIOOptions io{};
  io.format = mpi_parallel ? "pvtu" : "vtu";
  io.path = out_path.string();
  mesh.save_parallel(io);

  if (is_root) {
    oopCout() << "[svMultiPhysics::Application] Wrote VTK: " << out_path.string() << std::endl;
    if (pvd && !pvd->pvd_path.empty()) {
      std::error_code ec;
      auto rel = std::filesystem::relative(out_path, pvd->pvd_path.parent_path(), ec);
      if (ec) {
        rel = out_path.filename();
      }
      const std::string rel_file = rel.generic_string();
      if (pvd->entries.empty() || pvd->entries.back().file != rel_file) {
        pvd->entries.push_back(VtkTimeSeriesCollection::Entry{time, rel_file});
      }
    }
  }
  if (oopTraceEnabled() && is_root) {
    oopCout() << "[svMultiPhysics::Application] VTK output: done step=" << step << " time=" << time << std::endl;
  }
}

} // namespace core
} // namespace application
