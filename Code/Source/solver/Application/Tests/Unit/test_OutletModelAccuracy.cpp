#include <gtest/gtest.h>

#include "Application/Core/ApplicationDriver.h"

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#include <mpi.h>
#endif

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

// The RCR outlet of the pipe_RCR_3d deck against the exact solution of its
// 0D model.  The pipe is rigid and the flow incompressible, so the outlet
// flow equals the imposed inflow at every instant, and the capacitor
// pressure X of C dX/dt = Q - (X - Pd)/Rd has a closed form for an inflow
// that rises linearly to Q1 over [0, T1] and then holds.

namespace {

namespace fs = std::filesystem;

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
void finalizeMpiIfNeeded()
{
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}
#endif

void ensureMpiInitialized()
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    std::atexit(finalizeMpiIfNeeded);
  }
#endif
}

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const fs::path& path) : previous_(fs::current_path()) { fs::current_path(path); }
  ~ScopedCurrentPath() { fs::current_path(previous_); }
  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
  fs::path previous_;
};

struct HistoryRow {
  double time{0.0};
  double Q{0.0};
  double X{0.0};
  double P_out{0.0};
};

std::vector<HistoryRow> readHistory(const fs::path& path)
{
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  EXPECT_EQ(line, "step,time,rcr_windkessel_b2.Q,rcr_windkessel_b2.X,rcr_windkessel_b2.P_out");
  std::vector<HistoryRow> rows;
  while (std::getline(in, line)) {
    std::istringstream cells(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(cells, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    if (values.size() == 5u) {
      rows.push_back({values[1], values[2], values[3], values[4]});
    }
  }
  return rows;
}

} // namespace

TEST(OutletModelAccuracy, PipeRCRMatchesTheExactWindkesselResponse)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensureMpiInitialized();
  const fs::path deck_dir = fs::path(SVMP_REPOSITORY_ROOT) / "tests" / "cases" / "fluid" / "pipe_RCR_3d";
  ASSERT_TRUE(fs::exists(deck_dir / "mesh" / "mesh-complete.mesh.vtu"));
  const fs::path work = fs::temp_directory_path() / ("svmp_outlet_model_accuracy_" + std::to_string(::getpid()));
  fs::remove_all(work);
  fs::create_directories(work);

  constexpr double Rp = 121.0, C = 1.5e-5, Rd = 1212.0, Pd = 0.0;
  constexpr double Q1 = 10.0, T1 = 0.1, dt = 0.005;
  constexpr int steps = 40;
  {
    // Inflow (outward flux) -Q1 t / T1 up to T1, then -Q1.
    std::ofstream flow(work / "ramp_hold.flow");
    flow << "3 1\n0.0 0.0\n" << T1 << " " << -Q1 << "\n1.0 " << -Q1 << "\n";
  }
  {
    const auto mesh = (deck_dir / "mesh").string();
    std::ofstream xml(work / "solver.xml");
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<svMultiPhysicsFile version=\"0.1\">\n"
        << "<GeneralSimulationParameters>\n"
        << "  <Use_new_OOP_solver>true</Use_new_OOP_solver>\n"
        << "  <Continue_previous_simulation> false </Continue_previous_simulation>\n"
        << "  <Number_of_spatial_dimensions> 3 </Number_of_spatial_dimensions>\n"
        << "  <Number_of_time_steps> " << steps << " </Number_of_time_steps>\n"
        << "  <Time_step_size> " << dt << " </Time_step_size>\n"
        << "  <Spectral_radius_of_infinite_time_step> 0.50 </Spectral_radius_of_infinite_time_step>\n"
        << "  <Searched_file_name_to_trigger_stop> STOP_SIM </Searched_file_name_to_trigger_stop>\n"
        << "  <Save_results_to_VTK_format> 0 </Save_results_to_VTK_format>\n"
        << "  <Name_prefix_of_saved_VTK_files> result </Name_prefix_of_saved_VTK_files>\n"
        << "  <Increment_in_saving_VTK_files> " << steps << " </Increment_in_saving_VTK_files>\n"
        << "  <Start_saving_after_time_step> 1 </Start_saving_after_time_step>\n"
        << "  <Increment_in_saving_restart_files> 100 </Increment_in_saving_restart_files>\n"
        << "  <Convert_BIN_to_VTK_format> 0 </Convert_BIN_to_VTK_format>\n"
        << "  <Write_scalar_history> true </Write_scalar_history>\n"
        << "  <Newton_absolute_tolerance> 1e-8 </Newton_absolute_tolerance>\n"
        << "  <Verbose> 0 </Verbose>\n  <Warning> 0 </Warning>\n  <Debug> 0 </Debug>\n"
        << "</GeneralSimulationParameters>\n"
        << "<Add_mesh name=\"msh\" >\n"
        << "  <Mesh_file_path> " << mesh << "/mesh-complete.mesh.vtu </Mesh_file_path>\n"
        << "  <Add_face name=\"lumen_inlet\"> <Face_file_path> " << mesh
        << "/mesh-surfaces/lumen_inlet.vtp </Face_file_path> </Add_face>\n"
        << "  <Add_face name=\"lumen_outlet\"> <Face_file_path> " << mesh
        << "/mesh-surfaces/lumen_outlet.vtp </Face_file_path> </Add_face>\n"
        << "  <Add_face name=\"lumen_wall\"> <Face_file_path> " << mesh
        << "/mesh-surfaces/lumen_wall.vtp </Face_file_path> </Add_face>\n"
        << "</Add_mesh>\n"
        << "<Add_equation type=\"fluid\" >\n"
        << "   <Coupled> true </Coupled>\n   <Min_iterations> 3 </Min_iterations>\n"
        << "   <Max_iterations> 15 </Max_iterations>\n   <Tolerance> 1e-9 </Tolerance>\n"
        << "   <Backflow_stabilization_coefficient> 0.2 </Backflow_stabilization_coefficient>\n"
        << "   <Density> 1.06 </Density>\n"
        << "   <Viscosity model=\"Constant\" > <Value> 0.04 </Value> </Viscosity>\n"
        << "   <LS type=\"NS\" >\n      <Linear_algebra type=\"fsils\" > <Preconditioner> fsils </Preconditioner> "
           "</Linear_algebra>\n"
        << "      <Max_iterations> 15 </Max_iterations>\n      <NS_GM_max_iterations> 10 </NS_GM_max_iterations>\n"
        << "      <NS_CG_max_iterations> 300 </NS_CG_max_iterations>\n      <Tolerance> 1e-3 </Tolerance>\n"
        << "      <NS_GM_tolerance> 1e-3 </NS_GM_tolerance>\n      <NS_CG_tolerance> 1e-3 </NS_CG_tolerance>\n"
        << "      <Absolute_tolerance> 1e-17 </Absolute_tolerance>\n"
        << "      <Krylov_space_dimension> 250 </Krylov_space_dimension>\n   </LS>\n"
        << "   <Add_BC name=\"lumen_inlet\" >\n      <Type> Dir </Type>\n"
        << "      <Time_dependence> Unsteady </Time_dependence>\n"
        << "      <Temporal_values_file_path> ramp_hold.flow </Temporal_values_file_path>\n"
        << "      <Profile> Parabolic </Profile>\n      <Impose_flux> true </Impose_flux>\n   </Add_BC>\n"
        << "   <Add_BC name=\"lumen_outlet\" >\n      <Type> Neu </Type>\n      <Time_dependence> RCR </Time_dependence>\n"
        << "      <RCR_values>\n        <Capacitance> " << C << " </Capacitance>\n"
        << "        <Distal_resistance> " << Rd << " </Distal_resistance>\n"
        << "        <Proximal_resistance> " << Rp << " </Proximal_resistance>\n"
        << "        <Distal_pressure> " << Pd << " </Distal_pressure>\n"
        << "        <Initial_pressure> 0 </Initial_pressure>\n      </RCR_values>\n   </Add_BC>\n"
        << "   <Add_BC name=\"lumen_wall\" >\n      <Type> Dir </Type>\n"
        << "      <Time_dependence> Steady </Time_dependence>\n      <Value> 0.0 </Value>\n   </Add_BC>\n"
        << "</Add_equation>\n</svMultiPhysicsFile>\n";
  }
  {
    const ScopedCurrentPath cwd(work);
    application::core::ApplicationDriver::run((work / "solver.xml").string());
  }
  const auto rows = readHistory(work / "scalar_history.csv");
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(steps));

  const double tau = Rd * C;
  const auto inflow = [&](double t) { return t < T1 ? Q1 * t / T1 : Q1; };
  const auto exactX = [&](double t) {
    const auto ramp = [&](double s) { return Pd + (Q1 / T1) * Rd * (s - tau * (1.0 - std::exp(-s / tau))); };
    return t <= T1 ? ramp(t) : Pd + Rd * Q1 + (ramp(T1) - Pd - Rd * Q1) * std::exp(-(t - T1) / tau);
  };
  // The flow rate is recorded at the generalized-alpha stage t_n + alpha_f dt
  // (alpha_f = 2/3 for rho_inf = 1/2), the capacitor pressure at the step end.
  const double alpha_f = 1.0 / 1.5;
  for (const auto& row : rows) {
    const double t_stage = row.time - (1.0 - alpha_f) * dt;
    // The imposed inflow reaches the outlet at the requested rate, to the
    // nonlinear tolerance (measured 1.2e-9 of Q1; 1.5e-2 before the inflow
    // was normalized by its nodal profile).
    EXPECT_NEAR(row.Q, inflow(t_stage), 1e-7 * Q1) << "t=" << row.time;
    // Second order in time: within 1e-3 of Rd Q1 at this step (measured
    // 4.3e-4, at the second step, from the jump of d2X/dt2 at t = 0); a
    // scheme lagging the pressure by a fraction of the step is off by
    // about dt dX/dt, 5e-2 of Rd Q1.
    EXPECT_NEAR(row.X, exactX(row.time), 1e-3 * Rd * Q1) << "t=" << row.time;
  }
  // After the ramp the flow holds, and the outlet pressure approaches
  // Pd + (Rp + Rd) Q1, here within e^{-5.5} of the capacitor's share.
  const auto& last = rows.back();
  EXPECT_NEAR(last.P_out, exactX(last.time) + Rp * Q1, 1e-3 * Rd * Q1);
  EXPECT_NEAR(last.P_out, Pd + (Rp + Rd) * Q1, 0.005 * Rd * Q1);

  std::error_code ec;
  fs::remove_all(work, ec);
#endif
}
