// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "OneDSolverInterface.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include "Core/Exception.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

OneDSolverInterface::~OneDSolverInterface()
{
  if (library_handle_) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(library_handle_));
#else
    dlclose(library_handle_);
#endif
    library_handle_ = nullptr;
  }
}

void OneDSolverInterface::load_library(const std::string& interface_lib)
{
  if (library_handle_) throw std::runtime_error("The 1D interface library is already loaded");
#ifdef _WIN32
  library_handle_ = LoadLibraryA(interface_lib.c_str());
#else
  library_handle_ = dlopen(interface_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!library_handle_) {
#ifdef _WIN32
    const auto error = std::system_category().message(GetLastError());
#else
    const char* detail = dlerror();
    const std::string error = detail ? detail : "unknown error";
#endif
    throw svmp::CoreException(
        std::string("[OneDSolverInterface] Could not load shared library '") +
        interface_lib + "': " + error,
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }

  auto load_sym = [&](const char* name) {
#ifdef _WIN32
    auto sym = GetProcAddress(static_cast<HMODULE>(library_handle_), name);
#else
    auto sym = dlsym(library_handle_, name);
#endif
    if (!sym) {
      throw svmp::CoreException(
          std::string("[OneDSolverInterface] Could not find the 1D interface function '") +
          name + "' in the shared library '" + interface_lib + "'",
          svmp::StatusCode::DependencyError,
          __FILE__,
          __LINE__,
          __func__);
    }
    return sym;
  };

  initialize_1d_ = reinterpret_cast<decltype(initialize_1d_)>(load_sym("initialize_1d"));
  set_external_step_size_1d_ = reinterpret_cast<decltype(set_external_step_size_1d_)>(load_sym("set_external_step_size_1d"));
  return_1d_solution_ = reinterpret_cast<decltype(return_1d_solution_)>(load_sym("return_1d_solution"));
  update_1d_solution_ = reinterpret_cast<decltype(update_1d_solution_)>(load_sym("update_1d_solution"));
  run_1d_simulation_step_1d_ = reinterpret_cast<decltype(run_1d_simulation_step_1d_)>(load_sym("run_1d_simulation_step_1d"));
  extract_coupled_dof_ = reinterpret_cast<decltype(extract_coupled_dof_)>(load_sym("extract_coupled_dof"));
}

void OneDSolverInterface::initialize(const std::string& input_file,
                                     int& problem_id,
                                     int& system_size,
                                     const std::string& coupling_type)
{
  if (!initialize_1d_) {
    throw svmp::CoreException(
        "[OneDSolverInterface] Cannot initialize the 1D solver because the "
        "1D interface library has not been loaded. Call load_library() with "
        "the svOneDSolver interface shared library before initialize(), and "
        "check that the library path is correct.",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }
  initialize_1d_(input_file.c_str(), problem_id, system_size,
                 coupling_type.c_str());
  problem_id_  = problem_id;
  system_size_ = system_size;
}

void OneDSolverInterface::set_external_step_size(int problem_id, double dt)
{
  if (!set_external_step_size_1d_) {
    throw svmp::CoreException(
        "[OneDSolverInterface] set_external_step_size_1d not loaded",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }
  set_external_step_size_1d_(problem_id, dt);
}

void OneDSolverInterface::return_solution(int problem_id, double* solution, int size)
{
  if (!return_1d_solution_) {
    throw svmp::CoreException(
        "[OneDSolverInterface] return_1d_solution not loaded",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }
  return_1d_solution_(problem_id, solution, size);
}

void OneDSolverInterface::update_solution(int problem_id, double* solution, int size)
{
  if (!update_1d_solution_) {
    throw svmp::CoreException(
        "[OneDSolverInterface] update_1d_solution not loaded",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }
  update_1d_solution_(problem_id, solution, size);
}

void OneDSolverInterface::run_simulation(int problem_id, double current_time,
                                   int save_incr,
                                   const std::string& coupling_type,
                                   double* params, double* solution,
                                   double& cpl_value, char last_flag,
                                   int& error_code)
{
  if (!run_1d_simulation_step_1d_) {
    throw svmp::CoreException(
        "[OneDSolverInterface] Cannot run the 1D solver because the "
        "run_1d_simulation_step_1d interface function is not available. "
        "Check that the svOneDSolver interface shared library was loaded correctly.",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }

  std::vector<char> ctype_buf(coupling_type.begin(), coupling_type.end());
  ctype_buf.push_back('\0');

  char flag_buf[2] = { last_flag, '\0' };

  error_code = 0;

  run_1d_simulation_step_1d_(problem_id, current_time, save_incr,
                             ctype_buf.data(), params, solution,
                             cpl_value, flag_buf, error_code);

  if (error_code != 0) {
    throw svmp::CoreException(
        "[OneDSolverInterface] svOneDSolver failed while advancing the 1D model. "
        "The 1D solver returned error code " + std::to_string(error_code) + ".",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }
}

void OneDSolverInterface::extract_coupled_dof(int problem_id, int& coupled_dof,
                                              const std::string& coupling_type)
{
  if (!extract_coupled_dof_) {
    throw svmp::CoreException(
        "[OneDSolverInterface] extract_coupled_dof not loaded",
        svmp::StatusCode::DependencyError,
        __FILE__,
        __LINE__,
        __func__);
  }
  // Copy into a mutable buffer; the shared-library function signature uses
  // char* (not const char*) so we must pass a writable copy.
  std::vector<char> buf(coupling_type.begin(), coupling_type.end());
  buf.push_back('\0');
  extract_coupled_dof_(problem_id, coupled_dof, buf.data());
}
