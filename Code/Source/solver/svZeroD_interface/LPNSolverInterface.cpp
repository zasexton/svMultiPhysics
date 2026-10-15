// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "LPNSolverInterface.h"
#include <stdexcept>
#include <iostream>
#include <string>

//--------------------
// LPNSolverInterface
//--------------------
//
LPNSolverInterface::LPNSolverInterface()
{
//// Set the default names of the LPN interface functions.
  lpn_initialize_name_ = "initialize";
  lpn_increment_time_name_ = "increment_time";
  lpn_run_simulation_name_ = "run_simulation";
  lpn_update_block_params_name_ = "update_block_params";
  lpn_read_block_params_name_ = "read_block_params";
  lpn_get_block_node_IDs_name_ = "get_block_node_IDs";
  lpn_update_state_name_ = "update_state";
  lpn_return_ydot_name_ = "return_ydot";
  lpn_return_y_name_ = "return_y";
  lpn_set_external_step_size_name_ = "set_external_step_size";
}

LPNSolverInterface::~LPNSolverInterface()
{
  if (library_handle_) vtkDynamicLoader::CloseLibrary(library_handle_);
}

//--------------
// load_library
//--------------
// Load the LPN shared library and get pointers to its interface functions.
//
void LPNSolverInterface::load_library(const std::string& interface_lib)
{
  if (library_handle_) throw std::runtime_error("The 0D interface library is already loaded");
  library_handle_ = vtkDynamicLoader::OpenLibrary(interface_lib.c_str());
  if (!library_handle_) {
    const char* error = vtkDynamicLoader::LastError();
    throw std::runtime_error("Could not load 0D library '" + interface_lib + "': " + (error ? error : "unknown error"));
  }
  auto load_symbol = [&](const char* name) {
    auto symbol = vtkDynamicLoader::GetSymbolAddress(library_handle_, name);
    if (!symbol) {
      throw std::runtime_error("Missing 0D interface function '" + std::string(name) + "' in '" + interface_lib + "'");
    }
    return symbol;
  };
  lpn_initialize_ = reinterpret_cast<decltype(lpn_initialize_)>(load_symbol("initialize"));
  lpn_increment_time_ = reinterpret_cast<decltype(lpn_increment_time_)>(load_symbol("increment_time"));
  lpn_run_simulation_ = reinterpret_cast<decltype(lpn_run_simulation_)>(load_symbol("run_simulation"));
  lpn_update_block_params_ = reinterpret_cast<decltype(lpn_update_block_params_)>(load_symbol("update_block_params"));
  lpn_read_block_params_ = reinterpret_cast<decltype(lpn_read_block_params_)>(load_symbol("read_block_params"));
  lpn_get_block_node_IDs_ = reinterpret_cast<decltype(lpn_get_block_node_IDs_)>(load_symbol("get_block_node_IDs"));
  lpn_update_state_ = reinterpret_cast<decltype(lpn_update_state_)>(load_symbol("update_state"));
  lpn_return_y_ = reinterpret_cast<decltype(lpn_return_y_)>(load_symbol("return_y"));
  lpn_return_ydot_ = reinterpret_cast<decltype(lpn_return_ydot_)>(load_symbol("return_ydot"));
  lpn_set_external_step_size_ = reinterpret_cast<decltype(lpn_set_external_step_size_)>(load_symbol("set_external_step_size"));
}

// Initialze the LPN solver.
//
// Parameters:
//
//   file_name: The name of the LPN configuration file (JSON).
//
void LPNSolverInterface::initialize(std::string file_name)
{
  if (!lpn_initialize_) throw std::runtime_error("The 0D interface library has not been loaded");
  lpn_initialize_(file_name, problem_id_, pts_per_cycle_, num_cycles_, num_output_steps_, block_names_, variable_names_);
  std::cout << "[LPNSolverInterface::initialize] Problem ID: " << problem_id_ << std::endl;
  system_size_ = variable_names_.size();
  std::cout << "[LPNSolverInterface::initialize] System size: " << system_size_ << std::endl;
  //solution_.resize(system_size_*num_output_steps_);
}

// Set the external time step variable in the svZeroD interface.
//
// Parameters:
//
//   step_size: The time step in the 3D (external) solver. 
//
void LPNSolverInterface::set_external_step_size(double step_size)
{
  lpn_set_external_step_size_(problem_id_, step_size);
}

// Increment the LPN solution in time.
//
// Parameters:
//
//   time: The solution time.
//
//   solution: The returned LPN solution.
//
void LPNSolverInterface::increment_time(const double time, std::vector<double>& solution)
{
  lpn_increment_time_(problem_id_, time, solution);
}

// Run the 0D simulation
//
// Parameters:
//
//   time: The solution time in the 3D external solver
//
//   output_times: The time points at which 0D solutions are returned.
//
//   output_solutions: The returned 0D solutions at all time steps.
//
//   error_code: Either 0 or 1 depending on whether the 0D simulation diverged.
//
void LPNSolverInterface::run_simulation(const double time, std::vector<double>& output_times, std::vector<double>& output_solutions, int& error_code)
{
  lpn_run_simulation_(problem_id_, time, output_times, output_solutions, error_code);
}

// Update the parameters of a particular 0D block
//
// Parameters:
//
//   block_name: The name of the 0D block.
//
//   new_params: The new parameters for the 0D block.
//
void LPNSolverInterface::update_block_params(std::string block_name, std::vector<double>& new_params)
{
  lpn_update_block_params_(problem_id_, block_name, new_params);
}

// Read the paramaters of a particular 0D block
//
// Parameters:
//
//   block_name: The name of the 0D block.
//
//   new_params: The parameters for the 0D block.
//
void LPNSolverInterface::read_block_params(std::string block_name, std::vector<double>& block_params)
{
  lpn_read_block_params_(problem_id_, block_name, block_params);
}

// Get the IDs of the inlet/outlet variables of a given block in the state vector 
//
// Parameters:
//
//   block_name: The name of the 0D block.
//
//   IDs: The solution IDs of the inlet and outlet nodes for the block.
//
void LPNSolverInterface::get_block_node_IDs(std::string block_name, std::vector<int>& IDs)
{
  lpn_get_block_node_IDs_(problem_id_, block_name, IDs);
}

// Overwrite the y and ydot state vectors in the 0D solver
//
// Parameters:
//
//   state_y: The y state vector
//
//   state_ydot: The ydot state vector
void LPNSolverInterface::update_state(std::vector<double> state_y, std::vector<double> state_ydot)
{
  lpn_update_state_(problem_id_, state_y, state_ydot);
}

// Return the 0D y state vector
//
// Parameters:
//
//   y: The y state vector
//
void LPNSolverInterface::return_y(std::vector<double>& y)
{
  lpn_return_y_(problem_id_, y);
}

// Return the 0D ydot state vector
//
// Parameters:
//
//   ydot: The y state vector
//
void LPNSolverInterface::return_ydot(std::vector<double>& ydot)
{
  lpn_return_ydot_(problem_id_, ydot);
}
