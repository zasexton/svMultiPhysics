/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ODESolver.h"

/// \brief Default constructor for ODESolver.
ODESolver::ODESolver() {
    this->h_min = 1e-6;
    this->h_max = 1e-1;
    this->to = std::numeric_limits<double>::quiet_NaN();
    this->t = std::numeric_limits<double>::quiet_NaN();
    this->tn = std::numeric_limits<double>::quiet_NaN();
    this->t0 = std::numeric_limits<double>::quiet_NaN();
    this->t1 = std::numeric_limits<double>::quiet_NaN();
    this->ho = std::numeric_limits<double>::quiet_NaN();
    this->h = std::numeric_limits<double>::quiet_NaN();
    this->hn = std::numeric_limits<double>::quiet_NaN();
    this->status = 0;
    this->success = false;
    this->atol = 1e-6;
    this->rtol = 1e-3;
}

/// \brief Set the ODE system to be solved.
/// \param system The ODE system to be solved.
void ODESolver::set_ode_system(ODESystem system) {
    // [TODO] set the size of the system from the ode_system object
    this->ode_system = std::move(system);
}

/// \brief Set the initial condition for the ODE system.
/// \param initial_state A vector containing the initial state of the system.
void ODESolver::set_initial_condition(const Vector<double>& initial_state) {
    //
    this->y0 = initial_state;
    this->y = initial_state;
    this->state_variables.push_back(initial_state);
    this->dydt_old.resize(initial_state.size());
    this->dydt.resize(initial_state.size());
    this->dydt_new.resize(initial_state.size());
}

/// \brief Set the time interval for the ODE solution.
/// \param start The initial time.
/// \param end The final time.
void ODESolver::set_time_interval(double start, double end) {
    this->t0 = start;
    this->t1 = end;
}

/// \brief Set the step size for the ODE solver.
/// \param step_size The step size.
void ODESolver::set_step_size(double step_size) {
    this->h = step_size;
}

/// \brief Set the absolute tolerance for the ODE solver.
/// \param value The absolute tolerance.
void ODESolver::set_absolute_tolerance(double value) {
    this->atol = value;
}

/// \brief Set the relative tolerance for the ODE solver.
/// \param value The relative tolerance.
void ODESolver::set_relative_tolerance(double value) {
    this->rtol = value;
}

/// \brief Set the maximum step size for the ODE solver.
/// \param value The maximum step size.
void ODESolver::set_max_step_size(double value) {
    this->h_max = value;
}

/// \brief Set the minimum step size for the ODE solver.
/// \param value The minimum step size.
void ODESolver::set_min_step_size(double value) {
    this->h_min = value;
}

/// \brief Update constants used in the ODE solver.
void ODESolver::update_constants() {
    // [TODO] update the constants used in the ODE solver
    // this should take a functional input argument that uses
    // the state variables and other current time solutions to
    // update all of the intermediate values that may inform the
    // next time step.
}

/// \brief Solve the ODE system.
/// \return True if the solution was successful, false otherwise.
bool ODESolver::solve() {
    // Check if the solver is ready to solve
    this->check_solver();
    // Run the solver
    while (step()) {
        // [TODO] add a check for events
        // this->checkEvents();
    }
    // Return the final state of the solver
    if (this->status == 1) {
        this->success = true;
        this->t1 = this->t;
        this->y1 = this->y;
        return true;
    } else {
        this->success = false;
        this->t1 = this->t; // assign the last attempted time
        this->y1 = std::numeric_limits<double>::quiet_NaN(); // assign NaN to the last attempted state
        return false;
    }
}

/// \brief Perform a single step of the ODE solver.
/// \return True if the step was successful, false otherwise.
bool ODESolver::step() {
    // Default implementation for the step function
    // should return false since there is no specified
    // time-stepping method
    this->success = false;
    this->status = -1;
    return false;
}

/// \brief Get the current time of the ODE solver.
/// \return The current time.
double ODESolver::get_time() {
    return this->t;
}

/// \brief Get the current state of the ODE system.
/// \return A vector containing the current state.
Vector<double> ODESolver::get_state() {
    return this->y;
}

/// \brief Get the time points of the ODE solution.
/// \return A vector containing the time points.
Vector<double> ODESolver::get_times() {
    Vector<double> times(this->time_points.size());
    for (int i = 0; i < times.size(); ++i) {
        times(i) = this->time_points[static_cast<size_t>(i)];
    }
    return times;
}

/// \brief Get the state variables of the ODE solution.
/// \return An array containing the state variables.
Array<double> ODESolver::get_states() {
    if (this->state_variables.size() == 0) {
        return Array<double>();
    }
    int nvar = this->state_variables[0].size();
    int nstep = static_cast<int>(this->state_variables.size());
    Array<double> A(nvar, nstep);
    for (int j = 0; j < nstep; ++j) {
        const auto& v = this->state_variables[static_cast<size_t>(j)];
        for (int i = 0; i < nvar; ++i) {
            A(i,j) = v(i);
        }
    }
    return A;
}

/// \brief Check the solver's status and parameters.
///
/// This method checks if the solver is properly initialized and ready to solve the ODE system.
/// It verifies that the initial time, final time, and step size are not NaN, and that the state vector is not empty.
/// \throws std::runtime_error if any of the required parameters are not properly set.
void ODESolver::check_solver() {
    if (std::isnan(this->t0)) {
        throw std::runtime_error("Initial time (t0) is NaN");
    }
    if (std::isnan(this->t1)) {
        throw std::runtime_error("Final time (t1) is NaN");
    }
    if (std::isnan(this->h)) {
        throw std::runtime_error("Step size (h) is NaN");
    }
    if (this->y.size() == 0) {
        throw std::runtime_error("State vector (y) is empty");
    }
}
