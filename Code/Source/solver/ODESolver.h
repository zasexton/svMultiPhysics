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

#ifndef SV_TOP_ODESOLVER_H
#define SV_TOP_ODESOLVER_H

#include <vector>
#include <iostream>
#include <limits>
#include <stdexecpt>
#include <utility>

#include "Vector.h"
#include "Array.h"
#include "consts.h"
#include "odes.h"

/// \class ODESolver
/// \brief A class for solving ordinary differential equations (ODEs).
///
/// This class provides methods to set up and solve ODE
class ODESolver {
public:
    /// \brief Default constructor for ODESolver.
    ODESolver();

    /// \brief Virtual destructor for ODESolver.
    virtual ~ODESolver() {}

    /// \brief Set the ODE system to be solved.
    virtual void set_ode_system(ODESystem ode_system);

    /// \brief Set the initial condition for the ODE system.
    /// \param initial_state A vector containing the initial state of the system.
    virtual void set_initial_condition(const Vector<double>& initial_state);

    /// \brief Set the time interval for the ODE solution.
    /// \param t0 The initial time.
    /// \param t1 The final time.
    virtual void set_time_interval(double t0, double t1);

    /// \brief Set the step size for the ODE solver.
    /// \param step_size The step size.
    virtual void set_step_size(double step_size);

    /// \brief Set the absolute tolerance for the ODE solver.
    /// \param atol The absolute tolerance.
    virtual void set_absolute_tolerance(double atol);

    /// \brief Set the relative tolerance for the ODE solver.
    /// \param rtol The relative tolerance.
    virtual void set_relative_tolerance(double rtol);

    /// \brief Set the maximum step size for the ODE solver.
    /// \param h_max The maximum step size.
    virtual void set_max_step_size(double h_max);

    /// \brief Set the minimum step size for the ODE solver.
    /// \param h_min The minimum step size.
    virtual void set_min_step_size(double h_min);

    /// \brief Update constants used in the ODE solver.
    virtual void update_constants();

    /// \brief Solve the ODE system.
    /// \return True if the solution was successful, false otherwise.
    virtual bool solve();

    /// \brief Perform a single step of the ODE solver.
    /// \return True if the step was successful, false otherwise.
    virtual bool step();

    /// \brief Get the current time of the ODE solver.
    /// \return The current time.
    double get_time();

    /// \brief Get the current state of the ODE system.
    /// \return A vector containing the current state.
    Vector<double> get_state();

    /// \brief Get the time points of the ODE solution.
    /// \return A vector containing the time points.
    Vector<double> get_times();

    /// \brief Get the state variables of the ODE solution.
    /// \return An array containing the state variables.
    Array<double> get_states();

protected:
    bool success; ///< Indicates if the solver was successful.
    int status; ///< Status of the solver.
    Vector<double> dydt_old; ///< Derivative of the state at the previous time step.
    Vector<double> dydt; ///< Derivative of the state at the current time step.
    Vector<double> dydt_new; ///< Derivative of the state at the next time step.
    Vector<double> y0; ///< Initial state.
    Vector<double> yo; ///< State at the previous time step.
    Vector<double> y; ///< Current state.
    Vector<double> yn; ///< State at the next time step.
    Vector<double> y1; ///< State at the first time step.
    double to; ///< Initial time.
    double t; ///< Current time.
    double tn; ///< Next time.
    double t0; ///< Start time of the interval.
    double t1; ///< End time of the interval.
    double ho; ///< Step size at the previous time step.
    double h; ///< Current step size.
    double hn; ///< Step size at the next time step.
    double atol; ///< Absolute tolerance.
    double rtol; ///< Relative tolerance.
    double h_max; ///< Maximum step size.
    double h_min; ///< Minimum step size.
    std::vector<double> time_points; ///< Time points of the solution.
    std::vector<Vector<double>> state_variables; ///< State variables of the solution.

    /// \brief Check the solver's status and parameters.
    void check_solver();
};


#endif //SV_TOP_ODESOLVER_H