/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_TIME_LOOP_H
#define SVMP_FE_TIMESTEPPING_TIME_LOOP_H

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/StepController.h"

#include <functional>
#include <memory>
#include <optional>

namespace svmp {
namespace FE {
namespace timestepping {

enum class SchemeKind : std::uint8_t {
    BackwardEuler,
    BDF2,
    ThetaMethod,
    TRBDF2,
    GeneralizedAlpha,
    Newmark,
    VSVO_BDF,
    DG0,
    DG1,
    DG,
    CG1,
    CG2,
    CG
};

struct TimeLoopOptions {
    double t0{0.0};
    double t_end{0.0};
    double dt{0.0};
    int max_steps{1000000};

    bool adjust_last_step{true};

    SchemeKind scheme{SchemeKind::BackwardEuler};
    double theta{1.0};
    double trbdf2_gamma{0.5857864376269049}; // 2 - sqrt(2)

    // Generalized-α parameterization via spectral radius at infinity (ρ∞).
    // - For systems with temporalOrder()==1: Jansen–Whiting–Hulbert (first-order generalized-α).
    // - For systems with temporalOrder()==2: Chung–Hulbert generalized-α for structural dynamics.
    double generalized_alpha_rho_inf{1.0};

    // Newmark-β family parameters (structural dynamics).
    // - For systems with temporalOrder()==2, TimeLoop uses a displacement-only Newmark-β update
    //   and requires `TimeHistory` to store velocity/acceleration (`uDot`, `uDDot`).
    // - For systems with temporalOrder()<=1, SchemeKind::Newmark is treated as an alias of
    //   Crank–Nicolson (θ=0.5).
    double newmark_beta{0.25};
    double newmark_gamma{0.5};

    // cG/dG in time (collocation equivalents for first-order systems).
    // For SchemeKind::DG: degree k => (k+1)-stage Radau IIA (order 2k+1).
    // For SchemeKind::CG: degree k => k-stage Gauss collocation (order 2k).
    int dg_degree{1};
    int cg_degree{2};

    NewtonOptions newton{};

    // Optional adaptive step-size controller. If null, TimeLoop uses a fixed dt
    // (with optional last-step adjustment) and throws on nonlinear failure.
    std::shared_ptr<StepController> step_controller{};
};

struct TimeLoopCallbacks {
    std::function<void(const TimeHistory&)> on_step_start{};
    std::function<void(const TimeHistory&, const NewtonReport&)> on_nonlinear_done{};
    std::function<void(const TimeHistory&)> on_step_accepted{};

    std::function<void(const TimeHistory&, StepRejectReason, const NewtonReport&)> on_step_rejected{};
    std::function<void(double old_dt, double new_dt, int step_index, int attempt_index)> on_dt_updated{};
};

struct TimeLoopReport {
    bool success{true};
    int steps_taken{0};
    double final_time{0.0};
    std::string message{};
};

class TimeLoop {
public:
    explicit TimeLoop(TimeLoopOptions options);

    [[nodiscard]] const TimeLoopOptions& options() const noexcept { return options_; }

    [[nodiscard]] TimeLoopReport run(systems::TransientSystem& transient,
                                     const backends::BackendFactory& factory,
                                     backends::LinearSolver& linear,
                                     TimeHistory& history,
                                     const TimeLoopCallbacks& callbacks = {}) const;

private:
    TimeLoopOptions options_;
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_TIME_LOOP_H
