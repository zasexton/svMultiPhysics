#ifndef SV_TOP_RUNGEKUTTA45_H
#define SV_TOP_RUNGEKUTTA45_H

#include "ODESolver.h"
#include "mat_fun.h"
#include "utils.h"
#include "odes.h"

#include <thread>
#include <chrono>

class RungeKutta45 : public ODESolver {
public:
    explicit RungeKutta45(ODESystem ode_system);
    void setInitialCondition(const Vector<double>& initial_state) override;
    bool solve() override;
    bool step() override;
    void setStepSize(double step_size) override;
    //virtual ~RungeKuttaSolver45() {}
protected:
    int order = 5;
    int error_estimate_order = 4;
    int n_stages = 6;
    int n_equations{};
    double error_exponent{};
    double direction{};
    double safety = 0.9;
    double min_factor = 0.2;
    double max_factor = 10.0;
    double inf = std::numeric_limits<double>::infinity();
    double h_max = std::numeric_limits<double>::infinity();
    Vector<double> tmp_y;
    Vector<double> error;
    Vector<double> dy_new;
    Vector<double> dy;
    Vector<double> dyy;
    Vector<double> frk1, frk2, frk3, frk4;
    Vector<double> aa;
    Vector<double> scale;
    Vector<double> C = {0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0};
    Array<double> A = {{0.0, 0.0, 0.0, 0.0, 0.0},
                       {1.0/5, 0, 0, 0, 0},
                       {3.0/40, 9.0/40, 0, 0, 0},
                       {44.0/45, -56.0/15, 32.0/9, 0, 0},
                       {19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729, 0},
                       {9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656}};
    Vector<double> B = {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0};
    Vector<double> E = {-71.0/57600.0, 0.0, 71.0/16695.0, -71.0/1920.0, 17253.0/339200.0, -22.0/525.0, 1.0/40.0};
    Array<double> P = {{1, -8048581381.0/2820520608, 8663915743.0/2820520608,-12715105075.0/11282082432},
                       {0.0, 0.0, 0.0, 0.0},
                       {0, 131558114200.0/32700410799, -68118460800.0/10900136933,87487479700.0/32700410799},
                       {0, -1754552775.0/470086768, 14199869525.0/1410260304,-10690763975.0/1880347072},
                       {0, 127303824393.0/49829197408, -318862633887.0/49829197408, 701980252875.0 / 199316789632},
                       {0, -282668133.0/205662961, 2019193451.0/616988883, -1453857185.0/822651844},
                       {0, 40617522.0/29380423, -110615467.0/29380423, 69997945.0/29380423}};
    Array<double> K;
};


#endif //SV_TOP_RUNGEKUTTA45_H