#include "RungeKutta45.h"

RungeKutta45::RungeKutta45(ODESystem ode_system) {
    set_ode_system(std::move(ode_system));
}

void RungeKutta45::set_initial_condition(const Vector<double>& initial_state) {
    this->y0 = initial_state;
    this->y = initial_state;
    this->state_variables.push_back(initial_state);
    this->dydt_old.resize(initial_state.size());
    this->dydt.resize(initial_state.size());
    this->dydt_new.resize(initial_state.size());
    this->frk1.resize(initial_state.size());
    this->frk2.resize(initial_state.size());
    this->frk3.resize(initial_state.size());
    this->frk4.resize(initial_state.size());
    this->n_equations = initial_state.size();
    this->dy.resize(this->y.size());
    this->dyy.resize(this->y.size());
    this->dy_new.resize(this->y.size());
    this->tmp_y.resize(this->y.size());
    this->scale.resize(this->y.size());
}

void RungeKutta45::set_step_size(double step_size) {
    this->h = step_size;
}

bool RungeKutta45::solve() {
    // Check if the solver is ready to solve
    this->check_solver();
    std::cout << "Checks passed" << std::endl;
    // Begin inital calucations
    this->K.resize(this->n_stages + 1, this->n_equations);
    this->error_exponent = -1.0 / (this->error_estimate_order + 1);
    this->direction = (this->t1 - this->t0) > 0 ? 1 : -1;
    this->ode_system.derivative_function(this->t0, this->y0, this->dydt);
    // Select Initial Step

    if (this->y0.size() == 0) {
        this->h = this->inf;
    }
    double interval_length = std::fabs(this->t1 - this->t0);
    if (interval_length == 0.0) {
        this->h = 0.0;
    }
    Vector<double> tmp_scale = this->atol + this->rtol * this->y0.abs();
    Vector<double> D0(this->y0.size());
    Vector<double> D1(this->y0.size());
    Vector<double> D2(this->y0.size());
    Vector<double> f0(this->y0.size());
    Vector<double> f1(this->y0.size());
    Vector<double> yy1(this->y0.size());
    for (int i = 0; i < this->y0.size(); i++) {
        D0(i) = this->y0(i) / tmp_scale(i);
        D1(i) = this->dydt(i) / tmp_scale(i);
    }
    std::cout << "atol: " << this->atol << std::endl;
    std::cout << "rtol: " << this->rtol << std::endl;
    std::cout << "y0/scale: " << D0 << std::endl;
    double d0 = std::sqrt(utils::norm(D0)) / std::sqrt(this->y0.size());
    std::cout << "d0: " << d0 << std::endl;
    double d1 = std::sqrt(utils::norm(D1)) / std::sqrt(this->y0.size());
    std::cout << "d1: " << d1 << std::endl;
    double h0, h1, d2;
    if (d0 < 1e-5 || d1 < 1e-5) {
        h0 = 1e-6;
    } else {
        h0 = 0.01 * d0 / d1;
    }
    h0 = std::min(h0, interval_length);
    this->ode_system.derivative_function(this->t0, this->y0, f0);
    yy1 = this->y0 + h0 * this->direction * f0;
    this->ode_system.derivative_function(this->t0 + h0 * this->direction, yy1, f1);
    for (int i = 0; i < this->y0.size(); i++) {
        D2(i) = (f1(i) - f0(i)) / tmp_scale(i);
    }
    d2 = (std::sqrt(utils::norm(D2)) / std::sqrt(this->y0.size())) / h0;
    if (d1 <= 1e-15 && d2 <= 1e-15) {
        h1 = std::max(1e-6, h0 * 1e-3);
    } else {
        h1 = std::pow(0.01 / std::max(d1, d2),1.0/(this->order + 1.0));
    }
    Vector<double> hs = {100 * h0, h1, interval_length, this->h_max};
    this->h = hs.min();
    std::cout << "Initial step size: " << this->h << std::endl;

    // Select the initial step for the ode solve
    /*
    int ord = this->order;
    double d0, d1, d2;
    double h0, h1;
    double h_abs;
    Vector<double> f0(y0.size());
    Vector<double> scale(y0.size());
    Vector<double> yy1(y0.size());
    Vector<double> f1(y0.size());
    this->ode_system.derivative_function(this->t0, this->y0, f0);
    if (y0.size() == 0) {
        h_abs = this->inf;
    }
    scale = atol + rtol * y0.abs();
    d0 = 0.0;
    d1 = 0.0;
    for (int i = 0; i < y0.size(); ++i) {
        d0 += std::pow(y0(i) / scale(i), 2.0);
        d1 += std::pow(f0(i) / scale(i), 2.0);
    }
    d0 = std::sqrt(d0) / std::sqrt(y0.size());
    d1 = std::sqrt(d1) / std::sqrt(y0.size());
    if (d0 < 1e-5 || d1 < 1e-5) {
        h0 = 1e-6;
    } else {
        h0 = 0.01 * d0 / d1;
    }
    yy1 = y0 + h0 * direction * f0;
    this->ode_system.derivative_function(t+h0, yy1, f1);
    d2 = 0.0;
    Vector<double> diff(y0.size());
    for (int i = 0; i < f0.size(); ++i) {
        d2 += std::pow((f1(i) - f0(i)) / scale(i), 2.0);
        diff(i) = (f1(i) - f0(i))/scale(i);
    }
    //std::cout << "diff: " << diff << std::endl;
    d2 = std::sqrt(d2) / static_cast<double>(std::sqrt(f0.size()));
    //std::cout << "tmp d2: " << d2 << std::endl;
    d2 = d2 / h0;
    if (d1 <= 1e-15 && d2 <= 1e-15) {
        h1 = std::max(1e-6, h0 * 1e-3);
    } else {
        h1 = std::pow((0.01 / std::max(d1, d2)),(1.0 / (ord + 1.0)));
    }
    /*
    std::cout << "scale: " << scale << std::endl;
    std::cout << "d0: " << d0 << std::endl;
    std::cout << "d1: " << d1 << std::endl;
    std::cout << "d2: " << d2 << std::endl;
    std::cout << "h0: " << h0 << std::endl;
    std::cout << "h1: " << h1 << std::endl;
    std::cout << "y0: " << y0 << std::endl;
    std::cout << "f0: " << f0 << std::endl;
    std::cout << "f1: " << f1 << std::endl;
    std::cout << "y1: " << y1 << std::endl;
     */
    //h_abs = std::min(100.0 * h0, h1);

    // Run the solver
    while (step()) {
        continue;
    }
    std::cout << "Solver finished" << std::endl;
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

bool RungeKutta45::step() {
    // Check if the solver has already finished
    // [TODO] there should be a DPS triple scheme employed to obtain an efficiency and agreement with
    // standard ODE solvers.
    bool step_accepted = false;
    bool step_rejected = false;
    if (status == 1) {
        return false;
    } else if (status == -1) {
        return false;
    }
    // Check if the next time is greater than the final time
    if ((this->t + this->h) > this->t1) {
        this->h = this->t1 - this->t;
        this->status = 1;
    }
    double min_step = 10 * std::fabs(std::nextafter(this->t, this->direction * this->inf) - this->t);
    if (this->t == this->t0) {
        std::cout << "t0: " << this->t0 << std::endl;
        std::cout << "t: " << this->t << std::endl;
        std::cout << "t1: " << this->t1 << std::endl;
        std::cout << "h: " << this->h << std::endl;
        std::cout << "Min step: " << min_step << std::endl;
        std::cout << "Direction: " << this->direction << std::endl;
        std::cout << "h_max: " << this->h_max << std::endl;
        std::cout << "h_min: " << this->h_min << std::endl;
    }
    double h_abs = std::fabs(this->h);
    double error_norm;
    double factor;
    double tmp_t;

    //this->dy.resize(1, this->y.size());
    //this->dyy.resize(1, this->y.size());
    //std::cout << "Y size: " << this->y.size() << std::endl;
    //this->dy_new.resize(this->y.size());
    //this->tmp_y.resize(this->y.size());
    if (h_abs > this->h_max) {
        h_abs = this->h_max;
    } else if (h_abs < min_step) {
        h_abs = min_step;
    } else {
        h_abs = h_abs;
    }

    while (!step_accepted) {
        // Update the state derivatives
        if (h_abs < min_step) {
            this->status = -1;
            return false;
        }
        this->h = this->direction * h_abs;
        this->tn = this->t + this->h;
        if ((this->direction * (this->tn - this->t1)) > 0) {
            this->tn = this->t1;
        }
        this->h = this->tn - this->t;
        h_abs = std::fabs(this->h);
        //std::cout << "Starting derivative calculation" << std::endl;
        try {
            // scipy implementation
            this->K.set_row(0,this->dydt);
            //Array<double> dy;
            //Array<double> dyy;
            //dy.resize(1, this->y.size());
            //dyy.resize(1, this->y.size());
            //std::cout << "Passed K0" << std::endl;
            for (int i = 1; i < this->C.size(); i++) {
                this->aa.resize(i);
                for (int j = 0; j < i; j++) {
                    this->aa(j) = this->A(i,j);
                }
                this->dyy = mat_fun::mat_mul(mat_fun::transpose(this->K.rows(0, i-1)),this->aa);
                if (this->t == this->t0) {
                    std::cout << "aa[:s]: " << aa << std::endl;
                    std::cout << "c: " << C(i) << std::endl;
                }
                if (this->tn == this->t1) {
                    std::cout << "t: " << this->t << std::endl;
                    std::cout << "Passed mat_mul 1" << std::endl;
                }
                //std::cout << "Shape dy: " << dy.nrows() << "x" << dy.ncols() << std::endl;
                //std::cout << "Shape dyy: " << dyy.nrows() << "x" << dyy.ncols() << std::endl;
                //std::cout << dyy << std::endl;
                //std::cout << "Passed mat_mul" << std::endl;
                this->dy = this->dyy * this->h;
                //std::this_thread::sleep_for(std::chrono::seconds(2));
                //std::cout << "Passed dy" << std::endl;
                //std::cout << "Shape dy: " << dy.nrows() << "x" << dy.ncols() << std::endl;
                //std::this_thread::sleep_for(std::chrono::seconds(2));
                tmp_t = this->t + this->C(i) * this->h;
                //std::cout << "Passed tmp_t" << std::endl;
                tmp_y = this->y + this->dy;
                //std::cout << "Passed tmp_y" << std::endl;
                this->ode_system.derivative_function(tmp_t, tmp_y, dy_new);
                //std::cout << "Passed derivative function" << std::endl;
                this->K.set_row(i, this->dy_new);
            }
            this->dy = mat_fun::mat_mul(mat_fun::transpose(this->K.rows(0,this->K.nrows()-2)), this->B);
            if (this->tn == this->t1) {
                std::cout << "Passed mat_mul 2" << std::endl;
            }
            this->yn = this->y + this->h * dy;
            this->ode_system.derivative_function(this->t + this->h, this->yn, this->dydt_new);
            this->K.set_row(this->K.nrows()-1, this->dydt_new);
            //this->dydt_old = this->dydt;
            //this->yn = this->y;
            //this->ode_system.derivative_function(this->t, this->yn, this->frk1);
            //this->yn = this->y + this->h * (1.0 / 2.0) * this->frk1;
            //this->ode_system.derivative_function(this->t, this->yn, this->frk2);
            //this->yn = this->y + this->h * (1.0 / 2.0) * this->frk2;
            //this->ode_system.derivative_function(this->t, this->yn, this->frk3);
            //this->yn = this->y + this->h * this->frk3;
            //this->ode_system.derivative_function(this->t, this->yn, this->frk4);
            //this->yn = this->y + this->h * (1.0 / 6.0) * (frk1 + 2.0 * (frk2 + frk3) + frk4);
        } catch (const std::exception &e) {
            std::cerr << "Error in derivative function: " << e.what() << std::endl;
            this->status = -1; // Mark as failed
            return false;
        }
        //std::cout << "Passed derivative calculation" << std::endl;
        if (this->t == this->t0) {
            std::cout << "y: " << this->y << std::endl;
            std::cout << "max(abs(y)): " << this->y.abs().max() << std::endl;
            std::cout << "yn: " << this->yn << std::endl;
            std::cout << "max(abs(yn)): " << this->yn.abs().max() << std::endl;
        }
        for (int i = 0; i < this->y.size(); i++) {
            if (std::fabs(this->y(i)) > std::fabs(this->yn(i))) {
                scale(i) = this->atol + this->rtol * std::fabs(this->y(i));
            } else {
                scale(i) = this->atol + this->rtol * std::fabs(this->yn(i));
            }
        }
        // Estimate error norm
        //std::cout << "Begin error calculation" << std::endl;

        error = mat_fun::mat_mul(mat_fun::transpose(this->K), this->E) * this->h;
        for (int i = 0; i < this->y.size(); i++) {
            error(i) = error(i) / scale(i);
        }

        error_norm = std::sqrt(utils::norm(error)) / std::sqrt(this->y.size());

        if (error_norm < 1.0) {
            if (error_norm == 0.0) {
                factor = this->max_factor;
            } else {
                factor = std::min(this->max_factor, this->safety * std::pow(error_norm, this->error_exponent));
            }
            if (step_rejected) {
                factor = std::min(1.0, factor);
            }
            h_abs *= factor;
            step_accepted = true;
        } else {
            h_abs *= std::max(this->min_factor, this->safety * std::pow(error_norm, this->error_exponent));
            step_rejected = true;
        }

    }
    //std::cout << "Passed error calculation" << std::endl;
    this->tn = this->t + this->h;

    // Add the next state and time to the solution
    this->time_points.push_back(this->tn);
    this->state_variables.push_back(this->yn);

    // Update the current state and time
    this->yo = this->y;
    this->to = this->t;
    this->y = this->yn;
    this->t = this->tn;
    this->h = h_abs;
    this->dydt = this->dydt_new;
    //std::cout << "t: " << this->t << std::endl;
    // Return the running state of the solver
    if (this->status == 1) {
        return false;
    } else if (this->status == -1) {
        return false;
    } else {
        return true;
    }
}
