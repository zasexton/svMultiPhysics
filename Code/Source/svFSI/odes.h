#ifndef SV_TOP_ODES_H
#define SV_TOP_ODES_H

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <regex>

#include <symengine/expression.h>
#include <symengine/parser.h>
#include <symengine/symbol.h>
#include <symengine/matrix.h>
#include <symengine/derivative.h>

#include "exprtk.h"
#include "Vector.h"
#include "Array.h"

//using Eigen::VectorXd;
using SymEngine::Expression;
using SymEngine::Symbol;
using SymEngine::DenseMatrix;
using SymEngine::Basic;
using SymEngine::RCP;
using SymEngine::symbol;
using SymEngine::make_rcp;

typedef exprtk::symbol_table<double> symbol_table_t;
typedef exprtk::expression<double>   expression_t;
typedef exprtk::parser<double>       parser_t;

struct ODE {
    std::string name;
    std::string expression_str;
    double value;
    std::vector<std::string> state_variables;
    std::unordered_set<std::string> constants;
    Expression symbolic_expression;
    std::vector<expression_t> numeric_expression;
};

struct ODESystem {
    std::vector<ODE> odes;
    std::function<void(const double, const Vector<double> &, Vector<double> &)> derivative_function;
    std::function<void(const double, const Vector<double> &, Array<double> &)> jacobian_function;
    DenseMatrix jacobian;
    std::unordered_set<std::string> constants;
    std::vector<RCP<const Symbol>> state_vars;
    std::vector<std::string> state_variables;
    std::unordered_map<std::string, double> state_variable_values;
    std::unordered_map<std::string, double> constant_values;
    symbol_table_t symbol_table;
    std::vector<std::pair<size_t, size_t>> jac_sparsity_pattern;
};

void replaceAll(std::string& str, const std::string& from, const std::string& to);

Expression create_symbolic_expression(const std::string& expr_str, const std::unordered_map<std::string,
RCP<const Basic>>& symbols_map);

DenseMatrix compute_jacobian(const std::vector<ODE>& odes, const std::vector<RCP<const SymEngine::Symbol>>& state_vars);

std::vector<RCP<const SymEngine::Symbol>> create_state_variable_symbols(const std::unordered_map<std::string,
        double>& state_var_values);

void identify_constants(ODE& ode, const std::unordered_set<std::string>& state_variables,
                        std::unordered_set<std::string>& constants);

// Parse an ODE system text into a list of ODE entries and collect
// referenced state variables and constants. Initial values for state
// variables are returned in state_var_values (initialized to 0.0).
std::vector<ODE> build_ode_system(const std::string& ode_str,
                                  std::unordered_set<std::string>& constants,
                                  std::unordered_map<std::string, double>& state_var_values);

// Create a numeric RHS evaluator: takes time and state and fills derivatives.
// constant_values supplies named constants used in the ODE expressions.
std::function<void(const double, const Vector<double>&, Vector<double>&)> create_derivative_function(
        std::vector<ODE>& odes,
        const std::unordered_map<std::string,double>& constant_values);

std::vector<std::string> extract_symbolic_expressions(const DenseMatrix& J);

// Create a numeric Jacobian evaluator from the symbolic matrix J.
// constant_values supplies named constants used in the ODE expressions.
std::function<void(const double, const Vector<double>&, Array<double>&)> create_jacobian_function(
        std::vector<ODE>& odes,
        const DenseMatrix& J,
        const std::unordered_map<std::string,double>& constant_values);

void getEigenValues(ODESystem& ode_system);

void printSparsity(const DenseMatrix& J);

std::vector<std::pair<size_t, size_t>> getSparsityPattern(const DenseMatrix& J);


#endif //SV_TOP_ODES_H
