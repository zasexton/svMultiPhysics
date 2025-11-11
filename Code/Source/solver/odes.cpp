#include "odes.h"
#include <algorithm>

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty())
        return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

Expression create_symbolic_expression(const std::string& expr_str, const std::unordered_map<std::string, RCP<const Basic>>& symbols_map) {
    std::string parsed_expr = expr_str;
    for (const auto& pair : symbols_map) {
        std::string symbol_name = pair.first;
        std::string symbol_rep = SymEngine::str(*pair.second);
        size_t pos = 0;
        while ((pos = parsed_expr.find(symbol_name, pos)) != std::string::npos) {
            parsed_expr.replace(pos, symbol_name.length(), symbol_rep);
            pos += symbol_rep.length();
        }
    }
    return SymEngine::parse(parsed_expr);
}

DenseMatrix compute_jacobian(const std::vector<ODE>& odes, const std::vector<RCP<const SymEngine::Symbol>>& state_vars) {
    size_t n = odes.size();
    DenseMatrix J(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            J.set(i,j,SymEngine::diff(odes[i].symbolic_expression.get_basic(), state_vars[j]));
        }
    }
    return J;
}

std::vector<RCP<const SymEngine::Symbol>> create_state_variable_symbols(const std::unordered_map<std::string, double>& state_var_values) {
    std::vector<RCP<const Symbol>> state_vars;
    for (const auto& var_name : state_var_values) {
        state_vars.push_back(SymEngine::rcp_static_cast<const Symbol>(symbol(var_name.first)));
    }
    return state_vars;
}

void identify_constants(ODE& ode, const std::unordered_set<std::string>& state_variables, std::unordered_set<std::string>& constants) {
    auto free_syms = SymEngine::free_symbols(*ode.symbolic_expression.get_basic());
    for (const auto& sym : free_syms) {
        std::string name = SymEngine::str(*sym);
        if (state_variables.find(name) == state_variables.end()) {
            constants.insert(name);
            ode.constants.insert(name);
        }
    }
}

std::vector<ODE> build_ode_system(const std::string& ode_str,
                                  std::unordered_set<std::string>& constants,
                                  std::unordered_map<std::string, double>& state_var_values) {
    std::vector<ODE> odes;
    std::unordered_set<std::string> state_variables;
    std::istringstream stream(ode_str);
    std::string line;
    std::unordered_map<std::string, RCP<const Basic>> symbols_map;
    SymEngine::Symbol t("t");
    // Utility for trimming whitespace (in place)
    auto trim = [](std::string& s) {
        s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
        auto not_space = [](unsigned char ch){ return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    };

    // Extract state variables and parse ODEs
    while (std::getline(stream, line)) {
        trim(line);
        if (line.empty()) continue;                       // skip blank lines
        if (line[0] == '#') continue;                     // skip comments

        std::istringstream line_stream(line);
        std::string name, eq, expr;
        if (!(line_stream >> name)) continue;             // nothing useful
        if (!(line_stream >> eq)) continue;               // malformed line
        std::getline(line_stream, expr);
        if (expr.empty()) {
            throw std::runtime_error(std::string("Malformed ODE line (no expression): ") + line);
        }
        // Clean expression: trim leading spaces and '=' symbols
        trim(expr);
        while (!expr.empty() && (expr[0] == '=')) {
            expr.erase(0, 1);
            trim(expr);
        }
        if (expr.empty()) {
            throw std::runtime_error(std::string("Malformed ODE line (empty rhs): ") + line);
        }

        // Replace all instances of '**' with '^' for compatibility with ExprTK
        replaceAll(expr, "**", "^");
        // Remove semicolons from the expression
        expr.erase(std::remove(expr.begin(), expr.end(), ';'), expr.end());

        // Extract state variable name from pattern: d<var>_dt
        size_t pos_dt = name.find("_dt");
        if (name.size() < 4 || pos_dt == std::string::npos || name[0] != 'd' || pos_dt <= 1) {
            throw std::runtime_error(std::string("Malformed ODE name, expected d<var>_dt: ") + name);
        }
        std::string var_name = name.substr(1, pos_dt - 1);
        if (var_name.empty()) {
            throw std::runtime_error(std::string("Empty state variable name in ODE: ") + line);
        }

        state_variables.insert(var_name);
        // Create symbolic variables
        symbols_map[var_name] = SymEngine::symbol(var_name);

        ODE ode = {name, expr, 0.0, {var_name}, {}};
        odes.push_back(ode);
    }

    // Identify constants and update state variables in each ODE
    for (auto& ode : odes) {
        std::unordered_set<std::string> vars;
        std::regex var_regex("[a-zA-Z_][a-zA-Z_0-9]*");
        auto words_begin = std::sregex_iterator(ode.expression_str.begin(), ode.expression_str.end(), var_regex);
        auto words_end = std::sregex_iterator();

        for (auto it = words_begin; it != words_end; ++it) {
            std::string var_name = it->str();
            if (state_variables.find(var_name) == state_variables.end() && var_name != "t") {  // Exclude state variables and known built-in variables
                vars.insert(var_name);
            }
        }

        //for (const auto& v : vars) {
        //    constants.insert(v);
        //    ode.constants.insert(v);
        //}

        // Update state variables for each ODE
        for (const auto& sv : state_variables) {
            if (ode.expression_str.find(sv) != std::string::npos) {
                ode.state_variables.push_back(sv);
            }
        }

        // Create symbolic representation of the expression
        ode.symbolic_expression = create_symbolic_expression(ode.expression_str, symbols_map);
        identify_constants(ode, state_variables, constants);
    }

    // Initialize ODE variables with initial values
    for (const auto& var_name : state_variables) {
        state_var_values[var_name] = 0.0;  // Initial value can be set later
    }

    return odes;
}

std::function<void(const double, const Vector<double>&, Vector<double>&)> create_derivative_function(
        std::vector<ODE>& odes,
        const std::unordered_map<std::string,double>& constant_values
) {
    // Own storage for variables to ensure lifetime matches functor
    auto var_names = std::vector<std::string>(odes.size());
    for (size_t i = 0; i < odes.size(); ++i) var_names[i] = odes[i].state_variables[0];

    auto var_values = std::make_shared<std::vector<double>>(odes.size(), 0.0);
    auto time_value = std::make_shared<double>(0.0);

    // Local symbol table that binds variables to our owned storage
    auto symtab_ptr = std::make_shared<symbol_table_t>();
    for (size_t i = 0; i < var_names.size(); ++i) {
        symtab_ptr->add_variable(var_names[i], (*var_values)[i]);
    }
    symtab_ptr->add_variable("t", *time_value);
    for (const auto& kv : constant_values) {
        symtab_ptr->add_constant(kv.first, kv.second);
    }

    // Compile expressions against the local symbol table
    auto expressions = std::make_shared<std::vector<expression_t>>(odes.size());
    parser_t parser;
    for (size_t i = 0; i < odes.size(); ++i) {
        (*expressions)[i].register_symbol_table(*symtab_ptr);
        if (!parser.compile(odes[i].expression_str, (*expressions)[i])) {
            // Leave expression as zero if compilation fails
            // std::cerr << "ExprTK compile error: " << parser.error() << " in expression: " << odes[i].expression_str << std::endl;
        }
    }

    return [var_values, time_value, expressions, symtab_ptr, var_names, odes](const double t, const Vector<double>& state, Vector<double>& derivatives) {
        (void)var_names; // unused but kept for clarity
        const size_t n = std::min(static_cast<size_t>(state.size()), odes.size());

        // Update time and state variables
        *time_value = t;
        for (size_t i = 0; i < n; ++i) (*var_values)[i] = state(i);

        // Evaluate derivatives (zero for out-of-range entries)
        for (size_t i = 0; i < odes.size(); ++i) {
            derivatives(i) = (*expressions)[i].value();
        }
        for (size_t i = odes.size(); i < derivatives.size(); ++i) derivatives(i) = 0.0;
    };
}

std::vector<std::string> extract_symbolic_expressions(const DenseMatrix& J) {
    std::vector<std::string> expressions;
    for (size_t i = 0; i < J.nrows(); ++i) {
        for (size_t j = 0; j < J.ncols(); ++j) {
            std::string expr_str = SymEngine::str(*J.get(i, j));
            replaceAll(expr_str, "**", "^");
            //std::cout << "Expression at (" << i << ", " << j << "): " << expr_str << std::endl;  // Debug output
            expressions.push_back(expr_str);
        }
    }
    return expressions;
}

std::function<void(const double, const Vector<double>&, Array<double>&)> create_jacobian_function(
        std::vector<ODE>& odes,
        const DenseMatrix& J,
        const std::unordered_map<std::string,double>& constant_values) {

    std::vector<std::string> str_expressions = extract_symbolic_expressions(J);

    // Local owned storage for variables and time
    auto var_names = std::vector<std::string>(odes.size());
    for (size_t i = 0; i < odes.size(); ++i) var_names[i] = odes[i].state_variables[0];
    auto var_values = std::make_shared<std::vector<double>>(odes.size(), 0.0);
    auto time_value = std::make_shared<double>(0.0);
    auto symtab_ptr = std::make_shared<symbol_table_t>();
    for (size_t i = 0; i < var_names.size(); ++i) symtab_ptr->add_variable(var_names[i], (*var_values)[i]);
    symtab_ptr->add_variable("t", *time_value);
    for (const auto& kv : constant_values) symtab_ptr->add_constant(kv.first, kv.second);

    // Compile all Jacobian entry expressions
    auto expressions = std::make_shared<std::vector<expression_t>>(str_expressions.size());
    parser_t parser;
    for (size_t i = 0; i < str_expressions.size(); ++i) {
        (*expressions)[i].register_symbol_table(*symtab_ptr);
        parser.compile(str_expressions[i], (*expressions)[i]);
    }

    return [var_values, time_value, expressions, symtab_ptr, var_names, odes](const double t, const Vector<double>& state, Array<double>& jacobian_values) {
        // Initialize/resize Jacobian to the state dimension; fill with zeros
        const size_t m = state.size();
        jacobian_values.resize(m, m);
        for (size_t i = 0; i < m; ++i) for (size_t j = 0; j < m; ++j) jacobian_values(i,j) = 0.0;

        // Use the common dimension between state and ODE system
        const size_t n = std::min(m, odes.size());

        // Update time and state variables
        *time_value = t;
        for (size_t k = 0; k < n; ++k) (*var_values)[k] = state(k);

        // Fill the top-left n x n block of the Jacobian with evaluated expressions
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                size_t idx = i * odes.size() + j; // expressions are stored row-major for the ODE Jacobian
                if (idx < expressions->size()) jacobian_values(i, j) = (*expressions)[idx].value();
            }
        }
    };
}

void getEigenValues(ODESystem& ode_system) {
    auto eigenvalues = SymEngine::eigen_values(ode_system.jacobian);
    std::cout << "Eigenvalues of the Jacobian matrix:" << std::endl;
    const SymEngine::FiniteSet& fs = static_cast<const SymEngine::FiniteSet&>(*eigenvalues);
    for (const auto& elem : fs.get_container()) {
        std::cout << *elem << std::endl;
    }
}

void printSparsity(const DenseMatrix& J) {
    std::cout << "Sparsity pattern of the Jacobian matrix:" << std::endl;
    for (size_t i = 0; i < J.nrows(); ++i) {
        for (size_t j = 0; j < J.ncols(); ++j) {
            if (J.get(i, j)->__eq__(*SymEngine::zero)) {
                std::cout << "0 ";
            } else {
                std::cout << "1 ";
            }
        }
        std::cout << std::endl;
    }
}

std::vector<std::pair<size_t, size_t>> getSparsityPattern(const DenseMatrix& J) {
    std::vector<std::pair<size_t, size_t>> sparsity;
    for (size_t i = 0; i < J.nrows(); ++i) {
        for (size_t j = 0; j < J.ncols(); ++j) {
            if (!J.get(i, j)->__eq__(*SymEngine::zero)) {
                sparsity.push_back(std::make_pair(i, j));
            }
        }
    }
    return sparsity;
}
