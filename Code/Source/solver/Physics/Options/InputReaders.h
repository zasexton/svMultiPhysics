/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_OPTIONS_INPUT_READERS_H
#define SVMP_PHYSICS_OPTIONS_INPUT_READERS_H

/**
 * @file InputReaders.h
 * @brief Generic readers of equation input parameters
 *
 * Parameter-map access (find, defined, first defined over several
 * spellings), whole-text parsing of numbers, booleans, integer and
 * semicolon lists and three-vectors, and the element type and polynomial
 * order of a mesh.  Not specific to one formulation (decision D52: the
 * option readers live in Physics/Options, below no formulation).  Moved
 * unchanged from NavierStokesRegister.cpp.
 */

#include "FE/Core/Types.h"
#include "Physics/Core/EquationModuleInput.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
class MeshBase;
} // namespace svmp

namespace svmp::Physics::options {

std::string trim_copy(std::string s);
std::string lower_copy(std::string s);
bool parse_bool_relaxed(std::string_view raw);

/// SVMP_OOP_SOLVER_TRACE: trace lines of the input translation.
[[nodiscard]] bool navierStokesTraceEnabled() noexcept;
void navierStokesTraceLog(const std::string& message);
[[nodiscard]] bool temporalSpatialBcTraceEnabled() noexcept;

double parse_double(std::string_view raw, std::string_view context);
int parse_positive_int(std::string_view raw, std::string_view context);
svmp::FE::GlobalIndex parse_global_index(std::string_view raw, std::string_view context);
bool values_match(svmp::FE::Real a, svmp::FE::Real b);
std::vector<std::string> split_csv_line(std::string_view line);
std::vector<int> parse_int_list(std::string_view raw);
std::vector<std::string> split_semicolon_entries(std::string_view raw);
std::array<svmp::FE::Real, 3> parse_real_vector3(std::string_view raw,
                                                 std::string_view context);

const svmp::Physics::ParameterValue* find_param(const svmp::Physics::ParameterMap& params,
                                                std::string_view key);
bool has_nonempty_defined(const svmp::Physics::ParameterMap& params, std::string_view key);
std::optional<double> get_defined_double(const svmp::Physics::ParameterMap& params, std::string_view key);
std::optional<int> get_defined_int(const svmp::Physics::ParameterMap& params, std::string_view key);
std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params, std::string_view key);
std::optional<std::string> get_defined_string(const svmp::Physics::ParameterMap& params, std::string_view key);
std::optional<double> first_defined_double(const svmp::Physics::ParameterMap& params,
                                           std::initializer_list<std::string_view> keys);
std::optional<std::string> first_defined_string(const svmp::Physics::ParameterMap& params,
                                                std::initializer_list<std::string_view> keys);
std::optional<bool> first_defined_bool(const svmp::Physics::ParameterMap& params,
                                       std::initializer_list<std::string_view> keys);
std::optional<int> first_defined_int(const svmp::Physics::ParameterMap& params,
                                     std::initializer_list<std::string_view> keys);
std::size_t defined_parameter_count(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys);
bool any_parameter_defined(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys);
std::optional<std::vector<int>> first_defined_int_entries(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context);
std::optional<std::vector<std::array<svmp::FE::Real, 3>>>
first_defined_vector3_entries(const svmp::Physics::ParameterMap& params,
                              std::initializer_list<std::string_view> keys,
                              std::string_view context);

std::array<bool, 3> active_components_from_flags(const std::array<int, 3>& active, int dim);
svmp::FE::Real direction_component(const std::vector<int>& effective_dir, int component);

svmp::FE::ElementType infer_base_element_type(const svmp::MeshBase& mesh);
int infer_polynomial_order(const svmp::MeshBase& mesh);
int resolve_element_order(const svmp::Physics::EquationModuleInput& input, int inferred_order);
const svmp::Physics::DomainInput& select_single_domain(const svmp::Physics::EquationModuleInput& input,
                                                       std::string_view module_name);

} // namespace svmp::Physics::options

#endif // SVMP_PHYSICS_OPTIONS_INPUT_READERS_H
