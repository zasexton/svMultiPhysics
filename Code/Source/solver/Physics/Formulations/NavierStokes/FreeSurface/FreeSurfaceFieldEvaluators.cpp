/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFieldEvaluators.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Basis/BasisFunction.h"
#include "FE/Elements/Element.h"
#include "FE/Geometry/CutQuadratureMapping.h"
#include "FE/Geometry/GeometryMapping.h"
#include "FE/Geometry/LinearMapping.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/Vector.h"
#include "FE/Spaces/FunctionSpace.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using namespace detail;

svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
makeFreeSurfaceVelocityEvaluator(
    const FreeSurfaceSystemHandle& sim,
    svmp::FE::FieldId velocity_field,
    std::span<const svmp::FE::Real> solution,
    std::string_view diagnostic_context,
    bool allow_scalar_product_fast_path)
{
  const auto& velocity_record =
      sim.fe_system->fieldRecord(velocity_field);
  const auto velocity_offset =
      sim.fe_system->fieldDofOffset(velocity_field);
  const int dimension = sim.fe_system->meshAccess().dimension();
  if (!velocity_record.space || velocity_offset < 0 ||
      velocity_record.components < dimension ||
      velocity_record.space->value_dimension() < dimension) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " +
        std::string(diagnostic_context) +
        " velocity field is incompatible with its free-surface declaration.");
  }

  auto* const fe_system = sim.fe_system.get();
  auto* const mesh_access = &sim.fe_system->meshAccess();
  const auto velocity_space = velocity_record.space;
  const auto context = std::string(diagnostic_context);

  // The diagnostic loops visit all points of one parent cell in a row.  The
  // scratch below keeps that cell's coefficients and geometry mapping, the
  // inverse Jacobian of an affine LinearMapping (whose Jacobian ignores the
  // reference point), and reusable basis buffers, so a point no longer
  // allocates.  Every returned value keeps its exact arithmetic.  Like the
  // mapping cache before it, the scratch makes the evaluator single-threaded.
  struct VelocityEvaluationScratch {
    bool coefficients_valid{false};
    svmp::FE::GlobalIndex coefficient_cell{-1};
    std::vector<svmp::FE::Real> coefficients{};
    std::map<svmp::FE::GlobalIndex,
             std::shared_ptr<svmp::FE::geometry::GeometryMapping>>
        mappings{};
    bool mapping_valid{false};
    svmp::FE::GlobalIndex mapping_cell{-1};
    const svmp::FE::geometry::GeometryMapping* mapping{nullptr};
    bool mapping_affine{false};
    bool affine_inverse_valid{false};
    svmp::FE::math::Matrix<svmp::FE::Real, 3, 3> affine_inverse{};
    std::vector<svmp::FE::Real> basis_values{};
    std::vector<svmp::FE::basis::Gradient> basis_gradients{};
  };
  auto scratch = std::make_shared<VelocityEvaluationScratch>();
  const auto cell_coefficients =
      [fe_system, velocity_offset, velocity_field, solution, context, scratch](
          svmp::FE::GlobalIndex cell,
          const char* negative_dof_message,
          const char* short_solution_message)
      -> const std::vector<svmp::FE::Real>& {
        auto& state = *scratch;
        if (state.coefficients_valid && state.coefficient_cell == cell) {
          return state.coefficients;
        }
        state.coefficients_valid = false;
        const auto cell_dofs =
            fe_system->fieldDofHandler(velocity_field).getCellDofs(cell);
        state.coefficients.clear();
        state.coefficients.reserve(cell_dofs.size());
        for (const auto dof : cell_dofs) {
          if (dof < 0) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] " + context +
                negative_dof_message);
          }
          const auto index =
              static_cast<std::size_t>(velocity_offset + dof);
          if (index >= solution.size()) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] " + context +
                short_solution_message);
          }
          state.coefficients.push_back(solution[index]);
        }
        state.coefficient_cell = cell;
        state.coefficients_valid = true;
        return state.coefficients;
      };

  // A product of a plain H1 scalar space is evaluated exactly as
  // ProductSpace::evaluate / evaluate_jacobian do through the scalar
  // FunctionSpace::evaluate / evaluate_jacobian: the same basis call and the
  // same accumulation order per component, with one basis evaluation shared
  // by the components.  Any other space takes the generic path.
  std::shared_ptr<const svmp::FE::elements::Element> scalar_element;
  std::size_t scalar_dofs = 0u;
  std::size_t product_components = 0u;
  if (allow_scalar_product_fast_path) {
    const auto* product =
        dynamic_cast<const svmp::FE::spaces::ProductSpace*>(
            velocity_space.get());
    if (product != nullptr &&
        typeid(*product) == typeid(svmp::FE::spaces::ProductSpace) &&
        product->base_space() != nullptr) {
      const auto& base = *product->base_space();
      auto element = base.element_ptr();
      if (typeid(base) == typeid(svmp::FE::spaces::H1Space) &&
          element != nullptr && !element->basis().is_vector_valued() &&
          element->num_dofs() > 0u &&
          element->num_dofs() == product->scalar_dofs_per_component() &&
          product->value_dimension() >= 1 &&
          product->value_dimension() <= 3) {
        scalar_dofs = element->num_dofs();
        product_components =
            static_cast<std::size_t>(product->value_dimension());
        scalar_element = std::move(element);
      }
    }
  }

  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
      velocity;
  velocity.value =
      [velocity_space,
       scratch,
       cell_coefficients,
       scalar_element,
       scalar_dofs,
       product_components](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto& coefficients = cell_coefficients(
            cell,
            " velocity cell has a negative DOF.",
            " solution is too small for the velocity field.");
        const svmp::FE::spaces::FunctionSpace::Value reference_value{
            reference_point[0], reference_point[1], reference_point[2]};
        if (scalar_element != nullptr &&
            coefficients.size() == scalar_dofs * product_components) {
          auto& values = scratch->basis_values;
          values.assign(scalar_dofs, svmp::FE::Real{0.0});
          scalar_element->basis().evaluate_values(reference_value, values);
          std::array<svmp::FE::Real, 3> result{};
          for (std::size_t component = 0u;
               component < product_components;
               ++component) {
            const svmp::FE::Real* component_coefficients =
                coefficients.data() + component * scalar_dofs;
            svmp::FE::Real component_value{0.0};
            for (std::size_t i = 0u; i < scalar_dofs; ++i) {
              component_value += values[i] * component_coefficients[i];
            }
            result[component] = component_value;
          }
          return result;
        }
        const auto value =
            velocity_space->evaluate(reference_value, coefficients);
        return std::array<svmp::FE::Real, 3>{
            value[0], value[1], value[2]};
      };
  velocity.physical_gradient =
      [mesh_access,
       velocity_space,
       context,
       scratch,
       cell_coefficients,
       scalar_element,
       scalar_dofs,
       product_components](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto& coefficients = cell_coefficients(
            cell,
            " velocity-gradient cell has a negative DOF.",
            " solution is too small for the velocity gradient.");
        const svmp::FE::spaces::FunctionSpace::Value reference_value{
            reference_point[0], reference_point[1], reference_point[2]};
        std::array<std::array<svmp::FE::Real, 3>, 3> reference_jacobian{};
        if (scalar_element != nullptr &&
            coefficients.size() == scalar_dofs * product_components) {
          auto& gradients = scratch->basis_gradients;
          gradients.assign(scalar_dofs, svmp::FE::basis::Gradient{});
          scalar_element->basis().evaluate_gradients(
              reference_value, gradients);
          for (std::size_t component = 0u;
               component < product_components;
               ++component) {
            const svmp::FE::Real* component_coefficients =
                coefficients.data() + component * scalar_dofs;
            std::array<svmp::FE::Real, 3> row{};
            for (std::size_t i = 0u; i < scalar_dofs; ++i) {
              for (std::size_t direction = 0u; direction < 3u;
                   ++direction) {
                row[direction] +=
                    gradients[i][direction] * component_coefficients[i];
              }
            }
            reference_jacobian[component] = row;
          }
        } else {
          const auto jacobian = velocity_space->evaluate_jacobian(
              reference_value, coefficients);
          for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t column = 0u; column < 3u; ++column) {
              reference_jacobian[row][column] = jacobian(row, column);
            }
          }
        }
        auto& state = *scratch;
        if (!state.mapping_valid || state.mapping_cell != cell) {
          state.mapping_valid = false;
          state.affine_inverse_valid = false;
          auto mapping = state.mappings.find(cell);
          if (mapping == state.mappings.end()) {
            mapping =
                state.mappings
                    .emplace(cell,
                             createCellGeometryMapping(*mesh_access, cell))
                    .first;
          }
          if (!mapping->second) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] " + context +
                " could not construct the velocity-gradient geometry mapping.");
          }
          const auto& cell_mapping = *mapping->second;
          state.mapping = &cell_mapping;
          state.mapping_affine =
              typeid(cell_mapping) ==
              typeid(svmp::FE::geometry::LinearMapping);
          state.mapping_cell = cell;
          state.mapping_valid = true;
        }
        svmp::FE::math::Vector<svmp::FE::Real, 3> xi{};
        xi[0] = reference_point[0];
        xi[1] = reference_point[1];
        xi[2] = reference_point[2];
        if (state.mapping_affine && !state.affine_inverse_valid) {
          state.affine_inverse = state.mapping->jacobian_inverse(xi);
          state.affine_inverse_valid = true;
        }
        const auto inverse_jacobian = state.mapping_affine
            ? state.affine_inverse
            : state.mapping->jacobian_inverse(xi);
        svmp::FE::interfaces::
            FreeSurfaceDiscreteFunctionalPhysicalGradient
                physical_gradient{};
        for (std::size_t component = 0u; component < 3u;
             ++component) {
          for (std::size_t physical_direction = 0u;
               physical_direction < 3u; ++physical_direction) {
            for (std::size_t reference_direction = 0u;
                 reference_direction < 3u;
                 ++reference_direction) {
              physical_gradient[component][physical_direction] +=
                  reference_jacobian[component][reference_direction] *
                  inverse_jacobian(
                      reference_direction, physical_direction);
            }
          }
        }
        return physical_gradient;
      };
  return velocity;
}

svmp::FE::interfaces::FreeSurfaceGeometryScalarEvaluator
makeFreeSurfaceScalarValueEvaluator(
    const FreeSurfaceSystemHandle& sim,
    svmp::FE::FieldId scalar_field,
    std::span<const svmp::FE::Real> solution,
    std::string_view diagnostic_context)
{
  const auto& record = sim.fe_system->fieldRecord(scalar_field);
  const auto offset = sim.fe_system->fieldDofOffset(scalar_field);
  if (!record.space || offset < 0 || record.components != 1 ||
      record.space->value_dimension() != 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " +
        std::string(diagnostic_context) +
        " scalar field is incompatible with its diagnostic declaration.");
  }
  auto* const fe_system = sim.fe_system.get();
  const auto space = record.space;
  const auto context = std::string(diagnostic_context);
  const auto coefficients_for_cell =
      [fe_system,
       offset,
       scalar_field,
       solution,
       context](svmp::FE::GlobalIndex cell) {
        const auto cell_dofs =
            fe_system->fieldDofHandler(scalar_field).getCellDofs(cell);
        std::vector<svmp::FE::Real> coefficients;
        coefficients.reserve(cell_dofs.size());
        for (const auto dof : cell_dofs) {
          if (dof < 0) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] " + context +
                " scalar cell has a negative DOF.");
          }
          const auto index = static_cast<std::size_t>(offset + dof);
          if (index >= solution.size()) {
            throw std::runtime_error(
                "[svMultiPhysics::Application] " + context +
                " solution is too small for the scalar field.");
          }
          coefficients.push_back(solution[index]);
        }
        return coefficients;
      };
  svmp::FE::interfaces::FreeSurfaceGeometryScalarEvaluator evaluator;
  evaluator.value =
      [space, coefficients_for_cell](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto coefficients = coefficients_for_cell(cell);
        const svmp::FE::spaces::FunctionSpace::Value point{
            reference_point[0], reference_point[1], reference_point[2]};
        const auto value = space->evaluate(point, coefficients);
        return value[0];
      };
  evaluator.reference_gradient =
      [space, coefficients_for_cell](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto coefficients = coefficients_for_cell(cell);
        const svmp::FE::spaces::FunctionSpace::Value point{
            reference_point[0], reference_point[1], reference_point[2]};
        const auto gradient = space->evaluate_gradient(point, coefficients);
        return std::array<svmp::FE::Real, 3>{
            gradient[0], gradient[1], gradient[2]};
      };
  return evaluator;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
