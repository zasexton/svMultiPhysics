/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/SnapshotFieldEvaluator.h"

#include "Assembly/Assembler.h"
#include "Basis/BasisFunction.h"
#include "Elements/Element.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Geometry/GeometryMapping.h"
#include "Geometry/LinearMapping.h"
#include "Geometry/MappingFactory.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {

namespace {

std::shared_ptr<svmp::FE::geometry::GeometryMapping> createCellGeometryMapping(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell)
{
  if (cell < 0 || cell >= mesh.numCells()) {
    return nullptr;
  }

  std::vector<std::array<svmp::FE::Real, 3>> coords;
  mesh.getCellCoordinates(cell, coords);
  if (coords.empty()) {
    return nullptr;
  }

  std::vector<svmp::FE::math::Vector<svmp::FE::Real, 3>> nodes;
  nodes.reserve(coords.size());
  for (const auto& coord : coords) {
    svmp::FE::math::Vector<svmp::FE::Real, 3> node{};
    node[0] = coord[0];
    node[1] = coord[1];
    node[2] = coord[2];
    nodes.push_back(node);
  }

  svmp::FE::geometry::MappingRequest map_request;
  map_request.element_type = mesh.getCellType(cell);
  map_request.geometry_order = mesh.getCellGeometryOrder(cell);
  map_request.use_affine = map_request.geometry_order <= 1;
  return svmp::FE::geometry::MappingFactory::create(map_request, nodes);
}

} // namespace

svmp::FE::interfaces::SnapshotVectorEvaluator
makeSnapshotVectorFieldEvaluator(
    const SnapshotFieldAccess& record,
    std::span<const svmp::FE::Real> solution,
    std::string_view diagnostic_context,
    bool allow_scalar_product_fast_path,
    std::string_view field_name)
{
  const std::string context(diagnostic_context);
  const auto offset = record.dof_offset;
  const std::string name(field_name);
  const auto& field_mesh = record.mesh();
  const int dimension = field_mesh.dimension();
  if (!record.space || offset < 0 ||
      record.components < dimension ||
      record.space->value_dimension() < dimension) {
    throw std::runtime_error(
        context + " " + name + " field is incompatible with its level-set declaration.");
  }

  const auto cell_dofs_of = record.cell_dofs;
  auto* const mesh_access = &field_mesh;
  const auto space = record.space;

  // The diagnostic loops visit all points of one parent cell in a row.  The
  // scratch below keeps that cell's coefficients and geometry mapping, the
  // inverse Jacobian of an affine LinearMapping (whose Jacobian ignores the
  // reference point), and reusable basis buffers, so a point no longer
  // allocates.  Every returned value keeps its exact arithmetic.  Like the
  // mapping cache before it, the scratch makes the evaluator single-threaded.
  struct VectorEvaluationScratch {
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
  auto scratch = std::make_shared<VectorEvaluationScratch>();
  const auto cell_coefficients =
      [cell_dofs_of, offset, solution, context, name, scratch](
          svmp::FE::GlobalIndex cell,
          bool gradient)
      -> const std::vector<svmp::FE::Real>& {
        auto& state = *scratch;
        if (state.coefficients_valid && state.coefficient_cell == cell) {
          return state.coefficients;
        }
        state.coefficients_valid = false;
        const auto cell_dofs = cell_dofs_of(cell);
        state.coefficients.clear();
        state.coefficients.reserve(cell_dofs.size());
        for (const auto dof : cell_dofs) {
          if (dof < 0) {
            throw std::runtime_error(
                context + " " + name +
                (gradient ? "-gradient cell has a negative DOF." : " cell has a negative DOF."));
          }
          const auto index =
              static_cast<std::size_t>(offset + dof);
          if (index >= solution.size()) {
            throw std::runtime_error(
                context + " solution is too small for the " + name +
                (gradient ? " gradient." : " field."));
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
            space.get());
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

  SnapshotVectorEvaluator evaluator;
  evaluator.value =
      [space,
       scratch,
       cell_coefficients,
       scalar_element,
       scalar_dofs,
       product_components](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto& coefficients = cell_coefficients(cell, /*gradient=*/false);
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
            space->element_space(cell).evaluate(reference_value, coefficients);
        return std::array<svmp::FE::Real, 3>{
            value[0], value[1], value[2]};
      };
  evaluator.physical_gradient =
      [mesh_access,
       space,
       context,
       name,
       scratch,
       cell_coefficients,
       scalar_element,
       scalar_dofs,
       product_components](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto& coefficients = cell_coefficients(cell, /*gradient=*/true);
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
          const auto jacobian = space->element_space(cell).evaluate_jacobian(
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
                context + " could not construct the " + name + "-gradient geometry mapping.");
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
            SnapshotVectorGradient
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
  return evaluator;
}

svmp::FE::interfaces::LevelSetGeometryScalarEvaluator
makeSnapshotScalarFieldEvaluator(
    const SnapshotFieldAccess& record,
    std::span<const svmp::FE::Real> solution,
    std::string_view diagnostic_context)
{
  const std::string context(diagnostic_context);
  const auto offset = record.dof_offset;
  if (!record.space || offset < 0 || record.components != 1 ||
      record.space->value_dimension() != 1) {
    throw std::runtime_error(
        context +
        " scalar field is incompatible with its diagnostic declaration.");
  }
  const auto cell_dofs_of = record.cell_dofs;
  const auto space = record.space;
  const auto coefficients_for_cell =
      [cell_dofs_of,
       offset,
       solution,
       context](svmp::FE::GlobalIndex cell) {
        const auto cell_dofs = cell_dofs_of(cell);
        std::vector<svmp::FE::Real> coefficients;
        coefficients.reserve(cell_dofs.size());
        for (const auto dof : cell_dofs) {
          if (dof < 0) {
            throw std::runtime_error(
                context +
                " scalar cell has a negative DOF.");
          }
          const auto index = static_cast<std::size_t>(offset + dof);
          if (index >= solution.size()) {
            throw std::runtime_error(
                context +
                " solution is too small for the scalar field.");
          }
          coefficients.push_back(solution[index]);
        }
        return coefficients;
      };
  svmp::FE::interfaces::LevelSetGeometryScalarEvaluator evaluator;
  evaluator.value =
      [space, coefficients_for_cell](
          svmp::FE::GlobalIndex cell,
          const std::array<svmp::FE::Real, 3>& reference_point,
          const svmp::FE::geometry::CutQuadratureProvenance&) {
        const auto coefficients = coefficients_for_cell(cell);
        const svmp::FE::spaces::FunctionSpace::Value point{
            reference_point[0], reference_point[1], reference_point[2]};
        const auto value = space->element_space(cell).evaluate(point, coefficients);
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
        const auto gradient = space->element_space(cell).evaluate_gradient(point, coefficients);
        return std::array<svmp::FE::Real, 3>{
            gradient[0], gradient[1], gradient[2]};
      };
  return evaluator;
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
