/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLER_SELECTION_H
#define SVMP_FE_ASSEMBLY_ASSEMBLER_SELECTION_H

/**
 * @file AssemblerSelection.h
 * @brief Form/system characteristics used for assembler selection and validation
 */

#include "Core/Types.h"
#include "Assembly/AssemblyKernel.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Summary of compiled-form requirements relevant to assembly strategy selection
 *
 * This is intentionally conservative: it represents the union of requirements
 * across all kernels that may be assembled by a single Assembler instance.
 */
struct FormCharacteristics {
    bool has_cell_terms{false};
    bool has_boundary_terms{false};
    bool has_interior_face_terms{false};
    bool has_global_terms{false};
    bool has_field_requirements{false};
    bool has_parameter_specs{false};

    int max_time_derivative_order{0};
    RequiredData required_data{RequiredData::None};

    [[nodiscard]] bool isTransient() const noexcept { return max_time_derivative_order > 0; }

    [[nodiscard]] bool needsDG() const noexcept
    {
        return has_interior_face_terms ||
               hasFlag(required_data, RequiredData::NeighborData) ||
               hasFlag(required_data, RequiredData::FaceOrientations);
    }

    [[nodiscard]] bool needsMaterialState() const noexcept
    {
        return hasFlag(required_data, RequiredData::MaterialState);
    }

    [[nodiscard]] bool needsSolution() const noexcept
    {
        return hasFlag(required_data, RequiredData::SolutionCoefficients) ||
               hasFlag(required_data, RequiredData::SolutionValues) ||
               hasFlag(required_data, RequiredData::SolutionGradients) ||
               hasFlag(required_data, RequiredData::SolutionHessians) ||
               hasFlag(required_data, RequiredData::SolutionLaplacians);
    }

    [[nodiscard]] bool needsFieldSolutions() const noexcept { return has_field_requirements; }
};

/**
 * @brief System-level characteristics relevant to assembly selection
 */
struct SystemCharacteristics {
    std::size_t num_fields{0};
    GlobalIndex num_cells{0};
    int dimension{0};

    GlobalIndex num_dofs_total{0};
    LocalIndex max_dofs_per_cell{0};
    int max_polynomial_order{0};

    int num_threads{1};
    int mpi_world_size{1};
};

inline void mergeFormCharacteristics(FormCharacteristics& dst, const FormCharacteristics& src) noexcept
{
    dst.has_cell_terms = dst.has_cell_terms || src.has_cell_terms;
    dst.has_boundary_terms = dst.has_boundary_terms || src.has_boundary_terms;
    dst.has_interior_face_terms = dst.has_interior_face_terms || src.has_interior_face_terms;
    dst.has_global_terms = dst.has_global_terms || src.has_global_terms;
    dst.has_field_requirements = dst.has_field_requirements || src.has_field_requirements;
    dst.has_parameter_specs = dst.has_parameter_specs || src.has_parameter_specs;
    dst.max_time_derivative_order = std::max(dst.max_time_derivative_order, src.max_time_derivative_order);
    dst.required_data |= src.required_data;
}

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLER_SELECTION_H
