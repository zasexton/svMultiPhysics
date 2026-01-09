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

#ifndef SVMP_FE_ASSEMBLY_STANDARD_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_STANDARD_ASSEMBLER_H

/**
 * @file StandardAssembler.h
 * @brief Traditional element-by-element assembly
 *
 * StandardAssembler implements the classic FEM assembly algorithm:
 * 1. Loop over elements
 * 2. For each element:
 *    a) Get element DOFs from DofMap
 *    b) Prepare AssemblyContext with geometry/basis data
 *    c) Call kernel to compute local matrix/vector
 *    d) Insert local contributions into global system
 *
 * This is the simplest assembler strategy, suitable for:
 * - Sequential (single-threaded) assembly
 * - Small to medium-sized problems
 * - Debugging and verification
 * - Reference implementation for testing other strategies
 *
 * Features:
 * - Support for rectangular assembly (test_space != trial_space)
 * - Constraint-aware assembly via ConstraintDistributor
 * - Cell, boundary face, and interior face integration
 * - Deterministic results (stable insertion order)
 *
 * Limitations:
 * - Not thread-safe (use ColoredAssembler or WorkStreamAssembler for parallel)
 * - No communication hiding for distributed assembly
 *
 * @see ParallelAssembler for MPI-parallel extension
 * @see ColoredAssembler for thread-parallel assembly
 */

#include "Assembler.h"
#include "GlobalSystemView.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"

#include <memory>
#include <vector>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
}

namespace constraints {
    class AffineConstraints;
    class ConstraintDistributor;
}

namespace sparsity {
    class SparsityPattern;
}

namespace spaces {
    class FunctionSpace;
}

namespace elements {
    class Element;
}

namespace assembly {

/**
 * @brief Traditional element-by-element assembler
 *
 * StandardAssembler provides a straightforward implementation of the
 * finite element assembly algorithm. It iterates over mesh cells,
 * computes local element matrices/vectors, and inserts them into
 * the global system.
 *
 * Usage:
 * @code
 *   StandardAssembler assembler;
 *   assembler.setDofMap(dof_map);
 *   assembler.setOptions(options);
 *   assembler.initialize();
 *
 *   DenseSystemView system(n_dofs);
 *   PoissonKernel kernel(source_func);
 *
 *   auto result = assembler.assembleBoth(mesh, space, kernel, system, system);
 *
 *   assembler.finalize(&system, &system);
 * @endcode
 */
class StandardAssembler : public Assembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    StandardAssembler();

    /**
     * @brief Construct with options
     */
    explicit StandardAssembler(const AssemblyOptions& options);

    /**
     * @brief Destructor
     */
    ~StandardAssembler() override;

    /**
     * @brief Move constructor
     */
    StandardAssembler(StandardAssembler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    StandardAssembler& operator=(StandardAssembler&& other) noexcept;

    // Non-copyable
    StandardAssembler(const StandardAssembler&) = delete;
    StandardAssembler& operator=(const StandardAssembler&) = delete;

    // =========================================================================
    // Configuration (Assembler interface)
    // =========================================================================

    using Assembler::assembleBoth;
    using Assembler::assembleBoundaryFaces;
    using Assembler::assembleMatrix;

    void setDofMap(const dofs::DofMap& dof_map) override;
    void setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset = 0) override;
    void setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset = 0) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setConstraints(const constraints::AffineConstraints* constraints) override;
    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override;
    void setOptions(const AssemblyOptions& options) override;
    void setCurrentSolution(std::span<const Real> solution) override;
    void setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields) override;
    void setPreviousSolution(std::span<const Real> solution) override;
    void setPreviousSolution2(std::span<const Real> solution) override;
    void setPreviousSolutionK(int k, std::span<const Real> solution) override;
    void setTimeIntegrationContext(const TimeIntegrationContext* ctx) override;
    void setTime(Real time) override;
    void setTimeStep(Real dt) override;
    void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept override;
    void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept override;
    void setUserData(const void* user_data) noexcept override;
    void setJITConstants(std::span<const Real> constants) noexcept override;
    void setCoupledValues(std::span<const Real> integrals,
                          std::span<const Real> aux_state) noexcept override;
    void setMaterialStateProvider(IMaterialStateProvider* provider) noexcept override;
    [[nodiscard]] const AssemblyOptions& getOptions() const noexcept override;

    // =========================================================================
    // Matrix Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    // =========================================================================
    // Vector Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    // =========================================================================
    // Combined Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    // =========================================================================
    // Face Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    [[nodiscard]] AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    [[nodiscard]] AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void initialize() override;
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override;
    void reset() override;

    // =========================================================================
    // Query
    // =========================================================================

    [[nodiscard]] std::string name() const override { return "StandardAssembler"; }
    [[nodiscard]] bool isConfigured() const noexcept override;
    [[nodiscard]] bool supportsRectangular() const noexcept override { return true; }
    [[nodiscard]] bool supportsDG() const noexcept override { return true; }
    [[nodiscard]] bool supportsFullContext() const noexcept override { return true; }
    [[nodiscard]] bool supportsSolution() const noexcept override { return true; }
    [[nodiscard]] bool supportsSolutionHistory() const noexcept override { return true; }
    [[nodiscard]] bool supportsTimeIntegrationContext() const noexcept override { return true; }
    [[nodiscard]] bool supportsDofOffsets() const noexcept override { return true; }
    [[nodiscard]] bool supportsFieldRequirements() const noexcept override { return true; }
    [[nodiscard]] bool supportsMaterialState() const noexcept override { return true; }
    [[nodiscard]] bool isThreadSafe() const noexcept override { return false; }

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Core assembly loop for cells
     */
    AssemblyResult assembleCellsCore(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view,
        bool assemble_matrix,
        bool assemble_vector);

    /**
     * @brief Prepare assembly context for a cell
     *
     * This method fully populates the AssemblyContext with:
     * - Quadrature points and weights from the element's quadrature rule
     * - Physical points computed via geometry mapping
     * - Jacobians and determinants at each quadrature point
     * - Basis function values and reference gradients
     * - Physical gradients transformed via inverse Jacobian
     */
    void prepareContext(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data);

    /**
     * @brief Prepare assembly context for a face
     *
     * Similar to prepareContext but for boundary/interior face integration.
     * Additionally computes face normals for face contexts.
     */
    void prepareContextFace(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex face_id,
        GlobalIndex cell_id,
        LocalIndex local_face_id,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        ContextType type,
        std::span<const LocalIndex> align_facet_to_reference = {});

    const FieldSolutionAccess* findFieldSolutionAccess(FieldId field) const noexcept;

    void populateFieldSolutionData(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const std::vector<FieldRequirement>& requirements);

    /**
     * @brief Insert local contributions into global system
     */
    void insertLocal(
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    /**
     * @brief Insert with constraint distribution
     */
    void insertLocalConstrained(
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    /**
     * @brief Get element from function space for a cell
     */
    const elements::Element& getElement(
        const spaces::FunctionSpace& space,
        GlobalIndex cell_id,
        ElementType cell_type) const;

    /**
     * @brief Compute face normal for a given local face in reference coordinates
     *
     * Returns the outward-pointing unit normal for the specified face.
     * This is a simplified implementation that returns reference element normals;
     * physical normals require the Jacobian transformation.
     */
    AssemblyContext::Vector3D computeFaceNormal(
        LocalIndex local_face_id,
        ElementType cell_type,
        int dim) const;

    /**
     * @brief Compute surface measure and physical unit normal for face integration
     *
     * Given the reference-space normal n_ref, the inverse Jacobian J^{-1}, and
     * the Jacobian determinant det(J), computes:
     *
     * 1. Surface measure = ||J^{-T} * n_ref|| * |det(J)| * dS_ref
     *    This is the correct scaling for integrating over the physical face.
     *
     * 2. Physical unit normal = normalize(J^{-T} * n_ref)
     *    The transformation J^{-T} maps reference normals to physical normals
     *    (not unit length).
     *
     * Mathematical background:
     * - For a mapping x = F(xi) from reference to physical space,
     *   the surface area element transforms as:
     *     dS_phys = ||cof(J) * n_ref|| * dS_ref = ||J^{-T} * n_ref|| * |det(J)| * dS_ref
     * - The physical normal direction is given by J^{-T} * n_ref (unnormalized)
     *
     * @param n_ref Reference-space unit normal for the face
     * @param J_inv Inverse Jacobian matrix at the quadrature point
     * @param det_J Jacobian determinant at the quadrature point
     * @param dim Spatial dimension (2 or 3)
     * @param[out] surface_measure The computed surface integration weight factor
     * @param[out] n_phys The computed physical unit normal
     */
    void computeSurfaceMeasureAndNormal(
        const AssemblyContext::Vector3D& n_ref,
        const AssemblyContext::Matrix3x3& J_inv,
        Real det_J,
        int dim,
        Real& surface_measure,
        AssemblyContext::Vector3D& n_phys) const;

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    AssemblyOptions options_;
    const dofs::DofMap* row_dof_map_{nullptr};
    const dofs::DofMap* col_dof_map_{nullptr};
    GlobalIndex row_dof_offset_{0};
    GlobalIndex col_dof_offset_{0};
    const dofs::DofHandler* dof_handler_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};
    const sparsity::SparsityPattern* sparsity_{nullptr};
    std::unique_ptr<constraints::ConstraintDistributor> constraint_distributor_;

    // Working storage
    AssemblyContext context_;
    KernelOutput kernel_output_;
    std::vector<GlobalIndex> row_dofs_;
    std::vector<GlobalIndex> col_dofs_;

    // Scratch for constraint distribution
    std::vector<GlobalIndex> scratch_rows_;
    std::vector<GlobalIndex> scratch_cols_;
    std::vector<Real> scratch_matrix_;
    std::vector<Real> scratch_vector_;

    // Scratch for geometry/basis computations
    std::vector<std::array<Real, 3>> cell_coords_;
    std::vector<AssemblyContext::Point3D> scratch_quad_points_;
    std::vector<Real> scratch_quad_weights_;
    std::vector<AssemblyContext::Point3D> scratch_phys_points_;
    std::vector<AssemblyContext::Matrix3x3> scratch_jacobians_;
    std::vector<AssemblyContext::Matrix3x3> scratch_inv_jacobians_;
    std::vector<Real> scratch_jac_dets_;
    std::vector<Real> scratch_integration_weights_;
    std::vector<Real> scratch_basis_values_;
    std::vector<AssemblyContext::Vector3D> scratch_ref_gradients_;
    std::vector<AssemblyContext::Vector3D> scratch_phys_gradients_;
    std::vector<AssemblyContext::Matrix3x3> scratch_ref_hessians_;
    std::vector<AssemblyContext::Matrix3x3> scratch_phys_hessians_;
    std::vector<AssemblyContext::Vector3D> scratch_normals_;

    // State
    std::span<const Real> current_solution_{};
    std::vector<std::span<const Real>> previous_solutions_{};
    std::vector<Real> local_solution_coeffs_{};
    std::vector<std::vector<Real>> local_prev_solution_coeffs_{};
    Real time_{0.0};
    Real dt_{0.0};
    const std::function<std::optional<Real>(std::string_view)>* get_real_param_{nullptr};
    const std::function<std::optional<params::Value>(std::string_view)>* get_param_{nullptr};
    const void* user_data_{nullptr};
    std::span<const Real> jit_constants_{};
    std::span<const Real> coupled_integrals_{};
    std::span<const Real> coupled_aux_state_{};
    const TimeIntegrationContext* time_integration_{nullptr};
    IMaterialStateProvider* material_state_provider_{nullptr};
    bool initialized_{false};

    std::vector<FieldSolutionAccess> field_solution_access_{};
};

// ============================================================================
// Factory Function
// ============================================================================

/**
 * @brief Create a standard assembler with default options
 */
std::unique_ptr<Assembler> createStandardAssembler();

/**
 * @brief Create a standard assembler with specified options
 */
std::unique_ptr<Assembler> createStandardAssembler(const AssemblyOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_STANDARD_ASSEMBLER_H
