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
#include "Spaces/OrientationManager.h"
#include "Geometry/GeometryMapping.h"
#include "Basis/BasisCache.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
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

namespace quadrature {
    class QuadratureRule;
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
 * When the provided mesh view contains unowned/ghost cells,
 * StandardAssembler assembles through an owned-row filter. It does not
 * implement the ghost buffering and reverse-scatter exchange required by
 * GhostPolicy::ReverseScatter; callers that need that policy on ghosted
 * distributed meshes must use ParallelAssembler instead. On ownership-complete
 * mesh views (for example, a serial reference mesh), it can assemble all rows.
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
    void setRowDofMap(const dofs::DofMap& dof_map,
                      GlobalIndex row_offset = 0,
                      DofEntityScope row_scope = DofEntityScope::Cell) override;
    void setColDofMap(const dofs::DofMap& dof_map,
                      GlobalIndex col_offset = 0,
                      DofEntityScope col_scope = DofEntityScope::Cell) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setConstraints(const constraints::AffineConstraints* constraints) override;
    void setSuppressConstraintInhomogeneity(bool suppress) override;
    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override;
    void setOptions(const AssemblyOptions& options) override;
	    void setCurrentSolution(std::span<const Real> solution) override;
	    void setCurrentSolutionView(const GlobalSystemView* solution_view) override;
	    void setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields) override;
	    void setMeshMotionFieldAccess(const MeshMotionFieldAccess& fields) override;
	    void setPreviousSolution(std::span<const Real> solution) override;
	    void setPreviousSolution2(std::span<const Real> solution) override;
	    void setPreviousSolutionView(const GlobalSystemView* solution_view) override;
	    void setPreviousSolution2View(const GlobalSystemView* solution_view) override;
	    void setPreviousSolutionK(int k, std::span<const Real> solution) override;
	    void setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view) override;
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
    void setAuxiliaryValues(std::span<const Real> inputs,
                            std::span<const Real> state,
                            std::span<const Real> outputs = {}) noexcept override;
    void setAuxiliaryOutputBindings(
        std::span<const AuxiliaryOutputBinding> bindings) noexcept override;
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
    // Fused Multi-Term Cell Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleCellsFused(
        const IMeshAccess& mesh,
        std::span<const FusedCellTerm> terms) override;

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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    [[nodiscard]] AssemblyResult assembleInterfaceFaces(
        const IMeshAccess& mesh,
        const svmp::InterfaceMesh& interface_mesh,
        int interface_marker,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;
#endif

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void initialize() override;
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override;
    void reset() override;
    void invalidateGeometryCaches() override;
    void invalidateTopologyLayoutCaches() override;

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
    /**
     * @brief Apply global->local orientation transforms to vector-basis coefficients
     *
     * This handles H(curl)/H(div) entity orientations by transforming the gathered
     * element-local coefficient vector (indexed in element-local DOF order) into
     * a coefficient vector compatible with reference-element basis evaluation on
     * this cell.
     */
    void applyVectorBasisGlobalToLocal(
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const spaces::FunctionSpace& space,
        std::span<Real> coeffs);

    /**
     * @brief Apply local->global orientation transforms to a kernel output
     *
     * Transforms element-local residual/matrix entries computed in local basis
     * into the global-orientation DOF basis used by the system:
     *   r_g = P_test^T r_l
     *   A_g = P_test^T A_l P_trial
     */
    void applyVectorBasisOutputOrientation(
        const IMeshAccess& mesh,
        GlobalIndex test_cell_id,
        const spaces::FunctionSpace& test_space,
        GlobalIndex trial_cell_id,
        const spaces::FunctionSpace& trial_space,
        KernelOutput& output);

    struct FaceTransformKey {
        BasisType basis_type{BasisType::Lagrange};
        Continuity continuity{Continuity::C0};
        ElementType face_type{ElementType::Unknown};
        int poly_order{0};
        std::array<int, 4> vertex_perm{{-1, -1, -1, -1}};
        std::uint8_t n_verts{0};
        int sign{+1};

        [[nodiscard]] bool operator==(const FaceTransformKey& other) const noexcept
        {
            return basis_type == other.basis_type &&
                   continuity == other.continuity &&
                   face_type == other.face_type &&
                   poly_order == other.poly_order &&
                   vertex_perm == other.vertex_perm &&
                   n_verts == other.n_verts &&
                   sign == other.sign;
        }
    };

    struct FaceTransformKeyHash {
        [[nodiscard]] std::size_t operator()(const FaceTransformKey& key) const noexcept
        {
            std::size_t h = 0;
            auto mix = [&](std::size_t v) {
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };

            mix(static_cast<std::size_t>(key.basis_type));
            mix(static_cast<std::size_t>(key.continuity));
            mix(static_cast<std::size_t>(key.face_type));
            mix(static_cast<std::size_t>(key.poly_order));
            mix(static_cast<std::size_t>(key.n_verts));
            for (std::size_t i = 0; i < static_cast<std::size_t>(key.n_verts) && i < key.vertex_perm.size(); ++i) {
                mix(static_cast<std::size_t>(key.vertex_perm[i] + 17));
            }
            mix(static_cast<std::size_t>(key.sign + 3));
            return h;
        }
    };

    struct FaceTransform {
        LocalIndex n{0};
        std::vector<Real> P{};   // global->local
        std::vector<Real> PT{};  // transpose(P)
    };

    /**
     * @brief Saved node coordinates for fused-batched assembly.
     *
     * Geometry data in the AssemblyContext arena survives prepareBasis() calls
     * (configure() only clears hessians/vector-basis, not geometry arrays, and
     * the arena is pre-reserved so no reallocation occurs). Only
     * scratch_node_coords_ (an assembler member) needs saving per batch slot.
     */
    struct SavedCellNodeCoords {
        std::vector<math::Vector<Real, 3>> node_coords;
        Real entity_h{0.0};
        Real entity_volume{0.0};
    };

    /** Per-slot DOF indices for fused-batched assembly. */
    struct SlotDofs {
        std::span<const GlobalIndex> row_dofs{};
        std::span<const GlobalIndex> col_dofs{};
    };

    struct CombinedInsertBlockInfo {
        int row_comp_start{0};
        int col_comp_start{0};
        int row_comps{0};
        int col_comps{0};
    };

    struct CombinedInsertTarget {
        GlobalSystemView* matrix_view{nullptr};
        GlobalSystemView* vector_view{nullptr};
        bool assemble_matrix{false};
        bool assemble_vector{false};
    };

    /**
     * @brief Per-thread mutable state for prepareGeometry in colored parallel assembly.
     *
     * When passed to prepareGeometry, the function uses these fields instead of
     * member variables, making it safe for concurrent calls from multiple threads.
     */
    struct GeometryWorkspace {
        std::vector<std::array<Real, 3>> cell_coords;
        std::vector<math::Vector<Real, 3>> node_coords;
        std::shared_ptr<geometry::GeometryMapping> mapping;
        ElementType mapping_type{ElementType::Unknown};
        int mapping_order{-1};
        bool mapping_affine{false};
        Real geom_h{0.0};
        Real geom_volume{0.0};
        const basis::BasisCacheEntry* geom_bcache{nullptr};
    };

    /**
     * @brief Per-thread mutable scratch for populateFieldSolutionData in colored
     *        parallel assembly.
     */
    struct FieldSolutionWorkspace {
        std::vector<Real> scalar_values_at_pt;
        std::vector<basis::Gradient> scalar_gradients_at_pt;
        std::vector<basis::Hessian> scalar_hessians_at_pt;
        std::vector<Real> field_local_coeffs;
        std::vector<Real> fsd_scalar_values;
        std::vector<AssemblyContext::Vector3D> fsd_scalar_gradients;
        std::vector<AssemblyContext::Matrix3x3> fsd_scalar_hessians;
        std::vector<Real> fsd_scalar_laplacians;
        std::vector<AssemblyContext::Vector3D> fsd_vector_values;
        std::vector<AssemblyContext::Matrix3x3> fsd_vector_jacobians;
        std::vector<AssemblyContext::Matrix3x3> fsd_vector_comp_hessians;
        std::vector<Real> fsd_vector_comp_laplacians;
        std::vector<math::Vector<Real, 3>> vec_values_at_pt;
        std::vector<basis::VectorJacobian> vec_jacobians_at_pt;
        std::vector<math::Vector<Real, 3>> vec_curls_at_pt;
        std::vector<Real> vec_divs_at_pt;
    };

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
     *
     * Internally calls prepareGeometry() + prepareBasis().
     */
    void prepareContext(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data);

    /**
     * @brief Resolve quadrature rule for a test space element on a cell
     */
    std::shared_ptr<const quadrature::QuadratureRule> resolveQuadratureRule(
        const spaces::FunctionSpace& test_space,
        GlobalIndex cell_id,
        ElementType cell_type) const;

    /**
     * @brief Prepare geometry data for a cell (cell-only, space-independent)
     *
     * Computes: node coordinates, geometry mapping, Jacobians, physical points,
     * integration weights at all quadrature points. The results are stored in
     * scratch arrays and are ready for subsequent prepareBasis() calls.
     */
    void prepareGeometry(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const quadrature::QuadratureRule& quad_rule);

    /**
     * @brief Thread-safe variant of prepareGeometry using explicit workspace.
     *
     * When ws is non-null, all mutable state (cell_coords, node_coords,
     * mapping, geom_h, geom_volume) is read/written through the workspace
     * instead of member variables.
     */
    void prepareGeometry(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const quadrature::QuadratureRule& quad_rule,
        GeometryWorkspace& ws);

    /**
     * @brief Prepare basis data and configure context for test/trial spaces
     *
     * Evaluates test/trial basis functions at QPs, transforms gradients using
     * the Jacobians already computed by prepareGeometry(), and sets up the
     * AssemblyContext with all basis and geometry data.
     */
    void prepareBasis(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        const quadrature::QuadratureRule& quad_rule);

    struct FrameGeometryScratch {
        std::vector<AssemblyContext::Point3D> points{};
        std::vector<AssemblyContext::Matrix3x3> jacobians{};
        std::vector<AssemblyContext::Matrix3x3> inverse_jacobians{};
        std::vector<Real> jacobian_determinants{};
        std::vector<Real> measures{};
        std::vector<std::array<Real, 3>> cell_coords{};
        std::vector<math::Vector<Real, 3>> node_coords{};
    };

    struct FaceFrameGeometryScratch {
        FrameGeometryScratch cell{};
        std::vector<AssemblyContext::Point3D> cell_reference_points{};
        std::vector<AssemblyContext::Point3D> face_reference_points{};
        std::vector<Real> canonical_to_reference_measures{};
        std::vector<AssemblyContext::Vector3D> normals{};
        std::vector<Real> surface_measures{};
        std::vector<AssemblyContext::Matrix3x3> surface_jacobians{};
        std::vector<std::array<Real, 3>> cell_coords{};
    };

    void prepareFrameExplicitGeometry(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        ElementType cell_type,
        const quadrature::QuadratureRule& quad_rule,
        RequiredData required_data);

    void computeFrameGeometry(
        ElementType cell_type,
        const quadrature::QuadratureRule& quad_rule,
        FrameGeometryScratch& scratch);

    void computeFaceFrameGeometry(
        ElementType cell_type,
        LocalIndex local_face_id,
        ElementType face_type,
        const quadrature::QuadratureRule& quad_rule,
        std::span<const LocalIndex> align_facet_to_reference,
        FaceFrameGeometryScratch& scratch);

    [[nodiscard]] std::vector<FieldRequirement> effectiveFieldRequirements(
        RequiredData required_data,
        std::span<const FieldRequirement> kernel_requirements) const;

    void validateMovingDomainRequirements(RequiredData required_data,
                                          const char* error_prefix) const;
    void populateMovingDomainFieldData(AssemblyContext& context,
                                       RequiredData required_data,
                                       const char* error_prefix);


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
        std::span<const LocalIndex> align_facet_to_reference = {},
        bool force_test_face_reference_coords = false,
        bool force_trial_face_reference_coords = false);

    struct FieldAccessPlan;
    void ensureFieldAccessPlans(const IMeshAccess& mesh);
    [[nodiscard]] const FieldAccessPlan* findFieldAccessPlan(FieldId field) const noexcept;

    void populateFieldSolutionData(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const std::vector<FieldRequirement>& requirements);

    /**
     * @brief Thread-safe variant using explicit field solution workspace.
     */
    void populateFieldSolutionData(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const std::vector<FieldRequirement>& requirements,
        FieldSolutionWorkspace& ws);

    /**
     * @brief Insert local contributions into global system
     */
    void insertLocal(
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view,
        std::span<const GlobalIndex> resolved_matrix_entries = {},
        std::span<const GlobalIndex> resolved_vector_entries = {});

    /**
     * @brief Insert with constraint distribution
     */
    void insertLocalConstrained(
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    struct CellDofTable;
    void ensureCellDofTables(const IMeshAccess& mesh);
    [[nodiscard]] const CellDofTable& getCellDofTable(
        const IMeshAccess& mesh,
        const dofs::DofMap* dof_map,
        GlobalIndex dof_offset);
    [[nodiscard]] std::span<const GlobalIndex> getCellDofsCached(
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const dofs::DofMap* dof_map,
        GlobalIndex dof_offset);
    [[nodiscard]] std::span<const GlobalIndex> getCellDofsFromTable(
        const CellDofTable& table,
        GlobalIndex cell_id) const;
    void ensureResolvedVectorTables(const IMeshAccess& mesh);
    void ensureResolvedVectorTable(
        const IMeshAccess& mesh,
        const dofs::DofMap* dof_map,
        GlobalIndex dof_offset,
        const GlobalSystemView* view);
    void ensureResolvedMatrixTable(
        const IMeshAccess& mesh,
        const dofs::DofMap* row_dof_map,
        GlobalIndex row_dof_offset,
        const dofs::DofMap* col_dof_map,
        GlobalIndex col_dof_offset,
        const GlobalSystemView* view);
    void ensureCellConstrainedFlags(const IMeshAccess& mesh);
    [[nodiscard]] std::span<const GlobalIndex> getResolvedCellVectorEntries(
        GlobalIndex cell_id,
        const dofs::DofMap* dof_map,
        GlobalIndex dof_offset,
        const GlobalSystemView* view) const;
    [[nodiscard]] std::span<const GlobalIndex> getResolvedCellMatrixEntries(
        GlobalIndex cell_id,
        const dofs::DofMap* row_dof_map,
        GlobalIndex row_dof_offset,
        const dofs::DofMap* col_dof_map,
        GlobalIndex col_dof_offset,
        const GlobalSystemView* view) const;
    void gatherCellVectorCoefficients(
        GlobalIndex cell_id,
        const dofs::DofMap* dof_map,
        GlobalIndex dof_offset,
        std::span<const GlobalIndex> dofs,
        const GlobalSystemView* view,
        std::span<const Real> raw_values,
        std::vector<Real>& out,
        const char* error_prefix,
        bool validate_negative_dofs);
    void insertLocalForCell(
        GlobalIndex cell_id,
        const dofs::DofMap* row_dof_map,
        GlobalIndex row_dof_offset,
        const dofs::DofMap* col_dof_map,
        GlobalIndex col_dof_offset,
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);
    void resizeCombinedInsertScratch(std::size_t batch_size, int combined_n);
    void zeroCombinedInsertScratch(std::size_t active, int combined_n);
    void scatterCombinedInsertBlockOutput(
        std::size_t slot,
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        const CombinedInsertBlockInfo& info,
        int total_comps,
        int combined_n,
        bool want_matrix,
        bool want_vector);
    void flushCombinedInsertBatch(
        std::span<const GlobalIndex> batch_cell_ids,
        int combined_n,
        const CombinedInsertTarget& target);

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

    [[nodiscard]] const FaceTransform& getFaceTransform(
        BasisType basis_type,
        Continuity continuity,
        ElementType face_type,
        int poly_order,
        const spaces::OrientationManager::FaceOrientation& orientation,
        LocalIndex expected_size);

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    AssemblyOptions options_;
    const dofs::DofMap* row_dof_map_{nullptr};
    const dofs::DofMap* col_dof_map_{nullptr};
    GlobalIndex row_dof_offset_{0};
    GlobalIndex col_dof_offset_{0};
    DofEntityScope row_dof_scope_{DofEntityScope::Cell};
    DofEntityScope col_dof_scope_{DofEntityScope::Cell};
    const dofs::DofHandler* dof_handler_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};
    const sparsity::SparsityPattern* sparsity_{nullptr};
    std::unique_ptr<constraints::ConstraintDistributor> constraint_distributor_;
    bool suppress_constraint_inhomogeneity_{false};

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
    std::vector<Real> scratch_orient_in_;
    std::vector<Real> scratch_orient_out_;

    // Scratch for batch constraint expansion (Phase 2 optimization).
    // Buffered adapters collect expanded entries here; flushed in one batch call.
    std::vector<GlobalIndex> scratch_expanded_rows_;
    std::vector<GlobalIndex> scratch_expanded_cols_;
    std::vector<Real> scratch_expanded_matrix_vals_;
    std::vector<GlobalIndex> scratch_expanded_vec_dofs_;
    std::vector<Real> scratch_expanded_vec_vals_;

    // Scratch for geometry/basis computations
    std::vector<std::array<Real, 3>> cell_coords_;
    std::vector<AssemblyContext::Point3D> scratch_quad_points_;
    std::vector<AssemblyContext::Point3D> scratch_face_quad_points_;
    std::vector<Real> scratch_quad_weights_;
    std::vector<AssemblyContext::Point3D> scratch_phys_points_;
    std::vector<AssemblyContext::Matrix3x3> scratch_jacobians_;
    std::vector<AssemblyContext::Matrix3x3> scratch_inv_jacobians_;
    std::vector<Real> scratch_jac_dets_;
    std::vector<Real> scratch_integration_weights_;
    std::vector<Real> scratch_basis_values_;
    std::vector<AssemblyContext::Vector3D> scratch_basis_vector_values_;
    std::vector<AssemblyContext::Matrix3x3> scratch_basis_vector_jacobians_;
    std::vector<AssemblyContext::Vector3D> scratch_basis_curls_;
    std::vector<Real> scratch_basis_divergences_;
    std::vector<AssemblyContext::Vector3D> scratch_ref_gradients_;
    std::vector<AssemblyContext::Vector3D> scratch_phys_gradients_;
    std::vector<AssemblyContext::Matrix3x3> scratch_ref_hessians_;
    std::vector<AssemblyContext::Matrix3x3> scratch_phys_hessians_;
    // Trial-specific scratch for different-space fast path
    std::vector<Real> scratch_trial_basis_values_;
    std::vector<AssemblyContext::Vector3D> scratch_trial_ref_gradients_;
    std::vector<AssemblyContext::Vector3D> scratch_trial_phys_gradients_;
    std::vector<AssemblyContext::Matrix3x3> scratch_trial_ref_hessians_;
    std::vector<AssemblyContext::Matrix3x3> scratch_trial_phys_hessians_;
    std::vector<AssemblyContext::Vector3D> scratch_normals_;
    FrameGeometryScratch scratch_reference_geometry_;
    FrameGeometryScratch scratch_current_geometry_;
    FaceFrameGeometryScratch scratch_active_face_geometry_;
    FaceFrameGeometryScratch scratch_reference_face_geometry_;
    FaceFrameGeometryScratch scratch_current_face_geometry_;
    std::vector<AssemblyContext::Matrix3x3> scratch_configuration_transforms_;
    std::vector<AssemblyContext::Vector3D> scratch_mesh_motion_values_;
    std::vector<AssemblyContext::Matrix3x3> scratch_mesh_motion_jacobians_;

    // Point-wise evaluation scratch arrays
    std::vector<Real> scratch_scalar_values_at_pt_;
    std::vector<basis::Gradient> scratch_scalar_gradients_at_pt_;
    std::vector<basis::Hessian> scratch_scalar_hessians_at_pt_;
    std::vector<math::Vector<Real, 3>> scratch_vec_values_at_pt_;
    std::vector<basis::VectorJacobian> scratch_vec_jacobians_at_pt_;
    std::vector<math::Vector<Real, 3>> scratch_vec_curls_at_pt_;
    std::vector<Real> scratch_vec_divs_at_pt_;
    std::vector<Real> scratch_field_local_coeffs_;

	    // State
	    std::span<const Real> current_solution_{};
	    const GlobalSystemView* current_solution_view_{nullptr};
	    std::vector<std::span<const Real>> previous_solutions_{};
	    std::vector<const GlobalSystemView*> previous_solution_views_{};
	    std::vector<Real> local_solution_coeffs_{};
	    std::vector<std::vector<Real>> local_prev_solution_coeffs_{};
	    std::vector<GlobalIndex> field_dof_scratch_{};
    Real time_{0.0};
    Real dt_{0.0};
    const std::function<std::optional<Real>(std::string_view)>* get_real_param_{nullptr};
    const std::function<std::optional<params::Value>(std::string_view)>* get_param_{nullptr};
    const void* user_data_{nullptr};
    std::span<const Real> jit_constants_{};
    std::span<const Real> coupled_integrals_{};
    std::span<const Real> coupled_aux_state_{};
    std::span<const Real> auxiliary_inputs_{};
    std::span<const Real> auxiliary_state_{};
    std::span<const Real> auxiliary_outputs_{};
    std::span<const AuxiliaryOutputBinding> auxiliary_output_bindings_{};
    const TimeIntegrationContext* time_integration_{nullptr};
    IMaterialStateProvider* material_state_provider_{nullptr};
    bool initialized_{false};

    std::vector<FieldSolutionAccess> field_solution_access_{};
    MeshMotionFieldAccess mesh_motion_field_access_{};

    std::unordered_map<FaceTransformKey, FaceTransform, FaceTransformKeyHash> face_transform_cache_{};

    // Cached geometry mapping to avoid per-cell heap allocation
    std::shared_ptr<geometry::GeometryMapping> cached_mapping_;
    ElementType cached_mapping_type_{ElementType::Unknown};
    int cached_mapping_order_{-1};
    bool cached_mapping_affine_{false};

    // Entity measures computed during prepareGeometry (reused by prepareBasis fast path
    // to avoid accessing mapping->nodes() — enables skipping coordinate restore for affine
    // elements in the coupled batch path).
    Real cached_geom_h_{0.0};
    Real cached_geom_volume_{0.0};

    // Cached quad rule from prepareContext (used for BasisCache lookups in populateFieldSolutionData)
    std::shared_ptr<const quadrature::QuadratureRule> cached_quad_rule_;

    // Scratch storage for node coordinate conversion (avoids per-cell heap allocation)
    std::vector<math::Vector<Real, 3>> scratch_node_coords_;

    // Scratch storage for populateFieldSolutionData (avoids per-cell heap allocation)
    std::vector<Real> scratch_fsd_scalar_values_;
    std::vector<AssemblyContext::Vector3D> scratch_fsd_scalar_gradients_;
    std::vector<AssemblyContext::Matrix3x3> scratch_fsd_scalar_hessians_;
    std::vector<Real> scratch_fsd_scalar_laplacians_;
    std::vector<AssemblyContext::Vector3D> scratch_fsd_vector_values_;
    std::vector<AssemblyContext::Matrix3x3> scratch_fsd_vector_jacobians_;
    std::vector<AssemblyContext::Matrix3x3> scratch_fsd_vector_component_hessians_;
    std::vector<Real> scratch_fsd_vector_component_laplacians_;

    // Cached BasisCacheEntry pointers (hoisted out of per-cell loop).
    // Invalidated when element type or hessian requirement changes.
    const basis::BasisCacheEntry* cached_geom_bcache_{nullptr};
    const basis::BasisCacheEntry* cached_test_bcache_{nullptr};
    const basis::BasisCacheEntry* cached_trial_bcache_{nullptr};
    bool cached_need_hessians_{false};
    const void* cached_quad_rule_ptr_{nullptr}; // identity check for quad rule change

    // Per-cell basis caching: when cell type + quad rule + hessian requirement
    // match the previous cell, skip BasisCache reads and scratch copies —
    // only recompute physical gradients using new Jacobians.
    bool basis_scratch_valid_{false};
    ElementType cached_basis_cell_type_{ElementType::Unknown};
    LocalIndex cached_basis_n_test_dofs_{0};
    LocalIndex cached_basis_n_trial_dofs_{0};
    LocalIndex cached_basis_n_qpts_{0};
    bool cached_basis_test_is_vector_{false};
    bool cached_basis_trial_is_vector_{false};
    bool cached_basis_same_space_{false};
    bool cached_basis_has_hessians_{false};
    const spaces::FunctionSpace* cached_basis_test_space_ptr_{nullptr};
    const spaces::FunctionSpace* cached_basis_trial_space_ptr_{nullptr};

    // Reserved for future per-context basis caching
    bool cached_qpt_test_valid_{false};
    bool cached_qpt_trial_valid_{false};

    // Cached qpt-major basis data for fast-path transpose elimination.
    // Populated once per block in the slow path; reused for all subsequent
    // fast-path cells. Eliminates dof-major→qpt-major transposes that were
    // previously repeated for every cell.
    std::vector<Real> cached_qpt_test_values_;
    std::vector<AssemblyContext::Vector3D> cached_qpt_test_ref_grads_;
    std::vector<AssemblyContext::Matrix3x3> cached_qpt_test_ref_hess_;
    std::vector<Real> cached_qpt_trial_values_;
    std::vector<AssemblyContext::Vector3D> cached_qpt_trial_ref_grads_;
    std::vector<AssemblyContext::Matrix3x3> cached_qpt_trial_ref_hess_;
    bool cached_qpt_major_valid_{false};
    bool cached_qpt_major_same_space_{false};
    bool cached_qpt_major_has_hessians_{false};

    // Pre-computed coupled-block metadata to avoid virtual calls in fast path.
    // Populated once per block before the cell loop; indexed by block index.
    std::vector<AssemblyContext::CoupledBlockMetadata> cached_coupled_block_meta_;

    // When non-null, prepareBasis fast path uses this instead of configure().
    // Set by the coupled block loop before calling prepareBasis, cleared after.
    const AssemblyContext::CoupledBlockMetadata* active_coupled_block_meta_{nullptr};

    // Field-solution BasisCache: small flat cache keyed by (BasisFunction*, gradients, hessians).
    // Typically 1-2 entries (one per unique basis in coupled blocks). Invalidated with mapping type.
    struct FieldBCacheEntry {
        const basis::BasisFunction* basis{nullptr};
        const quadrature::QuadratureRule* quad{nullptr};
        bool gradients{false};
        bool hessians{false};
        const basis::BasisCacheEntry* entry{nullptr};
    };
    std::vector<FieldBCacheEntry> cached_field_bcache_;

    struct CellDofTable {
        const dofs::DofMap* dof_map{nullptr};
        GlobalIndex dof_offset{0};
        std::uint64_t dof_layout_revision{0};
        std::vector<GlobalIndex> cell_offsets{};
        std::vector<GlobalIndex> dofs{};
    };
    struct CellResolvedVectorTable {
        const void* layout_handle{nullptr};
        std::uint64_t layout_revision{0};
        const dofs::DofMap* dof_map{nullptr};
        GlobalIndex dof_offset{0};
        std::uint64_t dof_layout_revision{0};
        std::vector<GlobalIndex> resolved{};
    };
    struct CellResolvedMatrixTable {
        const void* layout_handle{nullptr};
        std::uint64_t layout_revision{0};
        const dofs::DofMap* row_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        std::uint64_t row_dof_layout_revision{0};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex col_dof_offset{0};
        std::uint64_t col_dof_layout_revision{0};
        std::vector<GlobalIndex> cell_offsets{};
        std::vector<GlobalIndex> resolved{};
    };
    struct FieldAccessPlan {
        FieldId field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* space{nullptr};
        const dofs::DofMap* dof_map{nullptr};
        GlobalIndex dof_offset{0};
        std::uint64_t dof_layout_revision{0};
        const CellDofTable* dof_table{nullptr};
        FieldType field_type{FieldType::Scalar};
        bool is_product{false};
        int value_dimension{1};
    };
    const IMeshAccess* cached_cell_dof_mesh_{nullptr};
    GlobalIndex cached_cell_dof_count_{0};
    bool cached_cell_dof_mesh_revisions_valid_{false};
    std::uint64_t cached_cell_dof_topology_revision_{0};
    std::uint64_t cached_cell_dof_ownership_revision_{0};
    std::uint64_t cached_cell_dof_numbering_revision_{0};
    std::uint64_t cached_cell_dof_field_layout_revision_{0};
    std::deque<CellDofTable> cell_dof_tables_;
    std::vector<CellResolvedVectorTable> cell_resolved_vector_tables_;
    std::vector<CellResolvedMatrixTable> cell_resolved_matrix_tables_;
    std::vector<FieldAccessPlan> field_access_plans_;

    // Scratch storage for assembleCellsFused batching (avoids per-batch heap allocation)
    std::vector<AssemblyContext> scratch_batch_contexts_;
    std::vector<KernelOutput> scratch_batch_outputs_;
    std::vector<const AssemblyContext*> scratch_batch_context_ptrs_;
    std::vector<SavedCellNodeCoords> scratch_saved_node_coords_;
    std::vector<std::vector<SlotDofs>> scratch_batch_dofs_;
    std::vector<std::vector<Real>> scratch_batch_sol_coeffs_;
    std::vector<std::vector<std::vector<Real>>> scratch_batch_prev_sol_coeffs_;
    std::size_t scratch_batch_reserved_dofs_{0};
    std::size_t scratch_batch_reserved_qpts_{0};
    int scratch_batch_reserved_dim_{0};

    // Scratch for fused coupled block insertion (combines per-block outputs
    // into a single combined matrix/vector per cell to hit the FSILS fast path).
    std::vector<Real> scratch_fused_matrices_;
    std::vector<Real> scratch_fused_vectors_;
    std::vector<GlobalIndex> scratch_fused_dofs_;

    // Pre-resolved CSR slots for fused combined DOF insertion.
    // Built once on first assembly call; persists across Newton iterations.
    // scratch_fused_resolved_[cell_offset..cell_offset+cn*cn] = resolved slots
    // for the interleaved combined DOF list of that cell.
    std::vector<GlobalIndex> scratch_fused_resolved_;
    std::vector<GlobalIndex> scratch_fused_resolved_offsets_;
    std::vector<GlobalIndex> scratch_fused_vector_resolved_;
    std::vector<GlobalIndex> scratch_fused_vector_resolved_offsets_;

    // Per-cell flag: whether any DOF in the cell is constrained.
    // Built once during first assembly call; avoids per-call hasConstrainedDofs().
    std::vector<std::uint8_t> cell_constrained_flags_;
    bool cell_constrained_flags_valid_{false};
    std::uint64_t cell_constrained_flags_constraint_revision_{0};
    std::vector<std::pair<const dofs::DofMap*, std::uint64_t>> cell_constrained_flags_dof_revisions_;

    // Graph coloring for parallel assembly (computed once, persists across Newton iterations)
    void ensureColoring(const IMeshAccess& mesh,
                        std::span<const dofs::DofMap* const> extra_dof_maps = {});
    std::vector<int> coloring_colors_;                        // color per element
    std::vector<std::vector<GlobalIndex>> coloring_cells_by_color_; // cells grouped by color
    int coloring_num_colors_{0};
    bool coloring_valid_{false};
    const IMeshAccess* coloring_mesh_{nullptr};
    bool coloring_mesh_revisions_valid_{false};
    std::uint64_t coloring_topology_revision_{0};
    std::uint64_t coloring_ownership_revision_{0};
    std::uint64_t coloring_numbering_revision_{0};
    std::vector<std::pair<const dofs::DofMap*, std::uint64_t>> coloring_dof_revisions_;

    // Coupled scalar basis cache: caches the n_scalar_dofs reference
    // gradients/hessians (e.g. 4 for P1 Tet4) so that block transitions
    // in the coupled loop don't trigger full slow-path BasisCache re-evaluation.
    // The scalar ref data is the same for all blocks sharing the same element basis.
    bool coupled_scalar_ref_valid_{false};
    LocalIndex coupled_scalar_n_dofs_{0};
    LocalIndex coupled_scalar_n_qpts_{0};
    bool coupled_scalar_has_hessians_{false};
    std::vector<AssemblyContext::Vector3D> coupled_scalar_ref_grads_;   // [i * n_qpts + q]
    std::vector<AssemblyContext::Matrix3x3> coupled_scalar_ref_hess_;   // [i * n_qpts + q]
    std::vector<Real> coupled_scalar_basis_values_;                     // [q * n_scalar + i]

    // Qpt-major basis value caches for each unique DOF count in coupled blocks.
    // Built once during coupled scalar cache initialization. Looked up by DOF count
    // at basis-setting time. Supports arbitrary multi-field coupled systems.
    struct CoupledSpaceQptCache {
        LocalIndex n_dofs{0};                 // DOFs per element for this space
        std::vector<Real> qpt_values;         // [q * n_dofs + i]
    };
    std::vector<CoupledSpaceQptCache> coupled_space_qpt_caches_;

    /// Look up the qpt-major basis cache for a given DOF count. Returns nullptr if not found.
    const std::vector<Real>* findCoupledQptCache(LocalIndex n_dofs) const noexcept {
        for (const auto& c : coupled_space_qpt_caches_) {
            if (c.n_dofs == n_dofs) return &c.qpt_values;
        }
        return nullptr;
    }

    // Per-slot scalar physical gradient/hessian cache.
    // Populated by block 0 of each cell in the coupled loop; reused by blocks 1-3.
    // Layout: [q * n_scalar + si]  (qpt-major, scalar-DOF minor)
    static constexpr std::size_t kMaxScalarDofsPerSlot = 8;
    static constexpr std::size_t kMaxQPtsPerSlot = 16;
    static constexpr std::size_t kMaxScalarEntriesPerSlot = kMaxScalarDofsPerSlot * kMaxQPtsPerSlot;
    struct CoupledSlotPhysCache {
        AssemblyContext::Vector3D phys_grads[kMaxScalarEntriesPerSlot];
        AssemblyContext::Matrix3x3 phys_hess[kMaxScalarEntriesPerSlot];
    };
    std::vector<CoupledSlotPhysCache> coupled_slot_phys_cache_;

    // =========================================================================
    // Tier 3: Flat Cell Data Table
    // Pre-computed per-cell flat arrays for fast assembly.  Built once on first
    // assembly call for a given mesh topology; avoids per-cell virtual dispatch,
    // hash lookups, and constraint checks during the hot assembly loop.
    // =========================================================================
    struct FlatCellCoords {
        bool valid{false};
        int dim{0};
        int nodes_per_cell{0};
        GlobalIndex cell_count{0};
        ElementType uniform_cell_type{ElementType::Unknown};
        bool dense_cell_ids{false};
        std::vector<Real> coords;         ///< [n_cells * nodes_per_cell * 3] flat xyz
        const IMeshAccess* mesh{nullptr};
        bool revision_tracking_available{false};
        std::uint64_t geometry_revision{0};
        std::uint64_t topology_revision{0};
        std::uint64_t ownership_revision{0};
        std::uint64_t numbering_revision{0};
        std::uint64_t active_configuration_epoch{0};
        std::uint64_t coordinate_configuration_key{0};
    };
    FlatCellCoords flat_cell_coords_;

    /// Build flat coordinate array from mesh. Called once per topology.
    void ensureFlatCellCoords(const IMeshAccess& mesh);

    // =========================================================================
    // Tier 1: Cached field evaluation recipe
    // Pre-computed per-field data for fast populateFieldSolutionData.
    // Avoids per-field findFieldAccessPlan, getElement, basis cache lookups.
    // =========================================================================
    struct CachedFieldRecipe {
        FieldId field_id{INVALID_FIELD_ID};
        const FieldAccessPlan* access{nullptr};
        const basis::BasisCacheEntry* bcache{nullptr};
        const basis::BasisFunction* basis{nullptr};
        ElementType cell_type{ElementType::Unknown};
        std::string basis_cache_identity{};
        bool is_product{false};
        FieldType field_type{FieldType::Scalar};
        int value_dim{1};
        LocalIndex n_dofs{0};
        LocalIndex n_scalar_dofs{0};
        bool want_values{false};
        bool want_gradients{false};
        bool want_hessians{false};
        bool want_laplacians{false};
    };
    std::vector<CachedFieldRecipe> cached_field_recipes_;
    bool cached_field_recipes_valid_{false};

    /// Build field evaluation recipes from current field requirements.
    void ensureFieldRecipes(const IMeshAccess& mesh,
                            const std::vector<FieldRequirement>& requirements);

    struct CellCoefficientCacheEntry {
        const dofs::DofMap* dof_map{nullptr};
        GlobalIndex dof_offset{0};
        const spaces::FunctionSpace* space{nullptr};
        int history_index{0}; // 0=current, k>0 previous solution state
        bool localized_vector_basis{false};
        std::vector<Real> coeffs{};
    };
    struct CellFieldEvaluationCacheEntry {
        FieldId field_id{INVALID_FIELD_ID};
        GlobalIndex cell_id{-1};
        int history_index{0}; // 0=current, k>0 previous solution state
        FieldType field_type{FieldType::Scalar};
        int value_dim{1};
        bool has_values{false};
        bool has_gradients{false};
        std::vector<Real> scalar_values{};
        std::vector<AssemblyContext::Vector3D> scalar_gradients{};
        std::vector<AssemblyContext::Vector3D> vector_values{};
        std::vector<AssemblyContext::Matrix3x3> vector_jacobians{};
    };
    [[nodiscard]] std::span<const Real> gatherCachedCellVectorCoefficients(
        std::deque<CellCoefficientCacheEntry>& cache,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const dofs::DofMap* dof_map,
        GlobalIndex dof_offset,
        const spaces::FunctionSpace* space,
        std::span<const GlobalIndex> dofs,
        int history_index,
        bool localized_vector_basis,
        const char* error_prefix);

    /// Optimized field solution population using cached recipes + flat coords.
    void populateFieldSolutionDataFast(
        AssemblyContext& context,
        const IMeshAccess& mesh,
        GlobalIndex cell_id,
        const std::vector<FieldRequirement>& requirements,
        std::deque<CellCoefficientCacheEntry>* coefficient_cache = nullptr,
        std::deque<CellFieldEvaluationCacheEntry>* field_eval_cache = nullptr);
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
