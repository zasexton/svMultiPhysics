/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FESYSTEM_H
#define SVMP_FE_SYSTEMS_FESYSTEM_H

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Systems/FieldRegistry.h"
#include "Systems/GlobalKernel.h"
#include "Systems/GlobalKernelStateProvider.h"
#include "Systems/OperatorRegistry.h"
#include "Systems/ParameterRegistry.h"
#include "Systems/SearchAccess.h"
#include "Systems/SystemConstraint.h"
#include "Systems/SystemState.h"
#include "Systems/SystemSetup.h"

#include "Assembly/Assembler.h"

#include "Backends/Interfaces/LinearSolver.h"

#include "Constraints/AffineConstraints.h"
#include "Constraints/Constraint.h"
#include "Constraints/GaugeRegistry.h"

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ConstraintAnalysisSummary.h"
#include "Analysis/InterfaceTopologyContext.h"

#include "Dofs/DofHandler.h"
#include "Dofs/FieldDofMap.h"
#include "Dofs/BlockDofMap.h"

#include "Sparsity/SparsityPattern.h"
#include "Sparsity/SparsityBuilder.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace svmp {
class InterfaceMesh;
}
#endif

namespace svmp {
namespace FE {

namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace assembly {
class GlobalSystemView;
struct MatrixFreeOptions;
class IMatrixFreeKernel;
class MatrixFreeOperator;
class FunctionalKernel;
}

namespace backends {
struct DofPermutation;
} // namespace backends

namespace systems {

using BoundaryId = int;
using InterfaceId = int;
class OperatorBackends;
class CoupledBoundaryManager;

struct SetupOptions {
    dofs::DofDistributionOptions dof_options{};
    assembly::AssemblyOptions assembly_options{};
    sparsity::SparsityBuildOptions sparsity_options{};

    std::string assembler_name{"StandardAssembler"};

    sparsity::CouplingMode coupling_mode{sparsity::CouplingMode::Full};
    std::vector<sparsity::FieldCoupling> custom_couplings{};

    bool use_constraints_in_assembly{true};

    // Iterative-solver leverage (explicit opt-in): auto-register eligible matrix-free operators.
    bool auto_register_matrix_free{false};

    /// When true, print a detailed GaugeRegistry diagnostic report to stderr
    /// after nullspace resolution during setup().  Useful for debugging
    /// nullspace detection and enforcement decisions.
    bool gauge_diagnostics{false};
};

struct AssemblyRequest {
    OperatorTag op;
    bool want_matrix{false};
    bool want_vector{false};
    bool zero_outputs{true};
    bool assemble_boundary_terms{true};
    bool assemble_interior_face_terms{true};
    bool assemble_interface_face_terms{true};
    bool assemble_global_terms{true};

    /// When true, constrained assembly distributes matrix and vector independently
    /// (suppressing the -K*g Dirichlet inhomogeneity correction that joint distribution
    /// adds).  Set to true for nonlinear Newton solves where the residual R(u) is already
    /// evaluated at the constrained state.
    bool suppress_constraint_inhomogeneity{false};
};

class FESystem {
public:
    explicit FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access);
    ~FESystem();

    FESystem(FESystem&&) noexcept = default;
    FESystem& operator=(FESystem&&) noexcept = default;

    FESystem(const FESystem&) = delete;
    FESystem& operator=(const FESystem&) = delete;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    explicit FESystem(std::shared_ptr<const svmp::Mesh> mesh,
                      svmp::Configuration coord_cfg = svmp::Configuration::Reference);
#endif

    // ---- Definition phase ----
    FieldId addField(FieldSpec spec);
    void addConstraint(std::unique_ptr<constraints::Constraint> c);
    void addSystemConstraint(std::unique_ptr<ISystemConstraint> c);

    void addOperator(OperatorTag name);

    /// @name Kernel registration (internal — do not use in physics modules)
    ///
    /// These methods are called by FormsInstaller internally. Physics modules
    /// should use the public FormsInstaller API instead:
    ///   - installFormulation()    for residual physics
    ///   - installMixedBilinear()  for mixed bilinear operators
    ///   - installMixedLinear()    for mixed linear operators
    /// @{

    void addCellKernel(OperatorTag op, FieldId field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId test_field, FieldId trial_field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInteriorFaceKernel(OperatorTag op, FieldId field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInteriorFaceKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId field,
                                std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId test_field, FieldId trial_field,
                                std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addGlobalKernel(OperatorTag op,
                         std::shared_ptr<GlobalKernel> kernel);

    /// @}

    // ---- Optional operator backends (Milestone 5) ----
    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel);
    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                             const assembly::MatrixFreeOptions& options);
    [[nodiscard]] std::shared_ptr<assembly::MatrixFreeOperator>
    matrixFreeOperator(const OperatorTag& op) const;

    void addFunctionalKernel(std::string tag,
                             std::shared_ptr<assembly::FunctionalKernel> kernel);
    [[nodiscard]] Real evaluateFunctional(const std::string& tag,
                                          const SystemStateView& state) const;
    [[nodiscard]] Real evaluateBoundaryFunctional(const std::string& tag,
                                                  int boundary_marker,
                                                  const SystemStateView& state) const;

    /**
     * @brief Enable coupled boundary-condition infrastructure for a primary field
     *
     * This creates (or returns) a CoupledBoundaryManager used to orchestrate
     * boundary functionals and auxiliary (0D) state updates. The primary field
     * is used as the discrete solution source for BoundaryFunctional evaluation.
     */
    CoupledBoundaryManager& coupledBoundaryManager(FieldId primary_field);
    [[nodiscard]] CoupledBoundaryManager* coupledBoundaryManager() noexcept { return coupled_boundary_.get(); }
    [[nodiscard]] const CoupledBoundaryManager* coupledBoundaryManager() const noexcept { return coupled_boundary_.get(); }

    // ---- Setup phase ----
    void setup(const SetupOptions& opts = {}, const SetupInputs& inputs = {});

    // ---- Constraints lifecycle ----
    void updateConstraints(double time, double dt = 0.0);

    // ---- Assembly phase ----
    assembly::AssemblyResult assemble(
        const AssemblyRequest& req,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);

    assembly::AssemblyResult assembleResidual(
        const SystemStateView& state,
        assembly::GlobalSystemView& rhs_out);

    assembly::AssemblyResult assembleJacobian(
        const SystemStateView& state,
        assembly::GlobalSystemView& jac_out);

    assembly::AssemblyResult assembleMass(
        const SystemStateView& state,
        assembly::GlobalSystemView& mass_out);

    // ---- Time stepping lifecycle (optional; for MaterialState history) ----
    void beginTimeStep();
    void commitTimeStep();

	    // ---- Accessors ----
	    [[nodiscard]] const assembly::IMeshAccess& meshAccess() const;
	    [[nodiscard]] std::string assemblerName() const;
	    [[nodiscard]] std::string assemblerSelectionReport() const;
	    [[nodiscard]] const ISearchAccess* searchAccess() const noexcept { return search_access_.get(); }
	    void setSearchAccess(std::shared_ptr<const ISearchAccess> access) { search_access_ = std::move(access); }

	    /**
	     * @brief Locate a physical point in the mesh using the configured search access.
	     */
	    [[nodiscard]] ISearchAccess::PointLocation locatePoint(const std::array<Real, 3>& point,
	                                                           GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;

	    /**
	     * @brief Evaluate a field at a physical point (search + reference-space interpolation).
	     *
	     * @return nullopt if no search access is configured or the point is not located in the mesh.
	     */
	    [[nodiscard]] std::optional<std::array<Real, 3>> evaluateFieldAtPoint(FieldId field,
	                                                                          const SystemStateView& state,
	                                                                          const std::array<Real, 3>& point,
	                                                                          GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;

	    /**
	     * @brief Evaluate a field at all mesh vertices by direct DOF coefficient lookup.
	     *
	     * For Lagrange elements, basis functions equal 1 at their associated node and 0
	     * at all others, so the field value at a vertex equals the DOF coefficient.
	     * This avoids the expensive locatePoint + evaluate per vertex.
	     *
	     * @param field       Field to evaluate
	     * @param state       Current system state
	     * @param n_vertices  Number of mesh vertices
	     * @param out         Output buffer, size >= n_vertices * max(1, components)
	     * @return true if direct nodal evaluation was used, false if not supported
	     *         (caller should fall back to evaluateFieldAtPoint)
	     */
	    [[nodiscard]] bool evaluateFieldAtVertices(FieldId field,
	                                               const SystemStateView& state,
	                                               GlobalIndex n_vertices,
	                                               std::span<double> out) const;

	    // ---- Global-kernel persistent state (optional) ----
	    [[nodiscard]] assembly::MaterialStateView globalKernelCellState(const GlobalKernel& kernel,
	                                                                    GlobalIndex cell_id,
	                                                                    LocalIndex num_qpts) const;
    [[nodiscard]] assembly::MaterialStateView globalKernelBoundaryFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const;
    [[nodiscard]] assembly::MaterialStateView globalKernelInteriorFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    [[nodiscard]] const svmp::Mesh* mesh() const noexcept { return mesh_.get(); }
	    [[nodiscard]] svmp::Configuration coordinateConfiguration() const noexcept { return coord_cfg_; }

	    void setInterfaceMesh(InterfaceId marker, std::shared_ptr<const svmp::InterfaceMesh> mesh);
	    [[nodiscard]] bool hasInterfaceMesh(InterfaceId marker) const noexcept;
	    [[nodiscard]] const svmp::InterfaceMesh& interfaceMesh(InterfaceId marker) const;

	    void setInterfaceMeshFromFaceSet(InterfaceId marker,
	                                     const std::string& face_set_name,
	                                     bool compute_orientation = true);
	    void setInterfaceMeshFromBoundaryLabel(InterfaceId marker,
	                                           int boundary_label,
	                                           bool compute_orientation = true);
#endif

    [[nodiscard]] const dofs::DofHandler& dofHandler() const noexcept { return dof_handler_; }
    [[nodiscard]] const FieldRecord& fieldRecord(FieldId field) const;
    [[nodiscard]] const dofs::DofHandler& fieldDofHandler(FieldId field) const;
    [[nodiscard]] GlobalIndex fieldDofOffset(FieldId field) const;
    [[nodiscard]] const dofs::FieldDofMap& fieldMap() const noexcept { return field_map_; }
    [[nodiscard]] const dofs::BlockDofMap* blockMap() const noexcept { return block_map_.get(); }
    [[nodiscard]] const constraints::AffineConstraints& constraints() const noexcept { return affine_constraints_; }
    [[nodiscard]] const sparsity::SparsityPattern& sparsity(const OperatorTag& op) const;
    [[nodiscard]] const sparsity::DistributedSparsityPattern* distributedSparsityIfAvailable(const OperatorTag& op) const noexcept;
    [[nodiscard]] std::shared_ptr<const backends::DofPermutation> dofPermutation() const noexcept { return dof_permutation_; }

		    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }
		    [[nodiscard]] int temporalOrder() const noexcept;
		    [[nodiscard]] bool isTransient() const noexcept { return temporalOrder() > 0; }
		    [[nodiscard]] std::vector<FieldId> timeDerivativeFields(const OperatorTag& op) const;
		    [[nodiscard]] std::vector<FieldId> timeDerivativeFields() const;

	    // ---- Parameter requirements (optional) ----
	    [[nodiscard]] const ParameterRegistry& parameterRegistry() const noexcept { return parameter_registry_; }

    // ---- Gauge / nullspace detection (optional) ----
    /**
     * @brief Access the GaugeRegistry, creating it on first call
     *
     * The GaugeRegistry is an optional component for automatic nullspace
     * detection and enforcement.  It is created lazily on first access.
     */
    [[nodiscard]] gauge::GaugeRegistry& gaugeRegistry();
    [[nodiscard]] const gauge::GaugeRegistry* gaugeRegistryIfPresent() const noexcept {
        return gauge_registry_.get();
    }
    [[nodiscard]] bool hasGaugeRegistry() const noexcept { return gauge_registry_ != nullptr; }

    // ---- Problem analysis subsystem ----

    void addFormulationRecord(analysis::FormulationRecord record);
    void addBoundaryConditionDescriptor(analysis::BoundaryConditionDescriptor desc);
    void addContribution(analysis::ContributionDescriptor desc);
    void addVariableDescriptor(analysis::VariableDescriptor desc);

    [[nodiscard]] const std::vector<analysis::FormulationRecord>& formulationRecords() const noexcept {
        return formulation_records_;
    }
    [[nodiscard]] const std::vector<analysis::BoundaryConditionDescriptor>& boundaryConditionDescriptors() const noexcept {
        return bc_descriptors_;
    }
    [[nodiscard]] const std::vector<analysis::VariableDescriptor>& variableDescriptors() const noexcept {
        return variable_descriptors_;
    }
    [[nodiscard]] const std::vector<analysis::ContributionDescriptor>& contributionDescriptors() const noexcept {
        return contributions_;
    }

    /// Build and store topology context from the mesh
    void buildTopologyContext();
    [[nodiscard]] const analysis::TopologyAnalysisContext* topologyContext() const noexcept {
        return topology_context_ ? &*topology_context_ : nullptr;
    }

    /// Build and store interface topology from registered InterfaceMesh objects
    void buildInterfaceTopologyContext();
    [[nodiscard]] const analysis::InterfaceTopologyContext* interfaceTopologyContext() const noexcept {
        return interface_topology_context_ ? &*interface_topology_context_ : nullptr;
    }

    /// Build and store constraint summary from current AffineConstraints
    void buildConstraintSummary();
    [[nodiscard]] const analysis::ConstraintAnalysisSummary* constraintSummary() const noexcept {
        return constraint_summary_ ? &*constraint_summary_ : nullptr;
    }

    /// Invalidate cached analysis report (called automatically by mutation methods)
    void invalidateAnalysisCache() noexcept;

    /// Run all analysis passes and return a fresh report
    [[nodiscard]] analysis::ProblemAnalysisReport runProblemAnalysis() const;

    /// Cached version — re-runs only if inputs have changed
    [[nodiscard]] const analysis::ProblemAnalysisReport& analysisReport() const;

    // ---- Operator registry query (for tests and diagnostics) ----

    /**
     * @brief Query the registered operator definition for an operator tag
     *
     * Returns the OperatorDefinition containing all registered cell, boundary,
     * interior-face, interface-face, and global terms. This is the structural
     * view of what was installed — useful for parity tests that verify mixed
     * and manual installation paths produce identical block structure.
     */
    [[nodiscard]] const OperatorDefinition& operatorDefinition(const OperatorTag& op) const {
        return operator_registry_.get(op);
    }

    [[nodiscard]] bool hasOperator(const OperatorTag& op) const noexcept {
        return operator_registry_.has(op);
    }

    // ---- Rank-1 updates from coupled Jacobian assembly ----
    [[nodiscard]] std::span<const backends::RankOneUpdate> lastRankOneUpdates() const noexcept;
    void clearRankOneUpdates() noexcept;

    /// @cond INTERNAL
    // Internal — used by FormsInstaller for transactional kernel registration.
    // Public only because C++ templates cannot be called from non-friend TUs
    // when private. Do not call from physics modules.
    template <typename Fn>
    auto executeWithOperatorRollback_(Fn&& fn) -> decltype(fn()) {
        auto snap = operator_registry_.snapshot();
        try {
            return fn();
        } catch (...) {
            operator_registry_.rollback(snap);
            throw;
        }
    }
    /// @endcond

private:

    struct PlannedCellTerm {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedBoundaryTerm {
        int marker{0};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedInteriorFaceTerm {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedInterfaceFaceTerm {
        int marker{0};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct OperatorAssemblyPlan {
        std::vector<PlannedCellTerm> cell_terms{};
        std::vector<PlannedBoundaryTerm> boundary_terms{};
        std::vector<PlannedInteriorFaceTerm> interior_terms{};
        std::vector<PlannedInterfaceFaceTerm> interface_terms{};
        std::vector<GlobalKernel*> global_terms{};
    };

    friend assembly::AssemblyResult assembleOperator(
        FESystem& system,
        const AssemblyRequest& request,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);
    friend class OperatorBackends;

    void invalidateSetup() noexcept;
    void requireSetup() const;
    void requireSingleFieldSetup() const;
    void buildAssemblyPlans();

    [[nodiscard]] const FieldRecord& singleField() const;

    std::shared_ptr<const assembly::IMeshAccess> mesh_access_;
    std::shared_ptr<const ISearchAccess> search_access_{};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    std::shared_ptr<const svmp::Mesh> mesh_{};
	    svmp::Configuration coord_cfg_{svmp::Configuration::Reference};
	    std::unordered_map<InterfaceId, std::shared_ptr<const svmp::InterfaceMesh>> interface_meshes_{};
#endif

    FieldRegistry field_registry_;
    OperatorRegistry operator_registry_;
    std::vector<std::unique_ptr<constraints::Constraint>> constraint_defs_;
    std::vector<std::unique_ptr<ISystemConstraint>> system_constraint_defs_;

    dofs::DofHandler dof_handler_{};
    std::vector<dofs::DofHandler> field_dof_handlers_{};
    std::vector<GlobalIndex> field_dof_offsets_{};
    dofs::FieldDofMap field_map_{};
    std::unique_ptr<dofs::BlockDofMap> block_map_{};
	    constraints::AffineConstraints affine_constraints_{};

		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::SparsityPattern>> sparsity_by_op_{};
		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::DistributedSparsityPattern>> distributed_sparsity_by_op_{};
		    std::shared_ptr<const backends::DofPermutation> dof_permutation_{};

		    std::unique_ptr<assembly::Assembler> assembler_{};
		    std::string assembler_selection_report_{};
		    std::unique_ptr<assembly::IMaterialStateProvider> material_state_provider_{};
	    std::unique_ptr<GlobalKernelStateProvider> global_kernel_state_provider_{};
    std::unique_ptr<OperatorBackends> operator_backends_{};
    std::unique_ptr<CoupledBoundaryManager> coupled_boundary_{};
	    ParameterRegistry parameter_registry_{};
    std::unique_ptr<gauge::GaugeRegistry> gauge_registry_{};
    std::vector<backends::RankOneUpdate> last_rank_one_updates_{};
    std::unordered_map<OperatorTag, OperatorAssemblyPlan> assembly_plan_by_op_{};

    // Cached coupled Jacobian results.
    // For time-invariant sensitivities (e.g., RCR/resistance BCs), keyed by dt only.
    // For time-variant sensitivities, keyed by (time, dt) pair.
    // Time-invariance is auto-detected on the first computation by walking the
    // FormExpr trees for solution/time references.
    struct CoupledJacobianCache {
        double time{-1e30};
        double dt{-1e30};
        bool valid{false};
        bool is_time_invariant{false};  ///< True if sensitivity depends only on geometry + dt
        std::vector<backends::RankOneUpdate> rank_one_updates{};
        // For non-symmetric cases: cached outer-product matrix entries.
        struct SparseEntry {
            GlobalIndex row;
            std::vector<GlobalIndex> col_dofs;
            std::vector<Real> col_vals;
        };
        std::vector<SparseEntry> outer_product_entries{};

        void clear() noexcept {
            time = -1e30;
            dt = -1e30;
            valid = false;
            is_time_invariant = false;
            rank_one_updates.clear();
            outer_product_entries.clear();
        }
    };
    CoupledJacobianCache coupled_jac_cache_{};

    // ---- Analysis subsystem storage ----
    std::vector<analysis::FormulationRecord> formulation_records_;
    std::vector<analysis::ContributionDescriptor> contributions_;
    std::size_t contributions_def_count_{0}; ///< Watermark for definition-phase contributions
    std::vector<analysis::BoundaryConditionDescriptor> bc_descriptors_;
    std::vector<analysis::VariableDescriptor> variable_descriptors_;

    std::optional<analysis::TopologyAnalysisContext> topology_context_;
    std::optional<analysis::InterfaceTopologyContext> interface_topology_context_;
    std::optional<analysis::ConstraintAnalysisSummary> constraint_summary_;
    mutable std::optional<analysis::ProblemAnalysisReport> analysis_report_cache_;
    mutable std::uint64_t analysis_inputs_version_{0};
    mutable std::uint64_t analysis_report_version_{std::numeric_limits<std::uint64_t>::max()};

	    bool is_setup_{false};
	};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FESYSTEM_H
