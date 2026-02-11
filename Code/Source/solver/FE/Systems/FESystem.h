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

#include "Constraints/AffineConstraints.h"
#include "Constraints/Constraint.h"

#include "Dofs/DofHandler.h"
#include "Dofs/FieldDofMap.h"
#include "Dofs/BlockDofMap.h"

#include "Sparsity/SparsityPattern.h"
#include "Sparsity/SparsityBuilder.h"

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

private:
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

	    bool is_setup_{false};
	};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FESYSTEM_H
