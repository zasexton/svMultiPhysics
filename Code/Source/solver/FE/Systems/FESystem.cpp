/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FESystem.h"

#include "Systems/SystemAssembly.h"
#include "Systems/OperatorBackends.h"
#include "Systems/CoupledBoundaryManager.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Backends/Interfaces/GenericVector.h"

#include "Spaces/FunctionSpace.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <algorithm>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/MeshAccess.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Systems/MeshSearchAccess.h"
#endif

namespace svmp {
namespace FE {
namespace systems {

FESystem::FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access)
    : mesh_access_(std::move(mesh_access))
{
    FE_CHECK_NOT_NULL(mesh_access_.get(), "FESystem::mesh_access");
    operator_backends_ = std::make_unique<OperatorBackends>();
}

FESystem::~FESystem() = default;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
FESystem::FESystem(std::shared_ptr<const svmp::Mesh> mesh, svmp::Configuration coord_cfg)
    : mesh_(std::move(mesh)), coord_cfg_(coord_cfg)
{
    FE_CHECK_NOT_NULL(mesh_.get(), "FESystem::mesh");
    mesh_access_ = std::make_shared<assembly::MeshAccess>(*mesh_, coord_cfg_);
    search_access_ = std::make_shared<MeshSearchAccess>(*mesh_, coord_cfg_);
    FE_CHECK_NOT_NULL(mesh_access_.get(), "FESystem::mesh_access");
    operator_backends_ = std::make_unique<OperatorBackends>();
}

void FESystem::setInterfaceMesh(InterfaceId marker, std::shared_ptr<const svmp::InterfaceMesh> mesh)
{
    invalidateSetup();
    FE_THROW_IF(marker < 0, InvalidArgumentException,
                "FESystem::setInterfaceMesh: marker must be >= 0");
    FE_CHECK_NOT_NULL(mesh.get(), "FESystem::setInterfaceMesh: mesh");
    interface_meshes_[marker] = std::move(mesh);
}

bool FESystem::hasInterfaceMesh(InterfaceId marker) const noexcept
{
    return interface_meshes_.find(marker) != interface_meshes_.end();
}

const svmp::InterfaceMesh& FESystem::interfaceMesh(InterfaceId marker) const
{
    auto it = interface_meshes_.find(marker);
    FE_THROW_IF(it == interface_meshes_.end() || !it->second, InvalidArgumentException,
                "FESystem::interfaceMesh: unknown interface marker " + std::to_string(marker));
    return *it->second;
}

void FESystem::setInterfaceMeshFromFaceSet(InterfaceId marker,
                                           const std::string& face_set_name,
                                           bool compute_orientation)
{
    FE_CHECK_NOT_NULL(mesh_.get(), "FESystem::setInterfaceMeshFromFaceSet: mesh");
    auto iface = std::make_shared<svmp::InterfaceMesh>(
        svmp::InterfaceMesh::build_from_face_set(mesh_->base(), face_set_name, compute_orientation));
    setInterfaceMesh(marker, std::move(iface));
}

void FESystem::setInterfaceMeshFromBoundaryLabel(InterfaceId marker,
                                                 int boundary_label,
                                                 bool compute_orientation)
{
    FE_CHECK_NOT_NULL(mesh_.get(), "FESystem::setInterfaceMeshFromBoundaryLabel: mesh");
    auto iface = std::make_shared<svmp::InterfaceMesh>(
        svmp::InterfaceMesh::build_from_boundary_label(mesh_->base(),
                                                       static_cast<svmp::label_t>(boundary_label),
                                                       compute_orientation));
    setInterfaceMesh(marker, std::move(iface));
}
#endif

void FESystem::invalidateSetup() noexcept
{
    is_setup_ = false;
    assembler_.reset();
    assembler_selection_report_.clear();
    material_state_provider_.reset();
    global_kernel_state_provider_.reset();
    sparsity_by_op_.clear();
    distributed_sparsity_by_op_.clear();
    parameter_registry_.clear();
    if (operator_backends_) {
        operator_backends_->invalidateCache();
    }
}

void FESystem::requireSetup() const
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "FESystem: setup() has not been called");
}

const FieldRecord& FESystem::singleField() const
{
    FE_THROW_IF(field_registry_.size() != 1u, NotImplementedException,
                "FESystem::singleField: this operation currently requires exactly one field");
    return field_registry_.records().front();
}

void FESystem::requireSingleFieldSetup() const
{
    requireSetup();
    (void)singleField();
}

FieldId FESystem::addField(FieldSpec spec)
{
    invalidateSetup();
    if (spec.components <= 0) {
        spec.components = spec.space ? spec.space->value_dimension() : 1;
    }
    if (spec.space) {
        FE_THROW_IF(spec.components != spec.space->value_dimension(), InvalidArgumentException,
                    "FESystem::addField: FieldSpec.components must match FunctionSpace::value_dimension()");
    }
    return field_registry_.add(std::move(spec));
}

void FESystem::addConstraint(std::unique_ptr<constraints::Constraint> c)
{
    invalidateSetup();
    FE_CHECK_NOT_NULL(c.get(), "FESystem::addConstraint: constraint");
    constraint_defs_.push_back(std::move(c));
}

void FESystem::addSystemConstraint(std::unique_ptr<ISystemConstraint> c)
{
    invalidateSetup();
    FE_CHECK_NOT_NULL(c.get(), "FESystem::addSystemConstraint: constraint");
    system_constraint_defs_.push_back(std::move(c));
}

void FESystem::addOperator(OperatorTag name)
{
    invalidateSetup();
    operator_registry_.addOperator(std::move(name));
}

void FESystem::addCellKernel(OperatorTag op, FieldId field,
                             std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addCellKernel(std::move(op), field, field, std::move(kernel));
}

void FESystem::addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                             std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.cells.push_back(CellTerm{test_field, trial_field, std::move(kernel)});
}

void FESystem::addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId field,
                                 std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addBoundaryKernel(std::move(op), boundary, field, field, std::move(kernel));
}

void FESystem::addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId test_field,
                                 FieldId trial_field, std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.boundary.push_back(BoundaryTerm{boundary, test_field, trial_field, std::move(kernel)});
}

void FESystem::addInteriorFaceKernel(OperatorTag op, FieldId field,
                                     std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addInteriorFaceKernel(std::move(op), field, field, std::move(kernel));
}

void FESystem::addInteriorFaceKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                                     std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.interior.push_back(InteriorFaceTerm{test_field, trial_field, std::move(kernel)});
}

void FESystem::addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId field,
                                      std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addInterfaceFaceKernel(std::move(op), interface_marker, field, field, std::move(kernel));
}

void FESystem::addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId test_field, FieldId trial_field,
                                      std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.interface_faces.push_back(InterfaceFaceTerm{interface_marker, test_field, trial_field, std::move(kernel)});
}

void FESystem::addGlobalKernel(OperatorTag op, std::shared_ptr<GlobalKernel> kernel)
{
    invalidateSetup();
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    FE_CHECK_NOT_NULL(kernel.get(), "FESystem::addGlobalKernel: kernel");
    operator_registry_.get(op).global.push_back(std::move(kernel));
}

void FESystem::addMatrixFreeKernel(OperatorTag op,
                                   std::shared_ptr<assembly::IMatrixFreeKernel> kernel)
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    operator_backends_->registerMatrixFree(std::move(op), std::move(kernel));
}

void FESystem::addMatrixFreeKernel(OperatorTag op,
                                   std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                                   const assembly::MatrixFreeOptions& options)
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    operator_backends_->registerMatrixFree(std::move(op), std::move(kernel), options);
}

std::shared_ptr<assembly::MatrixFreeOperator> FESystem::matrixFreeOperator(const OperatorTag& op) const
{
    requireSingleFieldSetup();
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->matrixFreeOperator(*this, op);
}

void FESystem::addFunctionalKernel(std::string tag,
                                   std::shared_ptr<assembly::FunctionalKernel> kernel)
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    operator_backends_->registerFunctional(std::move(tag), std::move(kernel));
}

Real FESystem::evaluateFunctional(const std::string& tag, const SystemStateView& state) const
{
    requireSingleFieldSetup();
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->evaluateFunctional(*this, tag, state);
}

Real FESystem::evaluateBoundaryFunctional(const std::string& tag,
                                          int boundary_marker,
                                          const SystemStateView& state) const
{
    requireSingleFieldSetup();
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->evaluateBoundaryFunctional(*this, tag, boundary_marker, state);
}

CoupledBoundaryManager& FESystem::coupledBoundaryManager(FieldId primary_field)
{
    if (!coupled_boundary_) {
        coupled_boundary_ = std::make_unique<CoupledBoundaryManager>(*this, primary_field);
        return *coupled_boundary_;
    }
    FE_THROW_IF(coupled_boundary_->primaryField() != primary_field, InvalidArgumentException,
                "FESystem::coupledBoundaryManager: manager already initialized with a different primary field");
    return *coupled_boundary_;
}

const assembly::IMeshAccess& FESystem::meshAccess() const
{
    FE_CHECK_NOT_NULL(mesh_access_.get(), "FESystem::meshAccess");
    return *mesh_access_;
}

std::string FESystem::assemblerName() const
{
    if (!assembler_) {
        return {};
    }
    return assembler_->name();
}

std::string FESystem::assemblerSelectionReport() const
{
    return assembler_selection_report_;
}

ISearchAccess::PointLocation FESystem::locatePoint(const std::array<Real, 3>& point,
                                                   GlobalIndex hint_cell) const
{
    if (!search_access_) {
        return {};
    }
    return search_access_->locatePoint(point, hint_cell);
}

std::optional<std::array<Real, 3>> FESystem::evaluateFieldAtPoint(FieldId field,
                                                                  const SystemStateView& state,
                                                                  const std::array<Real, 3>& point,
                                                                  GlobalIndex hint_cell) const
{
    requireSetup();

    const auto loc = locatePoint(point, hint_cell);
    if (!loc.found || loc.cell_id == INVALID_GLOBAL_INDEX) {
        return std::nullopt;
    }

    const auto& rec = field_registry_.get(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::evaluateFieldAtPoint: field.space");

    // Reference coordinates (as provided by the search layer).
    spaces::FunctionSpace::Value xi;
    xi[0] = loc.xi[0];
    xi[1] = loc.xi[1];
    xi[2] = loc.xi[2];

    const auto field_idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || field_idx >= field_dof_handlers_.size(), InvalidArgumentException,
                "FESystem::evaluateFieldAtPoint: invalid FieldId");

    const auto cell_dofs_local = field_dof_handlers_[field_idx].getDofMap().getCellDofs(loc.cell_id);
    std::vector<Real> coeffs;
    coeffs.reserve(cell_dofs_local.size());

    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        solution_view = vec->createAssemblyView();
    }

    const GlobalIndex offset = field_dof_offsets_[field_idx];
    for (const auto d_local : cell_dofs_local) {
        const GlobalIndex d = d_local + offset;
        FE_THROW_IF(d < 0, InvalidArgumentException,
                    "FESystem::evaluateFieldAtPoint: negative DOF index");
        if (solution_view) {
            coeffs.push_back(solution_view->getVectorEntry(d));
        } else {
            const auto idx = static_cast<std::size_t>(d);
            FE_THROW_IF(idx >= state.u.size(), InvalidArgumentException,
                        "FESystem::evaluateFieldAtPoint: state.u is smaller than required by DOF index");
            coeffs.push_back(state.u[idx]);
        }
    }

    const auto v = rec.space->evaluate(xi, coeffs);
    return std::array<Real, 3>{v[0], v[1], v[2]};
}

const FieldRecord& FESystem::fieldRecord(FieldId field) const
{
    return field_registry_.get(field);
}

assembly::MaterialStateView FESystem::globalKernelCellState(const GlobalKernel& kernel,
                                                            GlobalIndex cell_id,
                                                            LocalIndex num_qpts) const
{
    requireSetup();
    if (!global_kernel_state_provider_) return {};
    return global_kernel_state_provider_->getCellState(kernel, cell_id, num_qpts);
}

assembly::MaterialStateView FESystem::globalKernelBoundaryFaceState(const GlobalKernel& kernel,
                                                                    GlobalIndex face_id,
                                                                    LocalIndex num_qpts) const
{
    requireSetup();
    if (!global_kernel_state_provider_) return {};
    return global_kernel_state_provider_->getBoundaryFaceState(kernel, face_id, num_qpts);
}

assembly::MaterialStateView FESystem::globalKernelInteriorFaceState(const GlobalKernel& kernel,
                                                                    GlobalIndex face_id,
                                                                    LocalIndex num_qpts) const
{
    requireSetup();
    if (!global_kernel_state_provider_) return {};
    return global_kernel_state_provider_->getInteriorFaceState(kernel, face_id, num_qpts);
}

const sparsity::SparsityPattern& FESystem::sparsity(const OperatorTag& op) const
{
    requireSetup();
    auto it = sparsity_by_op_.find(op);
    FE_THROW_IF(it == sparsity_by_op_.end() || !it->second, InvalidArgumentException,
                "FESystem::sparsity: no sparsity pattern for operator '" + op + "'");
    return *it->second;
}

const sparsity::DistributedSparsityPattern*
FESystem::distributedSparsityIfAvailable(const OperatorTag& op) const noexcept
{
    if (!is_setup_) {
        return nullptr;
    }
    auto it = distributed_sparsity_by_op_.find(op);
    if (it == distributed_sparsity_by_op_.end()) {
        return nullptr;
    }
    return it->second.get();
}

int FESystem::temporalOrder() const noexcept
{
    int max_order = 0;
    for (const auto& tag : operator_registry_.list()) {
        const auto& def = operator_registry_.get(tag);
        for (const auto& term : def.cells) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
        for (const auto& term : def.boundary) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
        for (const auto& term : def.interior) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
    }
    return max_order;
}

const dofs::DofHandler& FESystem::fieldDofHandler(FieldId field) const
{
    const auto idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || idx >= field_dof_handlers_.size(), InvalidArgumentException,
                "FESystem::fieldDofHandler: invalid field id");
    FE_THROW_IF(!field_dof_handlers_[idx].isFinalized(), InvalidStateException,
                "FESystem::fieldDofHandler: field DOFs not finalized");
    return field_dof_handlers_[idx];
}

GlobalIndex FESystem::fieldDofOffset(FieldId field) const
{
    const auto idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || idx >= field_dof_offsets_.size(), InvalidArgumentException,
                "FESystem::fieldDofOffset: invalid field id");
    return field_dof_offsets_[idx];
}

assembly::AssemblyResult FESystem::assemble(
    const AssemblyRequest& req,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    return assembleOperator(*this, req, state, matrix_out, vector_out);
}

assembly::AssemblyResult FESystem::assembleResidual(
    const SystemStateView& state,
    assembly::GlobalSystemView& rhs_out)
{
    AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    return assemble(req, state, nullptr, &rhs_out);
}

assembly::AssemblyResult FESystem::assembleJacobian(
    const SystemStateView& state,
    assembly::GlobalSystemView& jac_out)
{
    AssemblyRequest req;
    req.op = "jacobian";
    req.want_matrix = true;
    return assemble(req, state, &jac_out, nullptr);
}

assembly::AssemblyResult FESystem::assembleMass(
    const SystemStateView& state,
    assembly::GlobalSystemView& mass_out)
{
    AssemblyRequest req;
    req.op = "mass";
    req.want_matrix = true;
    return assemble(req, state, &mass_out, nullptr);
}

void FESystem::beginTimeStep()
{
    requireSetup();
    if (material_state_provider_) {
        material_state_provider_->beginTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->beginTimeStep();
    }
    if (coupled_boundary_) {
        coupled_boundary_->beginTimeStep();
    }
}

void FESystem::commitTimeStep()
{
    requireSetup();
    if (material_state_provider_) {
        material_state_provider_->commitTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->commitTimeStep();
    }
    if (coupled_boundary_) {
        coupled_boundary_->commitTimeStep();
    }
}

void FESystem::updateConstraints(double time, double dt)
{
    requireSetup();

    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::updateConstraints: constraint");
        if (c->isTimeDependent()) {
            (void)c->updateValues(affine_constraints_, time);
        }
    }

    for (auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::updateConstraints: system constraint");
        if (c->isTimeDependent()) {
            (void)c->updateValues(*this, affine_constraints_, time, dt);
        }
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
