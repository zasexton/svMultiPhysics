/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FESystem.h"

#include "Systems/SystemAssembly.h"
#include "Systems/OperatorBackends.h"
#include "Systems/BoundaryReductionService.h"
#include "Systems/CoupledBoundaryManager.h"
#include "Systems/AuxiliaryStateManager.h"
#include "Systems/AuxiliaryOperatorRegistry.h"
#include "Systems/AuxiliaryInputRegistry.h"
#include "Systems/AuxiliaryBindings.h"
#include "Systems/AuxiliaryModelBuilder.h"
#include "Systems/AuxiliaryStateStepper.h"
#include "Systems/AuxiliaryMultirateScheduler.h"
#include "Forms/PointEvaluator.h"
#include "Systems/AuxiliaryDerivativeProvider.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Dofs/EntityDofMap.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include "Spaces/FunctionSpace.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"

#include <algorithm>
#include <unordered_set>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/MeshAccess.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Systems/MeshSearchAccess.h"
#endif

namespace svmp {
namespace FE {
namespace systems {

/// Walk an expression tree and collect all FieldIds referenced by
/// DiscreteField or StateField nodes.
static void gatherFieldIds(const forms::FormExprNode& node, std::vector<FieldId>& out)
{
    const auto fid = node.fieldId();
    if (fid.has_value()) {
        if (std::find(out.begin(), out.end(), *fid) == out.end()) {
            out.push_back(*fid);
        }
    }
    for (const auto& child : node.childrenShared()) {
        if (child) gatherFieldIds(*child, out);
    }
}

FESystem::FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access)
    : mesh_access_(std::move(mesh_access))
{
    // mesh_access_ may be null for auxiliary-only use (no FE field assembly).
    // Full FE operations (setup, assembly) require non-null mesh.
    operator_backends_ = std::make_unique<OperatorBackends>();
}

FESystem::~FESystem() = default;
FESystem::FESystem(FESystem&&) noexcept = default;
FESystem& FESystem::operator=(FESystem&&) noexcept = default;

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
    dof_permutation_.reset();
    parameter_registry_.clear();
    if (operator_backends_) {
        operator_backends_->invalidateCache();
    }
    assembly_plan_by_op_.clear();
    coupled_jac_cache_.clear();

    // Clear setup-time auxiliary state hooks (sync, transfer) but
    // preserve block definitions and data.
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->invalidateSetup();
    }
    // Clear operator registry layout (rebuilt during setup).
    if (auxiliary_operator_registry_) {
        auxiliary_operator_registry_->clear();
    }

    // Clear setup-time analysis data that is rebuilt during setup().
    // Formulation records, BC descriptors, and definition-time contributions
    // (from CoupledBoundaryManager) are NOT cleared.
    // Only setup-time contributions (from kernel analysisContributions()) are
    // removed by truncating back to the definition-time watermark.
    contributions_.resize(contributions_def_count_);
    topology_context_.reset();
    interface_topology_context_.reset();
    constraint_summary_.reset();

    // Note: GaugeRegistry is NOT cleared here. Candidate deduplication in
    // addCandidate() prevents accumulation on repeated setup(). Anchoring
    // evidence may accumulate from kernel sources, but resolve() overwrites
    // previous results. A full gauge lifecycle fix (clearing setup-time
    // evidence while preserving definition-time evidence) would require
    // a watermark pattern in GaugeRegistry itself.
    invalidateAnalysisCache();
}

void FESystem::requireSetup() const
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "FESystem: setup() has not been called");
}

gauge::GaugeRegistry& FESystem::gaugeRegistry()
{
    if (!gauge_registry_) {
        gauge_registry_ = std::make_unique<gauge::GaugeRegistry>();
    }
    return *gauge_registry_;
}

// ============================================================================
// Problem analysis subsystem
// ============================================================================

void FESystem::addFormulationRecord(analysis::FormulationRecord record) {
    formulation_records_.push_back(std::move(record));
    invalidateAnalysisCache();
}

void FESystem::addBoundaryConditionDescriptor(analysis::BoundaryConditionDescriptor desc) {
    bc_descriptors_.push_back(std::move(desc));
    invalidateAnalysisCache();
}

void FESystem::addContribution(analysis::ContributionDescriptor desc) {
    contributions_.push_back(std::move(desc));
    // Track the definition-time watermark so invalidateSetup() preserves
    // contributions added before setup(). During setup(), the watermark is
    // frozen at the pre-setup level and setup-time contributions are added
    // above it.
    if (!is_setup_) {
        contributions_def_count_ = contributions_.size();
    }
    invalidateAnalysisCache();
}

void FESystem::addVariableDescriptor(analysis::VariableDescriptor desc) {
    variable_descriptors_.push_back(std::move(desc));
    invalidateAnalysisCache();
}

void FESystem::buildTopologyContext() {
    topology_context_ = analysis::TopologyAnalysisContext::build(meshAccess());
    invalidateAnalysisCache();
}

void FESystem::buildInterfaceTopologyContext() {
    analysis::InterfaceTopologyContext ctx;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    for (const auto& [marker, imesh] : interface_meshes_) {
        if (!imesh) continue;

        const auto n_faces = static_cast<GlobalIndex>(imesh->n_faces());
        for (GlobalIndex f = 0; f < n_faces; ++f) {
            auto local_f = static_cast<MeshIndex>(f);
            analysis::InterfaceFaceRecord rec;
            rec.interface_marker = marker;

            auto cells = imesh->volume_cells(local_f);
            rec.minus_cell = static_cast<GlobalIndex>(cells[0]);
            rec.plus_cell = static_cast<GlobalIndex>(cells[1]);
            rec.is_two_sided = !imesh->is_boundary_face(local_f);
            rec.has_orientation = imesh->has_orientation();

            if (rec.is_two_sided) {
                rec.minus_local_face = imesh->local_face_in_cell_minus(local_f);
                rec.plus_local_face = imesh->local_face_in_cell_plus(local_f);
            } else {
                rec.minus_local_face = imesh->local_face_in_cell(local_f);
            }

            // Annotate with bulk region IDs if topology context is available
            if (topology_context_) {
                if (rec.minus_cell != INVALID_GLOBAL_INDEX) {
                    rec.minus_region = topology_context_->regionForCell(rec.minus_cell);
                }
                if (rec.plus_cell != INVALID_GLOBAL_INDEX) {
                    rec.plus_region = topology_context_->regionForCell(rec.plus_cell);
                }
            }

            auto face_idx = ctx.faces.size();
            ctx.faces.push_back(std::move(rec));
            ctx.marker_to_faces[marker].push_back(face_idx);
        }
    }
#endif

    interface_topology_context_ = std::move(ctx);
    invalidateAnalysisCache();
}

void FESystem::buildConstraintSummary() {
    std::vector<analysis::ConstraintAnalysisSummary::FieldDofRange> ranges;
    for (const auto& fr : field_registry_.records()) {
        analysis::ConstraintAnalysisSummary::FieldDofRange r;
        r.field_id = fr.id;
        // Field DOF offsets are only valid after setup
        if (is_setup_ && fr.id < field_dof_offsets_.size()) {
            r.dof_offset = field_dof_offsets_[fr.id];
            r.num_dofs = field_dof_handlers_[fr.id].getStatistics().total_dofs;
            r.num_components = fr.components;
        }
        ranges.push_back(r);
    }

    // Build a DOF→region provider when topology is available.
    // Uses the EntityDofMap to map DOF → entity → cell → region.
    // Handles vertex, edge, face, and cell entities.
    const auto* topo = topology_context_ ? &*topology_context_ : nullptr;
    analysis::ConstraintAnalysisSummary::DofRegionProvider dof_region;
    if (topo && topo->numRegions() > 1) {
        const auto* emap = dof_handler_.getEntityDofMap();
        if (emap && mesh_access_) {
            // Pre-build vertex→cell map for O(1) lookup instead of O(n_cells) per DOF
            const auto n_cells = meshAccess().numCells();
            auto vertex_to_cell = std::make_shared<std::unordered_map<GlobalIndex, GlobalIndex>>();
            {
                std::vector<GlobalIndex> nodes;
                for (GlobalIndex c = 0; c < n_cells; ++c) {
                    nodes.clear();
                    meshAccess().getCellNodes(c, nodes);
                    for (auto n : nodes) {
                        vertex_to_cell->emplace(n, c);  // first cell wins
                    }
                }
            }

            dof_region = [topo, emap, vertex_to_cell, n_cells, this](GlobalIndex dof) -> int {
                auto ent = emap->getDofEntity(dof);
                if (!ent) return -1;

                switch (ent->kind) {
                    case dofs::EntityKind::Vertex: {
                        auto it = vertex_to_cell->find(ent->id);
                        if (it != vertex_to_cell->end()) {
                            return topo->regionForCell(it->second);
                        }
                        return -1;
                    }
                    case dofs::EntityKind::Cell: {
                        // Cell DOF — entity ID is the cell index
                        return topo->regionForCell(ent->id);
                    }
                    default: {
                        // Edge/Face DOFs: find an incident cell by scanning.
                        // This is O(n_cells) per DOF but only runs once during
                        // constraint summary build. For large meshes, a
                        // pre-built edge/face→cell map would be more efficient.
                        const auto& dmap = dof_handler_.getDofMap();
                        for (GlobalIndex c = 0; c < n_cells; ++c) {
                            auto cell_dofs = dmap.getCellDofs(c);
                            for (auto cd : cell_dofs) {
                                if (cd == dof) {
                                    return topo->regionForCell(c);
                                }
                            }
                        }
                        return -1;
                    }
                }
            };
        }
    }

    // Build component DOF provider from FieldDofMap.
    // Uses getComponentDofs() which works for any layout (component-blocked,
    // interleaved, or vector-basis). Returns empty for VectorBasis fields
    // where component extraction is not defined.
    analysis::ConstraintAnalysisSummary::ComponentDofProvider comp_dofs;
    if (is_setup_ && field_map_.numFields() > 0) {
        comp_dofs = [this](FieldId fid, int component) -> std::vector<GlobalIndex> {
            auto field_idx = static_cast<std::size_t>(fid);
            if (field_idx >= field_map_.numFields()) return {};
            const auto& fd = field_map_.getField(field_idx);
            if (fd.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise) return {};
            if (component < 0 || static_cast<LocalIndex>(component) >= fd.n_components) return {};
            auto idx_set = field_map_.getComponentDofs(field_idx, static_cast<LocalIndex>(component));
            return idx_set.toVector();
        };
    }

    constraint_summary_ = analysis::ConstraintAnalysisSummary::build(
        affine_constraints_, ranges, topo, dof_region, comp_dofs);
    invalidateAnalysisCache();
}

void FESystem::invalidateAnalysisCache() noexcept {
    ++analysis_inputs_version_;
}

analysis::ProblemAnalysisReport FESystem::runProblemAnalysis() const {
    analysis::ProblemAnalysisContext ctx;

    // Populate field descriptors from FieldRegistry.
    for (const auto& fr : field_registry_.records()) {
        analysis::FieldDescriptor fd;
        fd.field_id = fr.id;
        fd.name = fr.name;
        fd.value_dimension = fr.components;
        fd.field_type = (fr.components > 1) ? FieldType::Vector : FieldType::Scalar;
        if (fr.space) {
            fd.polynomial_order = fr.space->polynomial_order();
            fd.topological_dimension = fr.space->topological_dimension();
            fd.continuity = fr.space->continuity();

            // Derive component_extractable from the function space continuity.
            // H(div) and H(curl) spaces use vector-valued basis functions where
            // DOFs are NOT per-component — component extraction is not defined.
            // This works both pre-setup and post-setup.
            if (fd.continuity == Continuity::H_div ||
                fd.continuity == Continuity::H_curl) {
                fd.component_extractable = false;
            }

            // Phase 21: space family and trace capabilities from continuity
            switch (fd.continuity) {
                case Continuity::C0:
                case Continuity::C1:
                    fd.space_family = analysis::SpaceFamily::H1;
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::Value
                                          | analysis::TraceCapabilityFlags::NormalFlux;
                    break;
                case Continuity::H_div:
                    fd.space_family = analysis::SpaceFamily::HDiv;
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::NormalComponent
                                          | analysis::TraceCapabilityFlags::NormalFlux;
                    fd.has_exact_sequence_structure = true;
                    fd.supports_local_balance_closure = true;
                    break;
                case Continuity::H_curl:
                    fd.space_family = analysis::SpaceFamily::HCurl;
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::TangentialComponent;
                    fd.has_exact_sequence_structure = true;
                    break;
                case Continuity::L2:
                    fd.space_family = analysis::SpaceFamily::L2;
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::Jump
                                          | analysis::TraceCapabilityFlags::Average;
                    break;
                default:
                    fd.space_family = analysis::SpaceFamily::Custom;
                    break;
            }
        }
        // Post-setup refinement: use the actual FieldDofMap layout descriptor
        // which is authoritative (handles edge cases like custom spaces).
        if (is_setup_ && fr.id < field_map_.numFields()) {
            const auto& fmd = field_map_.getField(static_cast<std::size_t>(fr.id));
            fd.component_extractable =
                (fmd.component_dof_layout == dofs::FieldComponentDofLayout::ComponentWise);
        }
        ctx.addFieldDescriptor(std::move(fd));
    }

    // Populate variable descriptors.
    for (const auto& vd : variable_descriptors_) {
        ctx.addVariableDescriptor(vd);
    }

    // Populate formulation records.
    for (const auto& rec : formulation_records_) {
        ctx.addFormulationRecord(rec);
    }

    // Populate normalized contributions.
    for (const auto& c : contributions_) {
        ctx.addContribution(c);
    }

    // Populate BC descriptors.
    for (const auto& desc : bc_descriptors_) {
        ctx.addBCDescriptor(desc);
    }

    // Populate topology context if available.
    if (topology_context_) {
        ctx.setTopologyContext(*topology_context_);
    }

    // Populate interface topology if available.
    if (interface_topology_context_) {
        ctx.setInterfaceTopologyContext(*interface_topology_context_);
    }

    // Populate constraint summary if available.
    if (constraint_summary_) {
        ctx.setConstraintSummary(*constraint_summary_);
    }

    auto analyzer = analysis::ProblemAnalyzer::createDefault();
    return analyzer.analyze(ctx);
}

const analysis::ProblemAnalysisReport& FESystem::analysisReport() const {
    if (analysis_report_version_ != analysis_inputs_version_) {
        analysis_report_cache_ = runProblemAnalysis();
        analysis_report_version_ = analysis_inputs_version_;
    }
    return *analysis_report_cache_;
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

AuxiliaryStateManager& FESystem::auxiliaryStateManager()
{
    if (!auxiliary_state_manager_) {
        auxiliary_state_manager_ = std::make_unique<AuxiliaryStateManager>();
    }
    return *auxiliary_state_manager_;
}

AuxiliaryOperatorRegistry& FESystem::auxiliaryOperatorRegistry()
{
    if (!auxiliary_operator_registry_) {
        auxiliary_operator_registry_ = std::make_unique<AuxiliaryOperatorRegistry>();
    }
    return *auxiliary_operator_registry_;
}

AuxiliaryInputRegistry& FESystem::auxiliaryInputRegistry()
{
    if (!auxiliary_input_registry_) {
        auxiliary_input_registry_ = std::make_unique<AuxiliaryInputRegistry>();
    }
    return *auxiliary_input_registry_;
}

FEQuantityRegistry& FESystem::feQuantityRegistry()
{
    if (!fe_quantity_registry_) {
        fe_quantity_registry_ = std::make_unique<FEQuantityRegistry>();
    }
    return *fe_quantity_registry_;
}

std::span<const backends::RankOneUpdate> FESystem::lastRankOneUpdates() const noexcept
{
    return last_rank_one_updates_;
}

void FESystem::clearRankOneUpdates() noexcept
{
    last_rank_one_updates_.clear();
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

bool FESystem::evaluateFieldAtVertices(FieldId field,
                                        const SystemStateView& state,
                                        GlobalIndex n_vertices,
                                        std::span<double> out) const
{
    requireSetup();

    if (n_vertices <= 0) {
        return false;
    }

    const auto field_idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || field_idx >= field_dof_handlers_.size(), InvalidArgumentException,
                "FESystem::evaluateFieldAtVertices: invalid FieldId");

    const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
    if (!entity_map) {
        return false;
    }

    if (entity_map->numVertices() < n_vertices) {
        return false; // Entity map doesn't cover all mesh vertices
    }

    const auto& rec = field_registry_.get(field);
    const auto ncomp = static_cast<std::size_t>(std::max(1, rec.components));

    FE_THROW_IF(out.size() < static_cast<std::size_t>(n_vertices) * ncomp, InvalidArgumentException,
                "FESystem::evaluateFieldAtVertices: output buffer too small");

    // Check that vertex DOFs exist and have the expected component count
    {
        const auto test_dofs = entity_map->getVertexDofs(0);
        if (test_dofs.empty()) {
            return false; // No vertex DOFs (e.g. DG elements)
        }
        if (test_dofs.size() != ncomp) {
            return false; // Component count mismatch
        }
    }

    const GlobalIndex offset = field_dof_offsets_[field_idx];

    // Create assembly view if backend vector is provided (MPI case)
    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        solution_view = vec->createAssemblyView();
    }

    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        const auto vdofs = entity_map->getVertexDofs(v);
        const auto out_base = static_cast<std::size_t>(v) * ncomp;
        for (std::size_t c = 0; c < ncomp; ++c) {
            const GlobalIndex d = vdofs[c] + offset;
            if (solution_view) {
                out[out_base + c] = solution_view->getVectorEntry(d);
            } else {
                const auto idx = static_cast<std::size_t>(d);
                FE_THROW_IF(idx >= state.u.size(), InvalidArgumentException,
                            "FESystem::evaluateFieldAtVertices: state.u too small");
                out[out_base + c] = state.u[idx];
            }
        }
    }

    return true;
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

namespace {

void gatherTimeDerivativeFieldsFromNode(const forms::FormExprNode& node,
                                        FieldId kernel_trial_field,
                                        std::unordered_set<FieldId>& out)
{
    if (node.type() == forms::FormExprType::TimeDerivative) {
        const auto children = node.childrenShared();
        if (!children.empty() && children.front()) {
            const auto& child = *children.front();
            if (child.type() == forms::FormExprType::TrialFunction) {
                if (kernel_trial_field != INVALID_FIELD_ID) {
                    out.insert(kernel_trial_field);
                }
            } else if (child.type() == forms::FormExprType::StateField ||
                       child.type() == forms::FormExprType::DiscreteField) {
                if (const auto fid = child.fieldId()) {
                    out.insert(*fid);
                }
            }
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            gatherTimeDerivativeFieldsFromNode(*child, kernel_trial_field, out);
        }
    }
}

void gatherTimeDerivativeFieldsFromIR(const forms::FormIR& ir,
                                      FieldId kernel_trial_field,
                                      std::unordered_set<FieldId>& out)
{
    for (const auto& term : ir.terms()) {
        const auto* root = term.integrand.node();
        if (!root) {
            continue;
        }
        gatherTimeDerivativeFieldsFromNode(*root, kernel_trial_field, out);
    }
}

void gatherTimeDerivativeFieldsFromKernel(const assembly::AssemblyKernel* kernel,
                                          FieldId kernel_trial_field,
                                          std::unordered_set<FieldId>& out)
{
    if (!kernel) {
        return;
    }

    if (const auto* k = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
        gatherTimeDerivativeFieldsFromKernel(&k->fallbackKernel(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->residualIR(), kernel_trial_field, out);
        gatherTimeDerivativeFieldsFromIR(k->tangentIR(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->residualIR(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::FormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->ir(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::LinearFormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->bilinearIR(), kernel_trial_field, out);
        if (k->linearIR().has_value()) {
            gatherTimeDerivativeFieldsFromIR(*k->linearIR(), kernel_trial_field, out);
        }
        return;
    }
}

std::vector<FieldId> sortedUnique(std::unordered_set<FieldId> ids)
{
    std::vector<FieldId> out;
    out.reserve(ids.size());
    for (const auto fid : ids) {
        out.push_back(fid);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

} // namespace

std::vector<FieldId> FESystem::timeDerivativeFields(const OperatorTag& op) const
{
    std::unordered_set<FieldId> fields;
    const auto& def = operator_registry_.get(op);

    for (const auto& term : def.cells) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.boundary) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.interior) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.interface_faces) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }

    return sortedUnique(std::move(fields));
}

std::vector<FieldId> FESystem::timeDerivativeFields() const
{
    std::unordered_set<FieldId> fields;
    for (const auto& op : operator_registry_.list()) {
        const auto& def = operator_registry_.get(op);
        for (const auto& term : def.cells) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.boundary) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.interior) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.interface_faces) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
    }
    return sortedUnique(std::move(fields));
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
    // requireSetup() is skipped for auxiliary-only use (no mesh/fields).
    // Material/global-kernel/coupled-boundary providers are null when not set up.
    if (material_state_provider_) {
        material_state_provider_->beginTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->beginTimeStep();
    }
    if (coupled_boundary_) {
        coupled_boundary_->beginTimeStep();
    }
    // Reset generalized auxiliary state to committed values.
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->resetAllToCommitted();
    }
    // Invalidate all auxiliary inputs for the new time step.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->invalidateAll();
    }
}

void FESystem::commitTimeStep()
{
    if (material_state_provider_) {
        material_state_provider_->commitTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->commitTimeStep();
    }
    if (coupled_boundary_) {
        coupled_boundary_->commitTimeStep();
    }
    // Commit generalized auxiliary state with the last-known time.
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->commitAll(last_auxiliary_advance_time_);
    }
}

// ---------------------------------------------------------------------------
//  System state cache for auxiliary input callbacks
// ---------------------------------------------------------------------------

void FESystem::cacheSystemState(const SystemStateView& state) const
{
    cached_solution_u_ = state.u;
    cached_solution_vector_ = state.u_vector;
    cached_solution_u_prev_ = state.u_prev;
    cached_solution_prev_vector_ = state.u_prev_vector;
    cached_solution_u_prev2_ = state.u_prev2;
    cached_solution_prev2_vector_ = state.u_prev2_vector;
    cached_time_integration_ = state.time_integration;
    cached_user_data_ = state.user_data;
}

// ---------------------------------------------------------------------------
//  Auxiliary lifecycle
// ---------------------------------------------------------------------------

void FESystem::prepareAuxiliaryForAssembly(const SystemStateView& state,
                                            bool is_nonlinear_iteration)
{
    // Resolve any deferred derived-input expressions and dependency edges
    // that were registered via derivedInput().  This runs at most once —
    // after finalization, both vectors are empty.
    finalizeDeferredInputDeps();

    // Cache the full system state for FE-coupled input callbacks.
    cacheSystemState(state);

    // Evaluate auxiliary input providers.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(state.time, state.dt, is_nonlinear_iteration);
    }

    // Evaluate outputs for deployed models via the base-class output interface.
    for (auto& entry : deployed_aux_entries_) {
        const auto n_outputs = static_cast<std::size_t>(entry.model->outputCount());
        if (n_outputs == 0) continue;
        if (!auxiliary_state_manager_ || !auxiliary_state_manager_->hasBlock(entry.instance_name))
            continue;

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);

        // Build param values: prefer base-class signature, then built-model, then map-key.
        std::vector<Real> params;
        auto declared_params = entry.model->declaredParameterNames();
        if (!declared_params.empty()) {
            params.resize(declared_params.size(), 0.0);
            for (std::size_t pi = 0; pi < declared_params.size(); ++pi) {
                auto it = entry.param_values.find(declared_params[pi]);
                if (it != entry.param_values.end()) params[pi] = it->second;
            }
        } else if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
            const auto& sig = built->signature();
            params.resize(sig.parameters.size(), 0.0);
            for (std::size_t pi = 0; pi < sig.parameters.size(); ++pi) {
                auto it = entry.param_values.find(sig.parameters[pi].name);
                if (it != entry.param_values.end()) params[pi] = it->second;
            }
        } else {
            for (const auto& [pname, pval] : entry.param_values) {
                params.push_back(pval);
            }
        }

        // Build bound inputs for output evaluation.
        // For built models: ordered by signature. For generic: ordered by binding key.
        std::vector<Real> bound_inputs;
        if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
            const auto& sig = built->signature();
            std::size_t total_input_size = 0;
            for (const auto& inp : sig.inputs) total_input_size += static_cast<std::size_t>(inp.size);
            bound_inputs.resize(total_input_size, 0.0);
            std::size_t inp_offset = 0;
            for (const auto& inp : sig.inputs) {
                auto bind_it = entry.input_bindings.find(inp.name);
                if (bind_it != entry.input_bindings.end() && auxiliary_input_registry_) {
                    auto reg_vals = auxiliary_input_registry_->valuesOf(bind_it->second);
                    for (std::size_t k = 0; k < std::min(reg_vals.size(), static_cast<std::size_t>(inp.size)); ++k)
                        bound_inputs[inp_offset + k] = reg_vals[k];
                }
                inp_offset += static_cast<std::size_t>(inp.size);
            }
        } else {
            // Non-built models: prefer declaredInputNames() with name:size
            // parsing, then map-key order.
            auto decl_in = entry.model->declaredInputNames();
            if (!decl_in.empty() && auxiliary_input_registry_) {
                for (const auto& raw : decl_in) {
                    auto [iname, input_size] = parseDeclaredInputName(raw);
                    auto bind_it = entry.input_bindings.find(iname);
                    if (bind_it != entry.input_bindings.end() &&
                        auxiliary_input_registry_->hasInput(bind_it->second)) {
                        auto vals = auxiliary_input_registry_->valuesOf(bind_it->second);
                        for (int k = 0; k < input_size; ++k) {
                            bound_inputs.push_back(
                                k < static_cast<int>(vals.size()) ? vals[static_cast<std::size_t>(k)] : 0.0);
                        }
                    } else {
                        bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(input_size), 0.0);
                    }
                }
            } else if (!entry.input_bindings.empty() && auxiliary_input_registry_) {
                for (const auto& [model_name, reg_name] : entry.input_bindings) {
                    if (auxiliary_input_registry_->hasInput(reg_name)) {
                        auto vals = auxiliary_input_registry_->valuesOf(reg_name);
                        bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                    }
                }
            }
        }

        const auto n_entities = blk.entityCount();
        entry.output_buffer.resize(n_entities * n_outputs);

        // Build xdot = 0 scratch for output evaluation.
        const auto dim = static_cast<std::size_t>(entry.model->dimension());
        std::vector<Real> xdot_zero(dim, 0.0);
        const auto& emap = entry.entity_map; // empty = identity mapping

        // Build per-entity history spans.
        std::vector<std::vector<Real>> hist_entity_data;
        std::vector<std::span<const Real>> hist_spans;

        // Detect entity-local bindings for output eval.
        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [model_name, reg_name] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(reg_name) &&
                    auxiliary_input_registry_->isEntityLocal(reg_name)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        for (std::size_t e = 0; e < n_entities; ++e) {
            // Layout-aware entity gather.
            auto entity_state_vec = blk.gatherEntityWork(e);

            // Rebuild bound inputs per entity when entity-local bindings exist.
            const auto orig_e = emap.empty() ? e : emap[e];
            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* b2 = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : b2->signature().inputs) {
                        auto bind_it = entry.input_bindings.find(inp.name);
                        if (bind_it != entry.input_bindings.end()) {
                            auto vals = auxiliary_input_registry_->valuesOf(bind_it->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                        } else {
                            bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            // Layout-aware history gather.
            hist_entity_data.clear();
            hist_spans.clear();
            for (std::size_t k = 0; k < blk.history().depth(); ++k) {
                hist_entity_data.push_back(blk.gatherEntityHistory(k, e));
                hist_spans.push_back(hist_entity_data.back());
            }

            // Populate field_values for models with direct FE field references.
            std::vector<FieldValueEntry> fv_prep;
            if (entry.deriv_provider) {
                const auto& art_prep = entry.deriv_provider->artifact();
                for (const auto fid : art_prep.referenced_fields) {
                    const auto fidx = static_cast<std::size_t>(fid);
                    if (fidx >= field_dof_offsets_.size() ||
                        fidx >= field_dof_handlers_.size()) continue;
                    const auto fld_off = field_dof_offsets_[fidx];
                    const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                    if (!femap) continue;
                    auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                    if (!vdofs.empty()) {
                        FieldValueEntry fve;
                        fve.field = fid;
                        fve.n_components = static_cast<int>(vdofs.size());
                        for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                            const auto gidx = static_cast<std::size_t>(vdofs[static_cast<std::size_t>(c)] + fld_off);
                            fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : 0.0;
                        }
                        fv_prep.push_back(fve);
                    }
                }
            }

            AuxiliaryLocalContext ctx;
            ctx.time = state.time;
            ctx.dt = state.dt;
            ctx.effective_dt = state.dt;
            ctx.x = entity_state_vec;
            ctx.xdot = xdot_zero;
            ctx.history = hist_spans;
            ctx.inputs = bound_inputs;
            ctx.params = params;
            ctx.entity_index = e;
            ctx.field_values = fv_prep;

            std::span<Real> out_span{
                entry.output_buffer.data() + e * n_outputs, n_outputs};
            entry.model->evaluateOutputs(ctx, out_span);
        }
    }
}

void FESystem::deployAuxiliaryModel(AuxiliaryDeployedInstance instance)
{
    auto diag = instance.validate();
    FE_THROW_IF(!diag.empty(), InvalidArgumentException,
                "FESystem::deployAuxiliaryModel: " + diag);

    // Validate declared input name suffixes at deployment time.
    validateDeclaredInputNames(*instance.model());

    DeployedAuxEntry entry;
    entry.model = instance.model();
    entry.instance_name = instance.instanceName();

    // Build spec from deployment configuration.
    entry.spec.name = instance.instanceName();
    entry.spec.size = instance.model()->dimension();
    entry.spec.scope = instance.getScope();
    entry.spec.solve_mode = instance.getSolveMode();
    entry.spec.schedule_mode = instance.getSchedule();
    entry.spec.layout_mode = instance.getLayoutMode();
    entry.spec.ordering = instance.getEntityOrdering();
    entry.spec.deployment_region = instance.getRegion();
    // Copy derivative policy: prefer explicit instance policy, then built-model policy.
    if (instance.hasExplicitDerivativePolicy()) {
        entry.spec.derivative_policy = instance.getDerivativePolicy();
    } else if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(instance.model().get())) {
        entry.spec.derivative_policy = built->derivativePolicy();
    }
    entry.stepper_spec = instance.getStepperSpec();
    entry.initial_values = instance.initialValues();
    for (const auto& [k, v] : instance.inputBindings())
        entry.input_bindings[k] = v;
    for (const auto& [k, v] : instance.coupledBindings())
        entry.coupled_bindings[k] = v;
    entry.param_values = instance.paramValues();
    entry.explicit_entity_count = instance.getEntityCount();

    deployed_aux_entries_.push_back(std::move(entry));
}

AuxiliaryInstanceHandle FESystem::deploy(AuxiliaryDeployedInstance instance)
{
    const std::string inst_name = instance.instanceName();
    deployAuxiliaryModel(std::move(instance));
    return AuxiliaryInstanceHandle(inst_name);
}

AuxiliaryInputHandle FESystem::boundaryIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    forms::BoundaryFunctional::Reduction reduction,
    AuxiliaryInputUpdateSchedule schedule)
{
    // Gather referenced fields before moving integrand.
    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;  // copy before move
    def->boundary_marker = boundary_marker;
    def->capabilities.explicit_evaluation = true;
    // Monolithic linearization for boundary integrals requires the
    // StandardAssembler to have a GlobalSystemView solution set.
    // This works in production (backends provide GenericVector) but
    // not in lightweight test configurations with raw span solutions.
    // Mark as supported — the runtime path is wired through
    // evaluateFunctionalGradient() → assembleBoundaryGradient().
    def->capabilities.monolithic_linearization = !refs.empty();

    registerBoundaryIntegralInput(input_name, std::move(integrand),
                                  boundary_marker, reduction, schedule);
    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::boundaryIntegral(
    const std::string& input_name,
    forms::BoundaryFunctional functional,
    AuxiliaryInputUpdateSchedule schedule)
{
    std::vector<FieldId> refs;
    if (const auto* root = functional.integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = functional.integrand;
    def->boundary_marker = functional.boundary_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    registerBoundaryIntegralInput(input_name, std::move(functional), schedule);
    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::derivedInput(
    const std::string& name,
    forms::FormExpr expr,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(name.empty(), InvalidArgumentException,
                "FESystem::derivedInput: empty name");

    auto& reg = auxiliaryInputRegistry();

    AuxiliaryInputSpec spec;
    spec.name = name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::FormulationCallback;
    spec.update_schedule = schedule;

    // Auto-discover dependencies by scanning the expression for AuxiliaryInputSymbol
    // nodes referencing other registry inputs.  Must do this BEFORE moving expr.
    std::vector<std::string> deps;
    if (const auto* root = expr.node()) {
        std::function<void(const forms::FormExprNode&)> scan =
            [&](const forms::FormExprNode& n) {
                if (n.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                    if (auto sym = n.symbolName()) {
                        deps.push_back(std::string(*sym));
                    }
                }
                for (const auto* child : n.children()) {
                    if (child) scan(*child);
                }
            };
        scan(*root);
    }

    // Reject self-references BEFORE any side effects (registration, deferred
    // expression storage).  A failed check must not leave a partially-registered
    // input in the registry or a dangling deferred expression.
    for (const auto& dep : deps) {
        FE_THROW_IF(dep == name, InvalidArgumentException,
                    "FESystem::derivedInput('" + name +
                        "'): expression references itself — "
                        "self-referential derived inputs are not allowed");
    }

    // Store the expression in a shared_ptr so it can be resolved to
    // slot-based refs during finalizeDeferredInputDeps() (after all inputs
    // are registered and slots are stable).
    auto resolved_expr = std::make_shared<forms::FormExpr>(std::move(expr));
    auto* reg_ptr = &reg;

    reg.registerInput(spec,
        [reg_ptr, resolved_expr](Real time, Real dt, std::span<Real> out) {
            forms::PointEvalContext pctx;
            pctx.time = time;
            pctx.dt = dt;
            pctx.auxiliary_inputs = reg_ptr->all();
            out[0] = forms::evaluateScalarAt(*resolved_expr, pctx);
        });

    // Store (name, shared_ptr) for deferred symbol resolution.
    deferred_derived_exprs_.emplace_back(name, resolved_expr);

    // Defer dependency wiring to finalizeDeferredInputDeps().
    // At registration time, referenced inputs may not yet exist.  Wiring
    // now would silently drop any forward references.  At finalization,
    // all inputs are registered, so any unresolved name is a real error.
    for (const auto& dep : deps) {
        deferred_input_deps_.emplace_back(name, dep);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = name;
    def->kind = FEQuantityKind::DerivedCallback;
    def->shape = FEQuantityShape::scalar();
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = false;

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(name, std::move(def));
}

AuxiliaryInputHandle FESystem::sampledField(
    const std::string& input_name,
    const std::string& field_name,
    std::size_t n_entities)
{
    registerSampledFieldInput(input_name, field_name, n_entities);

    // Determine field ID and components for the definition.
    const auto fid = field_registry_.findByName(field_name);
    const int components = field_registry_.get(fid).components;

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::SampledField;
    def->shape = (components == 1)
        ? FEQuantityShape::scalar()
        : FEQuantityShape::vector(components);
    def->referenced_fields = {fid};
    def->source_field_name = field_name;
    def->entity_count = n_entities;
    def->capabilities.explicit_evaluation = true;
    // Sampled field dI/du is identity at sampled DOFs.
    def->capabilities.monolithic_linearization = true;

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::boundaryNodalSum(
    const std::string& input_name,
    const std::string& field_name,
    int boundary_marker)
{
    registerBoundaryNodalSumInput(input_name, field_name, boundary_marker);

    const auto fid = field_registry_.findByName(field_name);
    const int components = field_registry_.get(fid).components;

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryNodalSum;
    def->shape = (components == 1)
        ? FEQuantityShape::scalar()
        : FEQuantityShape::vector(components);
    def->referenced_fields = {fid};
    def->source_field_name = field_name;
    def->boundary_marker = boundary_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = false;  // nodal sum, not quadrature-weighted

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::boundaryAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    AuxiliaryInputUpdateSchedule schedule)
{
    // Boundary average = boundary integral / boundary measure.
    // Use BoundaryFunctional with Average reduction mode.
    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->boundary_marker = boundary_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    registerBoundaryIntegralInput(input_name, std::move(integrand),
                                  boundary_marker,
                                  forms::BoundaryFunctional::Reduction::Average,
                                  schedule);
    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::domainIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::domainIntegral: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::DomainIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Domain integrals use the FunctionalAssembler over all cells.
    // Determine the primary field for mesh/space context.
    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "FESystem::domainIntegral('" + input_name +
                    "'): at least one FE field must be registered");

    auto& reg = auxiliaryInputRegistry();

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::DomainIntegral;
    spec.update_schedule = schedule;
    spec.requires_mpi_reduction = true;

    // Evaluate via the BoundaryReductionService using a cell-domain
    // functional.  The service's functional assembler handles both
    // boundary and cell assembly.  We use boundary_marker = -1 to
    // signal a domain (all-cells) integral.
    auto captured_integrand = integrand;
    const auto captured_fid = primary_fid;
    const std::string func_name = input_name;

    // Register a domain functional with the per-field reduction service.
    auto& svc = boundaryReductionService(captured_fid);
    forms::BoundaryFunctional domain_func;
    domain_func.name = func_name;
    domain_func.integrand = std::move(integrand);
    domain_func.boundary_marker = -1;  // domain (all cells)
    domain_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
    domain_func.is_domain_functional = true;
    svc.addBoundaryFunctional(domain_func);
    bindSecondaryFields(svc, captured_fid, refs);

    reg.registerInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::span<Real> out) {
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctional(func_name, state);
            } else {
                out[0] = 0.0;
            }
        });

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::domainAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::domainAverage: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::DomainAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Domain average = domain integral / domain measure.
    // Register two callbacks: the integral and the measure, then
    // combine in a derived callback.
    const std::string integral_name = input_name + "__integral";
    const std::string measure_name = input_name + "__measure";

    // Register the integral.
    domainIntegral(integral_name, integrand, schedule);

    // Register the measure (∫ 1 dx = total domain volume).
    domainIntegral(measure_name, forms::FormExpr::constant(1.0), schedule);

    // Register the average as a derived callback.
    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::DomainAverage;
    spec.update_schedule = schedule;

    auto* reg_ptr = &reg;
    const auto int_name = integral_name;
    const auto meas_name = measure_name;

    reg.registerInput(spec,
        [reg_ptr, int_name, meas_name](Real, Real, std::span<Real> out) {
            const Real integral = reg_ptr->get(int_name);
            const Real measure = reg_ptr->get(meas_name);
            out[0] = (measure > 0.0) ? integral / measure : 0.0;
        });
    reg.addDependency(input_name, integral_name);
    reg.addDependency(input_name, measure_name);

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::regionIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    int region_marker,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::regionIntegral: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::RegionIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->region_marker = region_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Region integrals use BoundaryReductionService with a domain functional
    // filtered by region marker (material/domain ID).
    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "FESystem::regionIntegral('" + input_name +
                    "'): at least one FE field must be registered");

    auto& svc = boundaryReductionService(primary_fid);
    forms::BoundaryFunctional region_func;
    region_func.name = input_name;
    region_func.integrand = integrand;
    region_func.boundary_marker = -1;
    region_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
    region_func.is_domain_functional = true;
    region_func.region_marker = region_marker;
    svc.addBoundaryFunctional(region_func);
    bindSecondaryFields(svc, primary_fid, refs);

    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::DomainIntegral;
    spec.update_schedule = schedule;
    spec.requires_mpi_reduction = true;

    const auto captured_fid = primary_fid;
    const std::string func_name = input_name;

    reg.registerInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::span<Real> out) {
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctional(func_name, state);
            } else {
                out[0] = 0.0;
            }
        });

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::regionAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    int region_marker,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::regionAverage: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::RegionAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->region_marker = region_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Region average = region integral / region measure.
    const std::string integral_name = input_name + "__integral";
    const std::string measure_name = input_name + "__measure";

    regionIntegral(integral_name, integrand, region_marker, schedule);
    regionIntegral(measure_name, forms::FormExpr::constant(1.0), region_marker, schedule);

    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::DomainAverage;
    spec.update_schedule = schedule;

    auto* reg_ptr = &reg;
    const auto int_name = integral_name;
    const auto meas_name = measure_name;

    reg.registerInput(spec,
        [reg_ptr, int_name, meas_name](Real, Real, std::span<Real> out) {
            const Real integral = reg_ptr->get(int_name);
            const Real measure = reg_ptr->get(meas_name);
            out[0] = (measure > 0.0) ? integral / measure : 0.0;
        });
    reg.addDependency(input_name, integral_name);
    reg.addDependency(input_name, measure_name);

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::feExpression(
    const std::string& input_name,
    forms::FormExpr expression,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::feExpression: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = expression.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::FEExpression;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = expression;
    def->capabilities.explicit_evaluation = true;
    // FE expressions that reference fields support monolithic linearization
    // through the same domain-functional gradient assembly path.
    def->capabilities.monolithic_linearization = !refs.empty();

    // Use the domain-functional path (same as domainIntegral) so the
    // expression gets proper FE evaluation with quadrature and field
    // binding, AND supports symbolic gradient assembly for dI/du.
    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }

    if (primary_fid != INVALID_FIELD_ID) {
        // Register as a domain functional through BoundaryReductionService.
        auto& svc = boundaryReductionService(primary_fid);
        forms::BoundaryFunctional domain_func;
        domain_func.name = input_name;
        domain_func.integrand = expression;
        domain_func.boundary_marker = -1;
        domain_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
        domain_func.is_domain_functional = true;
        svc.addBoundaryFunctional(domain_func);
        bindSecondaryFields(svc, primary_fid, refs);

        auto& reg = auxiliaryInputRegistry();
        AuxiliaryInputSpec spec;
        spec.name = input_name;
        spec.size = 1;
        spec.producer = AuxiliaryInputProducer::DomainIntegral;
        spec.update_schedule = schedule;
        spec.requires_mpi_reduction = true;

        const auto captured_fid = primary_fid;
        const std::string func_name = input_name;

        reg.registerInput(spec,
            [this, func_name, captured_fid]
            (Real time, Real dt, std::span<Real> out) {
                SystemStateView state;
                state.time = time;
                state.dt = dt;
                state.u = cached_solution_u_;
                state.u_vector = cached_solution_vector_;
                state.u_prev = cached_solution_u_prev_;
                state.u_prev_vector = cached_solution_prev_vector_;
                state.time_integration = cached_time_integration_;
                state.user_data = cached_user_data_;

                auto it = boundary_reduction_services_.find(captured_fid);
                if (it != boundary_reduction_services_.end() && it->second) {
                    out[0] = it->second->evaluateFunctional(func_name, state);
                } else {
                    out[0] = 0.0;
                }
            });
    } else {
        // No field references: use PointEvaluator as a simple callback.
        auto& reg = auxiliaryInputRegistry();
        AuxiliaryInputSpec spec;
        spec.name = input_name;
        spec.size = 1;
        spec.producer = AuxiliaryInputProducer::FormulationCallback;
        spec.update_schedule = schedule;

        auto captured_expr = std::move(expression);

        reg.registerInput(spec,
            [this, captured_expr](Real time, Real dt, std::span<Real> out) {
                forms::PointEvalContext pctx;
                pctx.time = time;
                pctx.dt = dt;
                if (auxiliary_input_registry_) {
                    pctx.auxiliary_inputs = auxiliary_input_registry_->all();
                }
                out[0] = forms::evaluateScalarAt(captured_expr, pctx);
            });
    }

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

void FESystem::advanceAuxiliaryState(const SystemStateView& state)
{
    advanceAuxiliaryState(state, /*is_nonlinear_iteration=*/false);
}

void FESystem::advanceAuxiliaryState(const SystemStateView& state,
                                     bool is_nonlinear_iteration)
{
    // Cache the full system state so boundary-integral input callbacks
    // (and other FE-coupled callbacks) can access the current solution.
    cacheSystemState(state);

    // Pre-refresh inputs using the caller's nonlinear-iteration semantics.
    // The Real/Real overload will reuse the cached values and no-op for
    // clean OncePerTimeStep inputs.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(
            static_cast<Real>(state.time),
            static_cast<Real>(state.dt),
            is_nonlinear_iteration);
    }

    advanceAuxiliaryState(static_cast<Real>(state.time), static_cast<Real>(state.dt));
}

void FESystem::advanceAuxiliaryState(Real time, Real dt)
{
    last_auxiliary_advance_time_ = time + dt;

    if (!auxiliary_state_manager_) return;

    // Ensure auxiliary inputs are evaluated before stepping reads them.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(time, dt);
    }

    // Check if any block uses Multirate scheduling (interleaved time ordering).
    bool has_multirate = false;
    for (const auto& entry : deployed_aux_entries_) {
        if (entry.spec.solve_mode == AuxiliarySolveMode::Partitioned &&
            entry.spec.schedule_mode == AuxiliaryScheduleMode::Multirate) {
            has_multirate = true;
            break;
        }
    }

    if (has_multirate && aux_scheduler_) {
        // Multirate dispatch: use planSubsteps() for interleaved cross-block
        // time ordering.  Each substep advances one block by one dt_sub using
        // advanceFromWork(), which does NOT reset from committed state.
        auto plan = aux_scheduler_->planSubsteps(time, dt);

        // Track per-block x_prev buffers for advanceFromWork().
        // x_prev starts as committed state for the first substep of each block.
        std::unordered_map<std::string, std::vector<Real>> block_x_prev;

        for (const auto& ss : plan) {
            // Find the entry for this block.
            DeployedAuxEntry* ep = nullptr;
            for (auto& entry : deployed_aux_entries_) {
                if (entry.instance_name == ss.block_name &&
                    entry.spec.solve_mode == AuxiliarySolveMode::Partitioned &&
                    entry.stepper && entry.deriv_provider) {
                    ep = &entry;
                    break;
                }
            }
            if (!ep) continue;

            auto& blk = auxiliary_state_manager_->getBlock(ep->instance_name);
            auto params = buildParamVector(*ep);
            auto bound_inputs = buildInputVector(*ep);
            const auto n_entities = blk.entityCount();
            const auto& emap = ep->entity_map;

            // Detect entity-local inputs (same as standard path).
            bool has_entity_local = false;
            if (auxiliary_input_registry_) {
                for (const auto& [mn, rn] : ep->input_bindings) {
                    if (auxiliary_input_registry_->hasInput(rn) &&
                        auxiliary_input_registry_->isEntityLocal(rn)) {
                        has_entity_local = true;
                        break;
                    }
                }
            }

            for (std::size_t e = 0; e < n_entities; ++e) {
                auto ew = blk.gatherEntityWork(e);
                const auto orig_e = emap.empty() ? e : emap[e];

                // Initialize x_prev from committed on first substep.
                auto key = ep->instance_name + "_" + std::to_string(e);
                auto it = block_x_prev.find(key);
                if (it == block_x_prev.end()) {
                    auto ec = blk.gatherEntityCommitted(e);
                    block_x_prev[key] = std::vector<Real>(ec.begin(), ec.end());
                    std::copy(block_x_prev[key].begin(), block_x_prev[key].end(), ew.begin());
                }
                auto& x_prev = block_x_prev[key];

                // Rebuild inputs per entity when entity-local.
                if (has_entity_local && auxiliary_input_registry_) {
                    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(ep->model.get())) {
                        bound_inputs.clear();
                        for (const auto& inp : built->signature().inputs) {
                            auto bi = ep->input_bindings.find(inp.name);
                            if (bi != ep->input_bindings.end()) {
                                auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                                bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                            } else {
                                bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                            }
                        }
                    } else {
                        rebuildGenericInputsForEntity(*ep, orig_e, bound_inputs);
                    }
                }

                // Build history spans (same as standard path).
                std::vector<std::vector<Real>> hd;
                std::vector<std::span<const Real>> hs;
                for (std::size_t k = 0; k < blk.history().depth(); ++k) {
                    hd.push_back(blk.gatherEntityHistory(k, e));
                    hs.push_back(hd.back());
                }

                ep->stepper->advanceFromWork(
                    *ep->model, *ep->deriv_provider,
                    ew, x_prev,
                    hs, bound_inputs, params,
                    ss.t_start, ss.dt_sub, e);

                std::copy(ew.begin(), ew.end(), x_prev.begin());
                blk.scatterEntityWork(e, ew);
            }
        }
    } else {
        // Standard dispatch: each partitioned block advances once for the
        // full dt.  The stepper's substep_count handles Subcycled scheduling.
        for (auto& entry : deployed_aux_entries_) {
            if (entry.spec.solve_mode != AuxiliarySolveMode::Partitioned) continue;
            if (!entry.stepper || !entry.deriv_provider) continue;
            advanceOneEntry(entry, time, dt, entry.stepper_spec.substep_count);
        }
    }
}

void FESystem::advanceOneEntry(DeployedAuxEntry& entry, Real time, Real dt, int substep_count)
{
    auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);

    auto params = buildParamVector(entry);
    auto bound_inputs = buildInputVector(entry);

    const auto n_entities = blk.entityCount();
    const auto& emap = entry.entity_map;

    bool has_entity_local_inputs = false;
    if (auxiliary_input_registry_) {
        for (const auto& [mn, rn] : entry.input_bindings) {
            if (auxiliary_input_registry_->hasInput(rn) && auxiliary_input_registry_->isEntityLocal(rn)) {
                has_entity_local_inputs = true;
                break;
            }
        }
    }

    for (std::size_t e = 0; e < n_entities; ++e) {
        auto ew = blk.gatherEntityWork(e);
        auto ec = blk.gatherEntityCommitted(e);
        const auto orig_e = emap.empty() ? e : emap[e];

        if (has_entity_local_inputs && auxiliary_input_registry_) {
            bound_inputs.clear();
            if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                for (const auto& inp : built->signature().inputs) {
                    auto bi = entry.input_bindings.find(inp.name);
                    if (bi != entry.input_bindings.end()) {
                        auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                        bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                    } else {
                        bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                    }
                }
            } else {
                rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
            }
        }

        std::vector<std::vector<Real>> hd;
        std::vector<std::span<const Real>> hs;
        for (std::size_t k = 0; k < blk.history().depth(); ++k) {
            hd.push_back(blk.gatherEntityHistory(k, e));
            hs.push_back(hd.back());
        }

        entry.stepper->advance(*entry.model, *entry.deriv_provider,
                                ew, ec, hs, bound_inputs, params,
                                time, dt, substep_count, e);
        blk.scatterEntityWork(e, ew);
    }
}

std::vector<Real> FESystem::buildParamVector(const DeployedAuxEntry& entry) const
{
    std::vector<Real> params;
    auto declared_params = entry.model->declaredParameterNames();
    if (!declared_params.empty()) {
        params.resize(declared_params.size(), 0.0);
        for (std::size_t i = 0; i < declared_params.size(); ++i) {
            auto it = entry.param_values.find(declared_params[i]);
            if (it != entry.param_values.end()) params[i] = it->second;
        }
    } else if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
        const auto& sig = built->signature();
        params.resize(sig.parameters.size(), 0.0);
        for (std::size_t i = 0; i < sig.parameters.size(); ++i) {
            auto it = entry.param_values.find(sig.parameters[i].name);
            if (it != entry.param_values.end()) params[i] = it->second;
        }
    } else {
        for (const auto& [pname, pval] : entry.param_values)
            params.push_back(pval);
    }
    return params;
}

std::vector<Real> FESystem::buildInputVector(const DeployedAuxEntry& entry) const
{
    std::vector<Real> bound_inputs;
    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
        const auto& sig = built->signature();
        std::size_t total = 0;
        for (const auto& inp : sig.inputs) total += static_cast<std::size_t>(inp.size);
        bound_inputs.resize(total, 0.0);
        std::size_t off = 0;
        for (const auto& inp : sig.inputs) {
            auto bi = entry.input_bindings.find(inp.name);
            if (bi != entry.input_bindings.end() && auxiliary_input_registry_) {
                auto v = auxiliary_input_registry_->valuesOf(bi->second);
                for (std::size_t k = 0; k < std::min(v.size(), static_cast<std::size_t>(inp.size)); ++k)
                    bound_inputs[off + k] = v[k];
            }
            off += static_cast<std::size_t>(inp.size);
        }
    } else {
        auto decl = entry.model->declaredInputNames();
        if (!decl.empty() && auxiliary_input_registry_) {
            for (const auto& raw : decl) {
                auto [iname, input_size] = parseDeclaredInputName(raw);
                auto bi = entry.input_bindings.find(iname);
                if (bi != entry.input_bindings.end() && auxiliary_input_registry_->hasInput(bi->second)) {
                    auto v = auxiliary_input_registry_->valuesOf(bi->second);
                    for (int k = 0; k < input_size; ++k) {
                        bound_inputs.push_back(
                            k < static_cast<int>(v.size()) ? v[static_cast<std::size_t>(k)] : 0.0);
                    }
                } else {
                    bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(input_size), 0.0);
                }
            }
        } else if (!entry.input_bindings.empty() && auxiliary_input_registry_) {
            for (const auto& [mn, rn] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(rn)) {
                    auto v = auxiliary_input_registry_->valuesOf(rn);
                    bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                }
            }
        }
    }
    return bound_inputs;
}

std::pair<std::string, int> FESystem::parseDeclaredInputName(const std::string& raw)
{
    FE_THROW_IF(raw.empty(), InvalidArgumentException,
                "Declared input name is empty");

    auto colon = raw.find(':');
    if (colon == std::string::npos)
        return {raw, 1};

    auto base = raw.substr(0, colon);
    FE_THROW_IF(base.empty(), InvalidArgumentException,
                "Declared input name '" + raw +
                "': base name before ':' must not be empty");

    auto size_str = raw.substr(colon + 1);
    int sz = 0;
    std::size_t pos = 0;
    try {
        sz = std::stoi(size_str, &pos);
    } catch (const std::exception&) {
        FE_THROW(InvalidArgumentException,
                 "Declared input name '" + raw +
                 "': suffix after ':' must be a positive integer, got '" +
                 size_str + "'");
    }
    FE_THROW_IF(pos != size_str.size(), InvalidArgumentException,
                "Declared input name '" + raw +
                "': suffix after ':' must be a positive integer, got '" +
                size_str + "' (trailing characters)");
    FE_THROW_IF(sz < 1, InvalidArgumentException,
                "Declared input name '" + raw +
                "': size must be >= 1, got " + std::to_string(sz));
    return {base, sz};
}

void FESystem::validateDeclaredInputNames(const AuxiliaryStateModel& model)
{
    for (const auto& raw : model.declaredInputNames()) {
        parseDeclaredInputName(raw); // throws on malformed suffix
    }
}

void FESystem::rebuildGenericInputsForEntity(
    const DeployedAuxEntry& entry, std::size_t entity_index,
    std::vector<Real>& out) const
{
    out.clear();
    auto decl = entry.model->declaredInputNames();
    if (!decl.empty() && auxiliary_input_registry_) {
        for (const auto& raw : decl) {
            auto [iname, input_size] = parseDeclaredInputName(raw);
            auto bi = entry.input_bindings.find(iname);
            if (bi != entry.input_bindings.end() && auxiliary_input_registry_->hasInput(bi->second)) {
                auto v = auxiliary_input_registry_->valuesOf(bi->second, entity_index);
                for (int k = 0; k < input_size; ++k) {
                    out.push_back(
                        k < static_cast<int>(v.size()) ? v[static_cast<std::size_t>(k)] : 0.0);
                }
            } else {
                out.resize(out.size() + static_cast<std::size_t>(input_size), 0.0);
            }
        }
    } else {
        for (const auto& [mn, rn] : entry.input_bindings) {
            if (auxiliary_input_registry_ && auxiliary_input_registry_->hasInput(rn)) {
                auto v = auxiliary_input_registry_->valuesOf(rn, entity_index);
                out.insert(out.end(), v.begin(), v.end());
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  FE-coupled auxiliary input providers
// ---------------------------------------------------------------------------

void FESystem::wireFECoupledInputProviders()
{
    // No-op: FE-coupled input providers are registered by the caller
    // before finalization via registerSampledFieldInput() etc.
}

void FESystem::registerSampledFieldInput(
    const std::string& input_name,
    const std::string& field_name,
    std::size_t n_entities)
{
    auto& reg = auxiliaryInputRegistry();

    // Look up the field.  Requires setup() to have been called.
    const FieldId fid = field_registry_.findByName(field_name);
    FE_THROW_IF(fid == INVALID_FIELD_ID, InvalidArgumentException,
                "registerSampledFieldInput: unknown field '" + field_name + "'");
    const auto fidx_check = static_cast<std::size_t>(fid);
    FE_THROW_IF(fidx_check >= field_dof_handlers_.size(), InvalidStateException,
                "registerSampledFieldInput: must be called after setup() "
                "so field DOF handlers are available");
    {
        const auto* emap = field_dof_handlers_[fidx_check].getEntityDofMap();
        FE_THROW_IF(!emap || emap->numVertices() == 0, InvalidStateException,
                    "registerSampledFieldInput: field '" + field_name +
                    "' has no entity DOF map");
        const auto test_dofs = emap->getVertexDofs(0);
        FE_THROW_IF(test_dofs.empty(), InvalidArgumentException,
                    "registerSampledFieldInput: field '" + field_name +
                    "' has no vertex DOFs (requires vertex-based Lagrange space)");
    }
    const int components = field_registry_.get(fid).components;

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = components;
    spec.entity_count = n_entities;
    spec.producer = AuxiliaryInputProducer::SampledStateField;
    spec.field_stage = AuxiliaryFieldStage::CurrentIterate;
    spec.source_field_name = field_name;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::EachNonlinearIteration;

    const auto field_idx = static_cast<std::size_t>(fid);
    const auto cap_comp = components;
    reg.registerEntityInput(spec,
        [this, field_idx, cap_comp]
        (Real /*t*/, Real /*dt*/, std::size_t entity_id, std::span<Real> out) {
            // Use per-field DOF handler and field-specific offset, matching
            // the logic in evaluateFieldAtVertices().
            std::fill(out.begin(), out.end(), 0.0);
            if (field_idx >= field_dof_handlers_.size()) return;

            const auto* emap = field_dof_handlers_[field_idx].getEntityDofMap();
            if (!emap) return;

            auto dofs = emap->getVertexDofs(static_cast<GlobalIndex>(entity_id));
            const GlobalIndex offset = (field_idx < field_dof_offsets_.size())
                ? field_dof_offsets_[field_idx] : 0;

            // Read from backend vector if available (MPI), else from cached span.
            for (int c = 0; c < cap_comp && c < static_cast<int>(out.size()); ++c) {
                if (c < static_cast<int>(dofs.size())) {
                    const GlobalIndex d = dofs[static_cast<std::size_t>(c)] + offset;
                    if (cached_solution_vector_) {
                        // MPI/distributed path: use backend vector for global access.
                        auto* vec = const_cast<backends::GenericVector*>(cached_solution_vector_);
                        auto view = vec->createAssemblyView();
                        out[static_cast<std::size_t>(c)] = view->getVectorEntry(d);
                    } else if (static_cast<std::size_t>(d) < cached_solution_u_.size()) {
                        out[static_cast<std::size_t>(c)] = cached_solution_u_[static_cast<std::size_t>(d)];
                    }
                }
            }
        });
}

void FESystem::registerBoundaryNodalSumInput(
    const std::string& input_name,
    const std::string& field_name,
    int boundary_marker)
{
    auto& reg = auxiliaryInputRegistry();

    const FieldId fid = field_registry_.findByName(field_name);
    FE_THROW_IF(fid == INVALID_FIELD_ID, InvalidArgumentException,
                "registerBoundaryNodalSumInput: unknown field '" + field_name + "'");

    // Validate vertex-DOF precondition: this helper requires setup() to have
    // been called (so DOF handlers are built) and the field to have vertex DOFs.
    const auto fidx = static_cast<std::size_t>(fid);
    FE_THROW_IF(fidx >= field_dof_handlers_.size(), InvalidStateException,
                "registerBoundaryNodalSumInput: must be called after setup() "
                "so field DOF handlers are available");
    const auto* emap = field_dof_handlers_[fidx].getEntityDofMap();
    FE_THROW_IF(!emap || emap->numVertices() == 0, InvalidStateException,
                "registerBoundaryNodalSumInput: field '" + field_name +
                "' has no entity DOF map (setup may not have completed)");
    {
        const auto test_dofs = emap->getVertexDofs(0);
        FE_THROW_IF(test_dofs.empty(), InvalidArgumentException,
                    "registerBoundaryNodalSumInput: field '" + field_name +
                    "' has no vertex DOFs (requires vertex-based Lagrange space)");
    }

    const int components = field_registry_.get(fid).components;

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = std::max(1, components);
    spec.producer = AuxiliaryInputProducer::SampledBoundaryReduction;
    spec.field_stage = AuxiliaryFieldStage::CurrentIterate;
    spec.boundary_marker = boundary_marker;
    spec.source_field_name = field_name;

    const auto field_idx = static_cast<std::size_t>(fid);
    const auto cap_marker = boundary_marker;
    reg.registerInput(spec,
        [this, field_idx, cap_marker]
        (Real /*t*/, Real /*dt*/, std::span<Real> out) {
            // Boundary-face nodal reduction: sum all field DOF components
            // at unique boundary face vertices.
            //
            // This is a nodal sum (not a quadrature-weighted boundary
            // integral).  For a true boundary integral, use
            // BoundaryFunctional + the assembly pipeline instead.
            // The output size equals the number of field components.
            const auto ncomp = static_cast<std::size_t>(
                field_registry_.get(static_cast<FieldId>(field_idx)).components);
            std::fill(out.begin(), out.end(), 0.0);
            if (!mesh_access_ || field_idx >= field_dof_handlers_.size()) return;

            const auto* emap = field_dof_handlers_[field_idx].getEntityDofMap();
            if (!emap) return;

            const GlobalIndex fld_offset = (field_idx < field_dof_offsets_.size())
                ? field_dof_offsets_[field_idx] : 0;

            std::unique_ptr<assembly::GlobalSystemView> solution_view;
            if (cached_solution_vector_) {
                auto* vec = const_cast<backends::GenericVector*>(cached_solution_vector_);
                solution_view = vec->createAssemblyView();
            }

            // Face-vertex maps for supported element types.
            static const std::vector<std::vector<int>> tet_faces =
                {{1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}};
            static const std::vector<std::vector<int>> tri_faces =
                {{0,1}, {1,2}, {2,0}};
            static const std::vector<std::vector<int>> hex_faces =
                {{0,3,2,1}, {4,5,6,7}, {0,1,5,4},
                 {1,2,6,5}, {2,3,7,6}, {3,0,4,7}};
            static const std::vector<std::vector<int>> quad_faces =
                {{0,1}, {1,2}, {2,3}, {3,0}};

            auto getFaceMap = [](ElementType et) -> const std::vector<std::vector<int>>* {
                if (et == ElementType::Tetra4) return &tet_faces;
                if (et == ElementType::Triangle3) return &tri_faces;
                if (et == ElementType::Hex8) return &hex_faces;
                if (et == ElementType::Quad4) return &quad_faces;
                return nullptr;
            };

            std::unordered_set<GlobalIndex> visited;
            mesh_access_->forEachBoundaryFace(cap_marker,
                [&](GlobalIndex face_id, GlobalIndex cell_id) {
                    const auto local_face = mesh_access_->getLocalFaceIndex(face_id, cell_id);
                    std::vector<GlobalIndex> cell_nodes;
                    mesh_access_->getCellNodes(cell_id, cell_nodes);

                    const auto* fmap = getFaceMap(mesh_access_->getCellType(cell_id));
                    if (!fmap || local_face < 0 ||
                        static_cast<std::size_t>(local_face) >= fmap->size()) {
                        return; // Skip unsupported element types.
                    }

                    const auto& local_ids = (*fmap)[static_cast<std::size_t>(local_face)];
                    for (int li : local_ids) {
                        if (static_cast<std::size_t>(li) >= cell_nodes.size()) continue;
                        const auto node_id = cell_nodes[static_cast<std::size_t>(li)];
                        if (!visited.insert(node_id).second) continue;

                        auto dofs = emap->getVertexDofs(node_id);
                        for (std::size_t c = 0; c < std::min(ncomp, dofs.size()); ++c) {
                            const GlobalIndex d = dofs[c] + fld_offset;
                            Real val = 0.0;
                            if (solution_view) {
                                val = solution_view->getVectorEntry(d);
                            } else if (static_cast<std::size_t>(d) < cached_solution_u_.size()) {
                                val = cached_solution_u_[static_cast<std::size_t>(d)];
                            }
                            if (c < out.size()) out[c] += val;
                        }
                    }
                });
        });
}

// ---------------------------------------------------------------------------
//  Boundary reduction service
// ---------------------------------------------------------------------------

BoundaryReductionService& FESystem::boundaryReductionService(FieldId primary_field)
{
    auto& svc = boundary_reduction_services_[primary_field];
    if (!svc) {
        svc = std::make_unique<BoundaryReductionService>(*this, primary_field);
    }
    return *svc;
}

// ---------------------------------------------------------------------------
//  registerBoundaryIntegralInput
// ---------------------------------------------------------------------------

namespace {


} // namespace

void FESystem::registerBoundaryIntegralInput(
    const std::string& input_name,
    forms::BoundaryFunctional functional,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "registerBoundaryIntegralInput: empty input_name");
    FE_THROW_IF(!functional.integrand.isValid(), InvalidArgumentException,
                "registerBoundaryIntegralInput: invalid integrand");
    FE_THROW_IF(functional.boundary_marker < 0, InvalidArgumentException,
                "registerBoundaryIntegralInput: boundary_marker must be >= 0");

    // The functional's name defaults to the input_name if not set.
    if (functional.name.empty()) {
        functional.name = input_name;
    }

    // Determine the primary field by scanning the integrand for field references.
    std::vector<FieldId> referenced_fields;
    if (const auto* root = functional.integrand.node()) {
        gatherFieldIds(*root, referenced_fields);
    }

    // Multi-field integrands are supported via secondary field bindings.
    // The primary field provides the DOF layout and mesh context; secondary
    // fields contribute solution data through the functional assembler's
    // field binding mechanism.

    FieldId primary_fid = INVALID_FIELD_ID;
    if (!referenced_fields.empty()) {
        primary_fid = referenced_fields.front();
    } else {
        // No field references in integrand (e.g., constant or geometry-only
        // integrand like ∫_Γ 1 ds).  The integrand doesn't depend on DOFs,
        // but quadrature requires a function space.  Use GEOMETRY_FIELD_ID
        // as a logical sentinel — resolved to the first registered field's
        // space for quadrature rule selection only.
        primary_fid = GEOMETRY_FIELD_ID;
    }

    // Resolve GEOMETRY_FIELD_ID: prefer the first registered field (for DOF
    // access in field-dependent code paths), but allow GEOMETRY_FIELD_ID to
    // pass through when no fields exist (geometry-only evaluation with a
    // default P1 space).
    if (primary_fid == GEOMETRY_FIELD_ID) {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) {
            // Use first field for richer DOF access; the integrand doesn't
            // reference it, so only the quadrature rule matters.
            primary_fid = static_cast<FieldId>(0);
        }
        // else: keep GEOMETRY_FIELD_ID — BoundaryReductionService will
        // create a default P1 space from the mesh element type.
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "registerBoundaryIntegralInput('" + input_name +
                    "'): internal error — could not resolve primary field");

    // Register the functional with the per-field boundary reduction service.
    auto& svc = boundaryReductionService(primary_fid);
    svc.addBoundaryFunctional(functional);

    // Bind secondary fields and set dof_per_node for multi-field evaluation.
    bindSecondaryFields(svc, primary_fid, referenced_fields);

    // Register the input in the AuxiliaryInputRegistry with a callback
    // that evaluates the functional via the BoundaryReductionService.
    auto& reg = auxiliaryInputRegistry();

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;  // boundary integrals are scalar
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.update_schedule = schedule;
    spec.boundary_marker = functional.boundary_marker;
    spec.requires_mpi_reduction = true;  // MPI reduction is handled inside the service

    const auto func_name = functional.name;
    const auto captured_fid = primary_fid;

    reg.registerInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::span<Real> out) {
            // Build a SystemStateView from the full cached system state.
            // cacheSystemState() is called by prepareAuxiliaryForAssembly(),
            // advanceAuxiliaryState(SystemStateView), and
            // assembleMixedAuxiliaryIntoGlobal() before evaluate() is invoked.
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.u_prev2 = cached_solution_u_prev2_;
            state.u_prev2_vector = cached_solution_prev2_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctional(func_name, state);
            } else {
                out[0] = 0.0;
            }
        });
}

void FESystem::registerBoundaryIntegralInput(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    forms::BoundaryFunctional::Reduction reduction,
    AuxiliaryInputUpdateSchedule schedule)
{
    forms::BoundaryFunctional functional;
    functional.name = input_name;
    functional.integrand = std::move(integrand);
    functional.boundary_marker = boundary_marker;
    functional.reduction = reduction;

    registerBoundaryIntegralInput(input_name, std::move(functional), schedule);
}

// ---------------------------------------------------------------------------
//  Mixed monolithic assembly into global system
// ---------------------------------------------------------------------------

void FESystem::assembleMixedAuxiliaryIntoGlobal(
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out,
    bool want_matrix, bool want_vector,
    std::size_t n_field_dofs,
    bool is_nonlinear_iteration)
{
    if (!auxiliary_state_manager_ || !auxiliary_operator_registry_) return;
    if (!auxiliary_operator_registry_->isLayoutFinalized()) return;

    const auto mixed = auxiliary_operator_registry_->composeMixedLayout(n_field_dofs);

    // Cache the full system state for FE-coupled input callbacks.
    cacheSystemState(state);

    // Evaluate inputs with nonlinear-iteration flag.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(state.time, state.dt, is_nonlinear_iteration);
    }

    // For each monolithic auxiliary block, assemble its per-entity
    // contributions into the global matrix/vector at the auxiliary DOF offsets.
    // This matches the standalone assembleMonolithicAuxiliary() logic for
    // entity-local inputs, xdot computation, and input refresh.
    for (auto& entry : deployed_aux_entries_) {
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
        if (!auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        const int dim = entry.spec.size;
        const auto n_entities = blk.entityCount();

        // Find this block's offset in the mixed layout.
        std::size_t block_offset = 0;
        for (const auto& bl : mixed.aux_layout.blocks) {
            if (bl.name == entry.instance_name) {
                block_offset = bl.offset + mixed.aux_layout.mixed_system_offset;
                break;
            }
        }

        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);

        // Detect entity-local inputs (same as standalone monolithic path).
        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [mn, rn] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(rn) &&
                    auxiliary_input_registry_->isEntityLocal(rn)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        const auto& emap = entry.entity_map;

        for (std::size_t e = 0; e < n_entities; ++e) {
            auto entity_x = blk.gatherEntityWork(e);
            auto entity_committed = blk.gatherEntityCommitted(e);
            const auto orig_e = emap.empty() ? e : emap[e];

            // Rebuild inputs per entity when entity-local bindings exist.
            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : built->signature().inputs) {
                        auto bi = entry.input_bindings.find(inp.name);
                        if (bi != entry.input_bindings.end()) {
                            auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                        } else {
                            bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            // BDF1 xdot.
            std::vector<Real> xdot(static_cast<std::size_t>(dim));
            if (state.dt > 0.0) {
                for (int i = 0; i < dim; ++i)
                    xdot[static_cast<std::size_t>(i)] =
                        (entity_x[static_cast<std::size_t>(i)] -
                         entity_committed[static_cast<std::size_t>(i)]) / state.dt;
            }

            // Populate field_values when the model directly references FE fields.
            // Needed for both residual evaluation and Jacobian evaluation via
            // PointEvaluator, which encounters DiscreteField/StateField terminals.
            std::vector<FieldValueEntry> field_vals;
            if (entry.deriv_provider) {
                const auto& art = entry.deriv_provider->artifact();
                if (!art.referenced_fields.empty()) {
                    field_vals.reserve(art.referenced_fields.size());
                    for (const auto fid : art.referenced_fields) {
                        const auto fidx = static_cast<std::size_t>(fid);
                        if (fidx >= field_dof_offsets_.size() ||
                            fidx >= field_dof_handlers_.size()) continue;
                        const auto fld_off = field_dof_offsets_[fidx];
                        const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                        if (!femap) continue;
                        auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                        if (!vdofs.empty()) {
                            FieldValueEntry fve;
                            fve.field = fid;
                            fve.n_components = static_cast<int>(vdofs.size());
                            for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                                const auto gidx = static_cast<std::size_t>(vdofs[static_cast<std::size_t>(c)] + fld_off);
                                fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : 0.0;
                            }
                            field_vals.push_back(fve);
                        }
                    }
                }
            }

            AuxiliaryLocalContext ctx;
            ctx.time = state.time; ctx.dt = state.dt; ctx.effective_dt = state.dt;
            ctx.x = entity_x; ctx.xdot = xdot;
            ctx.inputs = bound_inputs; ctx.params = params;
            ctx.entity_index = e;
            ctx.field_values = field_vals;

            // Build global DOF indices for this entity's auxiliary unknowns.
            std::vector<GlobalIndex> aux_dofs(static_cast<std::size_t>(dim));
            for (int i = 0; i < dim; ++i) {
                aux_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(
                    block_offset + e * static_cast<std::size_t>(dim) +
                    static_cast<std::size_t>(i));
            }

            // Residual.
            if (want_vector && vector_out) {
                std::vector<Real> entity_res(static_cast<std::size_t>(dim));
                AuxiliaryResidualRequest res_req;
                res_req.residual = entity_res;
                entry.model->evaluateResidual(ctx, res_req);
                vector_out->addVectorEntries(aux_dofs, entity_res);
            }

            // Jacobian (aux-aux self-coupling block).
            if (want_matrix && matrix_out && entry.deriv_provider) {
                const auto n_inp = static_cast<int>(entry.input_bindings.size());
                std::vector<Real> entity_jac(static_cast<std::size_t>(dim * dim));
                std::vector<Real> entity_dFdi(static_cast<std::size_t>(dim * n_inp));

                AuxiliaryJacobianRequest jac_req;
                jac_req.dF_dx = entity_jac;
                jac_req.n = dim;
                // Request dF/dinputs for chain-rule coupling.
                if (n_inp > 0 && !entry.coupled_bindings.empty()) {
                    jac_req.dF_dinputs = entity_dFdi;
                    jac_req.n_inputs = n_inp;
                }
                entry.deriv_provider->evaluateJacobian(*entry.model, ctx, jac_req);
                matrix_out->addMatrixEntries(aux_dofs, aux_dofs, entity_jac);

                // Chain-rule coupling: dF/du = dF/dI * dI/du.
                // For each coupled binding, compute the field-auxiliary
                // Jacobian block and insert it into the global matrix.
                if (n_inp > 0 && !entity_dFdi.empty()) {
                    int input_col = 0;
                    for (const auto& [model_input, reg_input] : entry.input_bindings) {
                        auto cb_it = entry.coupled_bindings.find(model_input);
                        if (cb_it != entry.coupled_bindings.end()) {
                            const auto& handle = cb_it->second;
                            if (handle.hasDefinition() &&
                                handle.supportsMonolithicLinearization()) {
                                // For sampled fields, dI/du is identity at sampled DOFs.
                                // For boundary integrals, dI/du comes from the
                                // BoundaryReductionService gradient assembly.
                                //
                                // For now, sampled-field chain rule is implemented:
                                // dF/du_j = dF/dI_k * delta(k, DOF_j)
                                // = dF/dI column for the k-th input, scattered to field DOFs.
                                if (handle.kind() == FEQuantityKind::SampledField) {
                                    const auto& ref_fields = handle.referencedFields();
                                    if (!ref_fields.empty()) {
                                        const auto fid = ref_fields[0];
                                        const auto fidx = static_cast<std::size_t>(fid);
                                        if (fidx < field_dof_offsets_.size() &&
                                            fidx < field_dof_handlers_.size()) {
                                            const auto fld_off = field_dof_offsets_[fidx];
                                            const auto* emap = field_dof_handlers_[fidx].getEntityDofMap();
                                            if (emap) {
                                                // dI/du for sampled field = identity at vertex DOFs.
                                                // Use the actual DOF map for vertex e.
                                                auto vertex_dofs = emap->getVertexDofs(
                                                    static_cast<GlobalIndex>(e));
                                                // Extract dF/dI column for this input.
                                                std::vector<Real> col(static_cast<std::size_t>(dim));
                                                for (int r = 0; r < dim; ++r) {
                                                    col[static_cast<std::size_t>(r)] =
                                                        entity_dFdi[static_cast<std::size_t>(
                                                            r * n_inp + input_col)];
                                                }
                                                // Each vertex DOF gets a column of dF/dI.
                                                for (const auto local_dof : vertex_dofs) {
                                                    const auto global_dof = static_cast<GlobalIndex>(
                                                        local_dof + fld_off);
                                                    std::vector<GlobalIndex> fd = {global_dof};
                                                    matrix_out->addMatrixEntries(aux_dofs, fd, col);
                                                }
                                            }
                                        }
                                    }
                                }
                                else if (handle.kind() == FEQuantityKind::BoundaryIntegral ||
                                         handle.kind() == FEQuantityKind::BoundaryAverage ||
                                         handle.kind() == FEQuantityKind::DomainIntegral ||
                                         handle.kind() == FEQuantityKind::DomainAverage ||
                                         handle.kind() == FEQuantityKind::RegionIntegral ||
                                         handle.kind() == FEQuantityKind::RegionAverage ||
                                         handle.kind() == FEQuantityKind::FEExpression) {
                                    // Integral dI/du via symbolic gradient assembly.
                                    // For average kinds (DomainAverage, RegionAverage,
                                    // BoundaryAverage), the public handle name is a
                                    // derived callback over __integral and __measure.
                                    // Use the __integral name for gradient lookup.
                                    // For DomainAverage/RegionAverage, the service only
                                    // knows about the __integral sub-functional.
                                    // BoundaryAverage is registered directly as a
                                    // BoundaryFunctional with Reduction::Average, so
                                    // its gradient is already correct without __integral.
                                    std::string func_name = handle.registryName();
                                    const bool is_domain_region_avg =
                                        handle.kind() == FEQuantityKind::DomainAverage ||
                                        handle.kind() == FEQuantityKind::RegionAverage;
                                    if (is_domain_region_avg) {
                                        func_name = handle.registryName() + "__integral";
                                    }

                                    const auto& ref_fields = handle.referencedFields();
                                    if (!ref_fields.empty()) {
                                        const auto svc_fid = ref_fields[0];
                                        auto svc_it = boundary_reduction_services_.find(svc_fid);
                                        if (svc_it != boundary_reduction_services_.end() && svc_it->second) {
                                            for (const auto target_fid : ref_fields) {
                                                auto grad = svc_it->second->evaluateFunctionalGradient(
                                                    func_name, target_fid, state);

                                                // For averages, apply quotient rule:
                                                // d(I/M)/du = (dI/du)/M  (measure M is constant w.r.t. u
                                                // for geometry-independent integrands; for u-dependent
                                                // measure, the full quotient rule would be needed).
                                                if (is_domain_region_avg && auxiliary_input_registry_) {
                                                    const std::string meas_name =
                                                        handle.registryName() + "__measure";
                                                    if (auxiliary_input_registry_->hasInput(meas_name)) {
                                                        const Real measure =
                                                            auxiliary_input_registry_->get(meas_name);
                                                        if (measure > 0.0) {
                                                            for (auto& se : grad) se.value /= measure;
                                                        }
                                                    }
                                                }

                                                for (const auto& se : grad) {
                                                    std::vector<Real> col(static_cast<std::size_t>(dim));
                                                    for (int r = 0; r < dim; ++r) {
                                                        col[static_cast<std::size_t>(r)] =
                                                            entity_dFdi[static_cast<std::size_t>(
                                                                r * n_inp + input_col)];
                                                    }
                                                    for (auto& c : col) c *= se.value;
                                                    std::vector<GlobalIndex> field_dof = {se.dof};
                                                    matrix_out->addMatrixEntries(aux_dofs, field_dof, col);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ++input_col;
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // ----------------------------------------------------------------
    // Direct field-derivative block: dF_aux/du from direct FE field
    // references in auxiliary residual expressions (not mediated through
    // AuxiliaryInputRef).  This handles models that directly reference
    // DiscreteField/StateField nodes in their expressions.
    //
    // For node-scoped models with Lagrange elements, the Kronecker
    // delta property gives φ_j(vertex_i) = δ_ij, so the contribution
    // at entity e is simply dF/d(field_value) scattered to vertex e's DOF.
    // The derivative expression may itself depend on the field value
    // (nonlinear case), so we populate field_values in the context.
    // ----------------------------------------------------------------
    if (want_matrix && matrix_out) {
        for (auto& entry : deployed_aux_entries_) {
            if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
            if (!entry.deriv_provider) continue;

            const auto& art = entry.deriv_provider->artifact();
            if (!art.valid || art.referenced_fields.empty()) continue;

            const int dim = entry.model->dimension();
            if (dim == 0) continue;

            if (!auxiliary_state_manager_ ||
                !auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

            auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
            const auto n_ent = blk.entityCount();

            // Find this block's offset in the mixed layout.
            std::size_t block_offset = 0;
            for (const auto& bl : mixed.aux_layout.blocks) {
                if (bl.name == entry.instance_name) {
                    block_offset = bl.offset + mixed.aux_layout.mixed_system_offset;
                    break;
                }
            }

            auto params = buildParamVector(entry);
            auto bound_inputs = buildInputVector(entry);

            // Entity-local input handling (same as chain-rule path).
            bool has_entity_local_inputs = false;
            if (auxiliary_input_registry_) {
                for (const auto& [mn, rn] : entry.input_bindings) {
                    if (auxiliary_input_registry_->hasInput(rn) &&
                        auxiliary_input_registry_->isEntityLocal(rn)) {
                        has_entity_local_inputs = true;
                        break;
                    }
                }
            }

            const auto& ent_map = entry.entity_map;

            for (std::size_t e = 0; e < n_ent; ++e) {
                auto entity_x = blk.gatherEntityWork(e);
                auto entity_committed = blk.gatherEntityCommitted(e);
                const auto orig_e = ent_map.empty() ? e : ent_map[e];

                // Rebuild inputs per entity when entity-local bindings exist.
                if (has_entity_local_inputs && auxiliary_input_registry_) {
                    bound_inputs.clear();
                    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                        for (const auto& inp : built->signature().inputs) {
                            auto bi = entry.input_bindings.find(inp.name);
                            if (bi != entry.input_bindings.end()) {
                                auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                                bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                            } else {
                                bound_inputs.resize(bound_inputs.size() +
                                    static_cast<std::size_t>(inp.size), 0.0);
                            }
                        }
                    } else {
                        rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                    }
                }

                // BDF1 xdot.
                std::vector<Real> xdot(static_cast<std::size_t>(dim));
                if (state.dt > 0.0) {
                    for (int i = 0; i < dim; ++i)
                        xdot[static_cast<std::size_t>(i)] =
                            (entity_x[static_cast<std::size_t>(i)] -
                             entity_committed[static_cast<std::size_t>(i)]) / state.dt;
                }

                // Build field_values from the global solution for this entity.
                // For vertex-based Lagrange elements, the field value at vertex
                // orig_e is simply the DOF coefficients (Kronecker delta property).
                std::vector<FieldValueEntry> field_vals;
                field_vals.reserve(art.referenced_fields.size());
                for (const auto fid : art.referenced_fields) {
                    const auto fidx = static_cast<std::size_t>(fid);
                    if (fidx >= field_dof_offsets_.size() ||
                        fidx >= field_dof_handlers_.size()) continue;
                    const auto fld_off = field_dof_offsets_[fidx];
                    const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                    if (!femap) continue;
                    auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                    if (!vdofs.empty()) {
                        FieldValueEntry fve;
                        fve.field = fid;
                        fve.n_components = static_cast<int>(vdofs.size());
                        for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                            const auto gidx = static_cast<std::size_t>(vdofs[static_cast<std::size_t>(c)] + fld_off);
                            fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : 0.0;
                        }
                        field_vals.push_back(fve);
                    }
                }

                AuxiliaryLocalContext ctx;
                ctx.time = state.time; ctx.dt = state.dt; ctx.effective_dt = state.dt;
                ctx.x = entity_x; ctx.xdot = xdot;
                ctx.inputs = bound_inputs; ctx.params = params;
                ctx.entity_index = e;
                ctx.field_values = field_vals;

                // Build global DOF indices for this entity's auxiliary unknowns.
                std::vector<GlobalIndex> aux_dofs(static_cast<std::size_t>(dim));
                for (int i = 0; i < dim; ++i) {
                    aux_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(
                        block_offset + e * static_cast<std::size_t>(dim) +
                        static_cast<std::size_t>(i));
                }

                // For each referenced FE field, evaluate dF/d(field_comp) and
                // scatter to per-component vertex DOFs.
                //
                // evaluateFieldDerivative returns n_rows * n_comp values,
                // row-major: [row * nc + comp].  Each vertex DOF c at vertex
                // orig_e gets the column dF_i/d(field_comp_c).
                for (const auto fid : art.referenced_fields) {
                    auto dF_dfield = entry.deriv_provider->evaluateFieldDerivative(fid, ctx);
                    if (dF_dfield.empty()) continue;

                    const auto fidx = static_cast<std::size_t>(fid);
                    if (fidx >= field_dof_offsets_.size() ||
                        fidx >= field_dof_handlers_.size()) continue;

                    const auto fld_off = field_dof_offsets_[fidx];
                    const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                    if (!femap) continue;

                    auto vertex_dofs = femap->getVertexDofs(
                        static_cast<GlobalIndex>(orig_e));
                    const auto nc = static_cast<int>(vertex_dofs.size());

                    for (int c = 0; c < nc; ++c) {
                        const auto global_dof = static_cast<GlobalIndex>(
                            vertex_dofs[static_cast<std::size_t>(c)] + fld_off);
                        std::vector<GlobalIndex> col = {global_dof};
                        std::vector<Real> col_vals(static_cast<std::size_t>(dim));
                        for (int i = 0; i < dim; ++i) {
                            const auto idx = static_cast<std::size_t>(i * nc + c);
                            col_vals[static_cast<std::size_t>(i)] =
                                (idx < dF_dfield.size()) ? dF_dfield[idx] : 0.0;
                        }
                        matrix_out->addMatrixEntries(aux_dofs, col, col_vals);
                    }
                }
            }
        }
    }

    // Transpose Jacobian block: dR_PDE/dx_aux.
    //
    // When PDE forms reference AuxiliaryOutput nodes, the PDE residual
    // depends on auxiliary state through the output expressions.
    // Chain rule: dR_PDE/dx_j = Σ_k (dR_PDE/d(output_k)) * (d(output_k)/dx_j)
    //
    // dR_PDE/d(output_k): computed by FD perturbation of the output value
    //   in the assembler context and re-assembling the PDE residual.
    // d(output_k)/dx_j: computed by FD perturbation of the auxiliary state
    //   and re-evaluating the output expressions.
    // ----------------------------------------------------------------
    if (want_matrix && matrix_out) {
        for (auto& entry : deployed_aux_entries_) {
            if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
            const auto n_outputs = static_cast<int>(entry.model->outputCount());
            const int dim = entry.model->dimension();
            if (n_outputs == 0 || dim == 0) continue;

            // Find this block's offset in the mixed system.
            std::size_t block_offset = n_field_dofs;
            for (const auto& e2 : deployed_aux_entries_) {
                if (&e2 == &entry) break;
                if (e2.spec.solve_mode == AuxiliarySolveMode::Monolithic) {
                    block_offset += static_cast<std::size_t>(e2.model->dimension());
                }
            }

            // Get current output values and auxiliary state.
            if (!auxiliary_state_manager_ ||
                !auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

            auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
            auto entity_x = blk.work();

            // Build context for output evaluation.
            std::vector<Real> params = buildParamVector(entry);
            std::vector<Real> inputs = buildInputVector(entry);
            std::vector<Real> xdot(static_cast<std::size_t>(dim), 0.0);
            if (state.dt > 0.0) {
                auto committed = blk.committed();
                for (int i = 0; i < dim; ++i)
                    xdot[static_cast<std::size_t>(i)] =
                        (entity_x[static_cast<std::size_t>(i)] -
                         committed[static_cast<std::size_t>(i)]) / state.dt;
            }

            AuxiliaryLocalContext ctx;
            ctx.time = state.time; ctx.dt = state.dt; ctx.effective_dt = state.dt;
            ctx.x = entity_x; ctx.xdot = xdot;
            ctx.inputs = inputs; ctx.params = params;

            // Compute base outputs.
            std::vector<Real> base_outputs(static_cast<std::size_t>(n_outputs));
            entry.model->evaluateOutputs(ctx, base_outputs);

            // Compute d(output_k)/dx_j symbolically.
            // For BuiltAuxiliaryModel, output expressions are FormExpr trees
            // with AuxiliaryStateRef(slot) terminals that can be differentiated
            // using the PointEvaluator.  For custom models, fall back to FD.
            const Real eps = 1e-7;
            std::vector<Real> dO_dx(static_cast<std::size_t>(n_outputs * dim), 0.0);

            // Use symbolic d(output)/d(state) from the derivative provider
            // if available.  This avoids any FD on the auxiliary model.
            if (entry.deriv_provider) {
                const auto& art = entry.deriv_provider->artifact();
                if (art.valid && !art.dOutput_dx_exprs.empty() &&
                    art.n_outputs == n_outputs) {
                    // Evaluate symbolic derivative expressions.
                    forms::PointEvalContext pctx;
                    pctx.time = ctx.time;
                    pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
                    pctx.coupled_aux = ctx.x;
                    pctx.auxiliary_inputs = ctx.inputs;
                    pctx.jit_constants = ctx.params;

                    for (int k = 0; k < n_outputs; ++k) {
                        for (int j = 0; j < dim; ++j) {
                            const auto idx = static_cast<std::size_t>(k * dim + j);
                            if (idx < art.dOutput_dx_exprs.size()) {
                                dO_dx[idx] = forms::evaluateScalarAt(
                                    art.dOutput_dx_exprs[idx], pctx);
                            }
                        }
                    }
                } else {
                    // FD fallback for custom models without symbolic output derivatives.
                    std::vector<Real> x_pert(entity_x.begin(), entity_x.end());
                    AuxiliaryLocalContext pert_ctx = ctx;
                    pert_ctx.x = x_pert;
                    std::vector<Real> pert_outputs(static_cast<std::size_t>(n_outputs));

                    for (int j = 0; j < dim; ++j) {
                        const Real orig = x_pert[static_cast<std::size_t>(j)];
                        x_pert[static_cast<std::size_t>(j)] = orig + eps;
                        entry.model->evaluateOutputs(pert_ctx, pert_outputs);
                        x_pert[static_cast<std::size_t>(j)] = orig;

                        for (int k = 0; k < n_outputs; ++k) {
                            dO_dx[static_cast<std::size_t>(k * dim + j)] =
                                (pert_outputs[static_cast<std::size_t>(k)] -
                                 base_outputs[static_cast<std::size_t>(k)]) / eps;
                        }
                    }
                }
            }

            // Compute dR_PDE/d(output_k) symbolically by differentiating
            // the PDE form w.r.t. each AuxiliaryOutputRef(slot).
            //
            // Scan the formulation records for AuxiliaryOutputRef nodes
            // matching this entry's output slots, differentiate, and
            // assemble the derivative linear form.
            const auto output_names = entry.model->outputNames();

            // Build auxiliary DOF indices.
            std::vector<GlobalIndex> aux_dofs(static_cast<std::size_t>(dim));
            for (int j = 0; j < dim; ++j) {
                aux_dofs[static_cast<std::size_t>(j)] = static_cast<GlobalIndex>(
                    block_offset + static_cast<std::size_t>(j));
            }

            for (int k = 0; k < n_outputs; ++k) {
                // Check if any dO_dx is nonzero for this output.
                bool has_sensitivity = false;
                for (int j = 0; j < dim; ++j) {
                    if (std::abs(dO_dx[static_cast<std::size_t>(k * dim + j)]) > 1e-14) {
                        has_sensitivity = true;
                        break;
                    }
                }
                if (!has_sensitivity) continue;

                // Find the output slot in the auxiliary output buffer.
                const auto qualified = entry.instance_name + "/" + output_names[static_cast<std::size_t>(k)];
                const auto slot = auxiliaryOutputSlotOf(qualified);
                if (slot == static_cast<std::size_t>(-1)) continue;
                const auto slot32 = static_cast<std::uint32_t>(slot);

                // For each PDE formulation record that references this output,
                // symbolically differentiate and assemble dR/d(output_k).
                for (const auto& frec : formulation_records_) {
                    if (!frec.residual_expr) continue;
                    const auto residual = forms::FormExpr(
                        std::const_pointer_cast<forms::FormExprNode>(frec.residual_expr));

                    // Check if this form references the output slot.
                    bool references_slot = false;
                    std::function<void(const forms::FormExprNode&)> scan_refs =
                        [&](const forms::FormExprNode& n) {
                            if (n.type() == forms::FormExprType::AuxiliaryOutputRef) {
                                const auto s = n.slotIndex();
                                if (s && *s == slot32) references_slot = true;
                            }
                            for (const auto* c : n.children()) {
                                if (c && !references_slot) scan_refs(*c);
                            }
                        };
                    scan_refs(*frec.residual_expr);
                    if (!references_slot) continue;

                    // Symbolically differentiate: dR/d(output_k).
                    // The result is a linear form where AuxiliaryOutputRef(slot)
                    // has been replaced by constant(1.0) via the Kronecker delta rule.
                    auto dR_dOk = forms::differentiateWrtAuxiliaryOutput(residual, slot32);
                    if (!dR_dOk.isValid()) continue;

                    // Compile the derivative linear form into an assembly kernel.
                    const auto n_total = static_cast<GlobalIndex>(dof_handler_.getNumDofs());
                    if (n_total <= 0 || !assembler_) continue;

                    try {
                        forms::FormCompiler compiler;
                        auto ir = compiler.compileLinear(dR_dOk);
                        forms::FormKernel deriv_kernel(std::move(ir));

                        // Assemble the derivative linear form into a vector.
                        // The GradAccumulator collects per-DOF contributions.
                        struct VecAccum final : public assembly::GlobalSystemView {
                            std::unordered_map<GlobalIndex, Real> entries;
                            GlobalIndex sz;
                            explicit VecAccum(GlobalIndex s) : sz(s) {}
                            void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
                            void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
                            void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
                            void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
                            void setDiagonal(GlobalIndex, Real) override {}
                            void zeroRows(std::span<const GlobalIndex>, bool) override {}
                            void addVectorEntries(std::span<const GlobalIndex> d, std::span<const Real> v, assembly::AddMode) override {
                                for (std::size_t i = 0; i < d.size(); ++i) {
                                    if (d[i] >= 0 && d[i] < sz) entries[d[i]] += v[i];
                                }
                            }
                            void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
                                if (d >= 0 && d < sz) entries[d] += v;
                            }
                            void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
                            void zeroVectorEntries(std::span<const GlobalIndex> d) override { for (auto x : d) entries.erase(x); }
                            [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
                                auto it = entries.find(d); return it != entries.end() ? it->second : 0.0;
                            }
                            void beginAssemblyPhase() override {}
                            void endAssemblyPhase() override {}
                            void finalizeAssembly() override {}
                            [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return assembly::AssemblyPhase::Building; }
                            [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
                            [[nodiscard]] bool hasVector() const noexcept override { return true; }
                            [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
                            [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
                            [[nodiscard]] std::string backendName() const override { return "VecAccum"; }
                            void zero() override { entries.clear(); }
                        };

                        VecAccum dR_vec(n_total);

                        // Disable constraints for raw derivative assembly.
                        assembler_->setConstraints(nullptr);

                        // Set solution for state-dependent derivative forms.
                        std::unique_ptr<assembly::GlobalSystemView> sol_view;
                        if (state.u_vector) {
                            auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
                            sol_view = vec->createAssemblyView();
                            assembler_->setCurrentSolutionView(sol_view.get());
                        }

                        // Assemble on appropriate domains.
                        if (deriv_kernel.hasCell()) {
                            // Use first active field's space for assembly.
                            if (!frec.active_fields.empty()) {
                                const auto& rec = fieldRecord(frec.active_fields[0]);
                                if (rec.space) {
                                    const auto foff = fieldDofOffset(frec.active_fields[0]);
                                    const auto& fdh = fieldDofHandler(frec.active_fields[0]);
                                    assembler_->setRowDofMap(fdh.getDofMap(), foff);
                                    assembler_->setColDofMap(fdh.getDofMap(), foff);
                                    assembler_->assembleVector(
                                        meshAccess(), *rec.space, deriv_kernel, dR_vec);
                                }
                            }
                        }
                        if (deriv_kernel.hasBoundaryFace()) {
                            if (!frec.active_fields.empty()) {
                                const auto& rec = fieldRecord(frec.active_fields[0]);
                                if (rec.space) {
                                    const auto foff = fieldDofOffset(frec.active_fields[0]);
                                    const auto& fdh = fieldDofHandler(frec.active_fields[0]);
                                    assembler_->setRowDofMap(fdh.getDofMap(), foff);
                                    assembler_->setColDofMap(fdh.getDofMap(), foff);
                                    // Assemble on all boundary markers used in the form.
                                    // The kernel's computeBoundaryFace filters by marker.
                                    const auto& mesh = meshAccess();
                                    for (int m = 0; m < 256; ++m) {
                                        assembler_->assembleBoundaryFaces(
                                            mesh, m, *rec.space, deriv_kernel,
                                            nullptr, &dR_vec);
                                    }
                                }
                            }
                        }

                        // Compose: dR_i/dx_j = dR_i/d(output_k) * d(output_k)/dx_j.
                        for (const auto& [dof_i, dRi_dOk] : dR_vec.entries) {
                            if (std::abs(dRi_dOk) < 1e-14) continue;

                            for (int j = 0; j < dim; ++j) {
                                const Real dOk_dxj = dO_dx[static_cast<std::size_t>(k * dim + j)];
                                if (std::abs(dOk_dxj) < 1e-14) continue;

                                const Real val = dRi_dOk * dOk_dxj;
                                std::vector<GlobalIndex> row = {dof_i};
                                std::vector<GlobalIndex> col = {aux_dofs[static_cast<std::size_t>(j)]};
                                std::vector<Real> mat = {val};
                                matrix_out->addMatrixEntries(row, col, mat);
                            }
                        }
                    } catch (const std::exception& ex) {
                        // If symbolic compilation fails (e.g., unsupported
                        // expression structure), skip this contribution.
                        // The derivative is still correct for the other terms.
                        (void)ex;
                    }
                }
            }
        }
    }

    // Assemble registered AuxiliaryOperator contributions.
    if (auxiliary_operator_registry_) {
        for (const auto& op_name : auxiliary_operator_registry_->operatorNames()) {
            const auto& op = auxiliary_operator_registry_->getOperator(op_name);
            if (!op.residual_fn && !op.jacobian_fn) continue;

            AuxiliaryOperatorContext op_ctx;
            op_ctx.time = state.time;
            op_ctx.dt = state.dt;

            // Helper to resolve an operator endpoint (source or target)
            // to data span, offset, and DOF count in the mixed system.
            // scratch_buf is per-endpoint to avoid overwriting when both
            // source and target are field references in the distributed case.
            auto resolveEndpoint = [&](const std::string& name,
                                       std::vector<Real>& scratch_buf,
                                       std::span<const Real>& data_out,
                                       std::size_t& entity_count_out,
                                       int& stride_out,
                                       std::size_t& offset_out,
                                       std::size_t& n_out) {
                // Check auxiliary block first.
                if (auxiliary_state_manager_->hasBlock(name)) {
                    auto& blk = auxiliary_state_manager_->getBlock(name);
                    data_out = blk.work();
                    entity_count_out = blk.entityCount();
                    stride_out = blk.componentStride();
                    for (const auto& bl : mixed.aux_layout.blocks) {
                        if (bl.name == name) {
                            offset_out = bl.offset + mixed.aux_layout.mixed_system_offset;
                            n_out = bl.n_unknowns;
                            return;
                        }
                    }
                }
                // Check if it's a field reference (possibly "field:name" syntax).
                std::string field_name = name;
                if (name.substr(0, 6) == "field:") {
                    field_name = name.substr(6);
                }
                const FieldId fid = field_registry_.findByName(field_name);
                if (fid != INVALID_FIELD_ID) {
                    const auto fidx = static_cast<std::size_t>(fid);
                    const auto& rec = field_registry_.get(fid);
                    stride_out = std::max(1, rec.components);

                    // Field DOF offset and count in the global system.
                    const std::size_t fld_off = (fidx < field_dof_offsets_.size())
                        ? static_cast<std::size_t>(field_dof_offsets_[fidx]) : 0;
                    offset_out = fld_off;
                    if (fidx < field_dof_handlers_.size()) {
                        n_out = static_cast<std::size_t>(
                            field_dof_handlers_[fidx].getNumDofs());
                    } else {
                        n_out = 0;
                    }
                    // DOF-tuple count: number of DOF groups of size `stride`.
                    // For vertex-based Lagrange: equals num vertices.
                    // For higher-order: equals total DOFs / components.
                    entity_count_out = (stride_out > 0) ? n_out / static_cast<std::size_t>(stride_out) : 0;

                    // Provide a field-local view into the solution vector.
                    if (!cached_solution_u_.empty() && fld_off + n_out <= cached_solution_u_.size()) {
                        data_out = cached_solution_u_.subspan(fld_off, n_out);
                    } else if (cached_solution_vector_ && n_out > 0) {
                        // Distributed case: materialize field DOFs from
                        // the backend vector into the per-endpoint scratch.
                        auto* vec = const_cast<backends::GenericVector*>(cached_solution_vector_);
                        auto view = vec->createAssemblyView();
                        scratch_buf.resize(n_out);
                        for (std::size_t i = 0; i < n_out; ++i) {
                            scratch_buf[i] = view->getVectorEntry(
                                static_cast<GlobalIndex>(fld_off + i));
                        }
                        data_out = scratch_buf;
                    } else {
                        data_out = {};
                    }
                }
            };

            std::size_t src_offset = 0, src_n = 0;
            {
                std::span<const Real> src_data;
                std::size_t src_ec = 0;
                int src_s = 0;
                resolveEndpoint(op.source_name, field_endpoint_scratch_src_,
                                src_data, src_ec, src_s,
                                src_offset, src_n);
                op_ctx.source_data = src_data;
                op_ctx.source_entity_count = src_ec;
                op_ctx.source_stride = src_s;
            }

            std::size_t tgt_offset = 0, tgt_n = 0;
            {
                std::span<const Real> tgt_data;
                std::size_t tgt_ec = 0;
                int tgt_s = 0;
                resolveEndpoint(op.target_name, field_endpoint_scratch_tgt_,
                                tgt_data, tgt_ec, tgt_s,
                                tgt_offset, tgt_n);
                op_ctx.target_data = tgt_data;
                op_ctx.target_entity_count = tgt_ec;
                op_ctx.target_stride = tgt_s;
            }

            // Residual contribution.
            if (want_vector && vector_out && op.residual_fn && tgt_n > 0) {
                std::vector<Real> op_res(tgt_n);
                op.residual_fn(op_ctx, op_res);
                std::vector<GlobalIndex> tgt_dofs(tgt_n);
                for (std::size_t i = 0; i < tgt_n; ++i)
                    tgt_dofs[i] = static_cast<GlobalIndex>(tgt_offset + i);
                vector_out->addVectorEntries(tgt_dofs, op_res);
            }

            // Jacobian contribution.
            if (want_matrix && matrix_out && op.jacobian_fn && tgt_n > 0 && src_n > 0) {
                std::vector<Real> op_jac(tgt_n * src_n);
                op.jacobian_fn(op_ctx, op_jac);
                std::vector<GlobalIndex> tgt_dofs(tgt_n), src_dofs(src_n);
                for (std::size_t i = 0; i < tgt_n; ++i)
                    tgt_dofs[i] = static_cast<GlobalIndex>(tgt_offset + i);
                for (std::size_t i = 0; i < src_n; ++i)
                    src_dofs[i] = static_cast<GlobalIndex>(src_offset + i);
                matrix_out->addMatrixEntries(tgt_dofs, src_dofs, op_jac);
            }
        }
    }
}

void FESystem::assembleMonolithicAuxiliary(
    Real time, Real dt,
    std::span<Real> residual_out,
    std::span<Real> jacobian_out,
    bool is_nonlinear_iteration)
{
    if (!auxiliary_state_manager_ || !auxiliary_operator_registry_) return;

    const auto& layout = auxiliary_operator_registry_->auxiliaryLayout();
    const auto n_total = layout.total_aux_unknowns;
    FE_THROW_IF(residual_out.size() < n_total, InvalidArgumentException,
                "assembleMonolithicAuxiliary: residual buffer too small");
    FE_THROW_IF(jacobian_out.size() < n_total * n_total, InvalidArgumentException,
                "assembleMonolithicAuxiliary: Jacobian buffer too small");

    std::fill(residual_out.begin(), residual_out.begin() + static_cast<std::ptrdiff_t>(n_total), 0.0);
    std::fill(jacobian_out.begin(), jacobian_out.begin() + static_cast<std::ptrdiff_t>(n_total * n_total), 0.0);

    // Ensure auxiliary inputs are evaluated for the current step.
    // Pass is_nonlinear_iteration so EachNonlinearIteration inputs refresh.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(time, dt, is_nonlinear_iteration);
    }

    // Assemble contributions from each monolithic deployed block.
    for (auto& entry : deployed_aux_entries_) {
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
        if (!auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        const int dim = entry.spec.size;
        const auto n_entities = blk.entityCount();

        // Find this block's offset in the mixed layout.
        std::size_t block_offset = 0;
        for (const auto& bl : layout.blocks) {
            if (bl.name == entry.instance_name) {
                block_offset = bl.offset;
                break;
            }
        }

        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);

        // Detect entity-local inputs (same logic as partitioned path).
        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [mn, rn] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(rn) &&
                    auxiliary_input_registry_->isEntityLocal(rn)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        const auto& emap = entry.entity_map;

        // Per-entity assembly.
        for (std::size_t e = 0; e < n_entities; ++e) {
            auto entity_x = blk.gatherEntityWork(e);
            auto entity_committed = blk.gatherEntityCommitted(e);
            const auto row_base = block_offset + e * static_cast<std::size_t>(dim);
            const auto orig_e = emap.empty() ? e : emap[e];

            // Rebuild inputs per entity when entity-local bindings exist.
            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : built->signature().inputs) {
                        auto bi = entry.input_bindings.find(inp.name);
                        if (bi != entry.input_bindings.end()) {
                            auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                        } else {
                            bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            // Compute xdot via BDF1: xdot ≈ (x_work - x_committed) / dt.
            // NOTE: This is a standalone BDF1 approximation, not coupled to
            // the FE time integrator.  Full monolithic integration with the
            // FE linear solver would use the time integrator's own xdot.
            std::vector<Real> xdot(static_cast<std::size_t>(dim));
            if (dt > 0.0) {
                for (int i = 0; i < dim; ++i) {
                    xdot[static_cast<std::size_t>(i)] =
                        (entity_x[static_cast<std::size_t>(i)] -
                         entity_committed[static_cast<std::size_t>(i)]) / dt;
                }
            }

            AuxiliaryLocalContext ctx;
            ctx.time = time;
            ctx.dt = dt;
            ctx.effective_dt = dt;
            ctx.x = entity_x;
            ctx.xdot = xdot;
            ctx.inputs = bound_inputs;
            ctx.params = params;
            ctx.entity_index = e;

            std::vector<Real> entity_res(static_cast<std::size_t>(dim));
            AuxiliaryResidualRequest res_req;
            res_req.residual = entity_res;
            entry.model->evaluateResidual(ctx, res_req);

            for (int i = 0; i < dim; ++i) {
                residual_out[row_base + static_cast<std::size_t>(i)] += entity_res[static_cast<std::size_t>(i)];
            }

            // Evaluate Jacobian (if derivative provider available).
            if (entry.deriv_provider) {
                std::vector<Real> entity_jac(static_cast<std::size_t>(dim * dim));
                AuxiliaryJacobianRequest jac_req;
                jac_req.dF_dx = entity_jac;
                jac_req.n = dim;
                entry.deriv_provider->evaluateJacobian(*entry.model, ctx, jac_req);

                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        const auto gi = row_base + static_cast<std::size_t>(i);
                        const auto gj = row_base + static_cast<std::size_t>(j);
                        jacobian_out[gi * n_total + gj] +=
                            entity_jac[static_cast<std::size_t>(i * dim + j)];
                    }
                }
            }
        }
    }
}

MixedSystemLayout FESystem::composeMixedSystemLayout(std::size_t n_field_unknowns) const
{
    if (auxiliary_operator_registry_ && auxiliary_operator_registry_->isLayoutFinalized()) {
        return auxiliary_operator_registry_->composeMixedLayout(n_field_unknowns);
    }
    MixedSystemLayout layout;
    layout.n_field_unknowns = n_field_unknowns;
    layout.total_unknowns = n_field_unknowns;
    return layout;
}

void FESystem::rollbackAuxiliaryState()
{
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->rollbackAll();
    }
}

void FESystem::finalizeAuxiliaryLayout()
{
    // Materialize deployed instances into blocks, steppers, and derivative providers.
    for (auto& entry : deployed_aux_entries_) {
        auto& mgr = auxiliaryStateManager();

        // Determine entity count: prefer explicit, then mesh-derived, then 1.
        std::size_t entity_count = entry.explicit_entity_count;
        if (entity_count == 0) {
            switch (entry.spec.scope) {
                case AuxiliaryStateScope::Global:
                    entity_count = 1;
                    break;
                case AuxiliaryStateScope::Node:
                    FE_THROW(InvalidStateException,
                             "FESystem::finalizeAuxiliaryLayout: Node scope requires "
                             "explicit .entityCount() — IMeshAccess does not expose "
                             "vertex count");
                    break;
                case AuxiliaryStateScope::Cell:
                    if (mesh_access_) {
                        entity_count = static_cast<std::size_t>(mesh_access_->numOwnedCells());
                    } else {
                        entity_count = 1;
                    }
                    break;
                case AuxiliaryStateScope::Boundary:
                    entity_count = 1;
                    break;
                case AuxiliaryStateScope::Facet:
                    if (mesh_access_) {
                        entity_count = static_cast<std::size_t>(mesh_access_->numBoundaryFaces());
                    } else {
                        entity_count = 1;
                    }
                    break;
                case AuxiliaryStateScope::QuadraturePoint:
                    FE_THROW(InvalidStateException,
                             "FESystem::finalizeAuxiliaryLayout: QuadraturePoint scope "
                             "requires explicit .entityCount()");
                    break;
            }
        }

        // Region-to-entity expansion.
        // If the deployment region restricts to a subset, build an entity map
        // and adjust entity_count to the restricted set size.
        const auto& region = entry.spec.deployment_region;
        if (region.isRestricted()) {
            if (!region.explicit_entities.empty()) {
                // Explicit entity set: use directly.
                entry.entity_map = region.explicit_entities;
            } else if (mesh_access_) {
                // Marker-based region: expand against mesh topology.
                switch (region.kind) {
                    case AuxiliaryRegionKind::CellSet:
                    case AuxiliaryRegionKind::MaterialIdSet: {
                        // Parse identity as integer domain/material ID.
                        int target_id = 0;
                        try { target_id = std::stoi(region.identity); }
                        catch (...) {
                            FE_THROW(InvalidArgumentException,
                                     "FESystem::finalizeAuxiliaryLayout: CellSet/"
                                     "MaterialIdSet identity must be an integer, got '"
                                     + region.identity + "'");
                        }
                        mesh_access_->forEachOwnedCell([&](GlobalIndex cell_id) {
                            if (mesh_access_->getCellDomainId(cell_id) == target_id) {
                                entry.entity_map.push_back(
                                    static_cast<std::size_t>(cell_id));
                            }
                        });
                        break;
                    }
                    case AuxiliaryRegionKind::BoundarySet: {
                        int marker = 0;
                        try { marker = std::stoi(region.identity); }
                        catch (...) {
                            FE_THROW(InvalidArgumentException,
                                     "FESystem::finalizeAuxiliaryLayout: BoundarySet "
                                     "identity must be an integer marker, got '"
                                     + region.identity + "'");
                        }
                        mesh_access_->forEachBoundaryFace(marker,
                            [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
                                entry.entity_map.push_back(
                                    static_cast<std::size_t>(face_id));
                            });
                        break;
                    }
                    case AuxiliaryRegionKind::InterfaceSet: {
                        int marker = 0;
                        try { marker = std::stoi(region.identity); }
                        catch (...) {
                            FE_THROW(InvalidArgumentException,
                                     "FESystem::finalizeAuxiliaryLayout: InterfaceSet "
                                     "identity must be an integer marker, got '"
                                     + region.identity + "'");
                        }
                        // Collect ALL interior faces.  IMeshAccess does not
                        // expose per-face interface markers, so we cannot
                        // filter by the requested marker.  The identity is
                        // stored for restart/remap metadata only.
                        (void)marker;
                        mesh_access_->forEachInteriorFace(
                            [&](GlobalIndex face_id, GlobalIndex /*c0*/, GlobalIndex /*c1*/) {
                                entry.entity_map.push_back(
                                    static_cast<std::size_t>(face_id));
                            });
                        break;
                    }
                    default:
                        break;
                }
                FE_THROW_IF(entry.entity_map.empty() &&
                            region.kind != AuxiliaryRegionKind::WholeDomain,
                            InvalidStateException,
                            "FESystem::finalizeAuxiliaryLayout: marker-based region '"
                            + region.identity + "' expanded to 0 entities");
            } else {
                FE_THROW(InvalidStateException,
                         "FESystem::finalizeAuxiliaryLayout: deployment region "
                         "kind '" + region.identity + "' requires mesh access "
                         "for marker-based entity expansion, but no mesh was "
                         "provided to FESystem");
            }
            if (!entry.entity_map.empty()) {
                // The block storage size is the restricted entity count.
                entity_count = entry.entity_map.size();
            }
        }

        // Register the block.
        // Build initial values: if provided values match dim (not total),
        // replicate per entity to fill the full storage.
        // For ByComponentThenEntity ordering, transpose to component-major.
        std::vector<Real> full_init;
        if (!entry.initial_values.empty()) {
            const auto dim_sz = static_cast<std::size_t>(entry.spec.size);
            if (entry.initial_values.size() == dim_sz && entity_count > 1) {
                full_init.resize(entity_count * dim_sz);
                if (entry.spec.ordering == AuxiliaryEntityOrdering::ByComponentThenEntity) {
                    // Component-major: [comp0_e0, comp0_e1, ..., comp1_e0, comp1_e1, ...]
                    for (std::size_t c = 0; c < dim_sz; ++c) {
                        for (std::size_t e = 0; e < entity_count; ++e) {
                            full_init[c * entity_count + e] = entry.initial_values[c];
                        }
                    }
                } else {
                    // Entity-major (default): [e0_c0, e0_c1, ..., e1_c0, e1_c1, ...]
                    for (std::size_t e = 0; e < entity_count; ++e) {
                        std::copy(entry.initial_values.begin(),
                                  entry.initial_values.end(),
                                  full_init.begin() + static_cast<std::ptrdiff_t>(e * dim_sz));
                    }
                }
            } else {
                full_init = entry.initial_values;
            }
        }
        if (entry.spec.layout_mode == AuxiliaryLayoutMode::Ragged) {
            // Ragged layout is not supported through the FESystem deployment
            // API.  Both Partitioned stepping and Monolithic assembly assume
            // fixed per-entity dimension (spec.size).  Ragged blocks must be
            // registered directly via AuxiliaryStateManager::registerBlockRagged().
            FE_THROW(NotImplementedException,
                     "FESystem::finalizeAuxiliaryLayout: ragged layout for '"
                     + entry.instance_name + "' is not supported through the "
                     "deployment API.  The stepper and monolithic assembly "
                     "paths assume fixed per-entity dimension.  Use "
                     "AuxiliaryStateManager::registerBlockRagged() directly.");
        } else {
            mgr.registerBlock(entry.spec, entity_count,
                              full_init.empty()
                                  ? std::span<const Real>{}
                                  : std::span<const Real>(full_init));
        }

        // Create stepper and derivative provider for partitioned blocks.
        if (entry.spec.solve_mode == AuxiliarySolveMode::Partitioned) {
            entry.stepper = createStepper(entry.stepper_spec.method_name);
            entry.stepper->setup(entry.spec.size, entry.stepper_spec);

            entry.deriv_provider = std::make_unique<AuxiliaryDerivativeProvider>();
            entry.deriv_provider->setup(*entry.model, entry.spec.derivative_policy);
        }

        // Register monolithic unknowns with the operator registry.
        if (entry.spec.solve_mode == AuxiliarySolveMode::Monolithic) {
            auxiliaryOperatorRegistry().registerMonolithicUnknowns(
                entry.instance_name, entity_count,
                entry.spec.size, entry.spec.scope);

            // Create derivative provider for monolithic blocks too.
            entry.deriv_provider = std::make_unique<AuxiliaryDerivativeProvider>();
            entry.deriv_provider->setup(*entry.model, entry.spec.derivative_policy);
        }

        // Validate direct FE field references in auxiliary residual expressions.
        if (entry.deriv_provider) {
            const auto& art = entry.deriv_provider->artifact();
            if (!art.referenced_fields.empty()) {
                // Reject non-Node scopes.  Direct DiscreteField/StateField nodes
                // are only meaningful for Node-scoped models, where the Kronecker
                // delta property of Lagrange elements gives exact field values.
                if (entry.spec.scope != AuxiliaryStateScope::Node) {
                    const char* scope_name = "unknown";
                    switch (entry.spec.scope) {
                        case AuxiliaryStateScope::Global: scope_name = "Global"; break;
                        case AuxiliaryStateScope::Boundary: scope_name = "Boundary"; break;
                        case AuxiliaryStateScope::Cell: scope_name = "Cell"; break;
                        case AuxiliaryStateScope::QuadraturePoint: scope_name = "QuadraturePoint"; break;
                        case AuxiliaryStateScope::Facet: scope_name = "Facet"; break;
                        case AuxiliaryStateScope::Node: break; // unreachable
                    }
                    FE_THROW(InvalidArgumentException,
                        "FESystem::finalizeAuxiliaryLayout: " + std::string(scope_name)
                        + "-scoped auxiliary model '" + entry.instance_name
                        + "' directly references FE field(s) via DiscreteField/StateField "
                        "nodes.  Direct field references are only supported for Node-scoped "
                        "models (Lagrange Kronecker delta).  Use sampledField(), "
                        "boundaryIntegral(), domainAverage(), or feExpression() to mediate "
                        "field access, then bind via bindCoupled().");
                }

                // Validate that referenced fields have vertex DOFs with Lagrange
                // Kronecker delta semantics (H1/C0 spaces).  Scalar, vector,
                // and tensor fields are all supported; non-vertex spaces and
                // fields exceeding MAX_FIELD_VALUE_COMPONENTS are not.
                for (const auto fid : art.referenced_fields) {
                    if (!field_registry_.has(fid)) continue;
                    const auto& rec = field_registry_.get(fid);
                    if (rec.components > MAX_FIELD_VALUE_COMPONENTS) {
                        FE_THROW(InvalidArgumentException,
                            "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                            + entry.instance_name + "' references "
                            + std::to_string(rec.components) + "-component field '"
                            + rec.name + "' which exceeds MAX_FIELD_VALUE_COMPONENTS ("
                            + std::to_string(MAX_FIELD_VALUE_COMPONENTS) + ").");
                    }
                    // Require C0-continuous (nodal Lagrange) space for direct
                    // field references.  The Kronecker delta property (DOF
                    // coefficients equal pointwise vertex values) is only valid
                    // for C0 nodal Lagrange interpolation.  This includes both
                    // scalar H1 spaces and Product spaces built from H1 components
                    // (e.g., VectorSpace(H1, ...)).  L2, H(curl), H(div), C1,
                    // and other continuity types do not have this property.
                    if (rec.space) {
                        const auto ct = rec.space->continuity();
                        if (ct != Continuity::C0) {
                            FE_THROW(InvalidArgumentException,
                                "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                                + entry.instance_name + "' directly references field '"
                                + rec.name + "' which has non-C0 continuity.  Direct "
                                "DiscreteField/StateField references require C0 (nodal "
                                "Lagrange) spaces for the Kronecker delta property.  "
                                "Use sampledField() or feExpression() for L2 (DG), "
                                "H(div), H(curl), C1, or other space types.");
                        }
                    }
                    // Verify that the field's DOF handler has vertex DOFs.
                    // This is a defensive check: all C0 spaces in the current
                    // library are Lagrange and have vertex DOFs, but if a future
                    // C0 space (e.g., Bernstein, hierarchical) is added without
                    // nodal Kronecker semantics, this catches it at setup.
                    {
                        const auto fidx2 = static_cast<std::size_t>(fid);
                        if (fidx2 < field_dof_handlers_.size()) {
                            const auto* femap = field_dof_handlers_[fidx2].getEntityDofMap();
                            if (!femap) {
                                FE_THROW(InvalidArgumentException,
                                    "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                                    + entry.instance_name + "' directly references field '"
                                    + rec.name + "' which has no EntityDofMap.  Direct "
                                    "field references require vertex-based DOF mapping.");
                            }
                            auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(0));
                            if (vdofs.empty()) {
                                FE_THROW(InvalidArgumentException,
                                    "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                                    + entry.instance_name + "' directly references field '"
                                    + rec.name + "' which has no vertex DOFs.  Direct "
                                    "field references require nodal Lagrange spaces with "
                                    "vertex-associated DOFs (Kronecker delta property).  "
                                    "Use sampledField() or feExpression() instead.");
                            }
                        }
                    }
                }
            }
        }
    }

    if (auxiliary_operator_registry_ &&
        !auxiliary_operator_registry_->isLayoutFinalized()) {
        auxiliary_operator_registry_->finalizeLayout();
    }

    // Wire FE-coupled auxiliary input providers (SampledStateField, etc.)
    wireFECoupledInputProviders();

    // Build multirate scheduler from deployed block schedule modes.
    aux_scheduler_ = std::make_unique<AuxiliaryMultirateScheduler>();
    for (const auto& entry : deployed_aux_entries_) {
        if (entry.spec.solve_mode != AuxiliarySolveMode::Partitioned) continue;

        MultirateBlockSchedule sched;
        sched.block_name = entry.instance_name;

        switch (entry.spec.schedule_mode) {
            case AuxiliaryScheduleMode::SingleRate:
                sched.rate_ratio = 1;
                break;
            case AuxiliaryScheduleMode::Subcycled:
                sched.rate_ratio = entry.stepper_spec.substep_count;
                break;
            case AuxiliaryScheduleMode::Multirate:
                sched.rate_ratio = entry.stepper_spec.substep_count;
                break;
        }

        aux_scheduler_->addBlockSchedule(std::move(sched));
    }

    finalizeDeferredInputDeps();
}

void FESystem::assembleMixedAuxiliaryDense(
    const SystemStateView& state,
    std::size_t n_field_dofs,
    std::vector<Real>& residual_out,
    std::vector<Real>& matrix_out)
{
    // Compute total mixed size from the operator registry layout,
    // which accounts for entity counts (n_unknowns = entity_count * stride).
    std::size_t n_aux = 0;
    if (auxiliary_operator_registry_ && auxiliary_operator_registry_->isLayoutFinalized()) {
        n_aux = auxiliary_operator_registry_->auxiliaryLayout().total_aux_unknowns;
    } else {
        for (const auto& entry : deployed_aux_entries_) {
            if (entry.spec.solve_mode == AuxiliarySolveMode::Monolithic) {
                n_aux += static_cast<std::size_t>(entry.model->dimension());
            }
        }
    }
    const auto n_total = n_field_dofs + n_aux;
    residual_out.assign(n_total, 0.0);
    matrix_out.assign(n_total * n_total, 0.0);

    // Dense GlobalSystemView that stores matrix and vector entries.
    struct DenseAccum final : public assembly::GlobalSystemView {
        std::vector<Real>& vec;
        std::vector<Real>& mat;
        GlobalIndex n;
        DenseAccum(std::vector<Real>& v, std::vector<Real>& m, GlobalIndex sz)
            : vec(v), mat(m), n(sz) {}

        void addMatrixEntries(std::span<const GlobalIndex> rows,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            // Square single-DOF-set: rows = cols.
            const auto nd = static_cast<int>(rows.size());
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j) {
                    auto r = rows[static_cast<std::size_t>(i)];
                    auto c = rows[static_cast<std::size_t>(j)];
                    if (r >= 0 && r < n && c >= 0 && c < n)
                        mat[static_cast<std::size_t>(r * n + c)] +=
                            vals[static_cast<std::size_t>(i * nd + j)];
                }
        }
        void addMatrixEntries(std::span<const GlobalIndex> rows,
                              std::span<const GlobalIndex> cols,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            const auto nr = static_cast<int>(rows.size());
            const auto nc = static_cast<int>(cols.size());
            for (int i = 0; i < nr; ++i)
                for (int j = 0; j < nc; ++j) {
                    auto r = rows[static_cast<std::size_t>(i)];
                    auto c = cols[static_cast<std::size_t>(j)];
                    if (r >= 0 && r < n && c >= 0 && c < n)
                        mat[static_cast<std::size_t>(r * n + c)] +=
                            vals[static_cast<std::size_t>(i * nc + j)];
                }
        }
        void addMatrixEntry(GlobalIndex r, GlobalIndex c, Real v,
                            assembly::AddMode) override {
            if (r >= 0 && r < n && c >= 0 && c < n)
                mat[static_cast<std::size_t>(r * n + c)] += v;
        }
        void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}
        void addVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                auto d = dofs[i];
                if (d >= 0 && d < n) vec[static_cast<std::size_t>(d)] += vals[i];
            }
        }
        void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
            if (d >= 0 && d < n) vec[static_cast<std::size_t>(d)] += v;
        }
        void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void zeroVectorEntries(std::span<const GlobalIndex>) override {}
        [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
            return (d >= 0 && d < n) ? vec[static_cast<std::size_t>(d)] : 0.0;
        }
        void beginAssemblyPhase() override {}
        void endAssemblyPhase() override {}
        void finalizeAssembly() override {}
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
            return assembly::AssemblyPhase::Building;
        }
        [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return n; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return n; }
        [[nodiscard]] std::string backendName() const override { return "DenseAccum"; }
        void zero() override {
            std::fill(vec.begin(), vec.end(), 0.0);
            std::fill(mat.begin(), mat.end(), 0.0);
        }
    };

    DenseAccum accum(residual_out, matrix_out, static_cast<GlobalIndex>(n_total));
    assembleMixedAuxiliaryIntoGlobal(state, &accum, &accum,
                                      true, true, n_field_dofs, false);
}

void FESystem::finalizeDeferredInputDeps()
{
    // Resolve deferred derived-input expressions: AuxiliaryInputSymbol → AuxiliaryInputRef.
    // Safe to call multiple times — both vectors are cleared after first run.
    if (auxiliary_input_registry_ && !deferred_derived_exprs_.empty()) {
        for (auto& pair : deferred_derived_exprs_) {
            const auto& derived_name = pair.first;
            auto& expr_ptr = pair.second;
            auto* reg = auxiliary_input_registry_.get();
            auto resolve = [reg, &derived_name](const forms::FormExprNode& node)
                -> std::optional<forms::FormExpr> {
                if (node.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                    if (auto sym = node.symbolName()) {
                        const std::string sname{*sym};
                        FE_THROW_IF(!reg->hasInput(sname),
                                    InvalidArgumentException,
                                    "FESystem: derived input '" + derived_name +
                                        "' references unknown input '" + sname + "'");
                        const auto slot = reg->slotOf(sname);
                        return forms::FormExpr::auxiliaryInputRef(
                            static_cast<std::uint32_t>(slot));
                    }
                }
                return std::nullopt;
            };
            *expr_ptr = expr_ptr->transformNodes(resolve);
        }
        deferred_derived_exprs_.clear();
    }

    // Wire deferred input dependencies.
    if (auxiliary_input_registry_ && !deferred_input_deps_.empty()) {
        for (const auto& pair : deferred_input_deps_) {
            const auto& dependent = pair.first;
            const auto& dependency = pair.second;
            FE_THROW_IF(!auxiliary_input_registry_->hasInput(dependency),
                        InvalidArgumentException,
                        "FESystem: derived input '" + dependent +
                            "' references unknown input '" + dependency +
                            "' — ensure all referenced inputs are "
                            "registered before setup()");
            auxiliary_input_registry_->addDependency(dependent, dependency);
        }
        deferred_input_deps_.clear();
    }
}

void FESystem::bindSecondaryFields(BoundaryReductionService& svc,
                                    FieldId primary_fid,
                                    const std::vector<FieldId>& referenced_fields)
{
    if (referenced_fields.size() <= 1) return;  // no secondary fields

    // Compute total dof_per_node from all registered fields.
    // For interleaved layouts, each node stores components from all fields.
    int total_dpn = 0;
    for (const auto& rec : field_registry_.records()) {
        total_dpn += rec.components;
    }
    if (total_dpn > 0) {
        svc.setDofPerNode(total_dpn);
    }

    // Compute per-field component_offset in the interleaved layout.
    // Fields are ordered by FieldId (registration order).
    std::unordered_map<FieldId, int> field_offsets;
    int offset = 0;
    for (std::size_t i = 0; i < field_registry_.records().size(); ++i) {
        const auto fid = static_cast<FieldId>(i);
        field_offsets[fid] = offset;
        offset += field_registry_.records()[i].components;
    }

    for (const auto fid : referenced_fields) {
        if (fid == primary_fid) continue;
        const auto& sec_rec = field_registry_.get(fid);
        if (!sec_rec.space) continue;

        assembly::FieldSolutionBinding binding;
        binding.field = fid;
        binding.space = sec_rec.space.get();
        binding.field_type = sec_rec.space->field_type();
        binding.value_dimension = sec_rec.components;
        binding.n_components = sec_rec.components;
        auto off_it = field_offsets.find(fid);
        binding.component_offset = (off_it != field_offsets.end()) ? off_it->second : 0;
        svc.registerSecondaryField(binding);
    }
}

std::vector<BoundaryReductionService::SensitivityEntry>
FESystem::assembleBoundaryGradient(FieldId field,
                                    const forms::FormExpr& integrand_trial,
                                    int boundary_marker,
                                    const SystemStateView& state)
{
    const auto& rec = fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(),
                      "FESystem::assembleBoundaryGradient: field space is null");

    if (!assembler_) return {};

    const auto& fdh = fieldDofHandler(field);
    const auto field_off = fieldDofOffset(field);

    // Create the gradient kernel (forward-mode AD for exact ∂(integrand)/∂(trial_dof_j)).
    forms::BoundaryFunctionalGradientKernel grad_kernel(
        integrand_trial, boundary_marker);

    // Assemble using the StandardAssembler's boundary face pipeline with a
    // lightweight sparse vector accumulator (same pattern as SystemAssembly.cpp).
    const auto n_total = static_cast<GlobalIndex>(dof_handler_.getNumDofs());
    if (n_total <= 0) return {};

    // Lightweight GlobalSystemView that only accumulates vector entries.
    struct GradAccumulator final : public assembly::GlobalSystemView {
        std::unordered_map<GlobalIndex, Real> entries;
        GlobalIndex sz;
        explicit GradAccumulator(GlobalIndex s) : sz(s) {}

        // Matrix ops: no-op (we only need vector).
        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>,
                              assembly::AddMode) override {}
        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>,
                              std::span<const Real>, assembly::AddMode) override {}
        void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
        void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}

        // Vector ops.
        void addVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                if (dofs[i] >= 0 && dofs[i] < sz) entries[dofs[i]] += vals[i];
            }
        }
        void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
            if (d >= 0 && d < sz) entries[d] += v;
        }
        void setVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> vals) override {
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                if (dofs[i] >= 0 && dofs[i] < sz) entries[dofs[i]] = vals[i];
            }
        }
        void zeroVectorEntries(std::span<const GlobalIndex> dofs) override {
            for (const auto d : dofs) entries.erase(d);
        }
        [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
            auto it = entries.find(d);
            return (it != entries.end()) ? it->second : 0.0;
        }

        // Lifecycle ops.
        void beginAssemblyPhase() override {}
        void endAssemblyPhase() override {}
        void finalizeAssembly() override {}
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
            return assembly::AssemblyPhase::Building;
        }
        [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
        [[nodiscard]] std::string backendName() const override { return "GradAccumulator"; }
        void zero() override { entries.clear(); }
    };

    GradAccumulator accum(n_total);

    // Configure the assembler for this field.
    // Disable constraints for gradient assembly — we want the raw dI/du
    // without constraint redistribution.
    assembler_->setConstraints(nullptr);
    assembler_->setRowDofMap(fdh.getDofMap(), field_off);
    assembler_->setColDofMap(fdh.getDofMap(), field_off);

    // Set the solution on the assembler so the gradient kernel can access
    // field values.  Use the GlobalSystemView from the cached solution vector
    // if available, otherwise create a temporary local-span view.
    // The StandardAssembler requires a GlobalSystemView for solution access.
    // Create one from whichever solution source is available.
    struct SpanSolutionView final : public assembly::GlobalSystemView {
        std::span<const Real> data;
        GlobalIndex sz;
        SpanSolutionView(std::span<const Real> d, GlobalIndex s) : data(d), sz(s) {}

        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>,
                              assembly::AddMode) override {}
        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>,
                              std::span<const Real>, assembly::AddMode) override {}
        void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
        void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}
        void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>,
                              assembly::AddMode) override {}
        void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
        void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void zeroVectorEntries(std::span<const GlobalIndex>) override {}
        [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
            if (d >= 0 && static_cast<std::size_t>(d) < data.size()) return data[static_cast<std::size_t>(d)];
            return 0.0;
        }
        void beginAssemblyPhase() override {}
        void endAssemblyPhase() override {}
        void finalizeAssembly() override {}
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
            return assembly::AssemblyPhase::Building;
        }
        [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
        [[nodiscard]] std::string backendName() const override { return "SpanSolutionView"; }
        void zero() override {}
    };

    std::unique_ptr<assembly::GlobalSystemView> temp_sol_view;
    std::unique_ptr<SpanSolutionView> span_sol_view;
    if (state.u_vector) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        temp_sol_view = vec->createAssemblyView();
        assembler_->setCurrentSolutionView(temp_sol_view.get());
    } else if (!state.u.empty()) {
        // Wrap the raw solution span as a GlobalSystemView so the
        // StandardAssembler can access field values during gradient assembly.
        span_sol_view = std::make_unique<SpanSolutionView>(state.u, n_total);
        assembler_->setCurrentSolutionView(span_sol_view.get());
    }

    if (boundary_marker >= 0) {
        // Boundary face gradient assembly.
        assembler_->assembleBoundaryFaces(
            meshAccess(), boundary_marker,
            *rec.space, grad_kernel,
            /*matrix_view=*/nullptr,
            /*vector_view=*/&accum);
    } else {
        // Domain (all-cells) gradient assembly.
        // BoundaryFunctionalGradientKernel has hasCell()=false, so we wrap
        // it in a cell-capable adapter that reuses its Dual-arithmetic
        // evaluation for cell QPs instead of boundary face QPs.
        struct CellGradKernelAdapter final : public assembly::AssemblyKernel {
            forms::BoundaryFunctionalGradientKernel& inner;
            explicit CellGradKernelAdapter(forms::BoundaryFunctionalGradientKernel& k)
                : inner(k) {}
            [[nodiscard]] bool hasCell() const noexcept override { return true; }
            [[nodiscard]] bool hasBoundaryFace() const noexcept override { return false; }
            [[nodiscard]] bool hasInteriorFace() const noexcept override { return false; }
            [[nodiscard]] bool hasInterfaceFace() const noexcept override { return false; }
            [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override {
                return inner.getRequiredData();
            }
            [[nodiscard]] std::vector<assembly::FieldRequirement>
            fieldRequirements() const override {
                return inner.fieldRequirements();
            }
            void computeCell(const assembly::AssemblyContext& ctx,
                             assembly::KernelOutput& output) override {
                // Reuse the boundary face computation logic (which uses
                // Dual arithmetic for per-DOF derivatives) but call it
                // for a cell context.  The gradient kernel's computeBoundaryFace
                // reads basis values and QP weights from the context, which
                // are also valid for cell QPs.
                inner.computeBoundaryFace(ctx, -1, output);
            }
        };

        CellGradKernelAdapter cell_adapter(grad_kernel);
        assembler_->assembleVector(
            meshAccess(), *rec.space, cell_adapter, accum);
    }

    // Convert to SensitivityEntry pairs.
    std::vector<BoundaryReductionService::SensitivityEntry> result;
    result.reserve(accum.entries.size());
    for (const auto& [dof, val] : accum.entries) {
        if (std::abs(val) > 1e-16) {
            result.push_back({dof, val});
        }
    }

    return result;
}

std::span<const Real> FESystem::auxiliaryOutputValues() const noexcept
{
    // Flatten output buffers from all deployed entries.
    aux_output_flat_.clear();
    for (const auto& entry : deployed_aux_entries_) {
        aux_output_flat_.insert(aux_output_flat_.end(),
                                 entry.output_buffer.begin(),
                                 entry.output_buffer.end());
    }
    return aux_output_flat_;
}

std::size_t FESystem::auxiliaryOutputSlotOf(std::string_view output_name) const
{
    // Two-pass: first check for ambiguity, then return the slot.
    int match_count = 0;
    std::string first_instance;
    for (const auto& entry : deployed_aux_entries_) {
        for (const auto& oname : entry.model->outputNames()) {
            if (oname == output_name) {
                ++match_count;
                if (match_count == 1) first_instance = entry.instance_name;
            }
        }
    }
    FE_THROW_IF(match_count > 1, InvalidArgumentException,
                "auxiliaryOutputSlotOf(\"" + std::string(output_name) +
                    "\"): ambiguous — " + std::to_string(match_count) +
                    " deployed models have this output name. "
                    "Use auxiliaryOutputSlotOf(instance_name, output_name) instead.");

    if (match_count == 0) return static_cast<std::size_t>(-1);
    return auxiliaryOutputSlotOf(first_instance, output_name);
}

std::size_t FESystem::auxiliaryOutputSlotOf(
    std::string_view instance_name, std::string_view output_name) const
{
    std::size_t slot = 0;
    for (const auto& entry : deployed_aux_entries_) {
        auto out_names = entry.model->outputNames();
        const auto n_outputs = out_names.size();
        if (n_outputs == 0) continue;

        std::size_t n_entities = 1;
        if (auxiliary_state_manager_ &&
            auxiliary_state_manager_->hasBlock(entry.instance_name)) {
            n_entities = auxiliary_state_manager_->getBlock(entry.instance_name).entityCount();
        } else if (entry.explicit_entity_count > 0) {
            n_entities = entry.explicit_entity_count;
        }

        if (entry.instance_name == instance_name) {
            for (std::size_t i = 0; i < n_outputs; ++i) {
                if (out_names[i] == output_name) {
                    return slot + i;
                }
            }
        }

        slot += n_entities * n_outputs;
    }
    return static_cast<std::size_t>(-1);
}

std::vector<Real> FESystem::checkpointAuxiliaryState() const
{
    if (auxiliary_state_manager_) {
        return auxiliary_state_manager_->packAll();
    }
    return {};
}

void FESystem::restoreAuxiliaryState(std::span<const Real> data)
{
    if (auxiliary_state_manager_ && !data.empty()) {
        auxiliary_state_manager_->unpackAll(data);
    }
}

FESystem::AuxiliaryAnalysisSummary FESystem::auxiliaryAnalysisSummary() const
{
    AuxiliaryAnalysisSummary summary;

    if (auxiliary_state_manager_) {
        summary.n_blocks = auxiliary_state_manager_->blockCount();
        for (std::size_t i = 0; i < summary.n_blocks; ++i) {
            const auto& blk = auxiliary_state_manager_->state().block(i);
            summary.block_names.push_back(blk.name());
            const auto& spec = auxiliary_state_manager_->getSpec(blk.name());
            if (spec.solve_mode == AuxiliarySolveMode::Partitioned) {
                ++summary.n_partitioned;
            } else {
                ++summary.n_monolithic;
            }
        }
    }

    if (auxiliary_operator_registry_ && auxiliary_operator_registry_->isLayoutFinalized()) {
        summary.total_aux_unknowns = auxiliary_operator_registry_->auxiliaryLayout().total_aux_unknowns;
    }

    if (auxiliary_input_registry_) {
        summary.n_inputs = auxiliary_input_registry_->inputCount();
        summary.input_names = auxiliary_input_registry_->inputNames();
    }

    return summary;
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
