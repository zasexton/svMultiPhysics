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
#include "Dofs/EntityDofMap.h"

#include "Forms/FormKernels.h"
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
    dof_permutation_.reset();
    parameter_registry_.clear();
    if (operator_backends_) {
        operator_backends_->invalidateCache();
    }
    assembly_plan_by_op_.clear();
    coupled_jac_cache_.clear();

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
