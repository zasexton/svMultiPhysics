/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemManifest.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/JIT/KernelArgs.h"
#include "Constraints/Constraint.h"
#include "Constraints/GaugeRegistry.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Forms/FormIR.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCacheKey.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/JIT/KernelIR.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MonolithicCellKernel.h"
#include "Systems/FESystem.h"
#include "Systems/GlobalKernel.h"
#include "Systems/OperatorRegistry.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <exception>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

namespace {

constexpr std::uint64_t kOffset = forms::jit::kCacheKeyFNVOffset;

[[nodiscard]] std::uint64_t hashString(std::string_view s) noexcept
{
    return forms::jit::hashStringForCacheKey(s);
}

[[nodiscard]] std::string hex(std::uint64_t v)
{
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << v;
    return oss.str();
}

[[nodiscard]] std::string demangledTypeName(const std::type_info& info)
{
    int status = 0;
    char* name = abi::__cxa_demangle(info.name(), nullptr, nullptr, &status);
    if (status != 0 || name == nullptr) {
        std::free(name);
        return info.name();
    }
    std::string out(name);
    std::free(name);
    return out;
}

[[nodiscard]] std::uint64_t realBits(Real value) noexcept
{
    std::uint64_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

using TreeHashMemo = std::unordered_map<const forms::FormExprNode*, std::uint64_t>;

/// Ordered structural hash: node type, the generic payload accessors and, for
/// leaves, the spelling; children in order.  Constants enter by their bits.
/// Shared subexpressions are hashed once (the hash depends on the subtree
/// only).
[[nodiscard]] std::uint64_t treeHash(const forms::FormExprNode& node, TreeHashMemo& memo)
{
    if (const auto it = memo.find(&node); it != memo.end()) {
        return it->second;
    }
    std::uint64_t h = kOffset;
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(node.type()));
    if (const auto value = node.constantValue(); value.has_value()) {
        forms::jit::mixCacheKey(h, 1u);
        forms::jit::mixCacheKey(h, realBits(*value));
    }
    if (const auto i = node.componentIndex0(); i.has_value()) {
        forms::jit::mixCacheKey(h, 2u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(*i)));
    }
    if (const auto j = node.componentIndex1(); j.has_value()) {
        forms::jit::mixCacheKey(h, 3u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(*j)));
    }
    if (const auto d = node.identityDim(); d.has_value()) {
        forms::jit::mixCacheKey(h, 4u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(*d)));
    }
    if (const auto m = node.boundaryMarker(); m.has_value()) {
        forms::jit::mixCacheKey(h, 5u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(*m)));
    }
    if (const auto m = node.interfaceMarker(); m.has_value()) {
        forms::jit::mixCacheKey(h, 6u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(*m)));
    }
    if (const auto side = node.cutVolumeSide(); side.has_value()) {
        forms::jit::mixCacheKey(h, 7u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(*side));
    }
    if (const auto* measure = node.exteriorBoundaryMeasure(); measure != nullptr) {
        forms::jit::mixCacheKey(h, 8u);
        forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(measure->scope()));
        forms::jit::mixCacheKey(
            h, static_cast<std::uint64_t>(static_cast<std::int64_t>(measure->physicalBoundaryMarker())));
        forms::jit::mixCacheKey(
            h,
            static_cast<std::uint64_t>(static_cast<std::int64_t>(measure->generatedActiveBoundaryMarker())));
    }
    const auto children = node.children();
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(children.size()));
    if (children.empty()) {
        forms::jit::mixCacheKey(h, hashString(node.toString()));
    }
    for (const auto* child : children) {
        forms::jit::mixCacheKey(h, child != nullptr ? treeHash(*child, memo) : 0u);
    }
    memo.emplace(&node, h);
    return h;
}

[[nodiscard]] const char* domainName(forms::IntegralDomain domain) noexcept
{
    switch (domain) {
    case forms::IntegralDomain::Cell:
        return "Cell";
    case forms::IntegralDomain::Boundary:
        return "Boundary";
    case forms::IntegralDomain::InteriorFace:
        return "InteriorFace";
    case forms::IntegralDomain::InterfaceFace:
        return "InterfaceFace";
    case forms::IntegralDomain::CutVolume:
        return "CutVolume";
    }
    return "Unknown";
}

[[nodiscard]] const char* sideName(forms::CutVolumeSide side) noexcept
{
    return side == forms::CutVolumeSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] const char* sideName(geometry::CutIntegrationSide side) noexcept
{
    switch (side) {
    case geometry::CutIntegrationSide::Negative:
        return "Negative";
    case geometry::CutIntegrationSide::Positive:
        return "Positive";
    case geometry::CutIntegrationSide::Interface:
        return "Interface";
    }
    return "Unknown";
}

[[nodiscard]] const char* formKindName(forms::FormKind kind) noexcept
{
    switch (kind) {
    case forms::FormKind::Linear:
        return "Linear";
    case forms::FormKind::Bilinear:
        return "Bilinear";
    case forms::FormKind::Residual:
        return "Residual";
    }
    return "Unknown";
}

[[nodiscard]] std::uint64_t spaceSignatureHash(
    const std::optional<forms::FormExprNode::SpaceSignature>& sig) noexcept
{
    // Same mixing as the JIT compiler's space-signature hash.
    std::uint64_t h = kOffset;
    if (!sig) {
        forms::jit::mixCacheKey(h, 0ULL);
        return h;
    }
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(sig->space_type));
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(sig->field_type));
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(sig->continuity));
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig->value_dimension)));
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig->topological_dimension)));
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig->polynomial_order)));
    forms::jit::mixCacheKey(h, static_cast<std::uint64_t>(sig->element_type));
    return h;
}

struct GroupKey {
    forms::IntegralDomain domain{forms::IntegralDomain::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
    std::optional<forms::CutVolumeSide> interior_facet_side{};
    forms::CutVolumeSide cut_volume_side{forms::CutVolumeSide::Negative};

    friend bool operator==(const GroupKey&, const GroupKey&) = default;
};

[[nodiscard]] GroupKey groupKeyOf(const forms::IntegralTerm& term)
{
    // Mirrors the JIT compiler's kernel grouping.
    GroupKey key;
    key.domain = term.domain;
    key.boundary_marker = term.boundary_marker;
    key.interface_marker = term.interface_marker;
    if (term.domain == forms::IntegralDomain::InteriorFace) {
        key.interior_facet_side = term.interior_facet_side;
    }
    if (term.domain == forms::IntegralDomain::CutVolume) {
        key.cut_volume_side = term.cut_volume_side;
    }
    return key;
}

struct TermHashes {
    std::uint64_t kir{0};
    bool kir_ok{false};
    std::uint64_t tree{0};
    std::uint64_t str{0};
};

[[nodiscard]] TermHashes termHashes(const forms::IntegralTerm& term)
{
    TermHashes out;
    if (!term.integrand.isValid()) {
        return out;
    }
    TreeHashMemo memo;
    out.tree = treeHash(*term.integrand.node(), memo);
    out.str = hashString(term.debug_string.empty() ? term.integrand.toString() : term.debug_string);
    try {
        auto lowered = forms::jit::lowerToKernelIR(term.integrand);
        lowered.ir.optimize();
        out.kir = lowered.ir.stableHash64();
        out.kir_ok = true;
    } catch (const std::exception&) {
        out.kir_ok = false;
    }
    return out;
}

void appendFormIR(std::vector<std::string>& lines,
                  const std::string& prefix,
                  std::string_view role,
                  const forms::FormIR& ir,
                  const forms::JITOptions* jit_options)
{
    std::ostringstream head;
    head << prefix << " ir=" << role;
    if (!ir.isCompiled()) {
        lines.push_back(head.str() + " compiled=0");
        return;
    }
    head << " kind=" << formKindName(ir.kind());
    const std::string ir_prefix = head.str();
    const auto& terms = ir.terms();
    lines.push_back(ir_prefix + " terms=" + std::to_string(terms.size()) +
                    " test_space=" + hex(spaceSignatureHash(ir.testSpace())) +
                    " trial_space=" + hex(spaceSignatureHash(ir.trialSpace())));

    std::vector<TermHashes> hashes;
    hashes.reserve(terms.size());
    for (std::size_t i = 0; i < terms.size(); ++i) {
        const auto& term = terms[i];
        hashes.push_back(termHashes(term));
        const auto& th = hashes.back();
        std::ostringstream oss;
        oss << ir_prefix << " term=" << i << " domain=" << domainName(term.domain)
            << " bm=" << term.boundary_marker << " im=" << term.interface_marker
            << " fside="
            << (term.interior_facet_side.has_value() ? sideName(*term.interior_facet_side) : "-")
            << " cside=" << sideName(term.cut_volume_side)
            << " tdo=" << term.time_derivative_order << " ext=";
        if (term.exterior_boundary_measure.has_value()) {
            const auto& m = *term.exterior_boundary_measure;
            oss << (m.isGeneratedActiveSubset() ? "GeneratedActiveSubset" : "FullPhysical") << ":"
                << m.physicalBoundaryMarker() << ":" << m.generatedActiveBoundaryMarker();
        } else {
            oss << "-";
        }
        oss << " tree=" << hex(th.tree) << " str=" << hex(th.str)
            << " kir=" << (th.kir_ok ? hex(th.kir) : std::string("unsupported"));
        lines.push_back(oss.str());
    }

    // Kernel groups in order of first appearance.
    std::vector<GroupKey> keys;
    std::vector<std::vector<std::size_t>> members;
    for (std::size_t i = 0; i < terms.size(); ++i) {
        const auto key = groupKeyOf(terms[i]);
        std::size_t g = 0;
        while (g < keys.size() && !(keys[g] == key)) {
            ++g;
        }
        if (g == keys.size()) {
            keys.push_back(key);
            members.emplace_back();
        }
        members[g].push_back(i);
    }

    const auto test_hash = spaceSignatureHash(ir.testSpace());
    const auto trial_hash = spaceSignatureHash(ir.trialSpace());
    const bool tensor_lowering =
        jit_options != nullptr && jit_options->tensor.mode != forms::TensorLoweringMode::Off;
    for (std::size_t g = 0; g < keys.size(); ++g) {
        const auto& key = keys[g];
        std::uint64_t combined = kOffset;
        bool ok = true;
        std::ostringstream ids;
        for (std::size_t m = 0; m < members[g].size(); ++m) {
            const auto idx = members[g][m];
            const auto& term = terms[idx];
            ids << (m == 0u ? "" : ",") << idx;
            ok = ok && hashes[idx].kir_ok;
            forms::jit::mixCacheKey(combined, hashes[idx].kir);
            forms::jit::mixCacheKey(
                combined,
                static_cast<std::uint64_t>(static_cast<std::int64_t>(term.time_derivative_order)));
            forms::jit::mixCacheKey(combined, term.exterior_boundary_measure.has_value() ? 1u : 0u);
            if (term.exterior_boundary_measure.has_value()) {
                const auto& measure = *term.exterior_boundary_measure;
                forms::jit::mixCacheKey(combined, static_cast<std::uint64_t>(measure.scope()));
                forms::jit::mixCacheKey(
                    combined,
                    static_cast<std::uint64_t>(
                        static_cast<std::int64_t>(measure.physicalBoundaryMarker())));
                forms::jit::mixCacheKey(
                    combined,
                    static_cast<std::uint64_t>(
                        static_cast<std::int64_t>(measure.generatedActiveBoundaryMarker())));
            }
        }
        std::ostringstream oss;
        oss << ir_prefix << " group=" << g << " domain=" << domainName(key.domain)
            << " bm=" << key.boundary_marker << " im=" << key.interface_marker << " fside="
            << (key.interior_facet_side.has_value() ? sideName(*key.interior_facet_side) : "-")
            << " cside=" << sideName(key.cut_volume_side) << " terms=" << ids.str()
            << " ir_hash=" << (ok ? hex(combined) : std::string("unsupported")) << " key=";
        if (jit_options == nullptr || !ok || tensor_lowering) {
            oss << (tensor_lowering ? "tensor-lowering" : "-");
        } else {
            const bool uses_cell_batch_abi =
                key.domain == forms::IntegralDomain::Cell && jit_options->vectorize;
            forms::jit::KernelCacheKeyInputs inputs{};
            inputs.abi_version = uses_cell_batch_abi ? assembly::jit::kCellKernelBatchArgsABIV1
                                                     : assembly::jit::kKernelArgsABIVersionV6;
            inputs.abi_layout_revision = assembly::jit::kKernelArgsABILayoutRevisionV6;
            inputs.form_kind = ir.kind();
            inputs.domain = key.domain;
            inputs.boundary_marker = key.boundary_marker;
            inputs.interface_marker = key.interface_marker;
            inputs.interior_facet_side = key.interior_facet_side;
            inputs.cut_volume_side = key.cut_volume_side;
            inputs.combined_ir_hash = combined;
            inputs.test_space_hash = test_hash;
            inputs.trial_space_hash = trial_hash;
            inputs.jit_options = *jit_options;
            oss << hex(forms::jit::computeKernelCacheKey(inputs));
        }
        lines.push_back(oss.str());
    }
}

void appendKernel(std::vector<std::string>& lines,
                  const std::string& prefix,
                  const assembly::AssemblyKernel* kernel,
                  const forms::JITOptions* jit_options,
                  int depth)
{
    if (kernel == nullptr) {
        lines.push_back(prefix + " kernel=<null>");
        return;
    }
    if (depth > 8) {
        lines.push_back(prefix + " kernel=<nesting limit>");
        return;
    }
    const std::string type = demangledTypeName(typeid(*kernel));
    lines.push_back(prefix + " type=" + type + " name=" + kernel->name());

    if (const auto* wrapper = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
        appendKernel(lines, prefix + " jit", &wrapper->fallbackKernel(), &wrapper->jitOptions(), depth + 1);
        return;
    }
    if (const auto* k = dynamic_cast<const forms::FormKernel*>(kernel)) {
        appendFormIR(lines, prefix, "form", k->ir(), jit_options);
        return;
    }
    if (const auto* k = dynamic_cast<const forms::LinearFormKernel*>(kernel)) {
        appendFormIR(lines, prefix, "bilinear", k->bilinearIR(), jit_options);
        if (k->linearIR().has_value()) {
            appendFormIR(lines, prefix, "linear", *k->linearIR(), jit_options);
        }
        return;
    }
    if (const auto* k = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
        appendFormIR(lines, prefix, "residual", k->residualIR(), jit_options);
        return;
    }
    if (const auto* k = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
        appendFormIR(lines, prefix, "residual", k->residualIR(), jit_options);
        appendFormIR(lines, prefix, "tangent", k->tangentIR(), jit_options);
        return;
    }
    if (const auto* k = dynamic_cast<const forms::MixedBlockKernelSet*>(kernel)) {
        for (std::size_t b = 0; b < k->numBlocks(); ++b) {
            const auto& spec = k->blockSpec(b);
            std::ostringstream oss;
            oss << prefix << " block=" << b << " test=" << spec.test_field
                << " trial=" << spec.trial_field << " matrix=" << (spec.want_matrix ? 1 : 0)
                << " vector=" << (spec.want_vector ? 1 : 0);
            appendKernel(lines, oss.str(), spec.fallback_kernel.get(), jit_options, depth + 1);
        }
        return;
    }
    if (const auto* k = dynamic_cast<const forms::MonolithicCellKernel*>(kernel)) {
        for (std::size_t b = 0; b < k->numBlocks(); ++b) {
            const auto& spec = k->blockSpec(b);
            std::ostringstream oss;
            oss << prefix << " block=" << b << " test=" << spec.test_field
                << " trial=" << spec.trial_field << " matrix=" << (spec.want_matrix ? 1 : 0)
                << " vector=" << (spec.want_vector ? 1 : 0);
            const std::string block_prefix = oss.str();
            if (spec.residual_ir.has_value()) {
                appendFormIR(lines, block_prefix, "residual", *spec.residual_ir, nullptr);
            }
            if (spec.tangent_ir.has_value()) {
                appendFormIR(lines, block_prefix, "tangent", *spec.tangent_ir, nullptr);
            }
            appendKernel(lines, block_prefix, spec.fallback_kernel.get(), jit_options, depth + 1);
        }
        return;
    }
}

[[nodiscard]] std::string fieldLabel(const FESystem& system, FieldId field)
{
    if (field == INVALID_FIELD_ID) {
        return "-";
    }
    try {
        return std::to_string(field) + ":" + system.fieldRecord(field).name;
    } catch (const std::exception&) {
        return std::to_string(field);
    }
}

[[nodiscard]] std::string slotPrefix(const FESystem& system,
                                     const std::string& op,
                                     std::string_view slot,
                                     std::size_t index,
                                     FieldId test,
                                     FieldId trial)
{
    std::ostringstream oss;
    oss << "kernel op=" << op << " slot=" << slot << "#" << index
        << " test=" << fieldLabel(system, test) << " trial=" << fieldLabel(system, trial);
    return oss.str();
}

} // namespace

std::uint64_t formExprTreeHash(const forms::FormExpr& expr)
{
    if (!expr.isValid()) {
        return 0u;
    }
    TreeHashMemo memo;
    return treeHash(*expr.node(), memo);
}

std::vector<std::string> kernelManifestLines(const FESystem& system)
{
    std::vector<std::string> lines;
    for (const auto& tag : system.operatorTags()) {
        const auto& def = system.operatorDefinition(tag);
        lines.push_back("operator " + tag + " cells=" + std::to_string(def.cells.size()) +
                        " boundary=" + std::to_string(def.boundary.size()) +
                        " interior=" + std::to_string(def.interior.size()) +
                        " interface=" + std::to_string(def.interface_faces.size()) +
                        " cut_volume=" + std::to_string(def.cut_volumes.size()) +
                        " global=" + std::to_string(def.global.size()));
        for (std::size_t i = 0; i < def.cells.size(); ++i) {
            const auto& t = def.cells[i];
            appendKernel(lines, slotPrefix(system, tag, "cell", i, t.test_field, t.trial_field),
                         t.kernel.get(), nullptr, 0);
        }
        for (std::size_t i = 0; i < def.boundary.size(); ++i) {
            const auto& t = def.boundary[i];
            appendKernel(lines,
                         slotPrefix(system, tag, "boundary", i, t.test_field, t.trial_field) +
                             " marker=" + std::to_string(t.marker),
                         t.kernel.get(), nullptr, 0);
        }
        for (std::size_t i = 0; i < def.interior.size(); ++i) {
            const auto& t = def.interior[i];
            appendKernel(lines,
                         slotPrefix(system, tag, "interior", i, t.test_field, t.trial_field) +
                             " marker=" + std::to_string(t.marker) + " side=" + sideName(t.side),
                         t.kernel.get(), nullptr, 0);
        }
        for (std::size_t i = 0; i < def.interface_faces.size(); ++i) {
            const auto& t = def.interface_faces[i];
            appendKernel(lines,
                         slotPrefix(system, tag, "interface", i, t.test_field, t.trial_field) +
                             " marker=" + std::to_string(t.marker),
                         t.kernel.get(), nullptr, 0);
        }
        for (std::size_t i = 0; i < def.cut_volumes.size(); ++i) {
            const auto& t = def.cut_volumes[i];
            appendKernel(lines,
                         slotPrefix(system, tag, "cut_volume", i, t.test_field, t.trial_field) +
                             " marker=" + std::to_string(t.marker) + " side=" + sideName(t.side) +
                             " source='" + t.source_component_tag + "'",
                         t.kernel.get(), nullptr, 0);
        }
        for (std::size_t i = 0; i < def.global.size(); ++i) {
            const auto& k = def.global[i];
            lines.push_back("kernel op=" + tag + " slot=global#" + std::to_string(i) + " type=" +
                            (k ? demangledTypeName(typeid(*k)) : std::string("<null>")) +
                            " name=" + (k ? k->name() : std::string("<null>")));
        }
    }
    return lines;
}

std::vector<std::string> constraintManifestLines(const FESystem& system)
{
    return system.constraintManifestLines();
}

std::vector<std::string> FESystem::constraintManifestLines() const
{
    std::vector<std::string> lines;
    for (std::size_t i = 0; i < constraint_defs_.size(); ++i) {
        const auto& c = constraint_defs_[i];
        lines.push_back("constraint algebraic#" + std::to_string(i) + " type=" +
                        (c ? demangledTypeName(typeid(*c)) : std::string("<null>")));
    }
    for (std::size_t i = 0; i < system_constraint_defs_.size(); ++i) {
        const auto& c = system_constraint_defs_[i];
        std::ostringstream oss;
        oss << "constraint system#" << i << " type=";
        if (!c) {
            oss << "<null>";
            lines.push_back(oss.str());
            continue;
        }
        oss << demangledTypeName(typeid(*c)) << " time_dependent=" << (c->isTimeDependent() ? 1 : 0);
        std::string description = c->manifestDescription();
        if (const auto* aggregation =
                dynamic_cast<const constraints::SmallCutAggregationConstraint*>(c.get())) {
            // The aggregation constraint is described here (FESystem is its
            // friend) so that its files stay untouched.
            std::ostringstream agg;
            agg << std::hexfloat << "field=" << aggregation->field_
                << " active_side=" << sideName(aggregation->active_side_)
                << " interface_marker=" << aggregation->interface_marker_ << " excluded_markers=";
            for (std::size_t m = 0; m < aggregation->excluded_boundary_markers_.size(); ++m) {
                agg << (m == 0u ? "" : ",") << aggregation->excluded_boundary_markers_[m];
            }
            agg << " excluded_vertices=";
            for (std::size_t m = 0; m < aggregation->excluded_vertices_.size(); ++m) {
                agg << (m == 0u ? "" : ",") << aggregation->excluded_vertices_[m];
            }
            const auto& guards = aggregation->guards_;
            agg << " max_root_path=" << guards.maximum_root_path_length
                << " max_extrapolation=" << guards.maximum_reference_extrapolation_distance
                << " max_coefficient=" << guards.maximum_absolute_coefficient
                << " max_row_l1=" << guards.maximum_row_l1_norm
                << " rootless_fallback=" << (guards.rootless_fallback ? 1 : 0);
            description = agg.str();
        }
        if (!description.empty()) {
            oss << " " << description;
        }
        lines.push_back(oss.str());
    }
    if (const auto* registry = gaugeRegistryIfPresent()) {
        const auto& anchoring = registry->anchoring();
        for (std::size_t i = 0; i < anchoring.size(); ++i) {
            const auto& a = anchoring[i];
            std::ostringstream oss;
            oss << "gauge anchoring#" << i << " field=" << a.field << " component=" << a.component
                << " region=" << a.region << " family="
                << (a.family.has_value() ? std::to_string(static_cast<int>(*a.family))
                                         : std::string("all"))
                << " verdict=" << static_cast<int>(a.verdict)
                << " boundary_marker=" << a.boundary_marker << " source='" << a.source << "'";
            lines.push_back(oss.str());
        }
    }
    for (std::size_t i = 0; i < cut_integration_context_update_callbacks_.size(); ++i) {
        lines.push_back("cut_context_callback#" + std::to_string(i) + " name='" +
                        cut_integration_context_update_callbacks_[i].name + "'");
    }
    {
        std::ostringstream oss;
        oss << "generated_embedded_interface_markers count="
            << generated_embedded_interface_markers_.size() << " markers=";
        bool first = true;
        for (const auto marker : generated_embedded_interface_markers_) {
            oss << (first ? "" : ",") << marker;
            first = false;
        }
        lines.push_back(oss.str());
    }
    return lines;
}

} // namespace systems
} // namespace FE
} // namespace svmp
