/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/FormKernels.h"

#include "Constitutive/StateLayout.h"

#include "Core/FEException.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/Dual.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"
#include "Forms/Value.h"

#include "Assembly/AssemblyContext.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace forms {

struct ConstitutiveStateLayout {
    struct Block {
        const ConstitutiveModel* model{nullptr};
        std::size_t offset_bytes{0};
        std::size_t bytes{0};
        std::size_t alignment{1};
    };

    std::vector<Block> blocks{};
    std::size_t bytes_per_qpt{0};
    std::size_t alignment{alignof(std::max_align_t)};

    [[nodiscard]] bool empty() const noexcept { return bytes_per_qpt == 0u; }

    [[nodiscard]] const Block* find(const ConstitutiveModel* model) const noexcept
    {
        for (const auto& b : blocks) {
            if (b.model == model) return &b;
        }
        return nullptr;
    }
};

namespace {

enum class Side : std::uint8_t { Minus, Plus };

const assembly::AssemblyContext& ctxForSide(const assembly::AssemblyContext& ctx_minus,
                                            const assembly::AssemblyContext* ctx_plus,
                                            Side side)
{
    if (side == Side::Minus) return ctx_minus;
    if (!ctx_plus) {
        throw FEException("Forms: requested plus-side context in non-DG evaluation",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return *ctx_plus;
}

template<typename Scalar>
using EvalValue = Value<Scalar>;

template<typename Scalar>
constexpr bool isScalarKind(typename EvalValue<Scalar>::Kind k) noexcept
{
    return k == EvalValue<Scalar>::Kind::Scalar;
}

template<typename Scalar>
constexpr bool isVectorKind(typename EvalValue<Scalar>::Kind k) noexcept
{
    return k == EvalValue<Scalar>::Kind::Vector;
}

bool containsTestOrTrial(const FormExprNode& node)
{
    if (node.type() == FormExprType::TestFunction || node.type() == FormExprType::TrialFunction) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsTestOrTrial(*child)) {
            return true;
        }
    }
    return false;
}

bool containsTestFunction(const FormExprNode& node)
{
    if (node.type() == FormExprType::TestFunction) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsTestFunction(*child)) return true;
    }
    return false;
}

bool containsTrialFunction(const FormExprNode& node)
{
    if (node.type() == FormExprType::TrialFunction) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsTrialFunction(*child)) return true;
    }
    return false;
}

template<typename Scalar>
constexpr bool isMatrixKind(typename EvalValue<Scalar>::Kind k) noexcept
{
    return k == EvalValue<Scalar>::Kind::Matrix ||
           k == EvalValue<Scalar>::Kind::SymmetricMatrix ||
           k == EvalValue<Scalar>::Kind::SkewMatrix;
}

template<typename Scalar>
constexpr bool isTensor4Kind(typename EvalValue<Scalar>::Kind k) noexcept
{
    return k == EvalValue<Scalar>::Kind::Tensor4;
}

template<typename Scalar>
constexpr bool isTensor3Kind(typename EvalValue<Scalar>::Kind k) noexcept
{
    return k == EvalValue<Scalar>::Kind::Tensor3;
}

template<typename Scalar>
constexpr bool sameCategory(typename EvalValue<Scalar>::Kind a,
                            typename EvalValue<Scalar>::Kind b) noexcept
{
    return (isScalarKind<Scalar>(a) && isScalarKind<Scalar>(b)) ||
           (isVectorKind<Scalar>(a) && isVectorKind<Scalar>(b)) ||
           (isMatrixKind<Scalar>(a) && isMatrixKind<Scalar>(b)) ||
           (isTensor3Kind<Scalar>(a) && isTensor3Kind<Scalar>(b)) ||
           (isTensor4Kind<Scalar>(a) && isTensor4Kind<Scalar>(b));
}

template<typename Scalar>
constexpr typename EvalValue<Scalar>::Kind addSubResultKind(typename EvalValue<Scalar>::Kind a,
                                                            typename EvalValue<Scalar>::Kind b) noexcept
{
    if (a == b) return a;
    if (isMatrixKind<Scalar>(a) && isMatrixKind<Scalar>(b)) {
        return EvalValue<Scalar>::Kind::Matrix;
    }
    if (isTensor3Kind<Scalar>(a) && isTensor3Kind<Scalar>(b)) {
        return EvalValue<Scalar>::Kind::Tensor3;
    }
    if (isTensor4Kind<Scalar>(a) && isTensor4Kind<Scalar>(b)) {
        return EvalValue<Scalar>::Kind::Tensor4;
    }
    return a;
}

constexpr std::size_t idx4(int i, int j, int k, int l) noexcept
{
    return static_cast<std::size_t>((((i * 3) + j) * 3 + k) * 3 + l);
}

[[nodiscard]] bool isPowerOfTwo(std::size_t x) noexcept
{
    return x != 0u && (x & (x - 1u)) == 0u;
}

[[nodiscard]] std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    FE_THROW_IF(alignment == 0u, InvalidArgumentException, "Forms: alignUp alignment must be non-zero");
    FE_THROW_IF(!isPowerOfTwo(alignment), InvalidArgumentException, "Forms: alignUp alignment must be power-of-two");
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void gatherConstitutiveModels(const FormExprNode& node, std::vector<const ConstitutiveModel*>& models)
{
    if (node.type() == FormExprType::Constitutive) {
        if (const auto* model = node.constitutiveModel(); model != nullptr) {
            const bool seen = std::any_of(models.begin(), models.end(),
                                          [&](const ConstitutiveModel* m) { return m == model; });
            if (!seen) models.push_back(model);
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) gatherConstitutiveModels(*child, models);
    }
}

void gatherParameterSymbols(const FormExprNode& node, std::vector<std::string_view>& names)
{
    if (node.type() == FormExprType::ParameterSymbol) {
        const auto nm = node.symbolName();
        if (nm && !nm->empty()) {
            names.push_back(*nm);
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) gatherParameterSymbols(*child, names);
    }
}

[[nodiscard]] std::vector<params::Spec> computeParameterSpecs(std::span<const FormIR* const> irs)
{
    std::vector<const ConstitutiveModel*> models;
    std::vector<std::string_view> param_names;

    for (const auto* ir : irs) {
        if (ir == nullptr) continue;
        for (const auto& term : ir->terms()) {
            const auto* root = term.integrand.node();
            if (!root) continue;
            gatherConstitutiveModels(*root, models);
            gatherParameterSymbols(*root, param_names);
        }
    }

    std::vector<params::Spec> out;

    for (const auto* m : models) {
        if (!m) continue;
        auto specs = m->parameterSpecs();
        out.insert(out.end(), specs.begin(), specs.end());
    }

    if (!param_names.empty()) {
        std::vector<std::string> keys;
        keys.reserve(param_names.size());
        for (const auto nm : param_names) {
            keys.emplace_back(nm);
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

        for (auto& key : keys) {
            out.push_back(params::Spec{.key = std::move(key),
                                       .type = params::ValueType::Real,
                                       .required = true});
        }
    }

    return out;
}

[[nodiscard]] std::shared_ptr<const ConstitutiveStateLayout> buildConstitutiveStateLayout(
    const FormIR& ir,
    assembly::MaterialStateSpec& spec_out)
{
    spec_out = {};

    std::vector<const ConstitutiveModel*> models;
    for (const auto& term : ir.terms()) {
        const auto* root = term.integrand.node();
        if (!root) continue;
        gatherConstitutiveModels(*root, models);
    }

    auto layout = std::make_shared<ConstitutiveStateLayout>();

    std::size_t cursor = 0;
    std::size_t max_align = alignof(std::max_align_t);
    for (const auto* model : models) {
        auto ss = model->stateSpec();
        const auto* state_layout = model->stateLayout();

        if (ss.empty()) {
            if (state_layout == nullptr || state_layout->empty()) continue;
            ss.bytes_per_qpt = state_layout->bytesPerPoint();
            ss.alignment = state_layout->alignment();
        } else if (state_layout != nullptr && !state_layout->empty()) {
            FE_THROW_IF(ss.bytes_per_qpt != state_layout->bytesPerPoint(), InvalidArgumentException,
                        "Forms: ConstitutiveModel stateSpec bytes_per_qpt does not match stateLayout bytesPerPoint");
            FE_THROW_IF(ss.alignment != state_layout->alignment(), InvalidArgumentException,
                        "Forms: ConstitutiveModel stateSpec alignment does not match stateLayout alignment");
        }

        FE_THROW_IF(ss.alignment == 0u, InvalidArgumentException, "Forms: ConstitutiveModel stateSpec alignment must be > 0");
        FE_THROW_IF(!isPowerOfTwo(ss.alignment), InvalidArgumentException,
                    "Forms: ConstitutiveModel stateSpec alignment must be power-of-two");

        cursor = alignUp(cursor, ss.alignment);
        layout->blocks.push_back(ConstitutiveStateLayout::Block{
            model,
            cursor,
            ss.bytes_per_qpt,
            ss.alignment,
        });
        cursor += ss.bytes_per_qpt;
        max_align = std::max(max_align, ss.alignment);
    }

    if (cursor == 0u) {
        return nullptr;
    }

    layout->bytes_per_qpt = cursor;
    layout->alignment = max_align;

    spec_out.bytes_per_qpt = layout->bytes_per_qpt;
    spec_out.alignment = layout->alignment;
    return layout;
}

[[nodiscard]] int inferTopologicalDim(const FormIR& ir) noexcept
{
    if (const auto& ts = ir.testSpace(); ts.has_value()) {
        return ts->topological_dimension;
    }
    if (const auto& tr = ir.trialSpace(); tr.has_value()) {
        return tr->topological_dimension;
    }
    return 0;
}

struct InlinedConstitutiveCall {
    std::vector<FormExpr> outputs{};
    std::vector<MaterialStateUpdate> updates{};
    std::vector<const FormExprNode*> dependencies{};
    std::uint64_t kind_id{0u};
    MaterialStateAccess state_access{MaterialStateAccess::None};
};

void inlineInlinableConstitutives(std::span<FormIR*> irs,
                                  const ConstitutiveStateLayout* state_layout,
                                  const assembly::MaterialStateSpec& state_spec,
                                  InlinedMaterialStateUpdateProgram& state_updates_out)
{
    state_updates_out.clear();

    std::unordered_map<const FormExprNode*, InlinedConstitutiveCall> memo;
    std::unordered_map<const FormExprNode*, std::vector<const FormExprNode*>> deps_by_call;
    std::vector<const FormExprNode*> expansion_stack;

    struct EmittedKey {
        const FormExprNode* call{nullptr};
        IntegralDomain domain{IntegralDomain::Cell};
        int boundary_marker{-1};

        [[nodiscard]] bool operator==(const EmittedKey& other) const noexcept
        {
            return call == other.call &&
                   domain == other.domain &&
                   boundary_marker == other.boundary_marker;
        }
    };

    struct EmittedKeyHash {
        [[nodiscard]] std::size_t operator()(const EmittedKey& k) const noexcept
        {
            std::size_t h = std::hash<const FormExprNode*>{}(k.call);
            auto mix = [&](std::size_t v) {
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            mix(static_cast<std::size_t>(k.domain));
            mix(static_cast<std::size_t>(static_cast<std::uint32_t>(k.boundary_marker)));
            return h;
        }
    };

    std::unordered_set<EmittedKey, EmittedKeyHash> emitted;

    int dim_hint = 0;
    bool allow_trial_in_updates = false;
    for (auto* ir : irs) {
        if (ir != nullptr) {
            dim_hint = std::max(dim_hint, inferTopologicalDim(*ir));
            allow_trial_in_updates = allow_trial_in_updates || (ir->kind() == FormKind::Residual);
        }
    }

    const auto appendUpdatesForTerm = [&](const std::vector<MaterialStateUpdate>& updates,
                                          const IntegralTerm& term) {
        if (updates.empty()) {
            return;
        }
        switch (term.domain) {
            case IntegralDomain::Cell:
                state_updates_out.cell.insert(state_updates_out.cell.end(), updates.begin(), updates.end());
                return;
            case IntegralDomain::Boundary:
                if (term.boundary_marker < 0) {
                    state_updates_out.boundary_all.insert(state_updates_out.boundary_all.end(), updates.begin(), updates.end());
                } else {
                    auto& bucket = state_updates_out.boundary_by_marker[term.boundary_marker];
                    bucket.insert(bucket.end(), updates.begin(), updates.end());
                }
                return;
            case IntegralDomain::InteriorFace:
                state_updates_out.interior_face.insert(state_updates_out.interior_face.end(), updates.begin(), updates.end());
                return;
            case IntegralDomain::InterfaceFace:
                state_updates_out.interface_face.insert(state_updates_out.interface_face.end(), updates.begin(), updates.end());
                return;
        }
    };

    std::function<void(const FormExprNode*, const IntegralTerm&)> emitUpdatesForCall;
    emitUpdatesForCall = [&](const FormExprNode* call_ptr, const IntegralTerm& term) -> void {
        if (call_ptr == nullptr) return;
        const auto it = memo.find(call_ptr);
        if (it == memo.end()) return;

        const EmittedKey key{call_ptr, term.domain, (term.domain == IntegralDomain::Boundary) ? term.boundary_marker : -1};
        if (!it->second.updates.empty()) {
            if (emitted.insert(key).second) {
                appendUpdatesForTerm(it->second.updates, term);
            }
        }

        for (const auto* dep : it->second.dependencies) {
            emitUpdatesForCall(dep, term);
        }
    };

    const auto recordDependency = [&](const FormExprNode* callee) {
        if (callee == nullptr || expansion_stack.empty()) {
            return;
        }
        const auto* parent = expansion_stack.back();
        if (parent == nullptr) {
            return;
        }
        auto& deps = deps_by_call[parent];
        const bool seen = std::any_of(deps.begin(), deps.end(),
                                      [&](const FormExprNode* p) { return p == callee; });
        if (!seen) deps.push_back(callee);
    };

    std::function<std::optional<FormExpr>(const FormExprNode&, const IntegralTerm&)> term_transform;

    const auto get_or_inline = [&](const FormExprNode& call_node,
                                   const IntegralTerm& term,
                                   const std::function<std::optional<FormExpr>(const FormExprNode&)>& node_transform)
        -> const InlinedConstitutiveCall* {
        if (call_node.type() != FormExprType::Constitutive) {
            return nullptr;
        }

        if (auto it = memo.find(&call_node); it != memo.end()) {
            emitUpdatesForCall(&call_node, term);
            return &it->second;
        }

        const auto* model = call_node.constitutiveModel();
        if (!model) {
            throw FEException("Forms: Constitutive node missing model during inlining",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }

        const auto* inlinable = model->inlinable();
        if (!inlinable) {
            return nullptr;
        }

        const auto kids = call_node.childrenShared();
        FE_THROW_IF(kids.empty(), InvalidArgumentException,
                    "Forms: Constitutive node must have at least 1 input for inlining");

        std::vector<FormExpr> inputs;
        inputs.reserve(kids.size());
        for (const auto& k : kids) {
            FE_THROW_IF(!k, InvalidArgumentException,
                        "Forms: Constitutive node has null input");
            inputs.emplace_back(k);
        }

        // Recursively inline nested constitutive calls inside inputs.
        expansion_stack.push_back(&call_node);
        deps_by_call[&call_node].clear();
        for (auto& in : inputs) {
            in = in.transformNodes(node_transform);
        }

        std::uint32_t base_offset_u32 = 0u;
        std::size_t model_bytes = 0u;

        if (state_layout != nullptr) {
            if (const auto* block = state_layout->find(model); block != nullptr) {
                FE_THROW_IF(block->offset_bytes > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                            InvalidArgumentException,
                            "Forms: constitutive state base offset overflows u32");
                base_offset_u32 = static_cast<std::uint32_t>(block->offset_bytes);
                model_bytes = block->bytes;
            }
        }

        const auto access = inlinable->stateAccess();
        if (access != MaterialStateAccess::None) {
            FE_THROW_IF(state_spec.bytes_per_qpt == 0u, InvalidArgumentException,
                        "Forms: inlinable constitutive model requests material state but kernel has no materialStateSpec");
            FE_THROW_IF(model_bytes == 0u, InvalidArgumentException,
                        "Forms: inlinable constitutive model requests material state but provides no stateSpec/stateLayout");
        }

        InlinableConstitutiveContext ctx;
        ctx.dim = dim_hint;
        ctx.state_base_offset_bytes = base_offset_u32;
        ctx.state_layout = model->stateLayout();

        auto expansion = inlinable->inlineExpand(inputs, ctx);

        const auto n_outputs = model->outputCount();
        FE_THROW_IF(expansion.outputs.size() != n_outputs, InvalidArgumentException,
                    "Forms: inlinable constitutive expansion outputs.size() does not match model->outputCount()");

        // Inline nested calls within expansion outputs + update expressions.
        for (auto& out : expansion.outputs) {
            out = out.transformNodes(node_transform);
        }
        for (auto& op : expansion.state_updates) {
            FE_THROW_IF(!op.value.isValid(), InvalidArgumentException,
                        "Forms: inlinable constitutive state update has invalid value expression");
            op.value = op.value.transformNodes(node_transform);
        }

        InlinedConstitutiveCall stored;
        stored.kind_id = inlinable->kindId();
        stored.state_access = access;
        stored.outputs = std::move(expansion.outputs);

        stored.updates.reserve(expansion.state_updates.size());
        for (const auto& op : expansion.state_updates) {
            FE_THROW_IF(containsTestFunction(*op.value.node()), InvalidArgumentException,
                        "Forms: inlinable constitutive state update must not depend on TestFunction");
            FE_THROW_IF(!allow_trial_in_updates && containsTrialFunction(*op.value.node()), InvalidArgumentException,
                        "Forms: inlinable constitutive state update must not depend on TrialFunction outside residual kernels");

            const std::size_t off = static_cast<std::size_t>(op.offset_bytes);
            FE_THROW_IF(off + sizeof(Real) > state_spec.bytes_per_qpt, InvalidArgumentException,
                        "Forms: inlinable constitutive state update offset out of bounds");
            stored.updates.push_back(MaterialStateUpdate{op.offset_bytes, op.value});
        }

        stored.dependencies = std::move(deps_by_call[&call_node]);
        expansion_stack.pop_back();

        auto [it, inserted] = memo.emplace(&call_node, std::move(stored));
        (void)inserted;
        emitUpdatesForCall(&call_node, term);

        return &it->second;
    };

    term_transform = [&](const FormExprNode& n, const IntegralTerm& term) -> std::optional<FormExpr> {
        const auto node_transform = [&](const FormExprNode& nn) -> std::optional<FormExpr> {
            return term_transform(nn, term);
        };

        if (n.type() == FormExprType::ConstitutiveOutput) {
            const auto kids = n.childrenShared();
            FE_THROW_IF(kids.size() != 1u || !kids[0], InvalidArgumentException,
                        "Forms: ConstitutiveOutput node must have exactly 1 child");

            const auto out_idx = n.constitutiveOutputIndex().value_or(0);
            FE_THROW_IF(out_idx < 0, InvalidArgumentException,
                        "Forms: ConstitutiveOutput node has negative output index");

            recordDependency(kids[0].get());
            const auto* call = get_or_inline(*kids[0], term, node_transform);
            if (!call) {
                return std::nullopt;
            }
            const auto idx = static_cast<std::size_t>(out_idx);
            FE_THROW_IF(idx >= call->outputs.size(), InvalidArgumentException,
                        "Forms: ConstitutiveOutput index out of range after inlining");
            return call->outputs[idx];
        }

        if (n.type() == FormExprType::Constitutive) {
            recordDependency(&n);
            const auto* call = get_or_inline(n, term, node_transform);
            if (!call) {
                return std::nullopt;
            }
            FE_THROW_IF(call->outputs.empty(), InvalidArgumentException,
                        "Forms: inlinable constitutive expansion produced no outputs");
            return call->outputs[0];
        }

        return std::nullopt;
    };

    for (auto* ir : irs) {
        if (ir != nullptr) ir->transformIntegrands(term_transform);
    }
}

// ============================================================================
// Numeric helpers for coefficient derivatives (finite differences)
// ============================================================================

inline constexpr Real kCoeffFDStep = 1e-7;
inline constexpr Real kCoeffFDStep2 = 1e-5;

std::array<Real, 3> fdGradScalar(const ScalarCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    std::array<Real, 3> g{0.0, 0.0, 0.0};
    const Real h = kCoeffFDStep;
    for (int d = 0; d < dim; ++d) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(d)] += h;
        xm[static_cast<std::size_t>(d)] -= h;
        const Real fp = f(xp[0], xp[1], xp[2]);
        const Real fm = f(xm[0], xm[1], xm[2]);
        g[static_cast<std::size_t>(d)] = (fp - fm) / (2.0 * h);
    }
    return g;
}

std::array<Real, 3> fdGradScalarTime(const TimeScalarCoefficient& f,
                                     const std::array<Real, 3>& x,
                                     Real t,
                                     int dim)
{
    std::array<Real, 3> g{0.0, 0.0, 0.0};
    const Real h = kCoeffFDStep;
    for (int d = 0; d < dim; ++d) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(d)] += h;
        xm[static_cast<std::size_t>(d)] -= h;
        const Real fp = f(xp[0], xp[1], xp[2], t);
        const Real fm = f(xm[0], xm[1], xm[2], t);
        g[static_cast<std::size_t>(d)] = (fp - fm) / (2.0 * h);
    }
    return g;
}

std::array<std::array<Real, 3>, 3> fdGradVector(const VectorCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    std::array<std::array<Real, 3>, 3> J{};
    const Real h = kCoeffFDStep;
    for (int d = 0; d < dim; ++d) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(d)] += h;
        xm[static_cast<std::size_t>(d)] -= h;
        const auto vp = f(xp[0], xp[1], xp[2]);
        const auto vm = f(xm[0], xm[1], xm[2]);
        for (int comp = 0; comp < 3; ++comp) {
            J[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d)] =
                (vp[static_cast<std::size_t>(comp)] - vm[static_cast<std::size_t>(comp)]) / (2.0 * h);
        }
    }
    return J;
}

std::array<Real, 27> fdGradMatrix(const MatrixCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    // Return (dA_{ij}/dx_k) stored as (i,j,k) with stride-3 packing.
    std::array<Real, 27> G{};
    const Real h = kCoeffFDStep;
    for (int k = 0; k < dim; ++k) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(k)] += h;
        xm[static_cast<std::size_t>(k)] -= h;
        const auto Ap = f(xp[0], xp[1], xp[2]);
        const auto Am = f(xm[0], xm[1], xm[2]);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                const std::size_t idx = static_cast<std::size_t>((i * 3 + j) * 3 + k);
                G[idx] = (Ap[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] -
                          Am[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) /
                         (2.0 * h);
            }
        }
    }
    return G;
}

std::array<std::array<Real, 3>, 3> fdHessScalar(const ScalarCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    std::array<std::array<Real, 3>, 3> H{};
    const Real h = kCoeffFDStep2;
    const Real f0 = f(x[0], x[1], x[2]);

    // Diagonal second derivatives
    for (int i = 0; i < dim; ++i) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(i)] += h;
        xm[static_cast<std::size_t>(i)] -= h;
        const Real fp = f(xp[0], xp[1], xp[2]);
        const Real fm = f(xm[0], xm[1], xm[2]);
        H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = (fp - 2.0 * f0 + fm) / (h * h);
    }

    // Mixed second derivatives (symmetric)
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            auto xpp = x;
            auto xpm = x;
            auto xmp = x;
            auto xmm = x;

            xpp[static_cast<std::size_t>(i)] += h;
            xpp[static_cast<std::size_t>(j)] += h;

            xpm[static_cast<std::size_t>(i)] += h;
            xpm[static_cast<std::size_t>(j)] -= h;

            xmp[static_cast<std::size_t>(i)] -= h;
            xmp[static_cast<std::size_t>(j)] += h;

            xmm[static_cast<std::size_t>(i)] -= h;
            xmm[static_cast<std::size_t>(j)] -= h;

            const Real fpp = f(xpp[0], xpp[1], xpp[2]);
            const Real fpm = f(xpm[0], xpm[1], xpm[2]);
            const Real fmp = f(xmp[0], xmp[1], xmp[2]);
            const Real fmm = f(xmm[0], xmm[1], xmm[2]);

            const Real val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = val;
            H[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = val;
        }
    }

    return H;
}

std::array<Real, 27> fdHessVector(const VectorCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    std::array<Real, 27> H{};
    const Real h = kCoeffFDStep2;
    const auto f0 = f(x[0], x[1], x[2]);

    // Diagonal second derivatives
    for (int i = 0; i < dim; ++i) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(i)] += h;
        xm[static_cast<std::size_t>(i)] -= h;
        const auto fp = f(xp[0], xp[1], xp[2]);
        const auto fm = f(xm[0], xm[1], xm[2]);
        for (int comp = 0; comp < 3; ++comp) {
            const std::size_t idx = static_cast<std::size_t>((comp * 3 + i) * 3 + i);
            H[idx] = (fp[static_cast<std::size_t>(comp)] - 2.0 * f0[static_cast<std::size_t>(comp)] +
                      fm[static_cast<std::size_t>(comp)]) /
                     (h * h);
        }
    }

    // Mixed second derivatives (symmetric)
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            auto xpp = x;
            auto xpm = x;
            auto xmp = x;
            auto xmm = x;

            xpp[static_cast<std::size_t>(i)] += h;
            xpp[static_cast<std::size_t>(j)] += h;

            xpm[static_cast<std::size_t>(i)] += h;
            xpm[static_cast<std::size_t>(j)] -= h;

            xmp[static_cast<std::size_t>(i)] -= h;
            xmp[static_cast<std::size_t>(j)] += h;

            xmm[static_cast<std::size_t>(i)] -= h;
            xmm[static_cast<std::size_t>(j)] -= h;

            const auto fpp = f(xpp[0], xpp[1], xpp[2]);
            const auto fpm = f(xpm[0], xpm[1], xpm[2]);
            const auto fmp = f(xmp[0], xmp[1], xmp[2]);
            const auto fmm = f(xmm[0], xmm[1], xmm[2]);

            for (int comp = 0; comp < 3; ++comp) {
                const Real val =
                    (fpp[static_cast<std::size_t>(comp)] - fpm[static_cast<std::size_t>(comp)] -
                     fmp[static_cast<std::size_t>(comp)] + fmm[static_cast<std::size_t>(comp)]) /
                    (4.0 * h * h);
                const std::size_t idx_ij = static_cast<std::size_t>((comp * 3 + i) * 3 + j);
                const std::size_t idx_ji = static_cast<std::size_t>((comp * 3 + j) * 3 + i);
                H[idx_ij] = val;
                H[idx_ji] = val;
            }
        }
    }

    return H;
}

std::array<std::array<Real, 3>, 3> fdHessScalarTime(const TimeScalarCoefficient& f,
                                                    const std::array<Real, 3>& x,
                                                    Real t,
                                                    int dim)
{
    std::array<std::array<Real, 3>, 3> H{};
    const Real h = kCoeffFDStep2;
    const Real f0 = f(x[0], x[1], x[2], t);

    // Diagonal second derivatives
    for (int i = 0; i < dim; ++i) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(i)] += h;
        xm[static_cast<std::size_t>(i)] -= h;
        const Real fp = f(xp[0], xp[1], xp[2], t);
        const Real fm = f(xm[0], xm[1], xm[2], t);
        H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = (fp - 2.0 * f0 + fm) / (h * h);
    }

    // Mixed second derivatives (symmetric)
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            auto xpp = x;
            auto xpm = x;
            auto xmp = x;
            auto xmm = x;

            xpp[static_cast<std::size_t>(i)] += h;
            xpp[static_cast<std::size_t>(j)] += h;

            xpm[static_cast<std::size_t>(i)] += h;
            xpm[static_cast<std::size_t>(j)] -= h;

            xmp[static_cast<std::size_t>(i)] -= h;
            xmp[static_cast<std::size_t>(j)] += h;

            xmm[static_cast<std::size_t>(i)] -= h;
            xmm[static_cast<std::size_t>(j)] -= h;

            const Real fpp = f(xpp[0], xpp[1], xpp[2], t);
            const Real fpm = f(xpm[0], xpm[1], xpm[2], t);
            const Real fmp = f(xmp[0], xmp[1], xmp[2], t);
            const Real fmm = f(xmm[0], xmm[1], xmm[2], t);

            const Real val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = val;
            H[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = val;
        }
    }

    return H;
}

Real fdDivVector(const VectorCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    const Real h = kCoeffFDStep;
    Real div = 0.0;
    for (int d = 0; d < dim; ++d) {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(d)] += h;
        xm[static_cast<std::size_t>(d)] -= h;
        const auto vp = f(xp[0], xp[1], xp[2]);
        const auto vm = f(xm[0], xm[1], xm[2]);
        div += (vp[static_cast<std::size_t>(d)] - vm[static_cast<std::size_t>(d)]) / (2.0 * h);
    }
    return div;
}

std::array<Real, 3> fdCurlVector(const VectorCoefficient& f, const std::array<Real, 3>& x, int dim)
{
    // curl(v) = [dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy]
    const Real h = kCoeffFDStep;

    auto d = [&](int comp, int wrt) -> Real {
        auto xp = x;
        auto xm = x;
        xp[static_cast<std::size_t>(wrt)] += h;
        xm[static_cast<std::size_t>(wrt)] -= h;
        const auto vp = f(xp[0], xp[1], xp[2]);
        const auto vm = f(xm[0], xm[1], xm[2]);
        return (vp[static_cast<std::size_t>(comp)] - vm[static_cast<std::size_t>(comp)]) / (2.0 * h);
    };

    std::array<Real, 3> c{0.0, 0.0, 0.0};
    if (dim == 2) {
        // 2D: return curl in z component
        c[2] = d(1, 0) - d(0, 1);
        return c;
    }

    c[0] = d(2, 1) - d(1, 2);
    c[1] = d(0, 2) - d(2, 0);
    c[2] = d(1, 0) - d(0, 1);
    return c;
}

// ============================================================================
// Spatial jets (value + spatial derivatives via chain rule)
// ============================================================================

struct EvalEnvReal;
struct EvalEnvDual;

template<typename Scalar, typename Env>
Scalar makeScalarConstant(Real value, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return value;
    } else {
        return makeDualConstant(value, env.ws->alloc());
    }
}

template<typename Scalar>
Real scalarNumericValue(const Scalar& v) noexcept
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        return v;
    } else {
        return v.value;
    }
}

template<typename Scalar, typename Env>
Scalar s_add(const Scalar& a, const Scalar& b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return a + b;
    } else {
        return add(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_sub(const Scalar& a, const Scalar& b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return a - b;
    } else {
        return sub(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_mul(const Scalar& a, const Scalar& b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return a * b;
    } else {
        return mul(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env, std::enable_if_t<!std::is_same_v<Scalar, Real>, int> = 0>
Scalar s_mul(const Scalar& a, Real b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return a * b;
    } else {
        return mul(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env, std::enable_if_t<!std::is_same_v<Scalar, Real>, int> = 0>
Scalar s_mul(Real a, const Scalar& b, const Env& env)
{
    return s_mul(b, a, env);
}

template<typename Scalar, typename Env>
Scalar s_div(const Scalar& a, const Scalar& b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return a / b;
    } else {
        return div(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env, std::enable_if_t<!std::is_same_v<Scalar, Real>, int> = 0>
Scalar s_div(const Scalar& a, Real b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return a / b;
    } else {
        return div(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_neg(const Scalar& a, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return -a;
    } else {
        return neg(a, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_sqrt(const Scalar& a, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return std::sqrt(a);
    } else {
        return sqrt(a, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_exp(const Scalar& a, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return std::exp(a);
    } else {
        return exp(a, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_log(const Scalar& a, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return std::log(a);
    } else {
        return log(a, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_pow(const Scalar& a, const Scalar& b, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return std::pow(a, b);
    } else {
        return pow(a, b, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_abs(const Scalar& a, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return std::abs(a);
    } else {
        return abs(a, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
Scalar s_sign(const Scalar& a, const Env& env)
{
    if constexpr (std::is_same_v<Scalar, Real>) {
        (void)env;
        return (a > 0.0) ? 1.0 : ((a < 0.0) ? -1.0 : 0.0);
    } else {
        return sign(a, makeDualConstant(0.0, env.ws->alloc()));
    }
}

template<typename Scalar, typename Env>
EvalValue<Scalar> zeroVector(std::size_t n, const Env& env)
{
    EvalValue<Scalar> out;
    out.kind = EvalValue<Scalar>::Kind::Vector;
    out.resizeVector(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.vectorAt(i) = makeScalarConstant<Scalar>(0.0, env);
    }
    return out;
}

template<typename Scalar, typename Env>
EvalValue<Scalar> zeroMatrix(std::size_t rows, std::size_t cols, const Env& env)
{
    EvalValue<Scalar> out;
    out.kind = EvalValue<Scalar>::Kind::Matrix;
    out.resizeMatrix(rows, cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out.matrixAt(r, c) = makeScalarConstant<Scalar>(0.0, env);
        }
    }
    return out;
}

template<typename Scalar, typename Env>
EvalValue<Scalar> zeroTensor3(std::size_t d0, std::size_t d1, std::size_t d2, const Env& env)
{
    EvalValue<Scalar> out;
    out.kind = EvalValue<Scalar>::Kind::Tensor3;
    out.resizeTensor3(d0, d1, d2);
    for (std::size_t i = 0; i < d0; ++i) {
        for (std::size_t j = 0; j < d1; ++j) {
            for (std::size_t k = 0; k < d2; ++k) {
                out.tensor3At(i, j, k) = makeScalarConstant<Scalar>(0.0, env);
            }
        }
    }
    return out;
}

template<typename Scalar>
struct SpatialJet {
    EvalValue<Scalar> value{};
    EvalValue<Scalar> grad{};
    EvalValue<Scalar> hess{};
    bool has_grad{false};
    bool has_hess{false};
};

template<typename Scalar, typename Env>
EvalValue<Scalar> addSubValue(const EvalValue<Scalar>& a, const EvalValue<Scalar>& b, bool add, const Env& env)
{
    if (!sameCategory<Scalar>(a.kind, b.kind)) {
        throw FEException("Forms: spatial jet add/sub kind mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (isVectorKind<Scalar>(a.kind) && a.vectorSize() != b.vectorSize()) {
        throw FEException("Forms: spatial jet add/sub vector size mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (isMatrixKind<Scalar>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
        throw FEException("Forms: spatial jet add/sub matrix shape mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (isTensor3Kind<Scalar>(a.kind) &&
        (a.tensor3Dim0() != b.tensor3Dim0() ||
         a.tensor3Dim1() != b.tensor3Dim1() ||
         a.tensor3Dim2() != b.tensor3Dim2())) {
        throw FEException("Forms: spatial jet add/sub tensor3 shape mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    EvalValue<Scalar> out;
    out.kind = addSubResultKind<Scalar>(a.kind, b.kind);

    if (isScalarKind<Scalar>(a.kind)) {
        out.s = add ? s_add(a.s, b.s, env) : s_sub(a.s, b.s, env);
        return out;
    }

    if (isVectorKind<Scalar>(a.kind)) {
        out.resizeVector(a.vectorSize());
        for (std::size_t d = 0; d < out.vectorSize(); ++d) {
            out.vectorAt(d) = add ? s_add(a.vectorAt(d), b.vectorAt(d), env)
                                  : s_sub(a.vectorAt(d), b.vectorAt(d), env);
        }
        return out;
    }

    if (isMatrixKind<Scalar>(a.kind)) {
        out.kind = EvalValue<Scalar>::Kind::Matrix;
        out.resizeMatrix(a.matrixRows(), a.matrixCols());
        for (std::size_t r = 0; r < out.matrixRows(); ++r) {
            for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                out.matrixAt(r, c) = add ? s_add(a.matrixAt(r, c), b.matrixAt(r, c), env)
                                         : s_sub(a.matrixAt(r, c), b.matrixAt(r, c), env);
            }
        }
        return out;
    }

    if (isTensor3Kind<Scalar>(a.kind)) {
        out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
        for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
            for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                    out.tensor3At(i, j, k) = add ? s_add(a.tensor3At(i, j, k), b.tensor3At(i, j, k), env)
                                                 : s_sub(a.tensor3At(i, j, k), b.tensor3At(i, j, k), env);
                }
            }
        }
        return out;
    }

    throw FEException("Forms: spatial jet add/sub unsupported kind",
                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
}

template<typename Scalar, typename Env>
SpatialJet<Scalar> evalSpatialJet(const FormExprNode& node,
                                  const Env& env,
                                  Side side,
                                  LocalIndex q,
                                  int order)
{
    FE_THROW_IF(order < 0 || order > 2, InvalidArgumentException,
                "Forms: spatial jet supports order 0..2 only");

    const auto& ctx = ctxForSide(env.minus, env.plus, side);
    const int dim = ctx.dimension();

    SpatialJet<Scalar> out;
    out.has_grad = (order >= 1);
    out.has_hess = (order >= 2);

    switch (node.type()) {
        case FormExprType::Constant: {
            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
            out.value.s = makeScalarConstant<Scalar>(node.constantValue().value_or(0.0), env);
            if (out.has_grad) {
                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
            }
            if (out.has_hess) {
                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
            }
            return out;
        }
        case FormExprType::Time: {
            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
            out.value.s = makeScalarConstant<Scalar>(ctx.time(), env);
            if (out.has_grad) {
                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
            }
            if (out.has_hess) {
                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
            }
            return out;
        }
	        case FormExprType::TimeStep: {
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(ctx.timeStep(), env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::CellDiameter: {
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(ctx.cellDiameter(), env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::CellVolume: {
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(ctx.cellVolume(), env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::FacetArea: {
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(ctx.facetArea(), env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::CellDomainId: {
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(static_cast<Real>(ctx.cellDomainId()), env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::Normal: {
	            out.value.kind = EvalValue<Scalar>::Kind::Vector;
	            out.value.vector_size = dim;
	            const auto n = ctx.normal(q);
	            for (int d = 0; d < 3; ++d) {
	                out.value.v[static_cast<std::size_t>(d)] =
	                    makeScalarConstant<Scalar>(n[static_cast<std::size_t>(d)], env);
	            }
	            if (out.has_grad) {
	                out.grad = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroTensor3<Scalar>(static_cast<std::size_t>(dim),
	                                               static_cast<std::size_t>(dim),
	                                               static_cast<std::size_t>(dim),
	                                               env);
	            }
	            return out;
	        }
	        case FormExprType::ParameterSymbol: {
	            const auto nm = node.symbolName();
	            FE_THROW_IF(!nm || nm->empty(), InvalidArgumentException,
	                        "Forms: ParameterSymbol node missing name (jet)");
	            const auto* get = ctx.realParameterGetter();
	            FE_THROW_IF(get == nullptr || !static_cast<bool>(*get), InvalidArgumentException,
	                        "Forms: ParameterSymbol requires a real parameter getter in AssemblyContext (jet)");
	            const auto v = (*get)(*nm);
	            FE_THROW_IF(!v.has_value(), InvalidArgumentException,
	                        "Forms: missing required parameter '" + std::string(*nm) + "' (jet)");
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(*v, env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::ParameterRef: {
	            const auto slot = node.slotIndex().value_or(0u);
	            const auto vals = ctx.jitConstants();
	            FE_THROW_IF(vals.empty(), InvalidArgumentException,
	                        "Forms: ParameterRef requires AssemblyContext::jitConstants() (jet)");
	            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
	                        "Forms: ParameterRef slot out of range (jet)");
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(vals[slot], env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::BoundaryIntegralRef: {
	            const auto slot = node.slotIndex().value_or(0u);
	            const auto vals = ctx.coupledIntegrals();
	            FE_THROW_IF(vals.empty(), InvalidArgumentException,
	                        "Forms: BoundaryIntegralRef requires coupled integrals in AssemblyContext (jet)");
	            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
	                        "Forms: BoundaryIntegralRef slot out of range (jet)");
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(vals[slot], env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::AuxiliaryStateRef: {
	            const auto slot = node.slotIndex().value_or(0u);
	            const auto vals = ctx.coupledAuxState();
	            FE_THROW_IF(vals.empty(), InvalidArgumentException,
	                        "Forms: AuxiliaryStateRef requires coupled auxiliary state in AssemblyContext (jet)");
	            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
	                        "Forms: AuxiliaryStateRef slot out of range (jet)");
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(vals[slot], env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::MaterialStateOldRef: {
	            const auto off = node.stateOffsetBytes();
	            FE_THROW_IF(!off.has_value(), InvalidArgumentException,
	                        "Forms: MaterialStateOldRef node missing offset (jet)");
	            FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
	                        "Forms: MaterialStateOldRef requires material state in AssemblyContext (jet)");
	            const auto state = ctx.materialStateOld(q);
	            const auto offset = static_cast<std::size_t>(*off);
	            FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
	                        "Forms: MaterialStateOldRef offset out of range (jet)");
	            Real v = 0.0;
	            std::memcpy(&v, state.data() + offset, sizeof(Real));
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(v, env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::MaterialStateWorkRef: {
	            const auto off = node.stateOffsetBytes();
	            FE_THROW_IF(!off.has_value(), InvalidArgumentException,
	                        "Forms: MaterialStateWorkRef node missing offset (jet)");
	            FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
	                        "Forms: MaterialStateWorkRef requires material state in AssemblyContext (jet)");
	            const auto state = ctx.materialStateWork(q);
	            const auto offset = static_cast<std::size_t>(*off);
	            FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
	                        "Forms: MaterialStateWorkRef offset out of range (jet)");
	            Real v = 0.0;
	            std::memcpy(&v, state.data() + offset, sizeof(Real));
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(v, env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::PreviousSolutionRef: {
	            const int k = node.historyIndex().value_or(1);
	            FE_THROW_IF(k <= 0, InvalidArgumentException,
	                        "Forms: PreviousSolutionRef requires k >= 1 (jet)");
	
	            const auto coeffs = ctx.previousSolutionCoefficientsRaw(k);
	            FE_THROW_IF(coeffs.empty(), InvalidArgumentException,
	                        "Forms: PreviousSolutionRef coefficients not available (jet)");
	
	            if (ctx.trialFieldType() == FieldType::Vector) {
	                const int vd = ctx.trialValueDimension();
	                FE_THROW_IF(vd <= 0 || vd > 3, InvalidArgumentException,
	                            "Forms: PreviousSolutionRef vector value_dimension must be 1..3 (jet)");
	
	                out.value.kind = EvalValue<Scalar>::Kind::Vector;
	                out.value.vector_size = vd;
	                const auto u_val = ctx.previousSolutionVectorValue(q, k);
	                for (int c = 0; c < 3; ++c) {
	                    out.value.v[static_cast<std::size_t>(c)] =
	                        makeScalarConstant<Scalar>(u_val[static_cast<std::size_t>(c)], env);
	                }
	
	                const LocalIndex n_trial = ctx.numTrialDofs();
	                FE_THROW_IF((n_trial % static_cast<LocalIndex>(vd)) != 0, InvalidArgumentException,
	                            "Forms: PreviousSolutionRef trial DOF count not divisible by value_dimension (jet)");
	                const LocalIndex dofs_per_component =
	                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
	
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
	                    out.grad.matrix_rows = vd;
	                    out.grad.matrix_cols = dim;
	                    for (int r = 0; r < 3; ++r) {
	                        for (int c = 0; c < 3; ++c) {
	                            out.grad.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
	                                makeScalarConstant<Scalar>(0.0, env);
	                        }
	                    }
	                    for (int comp = 0; comp < vd; ++comp) {
	                        const LocalIndex base = static_cast<LocalIndex>(comp) * dofs_per_component;
	                        for (LocalIndex jj = 0; jj < dofs_per_component; ++jj) {
	                            const LocalIndex j = base + jj;
	                            const auto grad_j = ctx.trialPhysicalGradient(j, q);
	                            const Real coef = coeffs[static_cast<std::size_t>(j)];
	                            for (int d = 0; d < dim; ++d) {
	                                out.grad.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d)] =
	                                    s_add(out.grad.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d)],
	                                          makeScalarConstant<Scalar>(coef * grad_j[static_cast<std::size_t>(d)], env),
	                                          env);
	                            }
	                        }
	                    }
	                }
	
	                if (out.has_hess) {
	                    out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
	                    out.hess.tensor3_dim0 = vd;
	                    out.hess.tensor3_dim1 = dim;
	                    out.hess.tensor3_dim2 = dim;
	                    for (std::size_t idx = 0; idx < out.hess.t3.size(); ++idx) {
	                        out.hess.t3[idx] = makeScalarConstant<Scalar>(0.0, env);
	                    }
	
	                    for (int comp = 0; comp < vd; ++comp) {
	                        const LocalIndex base = static_cast<LocalIndex>(comp) * dofs_per_component;
	                        for (LocalIndex jj = 0; jj < dofs_per_component; ++jj) {
	                            const LocalIndex j = base + jj;
	                            const auto Hj = ctx.trialPhysicalHessian(j, q);
	                            const Real coef = coeffs[static_cast<std::size_t>(j)];
	                            for (int i = 0; i < dim; ++i) {
	                                for (int j2 = 0; j2 < dim; ++j2) {
	                                    const std::size_t ii = static_cast<std::size_t>(i);
	                                    const std::size_t jj2 = static_cast<std::size_t>(j2);
	                                    out.hess.tensor3At(static_cast<std::size_t>(comp), ii, jj2) =
	                                        s_add(out.hess.tensor3At(static_cast<std::size_t>(comp), ii, jj2),
	                                              makeScalarConstant<Scalar>(coef * Hj[ii][jj2], env),
	                                              env);
	                                }
	                            }
	                        }
	                    }
	                }
	
	                return out;
	            }
	
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(ctx.previousSolutionValue(q, k), env);
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                out.grad.vector_size = dim;
	                for (int d = 0; d < 3; ++d) {
	                    out.grad.v[static_cast<std::size_t>(d)] = makeScalarConstant<Scalar>(0.0, env);
	                }
	                for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
	                    const auto grad_j = ctx.trialPhysicalGradient(j, q);
	                    const Real coef = coeffs[static_cast<std::size_t>(j)];
	                    for (int d = 0; d < dim; ++d) {
	                        out.grad.v[static_cast<std::size_t>(d)] =
	                            s_add(out.grad.v[static_cast<std::size_t>(d)],
	                                  makeScalarConstant<Scalar>(coef * grad_j[static_cast<std::size_t>(d)], env),
	                                  env);
	                    }
	                }
	            }
	            if (out.has_hess) {
	                out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
	                out.hess.matrix_rows = dim;
	                out.hess.matrix_cols = dim;
	                for (int r = 0; r < 3; ++r) {
	                    for (int c = 0; c < 3; ++c) {
	                        out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
	                            makeScalarConstant<Scalar>(0.0, env);
	                    }
	                }
	                for (LocalIndex j = 0; j < ctx.numTrialDofs(); ++j) {
	                    const auto Hj = ctx.trialPhysicalHessian(j, q);
	                    const Real coef = coeffs[static_cast<std::size_t>(j)];
	                    for (int r = 0; r < dim; ++r) {
	                        for (int c = 0; c < dim; ++c) {
	                            const std::size_t rr = static_cast<std::size_t>(r);
	                            const std::size_t cc = static_cast<std::size_t>(c);
	                            out.hess.m[rr][cc] =
	                                s_add(out.hess.m[rr][cc],
	                                      makeScalarConstant<Scalar>(coef * Hj[rr][cc], env),
	                                      env);
	                        }
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::Coordinate: {
	            out.value.kind = EvalValue<Scalar>::Kind::Vector;
	            out.value.vector_size = dim;
	            const auto x = ctx.physicalPoint(q);
            for (int d = 0; d < 3; ++d) {
                out.value.v[static_cast<std::size_t>(d)] = makeScalarConstant<Scalar>(x[static_cast<std::size_t>(d)], env);
            }
            if (out.has_grad) {
                out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                out.grad.matrix_rows = dim;
                out.grad.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.grad.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeScalarConstant<Scalar>((r < dim && c < dim && r == c) ? 1.0 : 0.0, env);
                    }
                }
            }
            if (out.has_hess) {
                out.hess = zeroTensor3<Scalar>(static_cast<std::size_t>(dim),
                                               static_cast<std::size_t>(dim),
                                               static_cast<std::size_t>(dim),
                                               env);
            }
            return out;
        }
        case FormExprType::Identity: {
            const int idim = node.identityDim().value_or(dim);
            FE_THROW_IF(idim <= 0, InvalidArgumentException, "Forms: identity dimension must be positive (jet)");
            out.value.kind = EvalValue<Scalar>::Kind::Matrix;
            out.value.resizeMatrix(static_cast<std::size_t>(idim), static_cast<std::size_t>(idim));
            for (int r = 0; r < idim; ++r) {
                for (int c = 0; c < idim; ++c) {
                    out.value.matrixAt(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) =
                        makeScalarConstant<Scalar>((r == c) ? 1.0 : 0.0, env);
                }
            }
            if (out.has_grad) {
                out.grad = zeroTensor3<Scalar>(static_cast<std::size_t>(idim),
                                               static_cast<std::size_t>(idim),
                                               static_cast<std::size_t>(dim),
                                               env);
            }
            if (out.has_hess) {
                throw FEException("Forms: H(identity) / second derivatives of matrix-valued constants are not supported",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            return out;
        }
        case FormExprType::Coefficient: {
            if (const auto* f = node.timeScalarCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = makeScalarConstant<Scalar>((*f)(x[0], x[1], x[2], ctx.time()), env);
                if (out.has_grad) {
                    const auto g = fdGradScalarTime(*f, x, ctx.time(), dim);
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    for (int d = 0; d < 3; ++d) {
                        out.grad.v[static_cast<std::size_t>(d)] = makeScalarConstant<Scalar>(g[static_cast<std::size_t>(d)], env);
                    }
                }
                if (out.has_hess) {
                    const auto H = fdHessScalarTime(*f, x, ctx.time(), dim);
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeScalarConstant<Scalar>(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                        }
                    }
                }
                return out;
            }
            if (const auto* f = node.scalarCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = makeScalarConstant<Scalar>((*f)(x[0], x[1], x[2]), env);
                if (out.has_grad) {
                    const auto g = fdGradScalar(*f, x, dim);
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    for (int d = 0; d < 3; ++d) {
                        out.grad.v[static_cast<std::size_t>(d)] = makeScalarConstant<Scalar>(g[static_cast<std::size_t>(d)], env);
                    }
                }
                if (out.has_hess) {
                    const auto H = fdHessScalar(*f, x, dim);
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeScalarConstant<Scalar>(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                        }
                    }
                }
                return out;
            }
            if (const auto* f = node.vectorCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto v = (*f)(x[0], x[1], x[2]);
                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.vector_size = dim;
                for (int c = 0; c < 3; ++c) {
                    out.value.v[static_cast<std::size_t>(c)] = makeScalarConstant<Scalar>(v[static_cast<std::size_t>(c)], env);
                }
                if (out.has_grad) {
                    const auto J = fdGradVector(*f, x, dim);
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.matrix_rows = dim;
                    out.grad.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.grad.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeScalarConstant<Scalar>(J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                        }
                    }
                }
                if (out.has_hess) {
                    const auto H = fdHessVector(*f, x, dim);
                    out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.hess.tensor3_dim0 = dim;
                    out.hess.tensor3_dim1 = dim;
                    out.hess.tensor3_dim2 = dim;
                    for (std::size_t idx = 0; idx < H.size(); ++idx) {
                        out.hess.t3[idx] = makeScalarConstant<Scalar>(H[idx], env);
                    }
                }
                return out;
            }
            if (const auto* f = node.matrixCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto A = (*f)(x[0], x[1], x[2]);
                out.value.kind = EvalValue<Scalar>::Kind::Matrix;
                out.value.matrix_rows = dim;
                out.value.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.value.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeScalarConstant<Scalar>(A[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                    }
                }
                if (out.has_grad) {
                    const auto G = fdGradMatrix(*f, x, dim);
                    out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.grad.tensor3_dim0 = dim;
                    out.grad.tensor3_dim1 = dim;
                    out.grad.tensor3_dim2 = dim;
                    for (std::size_t idx = 0; idx < G.size(); ++idx) {
                        out.grad.t3[idx] = makeScalarConstant<Scalar>(G[idx], env);
                    }
                }
                if (out.has_hess) {
                    throw FEException("Forms: H(matrix coefficient) is not supported (requires rank-4 tensor)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                return out;
            }
            if (const auto* f = node.tensor3Coefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto T = (*f)(x[0], x[1], x[2]);
                out.value.kind = EvalValue<Scalar>::Kind::Tensor3;
                out.value.tensor3_dim0 = dim;
                out.value.tensor3_dim1 = dim;
                out.value.tensor3_dim2 = dim;
                for (std::size_t idx = 0; idx < T.size(); ++idx) {
                    out.value.t3[idx] = makeScalarConstant<Scalar>(T[idx], env);
                }
                if (out.has_grad || out.has_hess) {
                    throw FEException("Forms: spatial derivatives of Tensor3Coefficient are not supported (requires rank-4)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                return out;
            }
            throw FEException("Forms: coefficient node has no callable (jet)",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        case FormExprType::TestFunction: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: TestFunction must be bound to a FunctionSpace (jet)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (sig->field_type == FieldType::Scalar) {
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                const Real phi = (env.test_active == side) ? ctx.basisValue(env.i, q) : 0.0;
                out.value.s = makeScalarConstant<Scalar>(phi, env);
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    const auto g = (env.test_active == side) ? ctx.physicalGradient(env.i, q)
                                                             : assembly::AssemblyContext::Vector3D{0.0, 0.0, 0.0};
                    for (int d = 0; d < 3; ++d) {
                        out.grad.v[static_cast<std::size_t>(d)] = makeScalarConstant<Scalar>(g[static_cast<std::size_t>(d)], env);
                    }
                }
                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    const auto H = (env.test_active == side) ? ctx.physicalHessian(env.i, q)
                                                             : assembly::AssemblyContext::Matrix3x3{};
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeScalarConstant<Scalar>(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                        }
                    }
                }
                return out;
            }
            if (sig->field_type == FieldType::Vector) {
                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: TestFunction vector value_dimension must be 1..3 (jet)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: vector TestFunction DOF count not divisible by value_dimension (jet)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: TestFunction vector DOF index out of range (jet)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.vector_size = vd;
                for (int c = 0; c < 3; ++c) {
                    out.value.v[static_cast<std::size_t>(c)] = makeScalarConstant<Scalar>(0.0, env);
                }
                if (env.test_active == side) {
                    out.value.v[static_cast<std::size_t>(comp)] = makeScalarConstant<Scalar>(ctx.basisValue(env.i, q), env);
                }

                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.matrix_rows = vd;
                    out.grad.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.grad.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = makeScalarConstant<Scalar>(0.0, env);
                        }
                    }
                    if (env.test_active == side) {
                        const auto g = ctx.physicalGradient(env.i, q);
                        for (int d = 0; d < dim; ++d) {
                            out.grad.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d)] =
                                makeScalarConstant<Scalar>(g[static_cast<std::size_t>(d)], env);
                        }
                    }
                }

                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.hess.tensor3_dim0 = vd;
                    out.hess.tensor3_dim1 = dim;
                    out.hess.tensor3_dim2 = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                out.hess.tensor3At(static_cast<std::size_t>(r),
                                                   static_cast<std::size_t>(i),
                                                   static_cast<std::size_t>(j)) = makeScalarConstant<Scalar>(0.0, env);
                            }
                        }
                    }
                    if (env.test_active == side) {
                        const auto H = ctx.physicalHessian(env.i, q);
                        for (int i = 0; i < dim; ++i) {
                            for (int j = 0; j < dim; ++j) {
                                out.hess.tensor3At(static_cast<std::size_t>(comp),
                                                   static_cast<std::size_t>(i),
                                                   static_cast<std::size_t>(j)) =
                                    makeScalarConstant<Scalar>(H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], env);
                            }
                        }
                    }
                }
                return out;
            }

            throw FEException("Forms: TestFunction field type not supported (jet)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::TrialFunction: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: TrialFunction must be bound to a FunctionSpace (jet)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if constexpr (std::is_same_v<Scalar, Real>) {
                if constexpr (std::is_same_v<Env, EvalEnvReal>) {
                    if (env.kind == FormKind::Residual) {
                        throw FEException("Forms: TrialFunction in residual form evaluated in variational mode (jet)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                }
            }

            if constexpr (std::is_same_v<Scalar, Real>) {
                // Variational: TrialFunction is a basis function indexed by env.j.
                if (sig->field_type == FieldType::Scalar) {
                    out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                    const Real phi = (env.trial_active == side) ? ctx.trialBasisValue(env.j, q) : 0.0;
                    out.value.s = phi;
                    if (out.has_grad) {
                        out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                        out.grad.vector_size = dim;
                        const auto g = (env.trial_active == side) ? ctx.trialPhysicalGradient(env.j, q)
                                                                  : assembly::AssemblyContext::Vector3D{0.0, 0.0, 0.0};
                        out.grad.v = g;
                        out.grad.vector_size = dim;
                    }
                    if (out.has_hess) {
                        out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                        out.hess.matrix_rows = dim;
                        out.hess.matrix_cols = dim;
                        if (env.trial_active == side) {
                            out.hess.m = ctx.trialPhysicalHessian(env.j, q);
                        }
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: TrialFunction vector value_dimension must be 1..3 (jet)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    out.value.kind = EvalValue<Scalar>::Kind::Vector;
                    out.value.vector_size = vd;
                    if (env.trial_active != side) {
                        if (out.has_grad) {
                            out.grad = zeroMatrix<Scalar>(static_cast<std::size_t>(vd), static_cast<std::size_t>(dim), env);
                        }
                        if (out.has_hess) {
                            out.hess = zeroTensor3<Scalar>(static_cast<std::size_t>(vd),
                                                           static_cast<std::size_t>(dim),
                                                           static_cast<std::size_t>(dim),
                                                           env);
                        }
                        return out;
                    }
                    const LocalIndex n_trial = ctx.numTrialDofs();
                    if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: vector TrialFunction DOF count not divisible by value_dimension (jet)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
                    const int comp = static_cast<int>(env.j / dofs_per_component);
                    if (comp < 0 || comp >= vd) {
                        throw FEException("Forms: TrialFunction vector DOF index out of range (jet)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    out.value.v[static_cast<std::size_t>(comp)] = ctx.trialBasisValue(env.j, q);

                    if (out.has_grad) {
                        out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                        out.grad.matrix_rows = vd;
                        out.grad.matrix_cols = dim;
                        const auto g = ctx.trialPhysicalGradient(env.j, q);
                        for (int d = 0; d < dim; ++d) {
                            out.grad.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d)] =
                                g[static_cast<std::size_t>(d)];
                        }
                    }

                    if (out.has_hess) {
                        out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                        out.hess.tensor3_dim0 = vd;
                        out.hess.tensor3_dim1 = dim;
                        out.hess.tensor3_dim2 = dim;
                        const auto H = ctx.trialPhysicalHessian(env.j, q);
                        for (int i = 0; i < dim; ++i) {
                            for (int j = 0; j < dim; ++j) {
                                out.hess.tensor3At(static_cast<std::size_t>(comp),
                                                   static_cast<std::size_t>(i),
                                                   static_cast<std::size_t>(j)) =
                                    H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                            }
                        }
                    }
                    return out;
                }

                throw FEException("Forms: TrialFunction field type not supported (jet)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            } else {
                // Residual/Jacobian: TrialFunction is the solution, seeded w.r.t trial DOFs.
                if (sig->field_type == FieldType::Scalar) {
                    out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                    Dual u = makeDualConstant(ctx.solutionValue(q), env.ws->alloc());
                    if (env.trial_active == side) {
                        for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                            u.deriv[j] = ctx.trialBasisValue(static_cast<LocalIndex>(j), q);
                        }
                    }
                    out.value.s = u;
                    if (out.has_grad) {
                        const auto gval = ctx.solutionGradient(q);
                        out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                        out.grad.vector_size = dim;
                        for (int d = 0; d < 3; ++d) {
                            Dual g = makeDualConstant(gval[static_cast<std::size_t>(d)], env.ws->alloc());
                            if (env.trial_active == side) {
                                for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                    const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);
                                    g.deriv[j] = grad_j[static_cast<std::size_t>(d)];
                                }
                            }
                            out.grad.v[static_cast<std::size_t>(d)] = g;
                        }
                    }
                    if (out.has_hess) {
                        const auto Hval = ctx.solutionHessian(q);
                        out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                        out.hess.matrix_rows = dim;
                        out.hess.matrix_cols = dim;
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                Dual h = makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                                if (env.trial_active == side) {
                                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                        const auto Hj = ctx.trialPhysicalHessian(static_cast<LocalIndex>(j), q);
                                        h.deriv[j] = Hj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                    }
                                }
                                out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = h;
                            }
                        }
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: TrialFunction vector value_dimension must be 1..3 (jet)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    out.value.kind = EvalValue<Scalar>::Kind::Vector;
                    out.value.vector_size = vd;
                    const auto u_val = ctx.solutionVectorValue(q);
                    for (int c = 0; c < 3; ++c) {
                        out.value.v[static_cast<std::size_t>(c)] = makeDualConstant(u_val[static_cast<std::size_t>(c)], env.ws->alloc());
                    }

                    const LocalIndex n_trial = ctx.numTrialDofs();
                    if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: TrialFunction DOF count not divisible by value_dimension (jet)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));

                    if (env.trial_active == side) {
                        for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                            const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                            out.value.v[static_cast<std::size_t>(comp_j)].deriv[j] =
                                ctx.trialBasisValue(static_cast<LocalIndex>(j), q);
                        }
                    }

                    if (out.has_grad) {
                        const auto Jval = ctx.solutionJacobian(q);
                        out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                        out.grad.matrix_rows = vd;
                        out.grad.matrix_cols = dim;
                        for (int r = 0; r < vd; ++r) {
                            for (int c = 0; c < dim; ++c) {
                                Dual Jij = makeDualConstant(Jval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)],
                                                            env.ws->alloc());
                                if (env.trial_active == side) {
                                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                        const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                                        if (comp_j == r) {
                                            const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);
                                            Jij.deriv[j] = grad_j[static_cast<std::size_t>(c)];
                                        }
                                    }
                                }
                                out.grad.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = Jij;
                            }
                        }
                    }

                    if (out.has_hess) {
                        out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                        out.hess.tensor3_dim0 = vd;
                        out.hess.tensor3_dim1 = dim;
                        out.hess.tensor3_dim2 = dim;
                        for (int comp = 0; comp < vd; ++comp) {
                            const auto Hc = ctx.solutionComponentHessian(q, comp);
                            for (int i = 0; i < dim; ++i) {
                                for (int j = 0; j < dim; ++j) {
                                    Dual h = makeDualConstant(Hc[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], env.ws->alloc());
                                    if (env.trial_active == side) {
                                        for (std::size_t dof = 0; dof < env.n_trial_dofs; ++dof) {
                                            const int comp_j = static_cast<int>(static_cast<LocalIndex>(dof) / dofs_per_component);
                                            if (comp_j == comp) {
                                                const auto Hj = ctx.trialPhysicalHessian(static_cast<LocalIndex>(dof), q);
                                                h.deriv[dof] = Hj[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                                            }
                                        }
                                    }
                                    out.hess.tensor3At(static_cast<std::size_t>(comp),
                                                       static_cast<std::size_t>(i),
                                                       static_cast<std::size_t>(j)) = h;
                                }
                            }
                        }
                    }

                    return out;
                }

                throw FEException("Forms: TrialFunction field type not supported (jet)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
        }
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: DiscreteField must be bound to a FunctionSpace (jet)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto fid = node.fieldId();
            if (!fid || *fid == INVALID_FIELD_ID) {
                throw FEException("Forms: DiscreteField node missing a valid FieldId (jet)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = makeScalarConstant<Scalar>(ctx.fieldValue(*fid, q), env);
                if (out.has_grad) {
                    const auto g = ctx.fieldGradient(*fid, q);
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    for (int d = 0; d < 3; ++d) {
                        out.grad.v[static_cast<std::size_t>(d)] = makeScalarConstant<Scalar>(g[static_cast<std::size_t>(d)], env);
                    }
                }
                if (out.has_hess) {
                    const auto H = ctx.fieldHessian(*fid, q);
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeScalarConstant<Scalar>(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                        }
                    }
                }
                return out;
            }

            if (sig->field_type == FieldType::Vector) {
                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: DiscreteField vector value_dimension must be 1..3 (jet)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto v = ctx.fieldVectorValue(*fid, q);
                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.vector_size = vd;
                for (int c = 0; c < 3; ++c) {
                    out.value.v[static_cast<std::size_t>(c)] = makeScalarConstant<Scalar>(v[static_cast<std::size_t>(c)], env);
                }
                if (out.has_grad) {
                    const auto J = ctx.fieldJacobian(*fid, q);
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.matrix_rows = vd;
                    out.grad.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.grad.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeScalarConstant<Scalar>(J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env);
                        }
                    }
                }
                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.hess.tensor3_dim0 = vd;
                    out.hess.tensor3_dim1 = dim;
                    out.hess.tensor3_dim2 = dim;
                    for (int comp = 0; comp < vd; ++comp) {
                        const auto Hc = ctx.fieldComponentHessian(*fid, q, comp);
                        for (int i = 0; i < dim; ++i) {
                            for (int j = 0; j < dim; ++j) {
                                out.hess.tensor3At(static_cast<std::size_t>(comp),
                                                   static_cast<std::size_t>(i),
                                                   static_cast<std::size_t>(j)) =
                                    makeScalarConstant<Scalar>(Hc[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], env);
                            }
                        }
                    }
                }
                return out;
            }

            throw FEException("Forms: DiscreteField field type not supported (jet)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::Negate: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::logic_error("Forms: negate must have 1 child (jet)");
            }
            auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);

            if (isScalarKind<Scalar>(a.value.kind)) {
                a.value.s = s_neg(a.value.s, env);
            } else if (isVectorKind<Scalar>(a.value.kind)) {
                for (std::size_t i = 0; i < a.value.vectorSize(); ++i) {
                    a.value.vectorAt(i) = s_neg(a.value.vectorAt(i), env);
                }
            } else if (isMatrixKind<Scalar>(a.value.kind)) {
                for (std::size_t r = 0; r < a.value.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < a.value.matrixCols(); ++c) {
                        a.value.matrixAt(r, c) = s_neg(a.value.matrixAt(r, c), env);
                    }
                }
            } else if (isTensor3Kind<Scalar>(a.value.kind)) {
                for (std::size_t i = 0; i < a.value.tensor3Dim0(); ++i) {
                    for (std::size_t j = 0; j < a.value.tensor3Dim1(); ++j) {
                        for (std::size_t k = 0; k < a.value.tensor3Dim2(); ++k) {
                            a.value.tensor3At(i, j, k) = s_neg(a.value.tensor3At(i, j, k), env);
                        }
                    }
                }
            } else {
                throw FEException("Forms: negate() kind not supported in spatial jets",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (a.has_grad) {
                if (isVectorKind<Scalar>(a.grad.kind)) {
                    for (std::size_t i = 0; i < a.grad.vectorSize(); ++i) {
                        a.grad.vectorAt(i) = s_neg(a.grad.vectorAt(i), env);
                    }
                } else if (isMatrixKind<Scalar>(a.grad.kind)) {
                    for (std::size_t r = 0; r < a.grad.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < a.grad.matrixCols(); ++c) {
                            a.grad.matrixAt(r, c) = s_neg(a.grad.matrixAt(r, c), env);
                        }
                    }
                } else if (isTensor3Kind<Scalar>(a.grad.kind)) {
                    for (std::size_t i = 0; i < a.grad.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.grad.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.grad.tensor3Dim2(); ++k) {
                                a.grad.tensor3At(i, j, k) = s_neg(a.grad.tensor3At(i, j, k), env);
                            }
                        }
                    }
                }
            }

            if (a.has_hess) {
                if (isMatrixKind<Scalar>(a.hess.kind)) {
                    for (std::size_t r = 0; r < a.hess.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < a.hess.matrixCols(); ++c) {
                            a.hess.matrixAt(r, c) = s_neg(a.hess.matrixAt(r, c), env);
                        }
                    }
                } else if (isTensor3Kind<Scalar>(a.hess.kind)) {
                    for (std::size_t i = 0; i < a.hess.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.hess.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.hess.tensor3Dim2(); ++k) {
                                a.hess.tensor3At(i, j, k) = s_neg(a.hess.tensor3At(i, j, k), env);
                            }
                        }
                    }
                }
            }

            return a;
        }
        case FormExprType::Add:
        case FormExprType::Subtract: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::logic_error("Forms: add/sub must have 2 children (jet)");
            }
            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
            const bool add = (node.type() == FormExprType::Add);
            out.value = addSubValue<Scalar>(a.value, b.value, add, env);
            if (out.has_grad) {
                out.grad = addSubValue<Scalar>(a.grad, b.grad, add, env);
            }
            if (out.has_hess) {
                out.hess = addSubValue<Scalar>(a.hess, b.hess, add, env);
            }
            return out;
        }
        case FormExprType::Multiply: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::logic_error("Forms: multiply must have 2 children (jet)");
            }
            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);

            // Scalar * Scalar
            if (isScalarKind<Scalar>(a.value.kind) && isScalarKind<Scalar>(b.value.kind)) {
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = s_mul(a.value.s, b.value.s, env);

                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    for (int d = 0; d < dim; ++d) {
                        const Scalar term1 = s_mul(a.grad.vectorAt(static_cast<std::size_t>(d)), b.value.s, env);
                        const Scalar term2 = s_mul(a.value.s, b.grad.vectorAt(static_cast<std::size_t>(d)), env);
                        out.grad.v[static_cast<std::size_t>(d)] = s_add(term1, term2, env);
                    }
                }

                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
                            Hij = s_add(Hij,
                                        s_mul(a.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                              b.value.s, env),
                                        env);
                            Hij = s_add(Hij,
                                        s_mul(b.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                              a.value.s, env),
                                        env);
                            Hij = s_add(Hij,
                                        s_mul(a.grad.vectorAt(static_cast<std::size_t>(i)),
                                              b.grad.vectorAt(static_cast<std::size_t>(j)), env),
                                        env);
                            Hij = s_add(Hij,
                                        s_mul(a.grad.vectorAt(static_cast<std::size_t>(j)),
                                              b.grad.vectorAt(static_cast<std::size_t>(i)), env),
                                        env);
                            out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Hij;
                        }
                    }
                }
                return out;
            }

            // Scalar * Vector or Vector * Scalar
            if ((isScalarKind<Scalar>(a.value.kind) && isVectorKind<Scalar>(b.value.kind)) ||
                (isVectorKind<Scalar>(a.value.kind) && isScalarKind<Scalar>(b.value.kind))) {
                const bool scalar_left = isScalarKind<Scalar>(a.value.kind);
                const auto& s = scalar_left ? a : b;
                const auto& v = scalar_left ? b : a;

                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.resizeVector(v.value.vectorSize());
                for (std::size_t comp = 0; comp < v.value.vectorSize(); ++comp) {
                    out.value.vectorAt(comp) = s_mul(s.value.s, v.value.vectorAt(comp), env);
                }

                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.resizeMatrix(v.value.vectorSize(), static_cast<std::size_t>(dim));
                    for (std::size_t comp = 0; comp < v.value.vectorSize(); ++comp) {
                        for (int d = 0; d < dim; ++d) {
                            const Scalar term1 = s_mul(v.grad.matrixAt(comp, static_cast<std::size_t>(d)), s.value.s, env);
                            const Scalar term2 = s_mul(v.value.vectorAt(comp), s.grad.vectorAt(static_cast<std::size_t>(d)), env);
                            out.grad.matrixAt(comp, static_cast<std::size_t>(d)) = s_add(term1, term2, env);
                        }
                    }
                }

                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.hess.resizeTensor3(v.value.vectorSize(),
                                           static_cast<std::size_t>(dim),
                                           static_cast<std::size_t>(dim));
                    for (std::size_t comp = 0; comp < v.value.vectorSize(); ++comp) {
                        for (int i = 0; i < dim; ++i) {
                            for (int j = 0; j < dim; ++j) {
                                Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
                                Hij = s_add(Hij,
                                            s_mul(v.hess.tensor3At(comp, static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                                  s.value.s, env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(v.grad.matrixAt(comp, static_cast<std::size_t>(i)),
                                                  s.grad.vectorAt(static_cast<std::size_t>(j)), env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(v.grad.matrixAt(comp, static_cast<std::size_t>(j)),
                                                  s.grad.vectorAt(static_cast<std::size_t>(i)), env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(v.value.vectorAt(comp),
                                                  s.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), env),
                                            env);
                                out.hess.tensor3At(comp, static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = Hij;
                            }
                        }
                    }
                }
                return out;
            }

            // Scalar * Matrix or Matrix * Scalar (order <= 1 only)
            if ((isScalarKind<Scalar>(a.value.kind) && isMatrixKind<Scalar>(b.value.kind)) ||
                (isMatrixKind<Scalar>(a.value.kind) && isScalarKind<Scalar>(b.value.kind))) {
                if (out.has_hess) {
                    throw FEException("Forms: Hessian of matrix-valued product is not supported (requires rank-4)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                const bool scalar_left = isScalarKind<Scalar>(a.value.kind);
                const auto& s = scalar_left ? a : b;
                const auto& A = scalar_left ? b : a;

                out.value.kind = EvalValue<Scalar>::Kind::Matrix;
                out.value.resizeMatrix(A.value.matrixRows(), A.value.matrixCols());
                for (std::size_t r = 0; r < A.value.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < A.value.matrixCols(); ++c) {
                        out.value.matrixAt(r, c) = s_mul(s.value.s, A.value.matrixAt(r, c), env);
                    }
                }

                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.grad.resizeTensor3(A.value.matrixRows(), A.value.matrixCols(), static_cast<std::size_t>(dim));
                    for (std::size_t r = 0; r < A.value.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < A.value.matrixCols(); ++c) {
                            for (int d = 0; d < dim; ++d) {
                                const Scalar term1 = s_mul(A.grad.tensor3At(r, c, static_cast<std::size_t>(d)), s.value.s, env);
                                const Scalar term2 = s_mul(A.value.matrixAt(r, c), s.grad.vectorAt(static_cast<std::size_t>(d)), env);
                                out.grad.tensor3At(r, c, static_cast<std::size_t>(d)) = s_add(term1, term2, env);
                            }
                        }
                    }
                }
                return out;
            }

            // Matrix * Vector (order <= 1 only)
            if (isMatrixKind<Scalar>(a.value.kind) && isVectorKind<Scalar>(b.value.kind)) {
                if (out.has_hess) {
                    throw FEException("Forms: Hessian of matrix-vector product is not supported in spatial jets",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                const auto rows = a.value.matrixRows();
                const auto cols = a.value.matrixCols();
                if (b.value.vectorSize() != cols) {
                    throw FEException("Forms: spatial jet matrix-vector multiplication shape mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.resizeVector(rows);
                for (std::size_t r = 0; r < rows; ++r) {
                    Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                    for (std::size_t c = 0; c < cols; ++c) {
                        sum = s_add(sum, s_mul(a.value.matrixAt(r, c), b.value.vectorAt(c), env), env);
                    }
                    out.value.vectorAt(r) = sum;
                }
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.resizeMatrix(rows, static_cast<std::size_t>(dim));
                    for (std::size_t r = 0; r < rows; ++r) {
                        for (int d = 0; d < dim; ++d) {
                            Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                            for (std::size_t c = 0; c < cols; ++c) {
                                sum = s_add(sum, s_mul(a.grad.tensor3At(r, c, static_cast<std::size_t>(d)), b.value.vectorAt(c), env), env);
                                sum = s_add(sum, s_mul(a.value.matrixAt(r, c), b.grad.matrixAt(c, static_cast<std::size_t>(d)), env), env);
                            }
                            out.grad.matrixAt(r, static_cast<std::size_t>(d)) = sum;
                        }
                    }
                }
                return out;
            }

            // Vector * Matrix (order <= 1 only)
            if (isVectorKind<Scalar>(a.value.kind) && isMatrixKind<Scalar>(b.value.kind)) {
                if (out.has_hess) {
                    throw FEException("Forms: Hessian of vector-matrix product is not supported in spatial jets",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                const auto rows = b.value.matrixRows();
                const auto cols = b.value.matrixCols();
                if (a.value.vectorSize() != rows) {
                    throw FEException("Forms: spatial jet vector-matrix multiplication shape mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.resizeVector(cols);
                for (std::size_t c = 0; c < cols; ++c) {
                    Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                    for (std::size_t r = 0; r < rows; ++r) {
                        sum = s_add(sum, s_mul(a.value.vectorAt(r), b.value.matrixAt(r, c), env), env);
                    }
                    out.value.vectorAt(c) = sum;
                }
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.resizeMatrix(cols, static_cast<std::size_t>(dim));
                    for (std::size_t c = 0; c < cols; ++c) {
                        for (int d = 0; d < dim; ++d) {
                            Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                            for (std::size_t r = 0; r < rows; ++r) {
                                sum = s_add(sum, s_mul(a.grad.matrixAt(r, static_cast<std::size_t>(d)), b.value.matrixAt(r, c), env), env);
                                sum = s_add(sum, s_mul(a.value.vectorAt(r), b.grad.tensor3At(r, c, static_cast<std::size_t>(d)), env), env);
                            }
                            out.grad.matrixAt(c, static_cast<std::size_t>(d)) = sum;
                        }
                    }
                }
                return out;
            }

            // Matrix * Matrix (order <= 1 only)
            if (isMatrixKind<Scalar>(a.value.kind) && isMatrixKind<Scalar>(b.value.kind)) {
                if (out.has_hess) {
                    throw FEException("Forms: Hessian of matrix-matrix product is not supported (requires rank-4)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                const auto rows = a.value.matrixRows();
                const auto inner_dim = a.value.matrixCols();
                const auto cols = b.value.matrixCols();
                if (b.value.matrixRows() != inner_dim) {
                    throw FEException("Forms: spatial jet matrix-matrix multiplication shape mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.value.kind = EvalValue<Scalar>::Kind::Matrix;
                out.value.resizeMatrix(rows, cols);
                for (std::size_t r = 0; r < rows; ++r) {
                    for (std::size_t c = 0; c < cols; ++c) {
                        Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                        for (std::size_t k = 0; k < inner_dim; ++k) {
                            sum = s_add(sum, s_mul(a.value.matrixAt(r, k), b.value.matrixAt(k, c), env), env);
                        }
                        out.value.matrixAt(r, c) = sum;
                    }
                }
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.grad.resizeTensor3(rows, cols, static_cast<std::size_t>(dim));
                    for (std::size_t r = 0; r < rows; ++r) {
                        for (std::size_t c = 0; c < cols; ++c) {
                            for (int d = 0; d < dim; ++d) {
                                Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                                for (std::size_t k = 0; k < inner_dim; ++k) {
                                    sum = s_add(sum, s_mul(a.grad.tensor3At(r, k, static_cast<std::size_t>(d)), b.value.matrixAt(k, c), env), env);
                                    sum = s_add(sum, s_mul(a.value.matrixAt(r, k), b.grad.tensor3At(k, c, static_cast<std::size_t>(d)), env), env);
                                }
                                out.grad.tensor3At(r, c, static_cast<std::size_t>(d)) = sum;
                            }
                        }
                    }
                }
                return out;
            }

            throw FEException("Forms: unsupported multiplication kinds in spatial jets",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::Divide: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::logic_error("Forms: divide must have 2 children (jet)");
            }
            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
            if (!isScalarKind<Scalar>(b.value.kind)) {
                throw FEException("Forms: spatial jet division denominator must be scalar",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            // Reciprocal r = 1/b
            SpatialJet<Scalar> r;
            r.has_grad = out.has_grad;
            r.has_hess = out.has_hess;
            r.value.kind = EvalValue<Scalar>::Kind::Scalar;
            r.value.s = s_div(makeScalarConstant<Scalar>(1.0, env), b.value.s, env);
            if (out.has_grad) {
                r.grad.kind = EvalValue<Scalar>::Kind::Vector;
                r.grad.vector_size = dim;
                const Scalar inv_b2 = s_div(makeScalarConstant<Scalar>(1.0, env),
                                            s_mul(b.value.s, b.value.s, env),
                                            env);
                for (int d = 0; d < dim; ++d) {
                    r.grad.v[static_cast<std::size_t>(d)] =
                        s_neg(s_mul(b.grad.vectorAt(static_cast<std::size_t>(d)), inv_b2, env), env);
                }
            }
            if (out.has_hess) {
                r.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                r.hess.matrix_rows = dim;
                r.hess.matrix_cols = dim;
                const Scalar inv_b2 = s_div(makeScalarConstant<Scalar>(1.0, env),
                                            s_mul(b.value.s, b.value.s, env),
                                            env);
                const Scalar inv_b3 = s_div(inv_b2, b.value.s, env);
                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        const Scalar term1 = s_mul(2.0,
                                                   s_mul(b.grad.vectorAt(static_cast<std::size_t>(i)),
                                                         b.grad.vectorAt(static_cast<std::size_t>(j)), env),
                                                   env);
                        const Scalar t1 = s_mul(term1, inv_b3, env);
                        const Scalar t2 = s_mul(b.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                                inv_b2, env);
                        r.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = s_sub(t1, t2, env);
                    }
                }
            }

            // Multiply numerator by reciprocal.
            if (isScalarKind<Scalar>(a.value.kind)) {
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = s_mul(a.value.s, r.value.s, env);
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    for (int d = 0; d < dim; ++d) {
                        const Scalar term1 = s_mul(a.grad.vectorAt(static_cast<std::size_t>(d)), r.value.s, env);
                        const Scalar term2 = s_mul(a.value.s, r.grad.vectorAt(static_cast<std::size_t>(d)), env);
                        out.grad.v[static_cast<std::size_t>(d)] = s_add(term1, term2, env);
                    }
                }
                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
                            Hij = s_add(Hij,
                                        s_mul(a.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                              r.value.s, env),
                                        env);
                            Hij = s_add(Hij,
                                        s_mul(r.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                              a.value.s, env),
                                        env);
                            Hij = s_add(Hij,
                                        s_mul(a.grad.vectorAt(static_cast<std::size_t>(i)),
                                              r.grad.vectorAt(static_cast<std::size_t>(j)), env),
                                        env);
                            Hij = s_add(Hij,
                                        s_mul(a.grad.vectorAt(static_cast<std::size_t>(j)),
                                              r.grad.vectorAt(static_cast<std::size_t>(i)), env),
                                        env);
                            out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Hij;
                        }
                    }
                }
                return out;
            }
            if (isVectorKind<Scalar>(a.value.kind)) {
                out.value.kind = EvalValue<Scalar>::Kind::Vector;
                out.value.resizeVector(a.value.vectorSize());
                for (std::size_t comp = 0; comp < a.value.vectorSize(); ++comp) {
                    out.value.vectorAt(comp) = s_mul(a.value.vectorAt(comp), r.value.s, env);
                }
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.grad.resizeMatrix(a.value.vectorSize(), static_cast<std::size_t>(dim));
                    for (std::size_t comp = 0; comp < a.value.vectorSize(); ++comp) {
                        for (int d = 0; d < dim; ++d) {
                            const Scalar term1 = s_mul(a.grad.matrixAt(comp, static_cast<std::size_t>(d)), r.value.s, env);
                            const Scalar term2 = s_mul(a.value.vectorAt(comp), r.grad.vectorAt(static_cast<std::size_t>(d)), env);
                            out.grad.matrixAt(comp, static_cast<std::size_t>(d)) = s_add(term1, term2, env);
                        }
                    }
                }
                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.hess.resizeTensor3(a.value.vectorSize(),
                                           static_cast<std::size_t>(dim),
                                           static_cast<std::size_t>(dim));
                    for (std::size_t comp = 0; comp < a.value.vectorSize(); ++comp) {
                        for (int i = 0; i < dim; ++i) {
                            for (int j = 0; j < dim; ++j) {
                                Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
                                Hij = s_add(Hij,
                                            s_mul(a.hess.tensor3At(comp, static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                                  r.value.s, env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(a.grad.matrixAt(comp, static_cast<std::size_t>(i)),
                                                  r.grad.vectorAt(static_cast<std::size_t>(j)), env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(a.grad.matrixAt(comp, static_cast<std::size_t>(j)),
                                                  r.grad.vectorAt(static_cast<std::size_t>(i)), env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(a.value.vectorAt(comp),
                                                  r.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), env),
                                            env);
                                out.hess.tensor3At(comp, static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = Hij;
                            }
                        }
                    }
                }
                return out;
            }

            if (isMatrixKind<Scalar>(a.value.kind)) {
                if (out.has_hess) {
                    throw FEException("Forms: Hessian of matrix/scalar division is not supported (requires rank-4)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                out.value.kind = EvalValue<Scalar>::Kind::Matrix;
                out.value.resizeMatrix(a.value.matrixRows(), a.value.matrixCols());
                for (std::size_t r0 = 0; r0 < a.value.matrixRows(); ++r0) {
                    for (std::size_t c0 = 0; c0 < a.value.matrixCols(); ++c0) {
                        out.value.matrixAt(r0, c0) = s_mul(a.value.matrixAt(r0, c0), r.value.s, env);
                    }
                }
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
                    out.grad.resizeTensor3(a.value.matrixRows(), a.value.matrixCols(), static_cast<std::size_t>(dim));
                    for (std::size_t r0 = 0; r0 < a.value.matrixRows(); ++r0) {
                        for (std::size_t c0 = 0; c0 < a.value.matrixCols(); ++c0) {
                            for (int d = 0; d < dim; ++d) {
                                const Scalar term1 = s_mul(a.grad.tensor3At(r0, c0, static_cast<std::size_t>(d)), r.value.s, env);
                                const Scalar term2 = s_mul(a.value.matrixAt(r0, c0), r.grad.vectorAt(static_cast<std::size_t>(d)), env);
                                out.grad.tensor3At(r0, c0, static_cast<std::size_t>(d)) = s_add(term1, term2, env);
                            }
                        }
                    }
                }
                return out;
            }

	            throw FEException("Forms: unsupported division numerator kind in spatial jets",
	                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
	        case FormExprType::InnerProduct:
	        case FormExprType::DoubleContraction: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 2u || !kids[0] || !kids[1]) {
	                throw std::logic_error("Forms: inner/doubleContraction must have 2 children (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
	
	            // Scalar · Scalar
	            if (isScalarKind<Scalar>(a.value.kind) && isScalarKind<Scalar>(b.value.kind)) {
	                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	                out.value.s = s_mul(a.value.s, b.value.s, env);
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                    out.grad.vector_size = dim;
	                    for (int d = 0; d < dim; ++d) {
	                        const Scalar term1 = s_mul(a.grad.vectorAt(static_cast<std::size_t>(d)), b.value.s, env);
	                        const Scalar term2 = s_mul(a.value.s, b.grad.vectorAt(static_cast<std::size_t>(d)), env);
	                        out.grad.v[static_cast<std::size_t>(d)] = s_add(term1, term2, env);
	                    }
	                }
	                if (out.has_hess) {
	                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
	                    out.hess.matrix_rows = dim;
	                    out.hess.matrix_cols = dim;
	                    for (int i = 0; i < dim; ++i) {
	                        for (int j = 0; j < dim; ++j) {
	                            Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
	                            Hij = s_add(Hij,
	                                        s_mul(a.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
	                                              b.value.s, env),
	                                        env);
	                            Hij = s_add(Hij,
	                                        s_mul(b.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
	                                              a.value.s, env),
	                                        env);
	                            Hij = s_add(Hij,
	                                        s_mul(a.grad.vectorAt(static_cast<std::size_t>(i)),
	                                              b.grad.vectorAt(static_cast<std::size_t>(j)), env),
	                                        env);
	                            Hij = s_add(Hij,
	                                        s_mul(a.grad.vectorAt(static_cast<std::size_t>(j)),
	                                              b.grad.vectorAt(static_cast<std::size_t>(i)), env),
	                                        env);
	                            out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Hij;
	                        }
	                    }
	                }
	                return out;
	            }
	
	            // Vector · Vector
	            if (isVectorKind<Scalar>(a.value.kind) && isVectorKind<Scalar>(b.value.kind)) {
	                if (a.value.vectorSize() != b.value.vectorSize()) {
	                    throw FEException("Forms: inner(vector,vector) size mismatch (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                const auto n = a.value.vectorSize();
	                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	                Scalar sum = makeScalarConstant<Scalar>(0.0, env);
	                for (std::size_t k = 0; k < n; ++k) {
	                    sum = s_add(sum, s_mul(a.value.vectorAt(k), b.value.vectorAt(k), env), env);
	                }
	                out.value.s = sum;
	
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                    out.grad.vector_size = dim;
	                    for (int d = 0; d < dim; ++d) {
	                        Scalar gd = makeScalarConstant<Scalar>(0.0, env);
	                        for (std::size_t k = 0; k < n; ++k) {
	                            gd = s_add(gd,
	                                       s_mul(a.grad.matrixAt(k, static_cast<std::size_t>(d)), b.value.vectorAt(k), env),
	                                       env);
	                            gd = s_add(gd,
	                                       s_mul(a.value.vectorAt(k), b.grad.matrixAt(k, static_cast<std::size_t>(d)), env),
	                                       env);
	                        }
	                        out.grad.v[static_cast<std::size_t>(d)] = gd;
	                    }
	                }
	
	                if (out.has_hess) {
	                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
	                    out.hess.matrix_rows = dim;
	                    out.hess.matrix_cols = dim;
	                    for (int i = 0; i < dim; ++i) {
	                        for (int j = 0; j < dim; ++j) {
	                            Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
	                            for (std::size_t k = 0; k < n; ++k) {
	                                Hij = s_add(Hij,
	                                            s_mul(a.hess.tensor3At(k, static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
	                                                  b.value.vectorAt(k), env),
	                                            env);
	                                Hij = s_add(Hij,
	                                            s_mul(b.hess.tensor3At(k, static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
	                                                  a.value.vectorAt(k), env),
	                                            env);
	                                Hij = s_add(Hij,
	                                            s_mul(a.grad.matrixAt(k, static_cast<std::size_t>(i)),
	                                                  b.grad.matrixAt(k, static_cast<std::size_t>(j)), env),
	                                            env);
	                                Hij = s_add(Hij,
	                                            s_mul(a.grad.matrixAt(k, static_cast<std::size_t>(j)),
	                                                  b.grad.matrixAt(k, static_cast<std::size_t>(i)), env),
	                                            env);
	                            }
	                            out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Hij;
	                        }
	                    }
	                }
	                return out;
	            }
	
	            // Matrix : Matrix (Frobenius)
	            if (isMatrixKind<Scalar>(a.value.kind) && isMatrixKind<Scalar>(b.value.kind)) {
	                if (a.value.matrixRows() != b.value.matrixRows() || a.value.matrixCols() != b.value.matrixCols()) {
	                    throw FEException("Forms: inner(matrix,matrix) shape mismatch (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                if (out.has_hess) {
	                    throw FEException("Forms: Hessian of inner(matrix,matrix) is not supported (requires rank-4)",
	                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	                }
	                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	                Scalar sum = makeScalarConstant<Scalar>(0.0, env);
	                for (std::size_t r = 0; r < a.value.matrixRows(); ++r) {
	                    for (std::size_t c = 0; c < a.value.matrixCols(); ++c) {
	                        sum = s_add(sum, s_mul(a.value.matrixAt(r, c), b.value.matrixAt(r, c), env), env);
	                    }
	                }
	                out.value.s = sum;
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                    out.grad.vector_size = dim;
	                    for (int d = 0; d < dim; ++d) {
	                        Scalar gd = makeScalarConstant<Scalar>(0.0, env);
	                        for (std::size_t r = 0; r < a.value.matrixRows(); ++r) {
	                            for (std::size_t c = 0; c < a.value.matrixCols(); ++c) {
	                                gd = s_add(gd,
	                                           s_mul(a.grad.tensor3At(r, c, static_cast<std::size_t>(d)),
	                                                 b.value.matrixAt(r, c), env),
	                                           env);
	                                gd = s_add(gd,
	                                           s_mul(a.value.matrixAt(r, c),
	                                                 b.grad.tensor3At(r, c, static_cast<std::size_t>(d)), env),
	                                           env);
	                            }
	                        }
	                        out.grad.v[static_cast<std::size_t>(d)] = gd;
	                    }
	                }
	                return out;
	            }
	
	            throw FEException("Forms: inner/doubleContraction operand kind not supported in spatial jets",
	                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
	        case FormExprType::Power: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 2u || !kids[0] || !kids[1]) {
	                throw std::logic_error("Forms: pow must have 2 children (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
	            if (!isScalarKind<Scalar>(a.value.kind) || !isScalarKind<Scalar>(b.value.kind)) {
	                throw FEException("Forms: pow() in spatial jets expects scalar arguments",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = s_pow(a.value.s, b.value.s, env);
	
	            if (out.has_grad || out.has_hess) {
	                const Scalar loga = s_log(a.value.s, env);
	                const Scalar inv_a = s_div(makeScalarConstant<Scalar>(1.0, env), a.value.s, env);
	
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                    out.grad.vector_size = dim;
	                    for (int d = 0; d < dim; ++d) {
	                        const Scalar g1 = s_mul(b.grad.vectorAt(static_cast<std::size_t>(d)), loga, env);
	                        const Scalar g2 = s_mul(b.value.s,
	                                                s_mul(a.grad.vectorAt(static_cast<std::size_t>(d)), inv_a, env),
	                                                env);
	                        const Scalar gd = s_add(g1, g2, env);
	                        out.grad.v[static_cast<std::size_t>(d)] = s_mul(out.value.s, gd, env);
	                    }
	                }
	
	                if (out.has_hess) {
	                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
	                    out.hess.matrix_rows = dim;
	                    out.hess.matrix_cols = dim;
	
	                    const Scalar inv_a2 = s_mul(inv_a, inv_a, env);
	                    std::array<Scalar, 3> grad_l{};
	                    for (int d = 0; d < dim; ++d) {
	                        grad_l[static_cast<std::size_t>(d)] =
	                            s_mul(a.grad.vectorAt(static_cast<std::size_t>(d)), inv_a, env);
	                    }
	
	                    for (int i = 0; i < dim; ++i) {
	                        for (int j = 0; j < dim; ++j) {
	                            const auto ii = static_cast<std::size_t>(i);
	                            const auto jj = static_cast<std::size_t>(j);
	
	                            const Scalar Hess_l =
	                                s_sub(s_mul(a.hess.matrixAt(ii, jj), inv_a, env),
	                                      s_mul(s_mul(a.grad.vectorAt(ii), a.grad.vectorAt(jj), env), inv_a2, env),
	                                      env);
	
	                            const Scalar Hess_b = s_mul(b.hess.matrixAt(ii, jj), loga, env);
	                            const Scalar cross1 = s_mul(b.grad.vectorAt(ii), grad_l[jj], env);
	                            const Scalar cross2 = s_mul(b.grad.vectorAt(jj), grad_l[ii], env);
	                            const Scalar Hess_g =
	                                s_add(s_add(Hess_b, s_add(cross1, cross2, env), env),
	                                      s_mul(b.value.s, Hess_l, env),
	                                      env);
	
	                            const Scalar gi =
	                                s_add(s_mul(b.grad.vectorAt(ii), loga, env),
	                                      s_mul(b.value.s, grad_l[ii], env),
	                                      env);
	                            const Scalar gj =
	                                s_add(s_mul(b.grad.vectorAt(jj), loga, env),
	                                      s_mul(b.value.s, grad_l[jj], env),
	                                      env);
	
	                            out.hess.m[ii][jj] =
	                                s_mul(out.value.s, s_add(Hess_g, s_mul(gi, gj, env), env), env);
	                        }
	                    }
	                }
	            }
	
	            return out;
	        }
	        case FormExprType::Minimum:
	        case FormExprType::Maximum: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 2u || !kids[0] || !kids[1]) {
	                throw std::logic_error("Forms: min/max must have 2 children (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
	            if (!isScalarKind<Scalar>(a.value.kind) || !isScalarKind<Scalar>(b.value.kind)) {
	                throw FEException("Forms: min/max in spatial jets expects scalar arguments",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const Real av = scalarNumericValue(a.value.s);
	            const Real bv = scalarNumericValue(b.value.s);
	            const bool pick_a = (node.type() == FormExprType::Minimum) ? (av <= bv) : (av >= bv);
	            return pick_a ? a : b;
	        }
	        case FormExprType::Less:
	        case FormExprType::LessEqual:
	        case FormExprType::Greater:
	        case FormExprType::GreaterEqual:
	        case FormExprType::Equal:
	        case FormExprType::NotEqual: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 2u || !kids[0] || !kids[1]) {
	                throw std::logic_error("Forms: comparison must have 2 children (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, 0);
	            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, 0);
	            if (!isScalarKind<Scalar>(a.value.kind) || !isScalarKind<Scalar>(b.value.kind)) {
	                throw FEException("Forms: comparisons in spatial jets expect scalar arguments",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const Real av = scalarNumericValue(a.value.s);
	            const Real bv = scalarNumericValue(b.value.s);
	            bool truth = false;
	            switch (node.type()) {
	                case FormExprType::Less: truth = (av < bv); break;
	                case FormExprType::LessEqual: truth = (av <= bv); break;
	                case FormExprType::Greater: truth = (av > bv); break;
	                case FormExprType::GreaterEqual: truth = (av >= bv); break;
	                case FormExprType::Equal: truth = (av == bv); break;
	                case FormExprType::NotEqual: truth = (av != bv); break;
	                default: break;
	            }
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = makeScalarConstant<Scalar>(truth ? 1.0 : 0.0, env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::Conditional: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
	                throw std::logic_error("Forms: conditional must have 3 children (jet)");
	            }
	            const auto cond = evalSpatialJet<Scalar>(*kids[0], env, side, q, 0);
	            if (!isScalarKind<Scalar>(cond.value.kind)) {
	                throw FEException("Forms: conditional condition must be scalar (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
	            const auto b = evalSpatialJet<Scalar>(*kids[2], env, side, q, order);
	            const bool take_a = scalarNumericValue(cond.value.s) > 0.0;
	            SpatialJet<Scalar> res = take_a ? a : b;
	            if (a.value.kind == b.value.kind) {
	                return res;
	            }
	            if (isMatrixKind<Scalar>(a.value.kind) && isMatrixKind<Scalar>(b.value.kind)) {
	                res.value.kind = EvalValue<Scalar>::Kind::Matrix;
	                return res;
	            }
	            return res;
	        }
	        case FormExprType::AsVector: {
	            const auto kids = node.childrenShared();
	            if (kids.empty()) {
	                throw FEException("Forms: as_vector expects at least 1 scalar component (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            out.value.kind = EvalValue<Scalar>::Kind::Vector;
	            out.value.resizeVector(kids.size());
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Matrix;
	                out.grad.resizeMatrix(kids.size(), static_cast<std::size_t>(dim));
	            }
	            if (out.has_hess) {
	                out.hess.kind = EvalValue<Scalar>::Kind::Tensor3;
	                out.hess.resizeTensor3(kids.size(),
	                                       static_cast<std::size_t>(dim),
	                                       static_cast<std::size_t>(dim));
	            }
	            for (std::size_t c = 0; c < kids.size(); ++c) {
	                if (!kids[c]) throw std::logic_error("Forms: as_vector has null child (jet)");
	                const auto v = evalSpatialJet<Scalar>(*kids[c], env, side, q, order);
	                if (!isScalarKind<Scalar>(v.value.kind)) {
	                    throw FEException("Forms: as_vector components must be scalar (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                out.value.vectorAt(c) = v.value.s;
	                if (out.has_grad) {
	                    for (int d = 0; d < dim; ++d) {
	                        out.grad.matrixAt(c, static_cast<std::size_t>(d)) = v.grad.vectorAt(static_cast<std::size_t>(d));
	                    }
	                }
	                if (out.has_hess) {
	                    for (int i = 0; i < dim; ++i) {
	                        for (int j = 0; j < dim; ++j) {
	                            out.hess.tensor3At(c, static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
	                                v.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
	                        }
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::AsTensor: {
	            if (out.has_hess) {
	                throw FEException("Forms: Hessian of as_tensor is not supported (requires rank-4)",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }
	            const auto rows = node.tensorRows().value_or(0);
	            const auto cols = node.tensorCols().value_or(0);
	            if (rows <= 0 || cols <= 0) {
	                throw FEException("Forms: as_tensor requires explicit shape with rows,cols >= 1 (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const auto kids = node.childrenShared();
	            if (kids.size() != static_cast<std::size_t>(rows * cols)) {
	                throw FEException("Forms: as_tensor child count does not match rows*cols (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            out.value.kind = EvalValue<Scalar>::Kind::Matrix;
	            out.value.resizeMatrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
	                out.grad.resizeTensor3(static_cast<std::size_t>(rows),
	                                       static_cast<std::size_t>(cols),
	                                       static_cast<std::size_t>(dim));
	            }
	            for (int r = 0; r < rows; ++r) {
	                for (int c = 0; c < cols; ++c) {
	                    const auto idx = static_cast<std::size_t>(r * cols + c);
	                    if (!kids[idx]) throw std::logic_error("Forms: as_tensor has null child (jet)");
	                    const auto v = evalSpatialJet<Scalar>(*kids[idx], env, side, q, order);
	                    if (!isScalarKind<Scalar>(v.value.kind)) {
	                        throw FEException("Forms: as_tensor entries must be scalar (jet)",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	                    out.value.matrixAt(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) = v.value.s;
	                    if (out.has_grad) {
	                        for (int d = 0; d < dim; ++d) {
	                            out.grad.tensor3At(static_cast<std::size_t>(r),
	                                               static_cast<std::size_t>(c),
	                                               static_cast<std::size_t>(d)) =
	                                v.grad.vectorAt(static_cast<std::size_t>(d));
	                        }
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::Component: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: component must have 1 child (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            const int i = node.componentIndex0().value_or(0);
	            const int j = node.componentIndex1().value_or(-1);
	            if (isScalarKind<Scalar>(a.value.kind)) {
	                if (i != 0 || j >= 0) {
	                    throw FEException("Forms: component() invalid indices for scalar (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                return a;
	            }
	            if (isVectorKind<Scalar>(a.value.kind)) {
	                if (j >= 0) {
	                    throw FEException("Forms: component(v,i,j) invalid for vector (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                const auto n = a.value.vectorSize();
	                if (i < 0 || static_cast<std::size_t>(i) >= n) {
	                    throw FEException("Forms: component(v,i) index out of range (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	                out.value.s = a.value.vectorAt(static_cast<std::size_t>(i));
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                    out.grad.vector_size = dim;
	                    for (int d = 0; d < dim; ++d) {
	                        out.grad.v[static_cast<std::size_t>(d)] =
	                            a.grad.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(d));
	                    }
	                }
	                if (out.has_hess) {
	                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
	                    out.hess.matrix_rows = dim;
	                    out.hess.matrix_cols = dim;
	                    for (int r = 0; r < dim; ++r) {
	                        for (int c = 0; c < dim; ++c) {
	                            out.hess.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
	                                a.hess.tensor3At(static_cast<std::size_t>(i),
	                                                 static_cast<std::size_t>(r),
	                                                 static_cast<std::size_t>(c));
	                        }
	                    }
	                }
	                return out;
	            }
	            if (isMatrixKind<Scalar>(a.value.kind)) {
	                if (j < 0) {
	                    throw FEException("Forms: component(A,i) missing column index for matrix (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                if (out.has_hess) {
	                    throw FEException("Forms: Hessian of component(matrix,i,j) is not supported (requires rank-4)",
	                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	                }
	                const auto rows = a.value.matrixRows();
	                const auto cols = a.value.matrixCols();
	                if (i < 0 || static_cast<std::size_t>(i) >= rows || j < 0 || static_cast<std::size_t>(j) >= cols) {
	                    throw FEException("Forms: component(A,i,j) index out of range (jet)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	                out.value.s = a.value.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
	                if (out.has_grad) {
	                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                    out.grad.vector_size = dim;
	                    for (int d = 0; d < dim; ++d) {
	                        out.grad.v[static_cast<std::size_t>(d)] =
	                            a.grad.tensor3At(static_cast<std::size_t>(i),
	                                             static_cast<std::size_t>(j),
	                                             static_cast<std::size_t>(d));
	                    }
	                }
	                return out;
	            }
	            throw FEException("Forms: component() operand kind not supported in spatial jets",
	                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
	        case FormExprType::Transpose: {
	            if (out.has_hess) {
	                throw FEException("Forms: Hessian of transpose(A) is not supported (requires rank-4)",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: transpose must have 1 child (jet)");
	            }
	            const auto A = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            if (!isMatrixKind<Scalar>(A.value.kind)) {
	                throw FEException("Forms: transpose() expects a matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const auto rows = A.value.matrixRows();
	            const auto cols = A.value.matrixCols();
	            out.value.kind = (rows == cols) ? A.value.kind : EvalValue<Scalar>::Kind::Matrix;
	            out.value.resizeMatrix(cols, rows);
	            for (std::size_t r = 0; r < cols; ++r) {
	                for (std::size_t c = 0; c < rows; ++c) {
	                    out.value.matrixAt(r, c) = A.value.matrixAt(c, r);
	                }
	            }
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
	                out.grad.resizeTensor3(cols, rows, static_cast<std::size_t>(dim));
	                for (std::size_t r = 0; r < cols; ++r) {
	                    for (std::size_t c = 0; c < rows; ++c) {
	                        for (int d = 0; d < dim; ++d) {
	                            out.grad.tensor3At(r, c, static_cast<std::size_t>(d)) =
	                                A.grad.tensor3At(c, r, static_cast<std::size_t>(d));
	                        }
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::Trace: {
	            if (out.has_hess) {
	                throw FEException("Forms: Hessian of trace(A) is not supported (requires rank-4)",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: trace must have 1 child (jet)");
	            }
	            const auto A = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            if (!isMatrixKind<Scalar>(A.value.kind)) {
	                throw FEException("Forms: trace() expects a matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const auto rows = A.value.matrixRows();
	            const auto cols = A.value.matrixCols();
	            if (rows != cols || rows == 0u) {
	                throw FEException("Forms: trace() expects a square matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            Scalar tr = makeScalarConstant<Scalar>(0.0, env);
	            for (std::size_t d = 0; d < rows; ++d) {
	                tr = s_add(tr, A.value.matrixAt(d, d), env);
	            }
	            out.value.s = tr;
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                out.grad.vector_size = dim;
	                for (int d = 0; d < dim; ++d) {
	                    Scalar gd = makeScalarConstant<Scalar>(0.0, env);
	                    for (std::size_t k = 0; k < rows; ++k) {
	                        gd = s_add(gd, A.grad.tensor3At(k, k, static_cast<std::size_t>(d)), env);
	                    }
	                    out.grad.v[static_cast<std::size_t>(d)] = gd;
	                }
	            }
	            return out;
	        }
	        case FormExprType::SymmetricPart:
	        case FormExprType::SkewPart: {
	            if (out.has_hess) {
	                throw FEException("Forms: Hessian of sym(A)/skew(A) is not supported (requires rank-4)",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: sym/skew must have 1 child (jet)");
	            }
	            const auto A = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            if (!isMatrixKind<Scalar>(A.value.kind)) {
	                throw FEException("Forms: sym/skew expects a matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const auto rows = A.value.matrixRows();
	            const auto cols = A.value.matrixCols();
	            if (rows != cols) {
	                throw FEException("Forms: sym/skew expects a square matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            out.value.kind = (node.type() == FormExprType::SymmetricPart)
	                ? EvalValue<Scalar>::Kind::SymmetricMatrix
	                : EvalValue<Scalar>::Kind::SkewMatrix;
	            out.value.resizeMatrix(rows, cols);
	            for (std::size_t r = 0; r < rows; ++r) {
	                for (std::size_t c = 0; c < cols; ++c) {
	                    const Scalar art = A.value.matrixAt(r, c);
	                    const Scalar atr = A.value.matrixAt(c, r);
	                    out.value.matrixAt(r, c) =
	                        (node.type() == FormExprType::SymmetricPart)
	                            ? s_mul(0.5, s_add(art, atr, env), env)
	                            : s_mul(0.5, s_sub(art, atr, env), env);
	                }
	            }
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
	                out.grad.resizeTensor3(rows, cols, static_cast<std::size_t>(dim));
	                for (std::size_t r = 0; r < rows; ++r) {
	                    for (std::size_t c = 0; c < cols; ++c) {
	                        for (int d = 0; d < dim; ++d) {
	                            const Scalar grc = A.grad.tensor3At(r, c, static_cast<std::size_t>(d));
	                            const Scalar gcr = A.grad.tensor3At(c, r, static_cast<std::size_t>(d));
	                            out.grad.tensor3At(r, c, static_cast<std::size_t>(d)) =
	                                (node.type() == FormExprType::SymmetricPart)
	                                    ? s_mul(0.5, s_add(grc, gcr, env), env)
	                                    : s_mul(0.5, s_sub(grc, gcr, env), env);
	                        }
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::Deviator: {
	            if (out.has_hess) {
	                throw FEException("Forms: Hessian of dev(A) is not supported (requires rank-4)",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: dev must have 1 child (jet)");
	            }
	            const auto A = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            if (!isMatrixKind<Scalar>(A.value.kind)) {
	                throw FEException("Forms: dev() expects a matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const auto rows = A.value.matrixRows();
	            const auto cols = A.value.matrixCols();
	            if (rows != cols || rows == 0u) {
	                throw FEException("Forms: dev() expects a square matrix (jet)",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            Scalar tr = makeScalarConstant<Scalar>(0.0, env);
	            for (std::size_t d = 0; d < rows; ++d) tr = s_add(tr, A.value.matrixAt(d, d), env);
	            const Scalar mean = s_mul(tr,
	                                      makeScalarConstant<Scalar>(1.0 / static_cast<Real>(rows), env),
	                                      env);
	
	            out.value = A.value;
	            out.value.kind = A.value.kind;
	            for (std::size_t d = 0; d < rows; ++d) {
	                out.value.matrixAt(d, d) = s_sub(out.value.matrixAt(d, d), mean, env);
	            }
	            if (out.has_grad) {
	                out.grad = A.grad;
	                for (int k = 0; k < dim; ++k) {
	                    Scalar gmean = makeScalarConstant<Scalar>(0.0, env);
	                    for (std::size_t d = 0; d < rows; ++d) {
	                        gmean = s_add(gmean, A.grad.tensor3At(d, d, static_cast<std::size_t>(k)), env);
	                    }
	                    gmean = s_mul(gmean,
	                                  makeScalarConstant<Scalar>(1.0 / static_cast<Real>(rows), env),
	                                  env);
	                    for (std::size_t d = 0; d < rows; ++d) {
	                        out.grad.tensor3At(d, d, static_cast<std::size_t>(k)) =
	                            s_sub(out.grad.tensor3At(d, d, static_cast<std::size_t>(k)), gmean, env);
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::AbsoluteValue: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: abs must have 1 child (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
	            if (!isScalarKind<Scalar>(a.value.kind)) {
	                throw FEException("Forms: abs() in spatial jets expects a scalar operand",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            const Real av = scalarNumericValue(a.value.s);
	            const Real sgn = (av > 0.0) ? 1.0 : ((av < 0.0) ? -1.0 : 0.0);
	            const Scalar sgn_s = makeScalarConstant<Scalar>(sgn, env);
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = s_abs(a.value.s, env);
	            if (out.has_grad) {
	                out.grad.kind = EvalValue<Scalar>::Kind::Vector;
	                out.grad.vector_size = dim;
	                for (int d = 0; d < dim; ++d) {
	                    out.grad.v[static_cast<std::size_t>(d)] =
	                        s_mul(sgn_s, a.grad.vectorAt(static_cast<std::size_t>(d)), env);
	                }
	            }
	            if (out.has_hess) {
	                out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
	                out.hess.matrix_rows = dim;
	                out.hess.matrix_cols = dim;
	                for (int i = 0; i < dim; ++i) {
	                    for (int j = 0; j < dim; ++j) {
	                        out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
	                            s_mul(sgn_s, a.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), env);
	                    }
	                }
	            }
	            return out;
	        }
	        case FormExprType::Sign: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
	                throw std::logic_error("Forms: sign must have 1 child (jet)");
	            }
	            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, 0);
	            if (!isScalarKind<Scalar>(a.value.kind)) {
	                throw FEException("Forms: sign() in spatial jets expects a scalar operand",
	                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	            }
	            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
	            out.value.s = s_sign(a.value.s, env);
	            if (out.has_grad) {
	                out.grad = zeroVector<Scalar>(static_cast<std::size_t>(dim), env);
	            }
	            if (out.has_hess) {
	                out.hess = zeroMatrix<Scalar>(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim), env);
	            }
	            return out;
	        }
	        case FormExprType::Sqrt:
	        case FormExprType::Exp:
	        case FormExprType::Log: {
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1u || !kids[0]) {
                throw std::logic_error("Forms: unary op must have 1 child (jet)");
            }
            const auto u = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
            if (!isScalarKind<Scalar>(u.value.kind)) {
                throw FEException("Forms: sqrt/exp/log in spatial jets expects a scalar operand",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            out.value.kind = EvalValue<Scalar>::Kind::Scalar;
            if (node.type() == FormExprType::Sqrt) out.value.s = s_sqrt(u.value.s, env);
            else if (node.type() == FormExprType::Exp) out.value.s = s_exp(u.value.s, env);
            else out.value.s = s_log(u.value.s, env);

            if (out.has_grad) {
                out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                out.grad.vector_size = dim;

                Scalar fp = makeScalarConstant<Scalar>(0.0, env);
                if (node.type() == FormExprType::Sqrt) {
                    fp = s_div(makeScalarConstant<Scalar>(0.5, env), out.value.s, env);
                } else if (node.type() == FormExprType::Exp) {
                    fp = out.value.s;
                } else {
                    fp = s_div(makeScalarConstant<Scalar>(1.0, env), u.value.s, env);
                }

                for (int d = 0; d < dim; ++d) {
                    out.grad.v[static_cast<std::size_t>(d)] =
                        s_mul(fp, u.grad.vectorAt(static_cast<std::size_t>(d)), env);
                }
            }

            if (out.has_hess) {
                out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                out.hess.matrix_rows = dim;
                out.hess.matrix_cols = dim;

                Scalar fp = makeScalarConstant<Scalar>(0.0, env);
                Scalar fpp = makeScalarConstant<Scalar>(0.0, env);
                if (node.type() == FormExprType::Sqrt) {
                    fp = s_div(makeScalarConstant<Scalar>(0.5, env), out.value.s, env);
                    const Scalar denom = s_mul(s_mul(out.value.s, out.value.s, env), out.value.s, env);
                    fpp = s_div(makeScalarConstant<Scalar>(-0.25, env), denom, env);
                } else if (node.type() == FormExprType::Exp) {
                    fp = out.value.s;
                    fpp = out.value.s;
                } else {
                    fp = s_div(makeScalarConstant<Scalar>(1.0, env), u.value.s, env);
                    const Scalar denom = s_mul(u.value.s, u.value.s, env);
                    fpp = s_div(makeScalarConstant<Scalar>(-1.0, env), denom, env);
                }

                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        const Scalar term1 =
                            s_mul(fp, u.hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), env);
                        const Scalar term2 =
                            s_mul(fpp,
                                  s_mul(u.grad.vectorAt(static_cast<std::size_t>(i)),
                                        u.grad.vectorAt(static_cast<std::size_t>(j)), env),
                                  env);
                        out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = s_add(term1, term2, env);
                    }
                }
            }

            return out;
        }
        case FormExprType::Norm: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::logic_error("Forms: norm must have 1 child (jet)");
            }
            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);

            if (isVectorKind<Scalar>(a.value.kind)) {
                // s = sum_i a_i^2
                Scalar s0 = makeScalarConstant<Scalar>(0.0, env);
                for (std::size_t i = 0; i < a.value.vectorSize(); ++i) {
                    s0 = s_add(s0, s_mul(a.value.vectorAt(i), a.value.vectorAt(i), env), env);
                }

                // grad(s) and H(s) if needed
                EvalValue<Scalar> s_grad{};
                EvalValue<Scalar> s_hess{};
                if (out.has_grad) {
                    s_grad.kind = EvalValue<Scalar>::Kind::Vector;
                    s_grad.vector_size = dim;
                    for (int d = 0; d < dim; ++d) {
                        Scalar g = makeScalarConstant<Scalar>(0.0, env);
                        for (std::size_t i = 0; i < a.value.vectorSize(); ++i) {
                            g = s_add(g,
                                      s_mul(2.0,
                                            s_mul(a.value.vectorAt(i), a.grad.matrixAt(i, static_cast<std::size_t>(d)), env),
                                            env),
                                      env);
                        }
                        s_grad.v[static_cast<std::size_t>(d)] = g;
                    }
                }
                if (out.has_hess) {
                    s_hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    s_hess.matrix_rows = dim;
                    s_hess.matrix_cols = dim;
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            Scalar Hij = makeScalarConstant<Scalar>(0.0, env);
                            for (std::size_t comp = 0; comp < a.value.vectorSize(); ++comp) {
                                Hij = s_add(Hij,
                                            s_mul(2.0,
                                                  s_mul(a.grad.matrixAt(comp, static_cast<std::size_t>(i)),
                                                        a.grad.matrixAt(comp, static_cast<std::size_t>(j)), env),
                                                  env),
                                            env);
                                Hij = s_add(Hij,
                                            s_mul(2.0,
                                                  s_mul(a.value.vectorAt(comp),
                                                        a.hess.tensor3At(comp, static_cast<std::size_t>(i), static_cast<std::size_t>(j)), env),
                                                  env),
                                            env);
                            }
                            s_hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Hij;
                        }
                    }
                }

                // norm = sqrt(s)
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = s_sqrt(s0, env);

                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    const Scalar fp = s_div(makeScalarConstant<Scalar>(0.5, env), out.value.s, env);
                    for (int d = 0; d < dim; ++d) {
                        out.grad.v[static_cast<std::size_t>(d)] = s_mul(fp, s_grad.vectorAt(static_cast<std::size_t>(d)), env);
                    }
                }
                if (out.has_hess) {
                    out.hess.kind = EvalValue<Scalar>::Kind::Matrix;
                    out.hess.matrix_rows = dim;
                    out.hess.matrix_cols = dim;
                    const Scalar fp = s_div(makeScalarConstant<Scalar>(0.5, env), out.value.s, env);
                    const Scalar denom = s_mul(s_mul(out.value.s, out.value.s, env), out.value.s, env);
                    const Scalar fpp = s_div(makeScalarConstant<Scalar>(-0.25, env), denom, env);
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            const Scalar term1 = s_mul(fp, s_hess.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), env);
                            const Scalar term2 = s_mul(fpp,
                                                       s_mul(s_grad.vectorAt(static_cast<std::size_t>(i)),
                                                             s_grad.vectorAt(static_cast<std::size_t>(j)), env),
                                                       env);
                            out.hess.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = s_add(term1, term2, env);
                        }
                    }
                }
                return out;
            }

            if (isMatrixKind<Scalar>(a.value.kind)) {
                // Only support first derivatives (no rank-4 support).
                if (out.has_hess) {
                    throw FEException("Forms: Hessian of norm(matrix) is not supported (requires rank-4)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                Scalar s0 = makeScalarConstant<Scalar>(0.0, env);
                for (std::size_t r = 0; r < a.value.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < a.value.matrixCols(); ++c) {
                        s0 = s_add(s0, s_mul(a.value.matrixAt(r, c), a.value.matrixAt(r, c), env), env);
                    }
                }
                out.value.kind = EvalValue<Scalar>::Kind::Scalar;
                out.value.s = s_sqrt(s0, env);
                if (out.has_grad) {
                    out.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    out.grad.vector_size = dim;
                    const Scalar inv_n = s_div(makeScalarConstant<Scalar>(1.0, env), out.value.s, env);
                    for (int d = 0; d < dim; ++d) {
                        Scalar g = makeScalarConstant<Scalar>(0.0, env);
                        for (std::size_t r = 0; r < a.value.matrixRows(); ++r) {
                            for (std::size_t c = 0; c < a.value.matrixCols(); ++c) {
                                g = s_add(g,
                                          s_mul(a.value.matrixAt(r, c),
                                                a.grad.tensor3At(r, c, static_cast<std::size_t>(d)), env),
                                          env);
                            }
                        }
                        out.grad.v[static_cast<std::size_t>(d)] = s_mul(inv_n, g, env);
                    }
                }
                return out;
            }

            throw FEException("Forms: norm() kind not supported in spatial jets",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::OuterProduct: {
            if (out.has_hess) {
                throw FEException("Forms: Hessian of outer(vector,vector) is not supported (requires rank-4)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::logic_error("Forms: outer must have 2 children (jet)");
            }
            const auto a = evalSpatialJet<Scalar>(*kids[0], env, side, q, order);
            const auto b = evalSpatialJet<Scalar>(*kids[1], env, side, q, order);
            if (!isVectorKind<Scalar>(a.value.kind) || !isVectorKind<Scalar>(b.value.kind)) {
                throw FEException("Forms: outer() in spatial jets supports vector-vector only",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            out.value.kind = EvalValue<Scalar>::Kind::Matrix;
            out.value.resizeMatrix(a.value.vectorSize(), b.value.vectorSize());
            for (std::size_t r = 0; r < a.value.vectorSize(); ++r) {
                for (std::size_t c = 0; c < b.value.vectorSize(); ++c) {
                    out.value.matrixAt(r, c) = s_mul(a.value.vectorAt(r), b.value.vectorAt(c), env);
                }
            }

            if (out.has_grad) {
                out.grad.kind = EvalValue<Scalar>::Kind::Tensor3;
                out.grad.resizeTensor3(a.value.vectorSize(), b.value.vectorSize(), static_cast<std::size_t>(dim));
                for (std::size_t r = 0; r < a.value.vectorSize(); ++r) {
                    for (std::size_t c = 0; c < b.value.vectorSize(); ++c) {
                        for (int d = 0; d < dim; ++d) {
                            const Scalar term1 =
                                s_mul(a.grad.matrixAt(r, static_cast<std::size_t>(d)), b.value.vectorAt(c), env);
                            const Scalar term2 =
                                s_mul(a.value.vectorAt(r), b.grad.matrixAt(c, static_cast<std::size_t>(d)), env);
                            out.grad.tensor3At(r, c, static_cast<std::size_t>(d)) = s_add(term1, term2, env);
                        }
                    }
                }
            }

            return out;
        }
        case FormExprType::Gradient: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::logic_error("Forms: grad must have 1 child (jet)");
            }
            if (order >= 2) {
                throw FEException("Forms: spatial jet does not support second derivatives of grad(·) (would require 3rd derivatives)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            const int child_order = order + 1;
            const auto u = evalSpatialJet<Scalar>(*kids[0], env, side, q, child_order);
            SpatialJet<Scalar> res;
            res.has_grad = (order >= 1);
            res.has_hess = false;
            res.value = u.grad;
            if (order >= 1) {
                res.grad = u.hess;
            }
            return res;
        }
        case FormExprType::Divergence: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::logic_error("Forms: div must have 1 child (jet)");
            }
            if (order >= 2) {
                throw FEException("Forms: spatial jet does not support second derivatives of div(·) (would require 3rd derivatives)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            const int child_order = order + 1;
            const auto u = evalSpatialJet<Scalar>(*kids[0], env, side, q, child_order);

            if (isVectorKind<Scalar>(u.value.kind)) {
                SpatialJet<Scalar> res;
                res.has_grad = (order >= 1);
                res.has_hess = false;
                res.value.kind = EvalValue<Scalar>::Kind::Scalar;
                Scalar divu = makeScalarConstant<Scalar>(0.0, env);
                const int n = std::min(dim, static_cast<int>(u.value.vectorSize()));
                for (int d = 0; d < n; ++d) {
                    divu = s_add(divu, u.grad.matrixAt(static_cast<std::size_t>(d), static_cast<std::size_t>(d)), env);
                }
                res.value.s = divu;

                if (order >= 1) {
                    res.grad.kind = EvalValue<Scalar>::Kind::Vector;
                    res.grad.vector_size = dim;
                    for (int k = 0; k < dim; ++k) {
                        Scalar gk = makeScalarConstant<Scalar>(0.0, env);
                        for (int d = 0; d < n; ++d) {
                            gk = s_add(gk,
                                       u.hess.tensor3At(static_cast<std::size_t>(d),
                                                        static_cast<std::size_t>(k),
                                                        static_cast<std::size_t>(d)),
                                       env);
                        }
                        res.grad.v[static_cast<std::size_t>(k)] = gk;
                    }
                }
                return res;
            }

            if (isMatrixKind<Scalar>(u.value.kind)) {
                SpatialJet<Scalar> res;
                res.has_grad = false;
                res.has_hess = false;
                res.value.kind = EvalValue<Scalar>::Kind::Vector;
                res.value.resizeVector(u.value.matrixRows());
                const int cols = static_cast<int>(u.value.matrixCols());
                const int n = std::min(dim, cols);
                for (std::size_t r = 0; r < u.value.matrixRows(); ++r) {
                    Scalar sum = makeScalarConstant<Scalar>(0.0, env);
                    for (int d = 0; d < n; ++d) {
                        sum = s_add(sum, u.grad.tensor3At(r, static_cast<std::size_t>(d), static_cast<std::size_t>(d)), env);
                    }
                    res.value.vectorAt(r) = sum;
                }
                if (order >= 1) {
                    throw FEException("Forms: grad(div(matrix)) is not supported (requires rank-4)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }
                return res;
            }

            throw FEException("Forms: div() operand kind not supported in spatial jets",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        case FormExprType::Hessian: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::logic_error("Forms: H must have 1 child (jet)");
            }
            if (order > 0) {
                throw FEException("Forms: spatial jet does not support derivatives of H(·) (would require 3rd derivatives)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            const auto u = evalSpatialJet<Scalar>(*kids[0], env, side, q, 2);
            SpatialJet<Scalar> res;
            res.has_grad = false;
            res.has_hess = false;
            res.value = u.hess;
            return res;
        }
        default:
            break;
    }

    throw FEException("Forms: spatial jet does not support this node type",
                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
}

// ============================================================================
// Real (variational) evaluation
// ============================================================================

struct ConstitutiveCallCacheReal;

struct EvalEnvReal {
    const assembly::AssemblyContext& minus;
    const assembly::AssemblyContext* plus{nullptr};
    FormKind kind{FormKind::Linear};
    Side test_active{Side::Minus};
    Side trial_active{Side::Minus};
    LocalIndex i{0};
    LocalIndex j{0};
    const ConstitutiveStateLayout* constitutive_state{nullptr};
    ConstitutiveCallCacheReal* constitutive_cache{nullptr};
};

EvalValue<Real> evalReal(const FormExprNode& node,
                         const EvalEnvReal& env,
                         Side side,
                         LocalIndex q);

struct ConstitutiveCallKey {
    const FormExprNode* node{nullptr};
    Side side{Side::Minus};
    LocalIndex q{0};
    LocalIndex i{0};
    LocalIndex j{0};

    [[nodiscard]] bool operator==(const ConstitutiveCallKey& other) const noexcept
    {
        return node == other.node &&
               side == other.side &&
               q == other.q &&
               i == other.i &&
               j == other.j;
    }
};

struct ConstitutiveCallKeyHash {
    [[nodiscard]] std::size_t operator()(const ConstitutiveCallKey& key) const noexcept
    {
        std::size_t h = std::hash<const FormExprNode*>{}(key.node);
        auto mix = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };

        mix(static_cast<std::size_t>(key.side));
        mix(static_cast<std::size_t>(key.q));
        mix(static_cast<std::size_t>(key.i));
        mix(static_cast<std::size_t>(key.j));
        return h;
    }
};

struct ConstitutiveCallCacheReal {
    std::unordered_map<ConstitutiveCallKey,
                       std::vector<EvalValue<Real>>,
                       ConstitutiveCallKeyHash> values{};
};

EvalValue<Real> evalRealUnary(const FormExprNode& node,
                              const EvalEnvReal& env,
                              Side side,
                              LocalIndex q)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 1 || !kids[0]) {
        throw std::logic_error("Forms: unary node must have exactly 1 child");
    }
    return evalReal(*kids[0], env, side, q);
}

EvalValue<Real> evalConstitutiveOutputReal(const FormExprNode& call_node,
                                           const EvalEnvReal& env,
                                           Side side,
                                           LocalIndex q,
                                           std::size_t output_index)
{
    if (call_node.type() != FormExprType::Constitutive) {
        throw FEException("Forms: evalConstitutiveOutputReal called on non-constitutive node",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto* model = call_node.constitutiveModel();
    if (!model) {
        throw FEException("Forms: Constitutive node has no model",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto n_outputs = model->outputCount();
    if (output_index >= n_outputs) {
        throw FEException("Forms: constitutive output index out of range",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (env.constitutive_cache != nullptr) {
        ConstitutiveCallKey key{&call_node, side, q, env.i, env.j};
        auto it = env.constitutive_cache->values.find(key);
        if (it != env.constitutive_cache->values.end()) {
            if (it->second.size() != n_outputs) {
                throw FEException("Forms: constitutive cache output size mismatch",
                                  __FILE__, __LINE__, __func__, FEStatus::Unknown);
            }
            return it->second[output_index];
        }
    }

    const auto& ctx = ctxForSide(env.minus, env.plus, side);
    const int dim = ctx.dimension();

    ConstitutiveEvalContext mctx;
    mctx.domain = [&]() {
        switch (ctx.contextType()) {
            case assembly::ContextType::Cell:
                return ConstitutiveEvalContext::Domain::Cell;
            case assembly::ContextType::BoundaryFace:
                return ConstitutiveEvalContext::Domain::BoundaryFace;
            case assembly::ContextType::InteriorFace:
                return ConstitutiveEvalContext::Domain::InteriorFace;
            default:
                return ConstitutiveEvalContext::Domain::Cell;
        }
    }();
    mctx.side = (side == Side::Plus) ? ConstitutiveEvalContext::TraceSide::Plus
                                     : ConstitutiveEvalContext::TraceSide::Minus;
    mctx.dim = dim;
    mctx.x = ctx.physicalPoint(q);
    mctx.time = ctx.time();
    mctx.dt = ctx.timeStep();
    mctx.cell_id = ctx.cellId();
    mctx.face_id = ctx.faceId();
    mctx.local_face_id = ctx.localFaceId();
    mctx.boundary_marker = ctx.boundaryMarker();
    mctx.q = q;
    mctx.num_qpts = ctx.numQuadraturePoints();
    mctx.get_real_param = ctx.realParameterGetter();
    mctx.get_param = ctx.parameterGetter();
    mctx.user_data = ctx.userData();

    std::size_t state_offset_bytes = 0u;
    std::size_t state_bytes = 0u;

    if (env.constitutive_state != nullptr) {
        if (const auto* block = env.constitutive_state->find(model); block != nullptr && block->bytes > 0u) {
            if (!ctx.hasMaterialState()) {
                throw FEException("Forms: Constitutive node requires material state but no state is bound in this context",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto state_old = ctx.materialStateOld(q);
            const auto state_work = ctx.materialStateWork(q);
            const auto need = block->offset_bytes + block->bytes;
            if (state_old.size() < need || state_work.size() < need) {
                throw FEException("Forms: bound material state buffer is smaller than required by constitutive state layout",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            state_offset_bytes = block->offset_bytes;
            state_bytes = block->bytes;
            mctx.state_old = state_old.subspan(block->offset_bytes, block->bytes);
            mctx.state_work = state_work.subspan(block->offset_bytes, block->bytes);
        }
    }

    const auto kids = call_node.childrenShared();
    if (kids.empty()) {
        throw FEException("Forms: Constitutive node must have at least 1 child",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    struct NonlocalRealCtx {
        const EvalEnvReal* env{nullptr};
        Side eval_side{Side::Minus};
        const std::vector<std::shared_ptr<FormExprNode>>* inputs{nullptr};
        const assembly::AssemblyContext* ctx{nullptr};
        std::size_t state_offset_bytes{0u};
        std::size_t state_bytes{0u};
    };

    ConstitutiveEvalContext::NonlocalAccess nonlocal{};
    NonlocalRealCtx nonlocal_ctx{&env, side, &kids, &ctx, state_offset_bytes, state_bytes};
    nonlocal.self = &nonlocal_ctx;
    nonlocal.input_real = +[](const void* self, std::size_t input_index, LocalIndex qpt) -> EvalValue<Real> {
        const auto& s = *static_cast<const NonlocalRealCtx*>(self);
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal input_real missing context");
        FE_THROW_IF(s.inputs == nullptr, FEException,
                    "Forms: nonlocal input_real missing inputs");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal input_real qpt out of range");
        FE_THROW_IF(input_index >= s.inputs->size(), InvalidArgumentException,
                    "Forms: nonlocal input_real input_index out of range");
        const auto& child = (*s.inputs)[input_index];
        FE_THROW_IF(!child, InvalidArgumentException,
                    "Forms: nonlocal input_real input node is null");
        FE_THROW_IF(s.env == nullptr, FEException,
                    "Forms: nonlocal input_real missing eval environment");
        return evalReal(*child, *s.env, s.eval_side, qpt);
    };
    nonlocal.state_old = +[](const void* self, LocalIndex qpt) -> std::span<const std::byte> {
        const auto& s = *static_cast<const NonlocalRealCtx*>(self);
        if (s.state_bytes == 0u) return {};
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal state_old missing context");
        FE_THROW_IF(!s.ctx->hasMaterialState(), InvalidArgumentException,
                    "Forms: nonlocal state_old requires bound material state");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal state_old qpt out of range");
        const auto state = s.ctx->materialStateOld(qpt);
        const auto need = s.state_offset_bytes + s.state_bytes;
        FE_THROW_IF(state.size() < need, InvalidArgumentException,
                    "Forms: nonlocal state_old buffer smaller than required");
        return state.subspan(s.state_offset_bytes, s.state_bytes);
    };
    nonlocal.state_work = +[](const void* self, LocalIndex qpt) -> std::span<std::byte> {
        const auto& s = *static_cast<const NonlocalRealCtx*>(self);
        if (s.state_bytes == 0u) return {};
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal state_work missing context");
        FE_THROW_IF(!s.ctx->hasMaterialState(), InvalidArgumentException,
                    "Forms: nonlocal state_work requires bound material state");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal state_work qpt out of range");
        const auto state = s.ctx->materialStateWork(qpt);
        const auto need = s.state_offset_bytes + s.state_bytes;
        FE_THROW_IF(state.size() < need, InvalidArgumentException,
                    "Forms: nonlocal state_work buffer smaller than required");
        return state.subspan(s.state_offset_bytes, s.state_bytes);
    };
    nonlocal.physical_point = +[](const void* self, LocalIndex qpt) -> std::array<Real, 3> {
        const auto& s = *static_cast<const NonlocalRealCtx*>(self);
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal physical_point missing context");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal physical_point qpt out of range");
        return s.ctx->physicalPoint(qpt);
    };
    nonlocal.integration_weight = +[](const void* self, LocalIndex qpt) -> Real {
        const auto& s = *static_cast<const NonlocalRealCtx*>(self);
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal integration_weight missing context");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal integration_weight qpt out of range");
        return s.ctx->integrationWeight(qpt);
    };

    mctx.nonlocal = &nonlocal;

    std::vector<EvalValue<Real>> inputs;
    inputs.reserve(kids.size());
    for (const auto& child : kids) {
        if (!child) {
            throw FEException("Forms: Constitutive node has a null child",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        inputs.push_back(evalReal(*child, env, side, q));
    }

    std::vector<EvalValue<Real>> outputs(n_outputs);
    model->evaluateNaryOutputs({inputs.data(), inputs.size()}, mctx, {outputs.data(), outputs.size()});

    if (env.constitutive_cache != nullptr) {
        ConstitutiveCallKey key{&call_node, side, q, env.i, env.j};
        auto [it, inserted] = env.constitutive_cache->values.emplace(key, std::move(outputs));
        (void)inserted;
        return it->second[output_index];
    }

    return outputs[output_index];
}

EvalValue<Real> evalReal(const FormExprNode& node,
                         const EvalEnvReal& env,
                         Side side,
                         LocalIndex q)
{
    const auto& ctx = ctxForSide(env.minus, env.plus, side);
    const int dim = ctx.dimension();

    switch (node.type()) {
        case FormExprType::Constant: {
            const Real v = node.constantValue().value_or(0.0);
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, v};
        }
        case FormExprType::Time:
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.time()};
        case FormExprType::TimeStep:
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.timeStep()};
        case FormExprType::Coordinate: {
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Vector;
            out.v = ctx.physicalPoint(q);
            out.vector_size = dim;
            return out;
        }
        case FormExprType::ReferenceCoordinate: {
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Vector;
            out.v = ctx.quadraturePoint(q);
            out.vector_size = dim;
            return out;
        }
        case FormExprType::Identity: {
            const int idim = node.identityDim().value_or(dim);
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Matrix;
            if (idim > 0) {
                out.resizeMatrix(static_cast<std::size_t>(idim), static_cast<std::size_t>(idim));
                for (int a = 0; a < idim; ++a) {
                    out.matrixAt(static_cast<std::size_t>(a), static_cast<std::size_t>(a)) = 1.0;
                }
            }
            return out;
        }
        case FormExprType::Jacobian: {
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Matrix;
            out.m = ctx.jacobian(q);
            out.matrix_rows = dim;
            out.matrix_cols = dim;
            return out;
        }
        case FormExprType::JacobianInverse: {
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Matrix;
            out.m = ctx.inverseJacobian(q);
            out.matrix_rows = dim;
            out.matrix_cols = dim;
            return out;
        }
        case FormExprType::JacobianDeterminant: {
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.jacobianDet(q)};
        }
        case FormExprType::Normal: {
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Vector;
            out.v = ctx.normal(q);
            out.vector_size = dim;
            return out;
        }
        case FormExprType::CellDiameter: {
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.cellDiameter()};
        }
        case FormExprType::CellVolume: {
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.cellVolume()};
        }
        case FormExprType::FacetArea: {
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.facetArea()};
        }
        case FormExprType::CellDomainId: {
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, static_cast<Real>(ctx.cellDomainId())};
        }
        case FormExprType::ParameterSymbol: {
            const auto nm = node.symbolName();
            FE_THROW_IF(!nm || nm->empty(), InvalidArgumentException,
                        "Forms: ParameterSymbol node missing name");
            const auto* get = ctx.realParameterGetter();
            FE_THROW_IF(get == nullptr || !static_cast<bool>(*get), InvalidArgumentException,
                        "Forms: ParameterSymbol requires a real parameter getter in AssemblyContext");
            const auto v = (*get)(*nm);
            FE_THROW_IF(!v.has_value(), InvalidArgumentException,
                        "Forms: missing required parameter '" + std::string(*nm) + "'");
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, *v};
        }
        case FormExprType::ParameterRef: {
            const auto slot = node.slotIndex().value_or(0u);
            const auto vals = ctx.jitConstants();
            FE_THROW_IF(vals.empty(), InvalidArgumentException,
                        "Forms: ParameterRef requires AssemblyContext::jitConstants()");
            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                        "Forms: ParameterRef slot out of range");
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, vals[slot]};
        }
        case FormExprType::BoundaryIntegralRef: {
            const auto slot = node.slotIndex().value_or(0u);
            const auto vals = ctx.coupledIntegrals();
            FE_THROW_IF(vals.empty(), InvalidArgumentException,
                        "Forms: BoundaryIntegralRef requires coupled integrals in AssemblyContext");
            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                        "Forms: BoundaryIntegralRef slot out of range");
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, vals[slot]};
        }
        case FormExprType::AuxiliaryStateRef: {
            const auto slot = node.slotIndex().value_or(0u);
            const auto vals = ctx.coupledAuxState();
            FE_THROW_IF(vals.empty(), InvalidArgumentException,
                        "Forms: AuxiliaryStateRef requires coupled auxiliary state in AssemblyContext");
            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                        "Forms: AuxiliaryStateRef slot out of range");
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, vals[slot]};
        }
        case FormExprType::MaterialStateOldRef: {
            const auto off = node.stateOffsetBytes();
            FE_THROW_IF(!off.has_value(), InvalidArgumentException,
                        "Forms: MaterialStateOldRef node missing offset");
            FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
                        "Forms: MaterialStateOldRef requires material state in AssemblyContext");
            const auto state = ctx.materialStateOld(q);
            const auto offset = static_cast<std::size_t>(*off);
            FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
                        "Forms: MaterialStateOldRef offset out of range");
            Real v = 0.0;
            std::memcpy(&v, state.data() + offset, sizeof(Real));
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, v};
        }
        case FormExprType::MaterialStateWorkRef: {
            const auto off = node.stateOffsetBytes();
            FE_THROW_IF(!off.has_value(), InvalidArgumentException,
                        "Forms: MaterialStateWorkRef node missing offset");
            FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
                        "Forms: MaterialStateWorkRef requires material state in AssemblyContext");
            const auto state = ctx.materialStateWork(q);
            const auto offset = static_cast<std::size_t>(*off);
            FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
                        "Forms: MaterialStateWorkRef offset out of range");
            Real v = 0.0;
            std::memcpy(&v, state.data() + offset, sizeof(Real));
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, v};
        }
        case FormExprType::PreviousSolutionRef: {
            const int k = node.historyIndex().value_or(1);
            FE_THROW_IF(k <= 0, InvalidArgumentException,
                        "Forms: PreviousSolutionRef requires k >= 1");
            if (ctx.trialFieldType() == FieldType::Vector) {
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.vector_size = ctx.trialValueDimension();
                out.v = ctx.previousSolutionVectorValue(q, k);
                return out;
            }
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.previousSolutionValue(q, k)};
        }
        case FormExprType::Coefficient: {
            if (const auto* f = node.timeScalarCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, (*f)(x[0], x[1], x[2], ctx.time())};
            }
            if (const auto* f = node.scalarCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, (*f)(x[0], x[1], x[2])};
            }
            if (const auto* f = node.vectorCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.v = (*f)(x[0], x[1], x[2]);
                out.vector_size = dim;
                return out;
            }
            if (const auto* f = node.matrixCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Matrix;
                out.m = (*f)(x[0], x[1], x[2]);
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                return out;
            }
            if (const auto* f = node.tensor3Coefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Tensor3;
                out.t3 = (*f)(x[0], x[1], x[2]);
                out.tensor3_dim0 = dim;
                out.tensor3_dim1 = dim;
                out.tensor3_dim2 = dim;
                return out;
            }
            if (const auto* f = node.tensor4Coefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Tensor4;
                out.t4 = (*f)(x[0], x[1], x[2]);
                return out;
            }
            throw FEException("Forms: coefficient node has no callable",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        case FormExprType::Constitutive:
            return evalConstitutiveOutputReal(node, env, side, q, 0u);
        case FormExprType::ConstitutiveOutput: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw FEException("Forms: ConstitutiveOutput node must have exactly 1 child",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto out_idx = node.constitutiveOutputIndex().value_or(0);
            FE_THROW_IF(out_idx < 0, InvalidArgumentException,
                        "Forms: ConstitutiveOutput node has negative output index");
            return evalConstitutiveOutputReal(*kids[0], env, side, q, static_cast<std::size_t>(out_idx));
        }
        case FormExprType::TestFunction: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: TestFunction must be bound to a FunctionSpace",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                if (env.test_active != side) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, 0.0};
                }
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.basisValue(env.i, q)};
            }

            if (sig->field_type == FieldType::Vector) {
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.vector_size = sig->value_dimension;
                if (env.test_active != side) {
                    return out;
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: TestFunction vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: vector TestFunction DOF count is not divisible by value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: TestFunction vector DOF index out of range for its value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.v[static_cast<std::size_t>(comp)] = ctx.basisValue(env.i, q);
                return out;
            }

            throw FEException("Forms: TestFunction field type not supported",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::TrialFunction: {
            if (env.kind == FormKind::Residual) {
                // Residual forms should be evaluated by NonlinearFormKernel.
                throw FEException("Forms: TrialFunction in residual form evaluated in variational mode",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: TrialFunction must be bound to a FunctionSpace",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                if (env.trial_active != side) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, 0.0};
                }
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.trialBasisValue(env.j, q)};
            }

            if (sig->field_type == FieldType::Vector) {
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.vector_size = sig->value_dimension;
                if (env.trial_active != side) {
                    return out;
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: TrialFunction vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_trial = ctx.numTrialDofs();
                if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: vector TrialFunction DOF count is not divisible by value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.j / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: TrialFunction vector DOF index out of range for its value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.v[static_cast<std::size_t>(comp)] = ctx.trialBasisValue(env.j, q);
                return out;
            }

            throw FEException("Forms: TrialFunction field type not supported",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: DiscreteField must be bound to a FunctionSpace",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto fid = node.fieldId();
            if (!fid || *fid == INVALID_FIELD_ID) {
                throw FEException("Forms: DiscreteField node missing a valid FieldId",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, ctx.fieldValue(*fid, q)};
            }
            if (sig->field_type == FieldType::Vector) {
                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: DiscreteField vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.vector_size = vd;
                out.v = ctx.fieldVectorValue(*fid, q);
                return out;
            }

            throw FEException("Forms: DiscreteField field type not supported",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
	        case FormExprType::Gradient: {
	            return evalSpatialJet<Real>(node, env, side, q, 0).value;
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1 || !kids[0]) throw std::logic_error("grad must have 1 child");
	            const auto& child = *kids[0];

            // Only support gradients of terminals in the initial implementation.
            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: grad(TestFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                if (sig->field_type == FieldType::Scalar) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.vector_size = dim;
                    if (env.test_active == side) {
                        out.v = ctx.physicalGradient(env.i, q);
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: grad(TestFunction) vector value_dimension must be 1..3",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    out.matrix_rows = vd;
                    out.matrix_cols = dim;
                    if (env.test_active != side) {
                        return out;
                    }
                    const LocalIndex n_test = ctx.numTestDofs();
                    if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: grad(TestFunction) DOF count is not divisible by value_dimension",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                    const int comp = static_cast<int>(env.i / dofs_per_component);
                    if (comp < 0 || comp >= vd) {
                        throw FEException("Forms: grad(TestFunction) vector DOF index out of range",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const auto g = ctx.physicalGradient(env.i, q);
                    for (int c = 0; c < dim; ++c) {
                        out.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(c)] = g[static_cast<std::size_t>(c)];
                    }
                    return out;
                }

                throw FEException("Forms: grad(TestFunction) field type not supported",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            if (child.type() == FormExprType::TrialFunction) {
                if (env.kind == FormKind::Residual) {
                    throw FEException("Forms: grad(TrialFunction) in residual form evaluated in variational mode",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: grad(TrialFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                if (sig->field_type == FieldType::Scalar) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.vector_size = dim;
                    if (env.trial_active == side) {
                        out.v = ctx.trialPhysicalGradient(env.j, q);
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: grad(TrialFunction) vector value_dimension must be 1..3",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    out.matrix_rows = vd;
                    out.matrix_cols = dim;
                    if (env.trial_active != side) {
                        return out;
                    }
                    const LocalIndex n_trial = ctx.numTrialDofs();
                    if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: grad(TrialFunction) DOF count is not divisible by value_dimension",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
                    const int comp = static_cast<int>(env.j / dofs_per_component);
                    if (comp < 0 || comp >= vd) {
                        throw FEException("Forms: grad(TrialFunction) vector DOF index out of range",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const auto g = ctx.trialPhysicalGradient(env.j, q);
                    for (int c = 0; c < dim; ++c) {
                        out.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(c)] = g[static_cast<std::size_t>(c)];
                    }
                    return out;
                }

                throw FEException("Forms: grad(TrialFunction) field type not supported",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: grad(DiscreteField) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto fid = child.fieldId();
                if (!fid || *fid == INVALID_FIELD_ID) {
                    throw FEException("Forms: grad(DiscreteField) missing a valid FieldId",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                if (sig->field_type == FieldType::Scalar) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.vector_size = dim;
                    out.v = ctx.fieldGradient(*fid, q);
                    return out;
                }
                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: grad(DiscreteField) vector value_dimension must be 1..3",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    out.matrix_rows = vd;
                    out.matrix_cols = dim;
                    out.m = ctx.fieldJacobian(*fid, q);
                    return out;
                }

                throw FEException("Forms: grad(DiscreteField) field type not supported",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            if (child.type() == FormExprType::Constant) {
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.vector_size = dim;
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.scalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto g = fdGradScalar(*child.scalarCoefficient(), x, dim);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.v = g;
                out.vector_size = dim;
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.timeScalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto g = fdGradScalarTime(*child.timeScalarCoefficient(), x, ctx.time(), dim);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.v = g;
                out.vector_size = dim;
                return out;
            }

            throw FEException("Forms: grad() currently supports TestFunction, TrialFunction, Constant, and scalar Coefficient only",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
	        case FormExprType::Hessian: {
	            return evalSpatialJet<Real>(node, env, side, q, 0).value;
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1 || !kids[0]) throw std::logic_error("H() must have 1 child");
	            const auto& child = *kids[0];

            if (child.type() == FormExprType::Component) {
                const auto ckids = child.childrenShared();
                if (ckids.size() != 1 || !ckids[0]) throw std::logic_error("component() must have 1 child");
                const auto& base = *ckids[0];
                const int comp = child.componentIndex0().value_or(0);
                const int col = child.componentIndex1().value_or(-1);
                if (col >= 0) {
                    throw FEException("Forms: H(component(A,i,j)) is not supported",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }

                if (base.type() == FormExprType::TestFunction) {
                    const auto* sig = base.spaceSignature();
                    if (!sig) {
                        throw FEException("Forms: H(component(TestFunction,i)) requires a bound FunctionSpace",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    if (sig->field_type == FieldType::Scalar) {
                        if (comp != 0) {
                            throw FEException("Forms: H(component(TestFunction,i)) invalid component index for scalar-valued TestFunction",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        EvalValue<Real> out;
                        out.kind = EvalValue<Real>::Kind::Matrix;
                        out.matrix_rows = dim;
                        out.matrix_cols = dim;
                        if (env.test_active == side) {
                            out.m = ctx.physicalHessian(env.i, q);
                        }
                        return out;
                    }
                    if (sig->field_type != FieldType::Vector) {
                        throw FEException("Forms: H(component(TestFunction,i)) field type not supported",
                                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                    }

                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: H(component(TestFunction,i)) vector value_dimension must be 1..3",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    if (comp < 0 || comp >= vd) {
                        throw FEException("Forms: H(component(TestFunction,i)) component index out of range",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }

                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    out.matrix_rows = dim;
                    out.matrix_cols = dim;
                    if (env.test_active != side) {
                        return out;
                    }

                    const LocalIndex n_test = ctx.numTestDofs();
                    if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: H(component(TestFunction,i)) DOF count is not divisible by value_dimension",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                    const int comp_i = static_cast<int>(env.i / dofs_per_component);
                    if (comp_i == comp) {
                        out.m = ctx.physicalHessian(env.i, q);
                    }
                    return out;
                }

	                if (base.type() == FormExprType::TrialFunction) {
	                    if (env.kind == FormKind::Residual) {
	                        throw FEException("Forms: H(component(TrialFunction,i)) in residual form evaluated in variational mode",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
                    const auto* sig = base.spaceSignature();
                    if (!sig) {
                        throw FEException("Forms: H(component(TrialFunction,i)) requires a bound FunctionSpace",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    if (sig->field_type == FieldType::Scalar) {
                        if (comp != 0) {
                            throw FEException("Forms: H(component(TrialFunction,i)) invalid component index for scalar-valued TrialFunction",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        EvalValue<Real> out;
                        out.kind = EvalValue<Real>::Kind::Matrix;
                        out.matrix_rows = dim;
                        out.matrix_cols = dim;
                        if (env.trial_active == side) {
                            out.m = ctx.trialPhysicalHessian(env.j, q);
                        }
                        return out;
                    }
                    if (sig->field_type != FieldType::Vector) {
                        throw FEException("Forms: H(component(TrialFunction,i)) field type not supported",
                                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                    }

                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: H(component(TrialFunction,i)) vector value_dimension must be 1..3",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    if (comp < 0 || comp >= vd) {
                        throw FEException("Forms: H(component(TrialFunction,i)) component index out of range",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }

                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    out.matrix_rows = dim;
                    out.matrix_cols = dim;
                    if (env.trial_active != side) {
                        return out;
                    }

                    const LocalIndex n_trial = ctx.numTrialDofs();
                    if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: H(component(TrialFunction,i)) DOF count is not divisible by value_dimension",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
                    const int comp_j = static_cast<int>(env.j / dofs_per_component);
                    if (comp_j == comp) {
                        out.m = ctx.trialPhysicalHessian(env.j, q);
                    }
	                    return out;
	                }

	                if (base.type() == FormExprType::DiscreteField || base.type() == FormExprType::StateField) {
	                    const auto* sig = base.spaceSignature();
	                    if (!sig) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) requires a bound FunctionSpace",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	                    const auto fid = base.fieldId();
	                    if (!fid || *fid == INVALID_FIELD_ID) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) missing a valid FieldId",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	                    if (sig->field_type == FieldType::Scalar) {
	                        if (comp != 0) {
	                            throw FEException("Forms: H(component(DiscreteField,i)) invalid component index for scalar-valued DiscreteField",
	                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                        }
	                        EvalValue<Real> out;
	                        out.kind = EvalValue<Real>::Kind::Matrix;
	                        out.matrix_rows = dim;
	                        out.matrix_cols = dim;
	                        out.m = ctx.fieldHessian(*fid, q);
	                        return out;
	                    }
	                    if (sig->field_type != FieldType::Vector) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) field type not supported",
	                                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	                    }
	
	                    const int vd = sig->value_dimension;
	                    if (vd <= 0 || vd > 3) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) vector value_dimension must be 1..3",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	                    if (comp < 0 || comp >= vd) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) component index out of range",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	
	                    EvalValue<Real> out;
	                    out.kind = EvalValue<Real>::Kind::Matrix;
	                    out.matrix_rows = dim;
	                    out.matrix_cols = dim;
	                    out.m = ctx.fieldComponentHessian(*fid, q, comp);
	                    return out;
	                }

	                throw FEException("Forms: H(component(...)) currently supports TestFunction, TrialFunction, and DiscreteField only",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }

            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: H(TestFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Scalar) {
                    throw FEException("Forms: H(TestFunction) requires a scalar-valued TestFunction; use H(component(v,i)) for vector-valued spaces",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                if (env.test_active == side) {
                    out.m = ctx.physicalHessian(env.i, q);
                }
                return out;
            }
	            if (child.type() == FormExprType::TrialFunction) {
	                if (env.kind == FormKind::Residual) {
	                    throw FEException("Forms: H(TrialFunction) in residual form evaluated in variational mode",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: H(TrialFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Scalar) {
                    throw FEException("Forms: H(TrialFunction) requires a scalar-valued TrialFunction; use H(component(u,i)) for vector-valued spaces",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                if (env.trial_active == side) {
                    out.m = ctx.trialPhysicalHessian(env.j, q);
	                }
	                return out;
	            }
	            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
	                const auto* sig = child.spaceSignature();
	                if (!sig) {
	                    throw FEException("Forms: H(DiscreteField) requires a bound FunctionSpace",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                const auto fid = child.fieldId();
	                if (!fid || *fid == INVALID_FIELD_ID) {
	                    throw FEException("Forms: H(DiscreteField) missing a valid FieldId",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                if (sig->field_type != FieldType::Scalar) {
	                    throw FEException("Forms: H(DiscreteField) requires a scalar-valued DiscreteField; use H(component(u,i)) for vector-valued spaces",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                EvalValue<Real> out;
	                out.kind = EvalValue<Real>::Kind::Matrix;
	                out.matrix_rows = dim;
	                out.matrix_cols = dim;
	                out.m = ctx.fieldHessian(*fid, q);
	                return out;
	            }
	            if (child.type() == FormExprType::Constant) {
	                EvalValue<Real> out;
	                out.kind = EvalValue<Real>::Kind::Matrix;
	                out.matrix_rows = dim;
                out.matrix_cols = dim;
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.scalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto H = fdHessScalar(*child.scalarCoefficient(), x, dim);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Matrix;
                out.m = H;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
	                return out;
	            }
            if (child.type() == FormExprType::Coefficient && child.timeScalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto H = fdHessScalarTime(*child.timeScalarCoefficient(), x, ctx.time(), dim);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Matrix;
                out.m = H;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                return out;
            }

	            throw FEException("Forms: H() currently supports TestFunction, TrialFunction, DiscreteField, Constant, and scalar Coefficient only",
	                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
        case FormExprType::TimeDerivative: {
            const int order = node.timeDerivativeOrder().value_or(1);
            if (order <= 0) {
                throw FEException("Forms: dt(·,k) requires k >= 1",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            const auto* time_ctx = ctx.timeIntegrationContext();
            if (!time_ctx) {
                throw FEException("dt(...) operator requires a transient time-integration context",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            const auto* stencil = time_ctx->stencil(order);
            if (!stencil) {
                throw FEException("Forms: dt(·," + std::to_string(order) + ") is not supported by time integrator '" +
                                      time_ctx->integrator_name + "'",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            // Bilinear forms contribute only the "current" coefficient (history belongs to Systems).
            const Real scale = stencil->coeff(0);
            const auto val = evalRealUnary(node, env, side, q);
            if (val.kind == EvalValue<Real>::Kind::Scalar) {
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, scale * val.s};
            }
            if (val.kind == EvalValue<Real>::Kind::Vector) {
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.resizeVector(val.vectorSize());
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    out.vectorAt(d) = scale * val.vectorAt(d);
                }
                return out;
            }
            throw FEException("Forms: dt() operand did not evaluate to a scalar or vector",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	        }
	        case FormExprType::Divergence: {
	            return evalSpatialJet<Real>(node, env, side, q, 0).value;
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1 || !kids[0]) throw std::logic_error("div must have 1 child");
	            const auto& child = *kids[0];
            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: div(TestFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: div(TestFunction) requires a vector-valued TestFunction",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (env.test_active != side) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, 0.0};
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: div(TestFunction) vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: div(TestFunction) DOF count is not divisible by value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: div(TestFunction) vector DOF index out of range",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto g = ctx.physicalGradient(env.i, q);
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, g[static_cast<std::size_t>(comp)]};
            }
            if (child.type() == FormExprType::TrialFunction) {
                if (env.kind == FormKind::Residual) {
                    throw FEException("Forms: div(TrialFunction) in residual form evaluated in variational mode",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: div(TrialFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: div(TrialFunction) requires a vector-valued TrialFunction",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (env.trial_active != side) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, 0.0};
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: div(TrialFunction) vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_trial = ctx.numTrialDofs();
                if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: div(TrialFunction) DOF count is not divisible by value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.j / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: div(TrialFunction) vector DOF index out of range",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto g = ctx.trialPhysicalGradient(env.j, q);
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, g[static_cast<std::size_t>(comp)]};
            }
            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: div(DiscreteField) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: div(DiscreteField) requires a vector-valued DiscreteField",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto fid = child.fieldId();
                if (!fid || *fid == INVALID_FIELD_ID) {
                    throw FEException("Forms: div(DiscreteField) missing a valid FieldId",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: div(DiscreteField) vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto J = ctx.fieldJacobian(*fid, q);
                Real div = 0.0;
                const int n = std::min(dim, vd);
                for (int d = 0; d < n; ++d) {
                    div += J[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                }
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, div};
            }
            if (child.type() == FormExprType::Coefficient && child.vectorCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const Real div = fdDivVector(*child.vectorCoefficient(), x, dim);
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, div};
            }
            if (child.type() == FormExprType::Constant) {
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, 0.0};
            }
            throw FEException("Forms: div() currently supports vector TestFunction/TrialFunction and vector Coefficient only",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::Curl: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("curl must have 1 child");
            const auto& child = *kids[0];
            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: curl(TestFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: curl(TestFunction) requires a vector-valued TestFunction",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                if (env.test_active != side) {
                    return out;
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: curl(TestFunction) vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: curl(TestFunction) DOF count is not divisible by value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: curl(TestFunction) vector DOF index out of range",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const auto g = ctx.physicalGradient(env.i, q);
                if (dim == 2) {
                    if (comp == 0) out.v[2] = -g[1];
                    else if (comp == 1) out.v[2] = g[0];
                    return out;
                }
                if (comp == 0) {
                    out.v[1] = g[2];
                    out.v[2] = -g[1];
                } else if (comp == 1) {
                    out.v[0] = -g[2];
                    out.v[2] = g[0];
                } else if (comp == 2) {
                    out.v[0] = g[1];
                    out.v[1] = -g[0];
                }
                return out;
            }
            if (child.type() == FormExprType::TrialFunction) {
                if (env.kind == FormKind::Residual) {
                    throw FEException("Forms: curl(TrialFunction) in residual form evaluated in variational mode",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: curl(TrialFunction) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: curl(TrialFunction) requires a vector-valued TrialFunction",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                if (env.trial_active != side) {
                    return out;
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: curl(TrialFunction) vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_trial = ctx.numTrialDofs();
                if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: curl(TrialFunction) DOF count is not divisible by value_dimension",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.j / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: curl(TrialFunction) vector DOF index out of range",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const auto g = ctx.trialPhysicalGradient(env.j, q);
                if (dim == 2) {
                    if (comp == 0) out.v[2] = -g[1];
                    else if (comp == 1) out.v[2] = g[0];
                    return out;
                }
                if (comp == 0) {
                    out.v[1] = g[2];
                    out.v[2] = -g[1];
                } else if (comp == 1) {
                    out.v[0] = -g[2];
                    out.v[2] = g[0];
                } else if (comp == 2) {
                    out.v[0] = g[1];
                    out.v[1] = -g[0];
                }
                return out;
            }
            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: curl(DiscreteField) requires a bound FunctionSpace",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: curl(DiscreteField) requires a vector-valued DiscreteField",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto fid = child.fieldId();
                if (!fid || *fid == INVALID_FIELD_ID) {
                    throw FEException("Forms: curl(DiscreteField) missing a valid FieldId",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: curl(DiscreteField) vector value_dimension must be 1..3",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const auto J = ctx.fieldJacobian(*fid, q);
                auto d = [&](int comp, int wrt) -> Real {
                    if (comp < 0 || comp >= vd) return 0.0;
                    if (wrt < 0 || wrt >= dim) return 0.0;
                    return J[static_cast<std::size_t>(comp)][static_cast<std::size_t>(wrt)];
                };

                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                if (dim == 2) {
                    out.v[2] = d(1, 0) - d(0, 1);
                    return out;
                }
                out.v[0] = d(2, 1) - d(1, 2);
                out.v[1] = d(0, 2) - d(2, 0);
                out.v[2] = d(1, 0) - d(0, 1);
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.vectorCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto c = fdCurlVector(*child.vectorCoefficient(), x, dim);
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                out.v = c;
                return out;
            }
            if (child.type() == FormExprType::Constant) {
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                return out;
            }
            throw FEException("Forms: curl() currently supports vector TestFunction/TrialFunction and vector Coefficient only",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::RestrictMinus: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("(-) must have 1 child");
            return evalReal(*kids[0], env, Side::Minus, q);
        }
        case FormExprType::RestrictPlus: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("(+) must have 1 child");
            return evalReal(*kids[0], env, Side::Plus, q);
        }
        case FormExprType::Jump: {
            if (!env.plus) {
                throw FEException("Forms: jump() used outside interior-face integral",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("jump must have 1 child");
            const auto& child = *kids[0];
            const auto a = evalReal(child, env, Side::Minus, q);
            const auto b = evalReal(child, env, Side::Plus, q);

            if (a.kind != b.kind) {
                throw FEException("Forms: jump() operand has inconsistent kinds across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isVectorKind<Real>(a.kind) && a.vectorSize() != b.vectorSize()) {
                throw FEException("Forms: jump() operand has inconsistent vector sizes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isMatrixKind<Real>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
                throw FEException("Forms: jump() operand has inconsistent matrix shapes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Real> out;
            out.kind = a.kind;
            if (a.kind == EvalValue<Real>::Kind::Scalar) out.s = a.s - b.s;
            else if (a.kind == EvalValue<Real>::Kind::Vector) {
                out.resizeVector(a.vectorSize());
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    out.vectorAt(d) = a.vectorAt(d) - b.vectorAt(d);
                }
            } else if (isMatrixKind<Real>(a.kind)) {
                out.resizeMatrix(a.matrixRows(), a.matrixCols());
                for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                        out.matrixAt(r, c) = a.matrixAt(r, c) - b.matrixAt(r, c);
                    }
                }
            } else if (isTensor4Kind<Real>(a.kind)) {
                for (std::size_t k = 0; k < out.t4.size(); ++k) {
                    out.t4[k] = a.t4[k] - b.t4[k];
                }
            }
            return out;
        }
        case FormExprType::Average: {
            if (!env.plus) {
                throw FEException("Forms: avg() used outside interior-face integral",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("avg must have 1 child");
            const auto& child = *kids[0];
            const auto a = evalReal(child, env, Side::Minus, q);
            const auto b = evalReal(child, env, Side::Plus, q);
            if (a.kind != b.kind) {
                throw FEException("Forms: avg() operand has inconsistent kinds across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isVectorKind<Real>(a.kind) && a.vectorSize() != b.vectorSize()) {
                throw FEException("Forms: avg() operand has inconsistent vector sizes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isMatrixKind<Real>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
                throw FEException("Forms: avg() operand has inconsistent matrix shapes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Real> out;
            out.kind = a.kind;
            if (isScalarKind<Real>(a.kind)) out.s = 0.5 * (a.s + b.s);
            else if (isVectorKind<Real>(a.kind)) {
                out.resizeVector(a.vectorSize());
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    out.vectorAt(d) = 0.5 * (a.vectorAt(d) + b.vectorAt(d));
                }
            } else if (isMatrixKind<Real>(a.kind)) {
                out.resizeMatrix(a.matrixRows(), a.matrixCols());
                for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                        out.matrixAt(r, c) = 0.5 * (a.matrixAt(r, c) + b.matrixAt(r, c));
                    }
                }
            } else {
                for (std::size_t k = 0; k < out.t4.size(); ++k) {
                    out.t4[k] = 0.5 * (a.t4[k] + b.t4[k]);
                }
            }
            return out;
        }
        case FormExprType::Negate: {
            const auto a = evalRealUnary(node, env, side, q);
            EvalValue<Real> out = a;
            if (isScalarKind<Real>(a.kind)) out.s = -a.s;
            else if (isVectorKind<Real>(a.kind)) {
                for (std::size_t d = 0; d < out.vectorSize(); ++d) out.vectorAt(d) = -a.vectorAt(d);
            } else if (isMatrixKind<Real>(a.kind)) {
                for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < out.matrixCols(); ++c) out.matrixAt(r, c) = -a.matrixAt(r, c);
                }
            } else {
                for (std::size_t k = 0; k < out.t4.size(); ++k) {
                    out.t4[k] = -a.t4[k];
                }
            }
            return out;
        }
        case FormExprType::Transpose: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: transpose() expects a matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            EvalValue<Real> out;
            out.kind = (rows == cols) ? a.kind : EvalValue<Real>::Kind::Matrix;
            out.resizeMatrix(cols, rows);
            for (std::size_t r = 0; r < cols; ++r) {
                for (std::size_t c = 0; c < rows; ++c) {
                    out.matrixAt(r, c) = a.matrixAt(c, r);
                }
            }
            return out;
        }
        case FormExprType::Trace: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: trace() expects a matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: trace() expects a square matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Real tr = 0.0;
            for (std::size_t d = 0; d < rows; ++d) tr += a.matrixAt(d, d);
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, tr};
        }
        case FormExprType::Determinant: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: det() expects a matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: det() expects a square matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Real detA = 0.0;
            if (rows == 1u) {
                detA = a.matrixAt(0, 0);
            } else if (rows == 2u) {
                detA = a.matrixAt(0, 0) * a.matrixAt(1, 1) - a.matrixAt(0, 1) * a.matrixAt(1, 0);
            } else if (rows == 3u) {
                detA =
                    a.matrixAt(0, 0) * (a.matrixAt(1, 1) * a.matrixAt(2, 2) - a.matrixAt(1, 2) * a.matrixAt(2, 1)) -
                    a.matrixAt(0, 1) * (a.matrixAt(1, 0) * a.matrixAt(2, 2) - a.matrixAt(1, 2) * a.matrixAt(2, 0)) +
                    a.matrixAt(0, 2) * (a.matrixAt(1, 0) * a.matrixAt(2, 1) - a.matrixAt(1, 1) * a.matrixAt(2, 0));
            } else {
                throw FEException("Forms: det() is implemented only for 1x1, 2x2, and 3x3 matrices",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, detA};
        }
        case FormExprType::Cofactor:
        case FormExprType::Inverse: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: inv()/cofactor() expects a matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: inv()/cofactor() expects a square matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (rows > 3u) {
                throw FEException("Forms: inv()/cofactor() is implemented only for 1x1, 2x2, and 3x3 matrices",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            EvalValue<Real> cof;
            cof.kind = EvalValue<Real>::Kind::Matrix;
            cof.resizeMatrix(rows, cols);
            if (rows == 1u) {
                cof.matrixAt(0, 0) = 1.0;
            } else if (rows == 2u) {
                cof.matrixAt(0, 0) = a.matrixAt(1, 1);
                cof.matrixAt(0, 1) = -a.matrixAt(1, 0);
                cof.matrixAt(1, 0) = -a.matrixAt(0, 1);
                cof.matrixAt(1, 1) = a.matrixAt(0, 0);
            } else {
                cof.matrixAt(0, 0) = a.matrixAt(1, 1) * a.matrixAt(2, 2) - a.matrixAt(1, 2) * a.matrixAt(2, 1);
                cof.matrixAt(0, 1) = -(a.matrixAt(1, 0) * a.matrixAt(2, 2) - a.matrixAt(1, 2) * a.matrixAt(2, 0));
                cof.matrixAt(0, 2) = a.matrixAt(1, 0) * a.matrixAt(2, 1) - a.matrixAt(1, 1) * a.matrixAt(2, 0);

                cof.matrixAt(1, 0) = -(a.matrixAt(0, 1) * a.matrixAt(2, 2) - a.matrixAt(0, 2) * a.matrixAt(2, 1));
                cof.matrixAt(1, 1) = a.matrixAt(0, 0) * a.matrixAt(2, 2) - a.matrixAt(0, 2) * a.matrixAt(2, 0);
                cof.matrixAt(1, 2) = -(a.matrixAt(0, 0) * a.matrixAt(2, 1) - a.matrixAt(0, 1) * a.matrixAt(2, 0));

                cof.matrixAt(2, 0) = a.matrixAt(0, 1) * a.matrixAt(1, 2) - a.matrixAt(0, 2) * a.matrixAt(1, 1);
                cof.matrixAt(2, 1) = -(a.matrixAt(0, 0) * a.matrixAt(1, 2) - a.matrixAt(0, 2) * a.matrixAt(1, 0));
                cof.matrixAt(2, 2) = a.matrixAt(0, 0) * a.matrixAt(1, 1) - a.matrixAt(0, 1) * a.matrixAt(1, 0);
            }

            if (node.type() == FormExprType::Cofactor) {
                return cof;
            }

            Real detA = 0.0;
            if (rows == 1u) {
                detA = a.matrixAt(0, 0);
            } else if (rows == 2u) {
                detA = a.matrixAt(0, 0) * a.matrixAt(1, 1) - a.matrixAt(0, 1) * a.matrixAt(1, 0);
            } else {
                detA = a.matrixAt(0, 0) * cof.matrixAt(0, 0) +
                       a.matrixAt(0, 1) * cof.matrixAt(0, 1) +
                       a.matrixAt(0, 2) * cof.matrixAt(0, 2);
            }
            if (detA == 0.0) {
                throw FEException("Forms: inv() encountered singular matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            EvalValue<Real> invA;
            invA.kind = EvalValue<Real>::Kind::Matrix;
            invA.resizeMatrix(rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    invA.matrixAt(r, c) = cof.matrixAt(c, r) / detA;
                }
            }
            return invA;
        }
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: sym()/skew() expects a matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols) {
                throw FEException("Forms: sym()/skew() expects a square matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Real> out;
            out.kind = (node.type() == FormExprType::SymmetricPart)
                ? EvalValue<Real>::Kind::SymmetricMatrix
                : EvalValue<Real>::Kind::SkewMatrix;
            out.resizeMatrix(rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    const Real art = a.matrixAt(r, c);
                    const Real atr = a.matrixAt(c, r);
                    out.matrixAt(r, c) = (node.type() == FormExprType::SymmetricPart) ? (0.5 * (art + atr))
                                                                                      : (0.5 * (art - atr));
                }
            }
            return out;
        }
        case FormExprType::Deviator: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: dev() expects a matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Real tr = 0.0;
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: dev() expects a square matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            for (std::size_t d = 0; d < rows; ++d) tr += a.matrixAt(d, d);
            const Real mean = tr / static_cast<Real>(rows);

            EvalValue<Real> out = a;
            out.kind = isMatrixKind<Real>(a.kind) ? a.kind : EvalValue<Real>::Kind::Matrix;
            for (std::size_t d = 0; d < rows; ++d) out.matrixAt(d, d) -= mean;
            return out;
        }
        case FormExprType::Norm: {
            const auto a = evalRealUnary(node, env, side, q);
            Real nrm = 0.0;
            if (isScalarKind<Real>(a.kind)) {
                nrm = std::abs(a.s);
            } else if (isVectorKind<Real>(a.kind)) {
                for (std::size_t d = 0; d < a.vectorSize(); ++d) nrm += a.vectorAt(d) * a.vectorAt(d);
                nrm = std::sqrt(nrm);
            } else if (isMatrixKind<Real>(a.kind)) {
                for (std::size_t r = 0; r < a.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < a.matrixCols(); ++c) {
                        const Real v = a.matrixAt(r, c);
                        nrm += v * v;
                    }
                }
                nrm = std::sqrt(nrm);
            } else if (isTensor3Kind<Real>(a.kind)) {
                for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                    for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                        for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                            const Real v = a.tensor3At(i, j, k);
                            nrm += v * v;
                        }
                    }
                }
                nrm = std::sqrt(nrm);
            } else {
                for (std::size_t k = 0; k < a.t4.size(); ++k) {
                    nrm += a.t4[k] * a.t4[k];
                }
                nrm = std::sqrt(nrm);
            }
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, nrm};
        }
        case FormExprType::Normalize: {
            const auto a = evalRealUnary(node, env, side, q);
            if (!isVectorKind<Real>(a.kind)) {
                throw FEException("Forms: normalize() expects a vector",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Real nrm = 0.0;
            for (std::size_t d = 0; d < a.vectorSize(); ++d) nrm += a.vectorAt(d) * a.vectorAt(d);
            nrm = std::sqrt(nrm);
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Vector;
            out.resizeVector(a.vectorSize());
            if (nrm == 0.0) return out;
            for (std::size_t d = 0; d < out.vectorSize(); ++d) out.vectorAt(d) = a.vectorAt(d) / nrm;
            return out;
        }
        case FormExprType::AbsoluteValue: {
            const auto a = evalRealUnary(node, env, side, q);
            if (a.kind != EvalValue<Real>::Kind::Scalar) {
                throw FEException("Forms: abs() expects a scalar",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, std::abs(a.s)};
        }
        case FormExprType::Sign: {
            const auto a = evalRealUnary(node, env, side, q);
            if (a.kind != EvalValue<Real>::Kind::Scalar) {
                throw FEException("Forms: sign() expects a scalar",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const Real s = (a.s > 0.0) ? 1.0 : ((a.s < 0.0) ? -1.0 : 0.0);
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, s};
        }
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log: {
            const auto a = evalRealUnary(node, env, side, q);
            if (a.kind != EvalValue<Real>::Kind::Scalar) {
                throw FEException("Forms: sqrt/exp/log expects a scalar",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const Real v = (node.type() == FormExprType::Sqrt) ? std::sqrt(a.s)
                         : (node.type() == FormExprType::Exp)  ? std::exp(a.s)
                                                              : std::log(a.s);
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, v};
        }
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct:
        case FormExprType::Power:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2 || !kids[0] || !kids[1]) {
                throw std::logic_error("Forms: binary node must have 2 children");
            }
            const auto a = evalReal(*kids[0], env, side, q);
            const auto b = evalReal(*kids[1], env, side, q);

            if (node.type() == FormExprType::Add || node.type() == FormExprType::Subtract) {
                if (!sameCategory<Real>(a.kind, b.kind)) {
                    throw FEException("Forms: add/sub kind mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isVectorKind<Real>(a.kind) && a.vectorSize() != b.vectorSize()) {
                    throw FEException("Forms: add/sub vector size mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isMatrixKind<Real>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
                    throw FEException("Forms: add/sub matrix shape mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isTensor3Kind<Real>(a.kind) &&
                    (a.tensor3Dim0() != b.tensor3Dim0() ||
                     a.tensor3Dim1() != b.tensor3Dim1() ||
                     a.tensor3Dim2() != b.tensor3Dim2())) {
                    throw FEException("Forms: add/sub tensor3 shape mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = addSubResultKind<Real>(a.kind, b.kind);
                const Real sgn = (node.type() == FormExprType::Add) ? 1.0 : -1.0;
                if (isScalarKind<Real>(a.kind)) out.s = a.s + sgn * b.s;
                else if (isVectorKind<Real>(a.kind)) {
                    out.resizeVector(a.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                        out.vectorAt(d) = a.vectorAt(d) + sgn * b.vectorAt(d);
                    }
                } else if (isMatrixKind<Real>(a.kind)) {
                    out.resizeMatrix(a.matrixRows(), a.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = a.matrixAt(r, c) + sgn * b.matrixAt(r, c);
                        }
                    }
                } else if (isTensor3Kind<Real>(a.kind)) {
                    out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = a.tensor3At(i, j, k) + sgn * b.tensor3At(i, j, k);
                            }
                        }
                    }
                } else {
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = a.t4[k] + sgn * b.t4[k];
                    }
                }
                return out;
            }

            if (node.type() == FormExprType::Multiply) {
                // Scalar * Scalar / Scalar * Vector / Scalar * Matrix
                if (isScalarKind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, a.s * b.s};
                }
                if (isScalarKind<Real>(a.kind) && isVectorKind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.resizeVector(b.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) out.vectorAt(d) = a.s * b.vectorAt(d);
                    return out;
                }
                if (isVectorKind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.resizeVector(a.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) out.vectorAt(d) = a.vectorAt(d) * b.s;
                    return out;
                }
                if (isScalarKind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = b.kind;
                    out.resizeMatrix(b.matrixRows(), b.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = a.s * b.matrixAt(r, c);
                        }
                    }
                    return out;
                }
                if (isMatrixKind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = a.kind;
                    out.resizeMatrix(a.matrixRows(), a.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = a.matrixAt(r, c) * b.s;
                        }
                    }
                    return out;
                }
                // Matrix * Vector / Vector * Matrix / Matrix * Matrix
                if (isMatrixKind<Real>(a.kind) && isVectorKind<Real>(b.kind)) {
                    const auto rows = a.matrixRows();
                    const auto cols = a.matrixCols();
                    if (b.vectorSize() != cols) {
                        throw FEException("Forms: matrix-vector multiplication shape mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.resizeVector(rows);
                    for (std::size_t r = 0; r < rows; ++r) {
                        Real sum = 0.0;
                        for (std::size_t c = 0; c < cols; ++c) {
                            sum += a.matrixAt(r, c) * b.vectorAt(c);
                        }
                        out.vectorAt(r) = sum;
                    }
                    return out;
                }
                if (isVectorKind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
                    const auto rows = b.matrixRows();
                    const auto cols = b.matrixCols();
                    if (a.vectorSize() != rows) {
                        throw FEException("Forms: vector-matrix multiplication shape mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.resizeVector(cols);
                    for (std::size_t c = 0; c < cols; ++c) {
                        Real sum = 0.0;
                        for (std::size_t r = 0; r < rows; ++r) {
                            sum += a.vectorAt(r) * b.matrixAt(r, c);
                        }
                        out.vectorAt(c) = sum;
                    }
                    return out;
                }
                if (isMatrixKind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
                    const auto rows = a.matrixRows();
                    const auto inner_dim = a.matrixCols();
                    const auto cols = b.matrixCols();
                    if (b.matrixRows() != inner_dim) {
                        throw FEException("Forms: matrix-matrix multiplication shape mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    out.resizeMatrix(rows, cols);
                    for (std::size_t r = 0; r < rows; ++r) {
                        for (std::size_t c = 0; c < cols; ++c) {
                            Real sum = 0.0;
                            for (std::size_t k = 0; k < inner_dim; ++k) {
                                sum += a.matrixAt(r, k) * b.matrixAt(k, c);
                            }
                            out.matrixAt(r, c) = sum;
                        }
                    }
                    return out;
                }
                if (isScalarKind<Real>(a.kind) && isTensor3Kind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Tensor3;
                    out.resizeTensor3(b.tensor3Dim0(), b.tensor3Dim1(), b.tensor3Dim2());
                    for (std::size_t i = 0; i < b.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < b.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < b.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = a.s * b.tensor3At(i, j, k);
                            }
                        }
                    }
                    return out;
                }
                if (isTensor3Kind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Tensor3;
                    out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = a.tensor3At(i, j, k) * b.s;
                            }
                        }
                    }
                    return out;
                }
                if (isScalarKind<Real>(a.kind) && isTensor4Kind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Tensor4;
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = a.s * b.t4[k];
                    }
                    return out;
                }
                if (isTensor4Kind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Tensor4;
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = a.t4[k] * b.s;
                    }
                    return out;
                }
                throw FEException("Forms: unsupported multiplication kinds",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::Divide) {
                if (!isScalarKind<Real>(b.kind)) {
                    throw FEException("Forms: division denominator must be scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isScalarKind<Real>(a.kind)) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, a.s / b.s};
                }
                if (isVectorKind<Real>(a.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Vector;
                    out.resizeVector(a.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) out.vectorAt(d) = a.vectorAt(d) / b.s;
                    return out;
                }
                if (isMatrixKind<Real>(a.kind)) {
                    EvalValue<Real> out;
                    out.kind = a.kind;
                    out.resizeMatrix(a.matrixRows(), a.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = a.matrixAt(r, c) / b.s;
                        }
                    }
                    return out;
                }
                if (isTensor3Kind<Real>(a.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Tensor3;
                    out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = a.tensor3At(i, j, k) / b.s;
                            }
                        }
                    }
                    return out;
                }
                if (isTensor4Kind<Real>(a.kind)) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Tensor4;
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = a.t4[k] / b.s;
                    }
                    return out;
                }
                throw FEException("Forms: unsupported division kinds",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::InnerProduct) {
                if (isScalarKind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, a.s * b.s};
                }
                if (isVectorKind<Real>(a.kind) && isVectorKind<Real>(b.kind)) {
                    if (a.vectorSize() != b.vectorSize()) {
                        throw FEException("Forms: inner(vector,vector) size mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    Real dot = 0.0;
                    for (std::size_t d = 0; d < a.vectorSize(); ++d) {
                        dot += a.vectorAt(d) * b.vectorAt(d);
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, dot};
                }
                if (isMatrixKind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
                    if (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols()) {
                        throw FEException("Forms: inner(matrix,matrix) shape mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    Real sum = 0.0;
                    for (std::size_t r = 0; r < a.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < a.matrixCols(); ++c) {
                            sum += a.matrixAt(r, c) * b.matrixAt(r, c);
                        }
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, sum};
                }
                if (isTensor3Kind<Real>(a.kind) && isTensor3Kind<Real>(b.kind)) {
                    if (a.tensor3Dim0() != b.tensor3Dim0() ||
                        a.tensor3Dim1() != b.tensor3Dim1() ||
                        a.tensor3Dim2() != b.tensor3Dim2()) {
                        throw FEException("Forms: inner(tensor3,tensor3) shape mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    Real sum = 0.0;
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                sum += a.tensor3At(i, j, k) * b.tensor3At(i, j, k);
                            }
                        }
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, sum};
                }
                if (isTensor4Kind<Real>(a.kind) && isTensor4Kind<Real>(b.kind)) {
                    Real sum = 0.0;
                    for (std::size_t k = 0; k < a.t4.size(); ++k) {
                        sum += a.t4[k] * b.t4[k];
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, sum};
                }
                throw FEException("Forms: unsupported inner() kinds",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

	            if (node.type() == FormExprType::DoubleContraction) {
	                if (isTensor4Kind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
	                    EvalValue<Real> out;
	                    out.kind = EvalValue<Real>::Kind::Matrix;
	                    out.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));
	                    for (int i = 0; i < dim; ++i) {
	                        for (int j = 0; j < dim; ++j) {
	                            Real sum = 0.0;
	                            for (int k = 0; k < dim; ++k) {
	                                for (int l = 0; l < dim; ++l) {
	                                    sum += a.t4[idx4(i, j, k, l)] *
	                                           b.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(l));
	                                }
	                            }
	                            out.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = sum;
	                        }
	                    }
	                    return out;
	                }

	                if (isMatrixKind<Real>(a.kind) && isTensor4Kind<Real>(b.kind)) {
	                    EvalValue<Real> out;
	                    out.kind = EvalValue<Real>::Kind::Matrix;
	                    out.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));
	                    for (int k = 0; k < dim; ++k) {
	                        for (int l = 0; l < dim; ++l) {
	                            Real sum = 0.0;
	                            for (int i = 0; i < dim; ++i) {
	                                for (int j = 0; j < dim; ++j) {
	                                    sum += a.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) *
	                                           b.t4[idx4(i, j, k, l)];
	                                }
	                            }
	                            out.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(l)) = sum;
	                        }
	                    }
	                    return out;
	                }

                // Fall back to full contraction for matching shapes.
                if (isScalarKind<Real>(a.kind) && isScalarKind<Real>(b.kind)) {
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, a.s * b.s};
                }
                if (isVectorKind<Real>(a.kind) && isVectorKind<Real>(b.kind)) {
                    if (a.vectorSize() != b.vectorSize()) {
                        throw FEException("Forms: doubleContraction(vector,vector) size mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    Real dot = 0.0;
                    for (std::size_t d = 0; d < a.vectorSize(); ++d) {
                        dot += a.vectorAt(d) * b.vectorAt(d);
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, dot};
                }
                if (isMatrixKind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
                    if (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols()) {
                        throw FEException("Forms: doubleContraction(matrix,matrix) shape mismatch",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    Real sum = 0.0;
                    for (std::size_t r = 0; r < a.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < a.matrixCols(); ++c) {
                            sum += a.matrixAt(r, c) * b.matrixAt(r, c);
                        }
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, sum};
                }
                if (isTensor4Kind<Real>(a.kind) && isTensor4Kind<Real>(b.kind)) {
                    Real sum = 0.0;
                    for (std::size_t k = 0; k < a.t4.size(); ++k) {
                        sum += a.t4[k] * b.t4[k];
                    }
                    return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, sum};
                }

                throw FEException("Forms: unsupported doubleContraction() kinds",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::OuterProduct) {
                if (a.kind == EvalValue<Real>::Kind::Vector && b.kind == EvalValue<Real>::Kind::Vector) {
                    EvalValue<Real> out;
                    out.kind = EvalValue<Real>::Kind::Matrix;
                    const auto rows = a.vectorSize();
                    const auto cols = b.vectorSize();
                    out.resizeMatrix(rows, cols);
                    for (std::size_t r = 0; r < rows; ++r) {
                        for (std::size_t c = 0; c < cols; ++c) {
                            out.matrixAt(r, c) = a.vectorAt(r) * b.vectorAt(c);
                        }
                    }
                    return out;
                }
                throw FEException("Forms: outer() currently supports vector-vector only",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::CrossProduct) {
                if (a.kind != EvalValue<Real>::Kind::Vector || b.kind != EvalValue<Real>::Kind::Vector) {
                    throw FEException("Forms: cross() expects vector arguments",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Real> out;
                out.kind = EvalValue<Real>::Kind::Vector;
                const Real ax = a.v[0], ay = a.v[1], az = a.v[2];
                const Real bx = b.v[0], by = b.v[1], bz = b.v[2];
                out.v[0] = ay * bz - az * by;
                out.v[1] = az * bx - ax * bz;
                out.v[2] = ax * by - ay * bx;
                return out;
            }

            if (node.type() == FormExprType::Power) {
                if (a.kind != EvalValue<Real>::Kind::Scalar || b.kind != EvalValue<Real>::Kind::Scalar) {
                    throw FEException("Forms: pow() expects scalar arguments",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, std::pow(a.s, b.s)};
            }

            if (node.type() == FormExprType::Minimum || node.type() == FormExprType::Maximum) {
                if (a.kind != EvalValue<Real>::Kind::Scalar || b.kind != EvalValue<Real>::Kind::Scalar) {
                    throw FEException("Forms: min/max expects scalar arguments",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const Real v = (node.type() == FormExprType::Minimum) ? std::min(a.s, b.s) : std::max(a.s, b.s);
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, v};
            }

            if (node.type() == FormExprType::Less || node.type() == FormExprType::LessEqual ||
                node.type() == FormExprType::Greater || node.type() == FormExprType::GreaterEqual ||
                node.type() == FormExprType::Equal || node.type() == FormExprType::NotEqual) {
                if (a.kind != EvalValue<Real>::Kind::Scalar || b.kind != EvalValue<Real>::Kind::Scalar) {
                    throw FEException("Forms: comparisons expect scalar arguments",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                bool truth = false;
                switch (node.type()) {
                    case FormExprType::Less: truth = (a.s < b.s); break;
                    case FormExprType::LessEqual: truth = (a.s <= b.s); break;
                    case FormExprType::Greater: truth = (a.s > b.s); break;
                    case FormExprType::GreaterEqual: truth = (a.s >= b.s); break;
                    case FormExprType::Equal: truth = (a.s == b.s); break;
                    case FormExprType::NotEqual: truth = (a.s != b.s); break;
                    default: break;
                }
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, truth ? 1.0 : 0.0};
            }

            throw FEException("Forms: unreachable binary operation",
                              __FILE__, __LINE__, __func__, FEStatus::Unknown);
        }
        case FormExprType::Conditional: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3 || !kids[0] || !kids[1] || !kids[2]) {
                throw std::logic_error("Forms: conditional must have 3 children");
            }
            const auto cond = evalReal(*kids[0], env, side, q);
            if (cond.kind != EvalValue<Real>::Kind::Scalar) {
                throw FEException("Forms: conditional condition must be scalar",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto a = evalReal(*kids[1], env, side, q);
            const auto b = evalReal(*kids[2], env, side, q);
            if (!sameCategory<Real>(a.kind, b.kind)) {
                throw FEException("Forms: conditional branch kind mismatch",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (a.kind == b.kind) {
                return (cond.s > 0.0) ? a : b;
            }
            // Matrix sub-kinds: return as a generic matrix to keep downstream typing stable.
            if (isMatrixKind<Real>(a.kind) && isMatrixKind<Real>(b.kind)) {
                auto out = (cond.s > 0.0) ? a : b;
                out.kind = EvalValue<Real>::Kind::Matrix;
                return out;
            }
            return (cond.s > 0.0) ? a : b;
        }
        case FormExprType::AsVector: {
            const auto kids = node.childrenShared();
            if (kids.empty()) {
                throw FEException("Forms: as_vector expects at least 1 scalar component",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Vector;
            out.resizeVector(kids.size());
            for (std::size_t c = 0; c < kids.size(); ++c) {
                if (!kids[c]) throw std::logic_error("Forms: as_vector has null child");
                const auto v = evalReal(*kids[c], env, side, q);
                if (v.kind != EvalValue<Real>::Kind::Scalar) {
                    throw FEException("Forms: as_vector components must be scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.vectorAt(c) = v.s;
            }
            return out;
        }
        case FormExprType::AsTensor: {
            const auto rows = node.tensorRows().value_or(0);
            const auto cols = node.tensorCols().value_or(0);
            if (rows <= 0 || cols <= 0) {
                throw FEException("Forms: as_tensor requires explicit shape with rows,cols >= 1",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != static_cast<std::size_t>(rows * cols)) {
                throw FEException("Forms: as_tensor child count does not match rows*cols",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Real> out;
            out.kind = EvalValue<Real>::Kind::Matrix;
            out.resizeMatrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const auto idx = static_cast<std::size_t>(r * cols + c);
                    if (!kids[idx]) throw std::logic_error("Forms: as_tensor has null child");
                    const auto v = evalReal(*kids[idx], env, side, q);
                    if (v.kind != EvalValue<Real>::Kind::Scalar) {
                        throw FEException("Forms: as_tensor entries must be scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    out.matrixAt(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) = v.s;
                }
            }
            return out;
        }
        case FormExprType::Component: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("component() must have 1 child");
            const auto a = evalReal(*kids[0], env, side, q);
            const int i = node.componentIndex0().value_or(0);
            const int j = node.componentIndex1().value_or(-1);
            if (isScalarKind<Real>(a.kind)) {
                if (i != 0 || j >= 0) {
                    throw FEException("Forms: component() invalid indices for scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                return a;
            }
            if (isVectorKind<Real>(a.kind)) {
                if (j >= 0) {
                    throw FEException("Forms: component(v,i,j) invalid for vector",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto n = a.vectorSize();
                if (i < 0 || static_cast<std::size_t>(i) >= n) {
                    throw FEException("Forms: component(v,i) index out of range",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                return EvalValue<Real>{EvalValue<Real>::Kind::Scalar, a.vectorAt(static_cast<std::size_t>(i))};
            }
            if (!isMatrixKind<Real>(a.kind)) {
                throw FEException("Forms: component() is not defined for this operand kind",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            // Matrix
            if (j < 0) {
                throw FEException("Forms: component(A,i) missing column index for matrix",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (i < 0 || static_cast<std::size_t>(i) >= rows || j < 0 || static_cast<std::size_t>(j) >= cols) {
                throw FEException("Forms: component(A,i,j) index out of range",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return EvalValue<Real>{EvalValue<Real>::Kind::Scalar,
                                   a.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j))};
        }
        case FormExprType::IndexedAccess:
            throw FEException("Forms: indexed access must be lowered via forms::einsum(...) before compilation/evaluation",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
            break;
    }

    throw FEException("Forms: unsupported expression node in evaluation",
                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
}

// ============================================================================
// Dual (residual/Jacobian) evaluation
// ============================================================================

struct ConstitutiveCallCacheDual;

struct EvalEnvDual {
    const assembly::AssemblyContext& minus;
    const assembly::AssemblyContext* plus{nullptr};
    Side test_active{Side::Minus};
    Side trial_active{Side::Minus};
    LocalIndex i{0};
    std::size_t n_trial_dofs{0};
    DualWorkspace* ws{nullptr};
    const ConstitutiveStateLayout* constitutive_state{nullptr};
    ConstitutiveCallCacheDual* constitutive_cache{nullptr};

    // Optional coupled-scalar seeding (for sensitivity assembly).
    std::optional<std::uint32_t> coupled_integral_seed_slot{};
    // Row-major matrix d(aux[slot])/dQ[col] with num_integrals == coupled_aux_dseed_cols.
    std::span<const Real> coupled_aux_dseed{};
    std::size_t coupled_aux_dseed_cols{0u};
};

EvalValue<Dual> evalDual(const FormExprNode& node,
                         const EvalEnvDual& env,
                         Side side,
                         LocalIndex q);

struct ConstitutiveCallCacheDual {
    std::unordered_map<ConstitutiveCallKey,
                       std::vector<EvalValue<Dual>>,
                       ConstitutiveCallKeyHash> values{};
};

EvalValue<Dual> evalDualUnary(const FormExprNode& node,
                              const EvalEnvDual& env,
                              Side side,
                              LocalIndex q)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 1 || !kids[0]) {
        throw std::logic_error("Forms: unary node must have exactly 1 child");
    }
    return evalDual(*kids[0], env, side, q);
}

EvalValue<Dual> evalConstitutiveOutputDual(const FormExprNode& call_node,
                                           const EvalEnvDual& env,
                                           Side side,
                                           LocalIndex q,
                                           std::size_t output_index)
{
    if (call_node.type() != FormExprType::Constitutive) {
        throw FEException("Forms: evalConstitutiveOutputDual called on non-constitutive node",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto* model = call_node.constitutiveModel();
    if (!model) {
        throw FEException("Forms: Constitutive node has no model (dual)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto n_outputs = model->outputCount();
    if (output_index >= n_outputs) {
        throw FEException("Forms: constitutive output index out of range (dual)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    FE_CHECK_NOT_NULL(env.ws, "Forms: evalConstitutiveOutputDual: workspace");

    if (env.constitutive_cache != nullptr) {
        ConstitutiveCallKey key{&call_node, side, q, env.i, 0};
        auto it = env.constitutive_cache->values.find(key);
        if (it != env.constitutive_cache->values.end()) {
            if (it->second.size() != n_outputs) {
                throw FEException("Forms: constitutive cache output size mismatch (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::Unknown);
            }
            return it->second[output_index];
        }
    }

    const auto& ctx = ctxForSide(env.minus, env.plus, side);
    const int dim = ctx.dimension();

    ConstitutiveEvalContext mctx;
    mctx.domain = [&]() {
        switch (ctx.contextType()) {
            case assembly::ContextType::Cell:
                return ConstitutiveEvalContext::Domain::Cell;
            case assembly::ContextType::BoundaryFace:
                return ConstitutiveEvalContext::Domain::BoundaryFace;
            case assembly::ContextType::InteriorFace:
                return ConstitutiveEvalContext::Domain::InteriorFace;
            default:
                return ConstitutiveEvalContext::Domain::Cell;
        }
    }();
    mctx.side = (side == Side::Plus) ? ConstitutiveEvalContext::TraceSide::Plus
                                     : ConstitutiveEvalContext::TraceSide::Minus;
    mctx.dim = dim;
    mctx.x = ctx.physicalPoint(q);
    mctx.time = ctx.time();
    mctx.dt = ctx.timeStep();
    mctx.cell_id = ctx.cellId();
    mctx.face_id = ctx.faceId();
    mctx.local_face_id = ctx.localFaceId();
    mctx.boundary_marker = ctx.boundaryMarker();
    mctx.q = q;
    mctx.num_qpts = ctx.numQuadraturePoints();
    mctx.get_real_param = ctx.realParameterGetter();
    mctx.get_param = ctx.parameterGetter();
    mctx.user_data = ctx.userData();

    std::size_t state_offset_bytes = 0u;
    std::size_t state_bytes = 0u;

    if (env.constitutive_state != nullptr) {
        if (const auto* block = env.constitutive_state->find(model); block != nullptr && block->bytes > 0u) {
            if (!ctx.hasMaterialState()) {
                throw FEException("Forms: Constitutive node requires material state but no state is bound in this context (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto state_old = ctx.materialStateOld(q);
            const auto state_work = ctx.materialStateWork(q);
            const auto need = block->offset_bytes + block->bytes;
            if (state_old.size() < need || state_work.size() < need) {
                throw FEException("Forms: bound material state buffer is smaller than required by constitutive state layout (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            state_offset_bytes = block->offset_bytes;
            state_bytes = block->bytes;
            mctx.state_old = state_old.subspan(block->offset_bytes, block->bytes);
            mctx.state_work = state_work.subspan(block->offset_bytes, block->bytes);
        }
    }

    const auto kids = call_node.childrenShared();
    if (kids.empty()) {
        throw FEException("Forms: Constitutive node must have at least 1 child (dual)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    struct NonlocalDualCtx {
        const EvalEnvDual* env{nullptr};
        Side eval_side{Side::Minus};
        const std::vector<std::shared_ptr<FormExprNode>>* inputs{nullptr};
        const assembly::AssemblyContext* ctx{nullptr};
        std::size_t state_offset_bytes{0u};
        std::size_t state_bytes{0u};
    };

    ConstitutiveEvalContext::NonlocalAccess nonlocal{};
    NonlocalDualCtx nonlocal_ctx{&env, side, &kids, &ctx, state_offset_bytes, state_bytes};
    nonlocal.self = &nonlocal_ctx;
    nonlocal.input_dual = +[](const void* self,
                              std::size_t input_index,
                              LocalIndex qpt,
                              DualWorkspace& workspace) -> EvalValue<Dual> {
        const auto& s = *static_cast<const NonlocalDualCtx*>(self);
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal input_dual missing context");
        FE_THROW_IF(s.inputs == nullptr, FEException,
                    "Forms: nonlocal input_dual missing inputs");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal input_dual qpt out of range");
        FE_THROW_IF(input_index >= s.inputs->size(), InvalidArgumentException,
                    "Forms: nonlocal input_dual input_index out of range");
        const auto& child = (*s.inputs)[input_index];
        FE_THROW_IF(!child, InvalidArgumentException,
                    "Forms: nonlocal input_dual input node is null");
        FE_THROW_IF(s.env == nullptr, FEException,
                    "Forms: nonlocal input_dual missing eval environment");
        FE_THROW_IF(s.env->ws == nullptr, FEException,
                    "Forms: nonlocal input_dual missing workspace pointer");
        FE_THROW_IF(&workspace != s.env->ws, InvalidArgumentException,
                    "Forms: nonlocal input_dual workspace mismatch");
        return evalDual(*child, *s.env, s.eval_side, qpt);
    };
    nonlocal.state_old = +[](const void* self, LocalIndex qpt) -> std::span<const std::byte> {
        const auto& s = *static_cast<const NonlocalDualCtx*>(self);
        if (s.state_bytes == 0u) return {};
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal state_old missing context (dual)");
        FE_THROW_IF(!s.ctx->hasMaterialState(), InvalidArgumentException,
                    "Forms: nonlocal state_old requires bound material state (dual)");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal state_old qpt out of range (dual)");
        const auto state = s.ctx->materialStateOld(qpt);
        const auto need = s.state_offset_bytes + s.state_bytes;
        FE_THROW_IF(state.size() < need, InvalidArgumentException,
                    "Forms: nonlocal state_old buffer smaller than required (dual)");
        return state.subspan(s.state_offset_bytes, s.state_bytes);
    };
    nonlocal.state_work = +[](const void* self, LocalIndex qpt) -> std::span<std::byte> {
        const auto& s = *static_cast<const NonlocalDualCtx*>(self);
        if (s.state_bytes == 0u) return {};
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal state_work missing context (dual)");
        FE_THROW_IF(!s.ctx->hasMaterialState(), InvalidArgumentException,
                    "Forms: nonlocal state_work requires bound material state (dual)");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal state_work qpt out of range (dual)");
        const auto state = s.ctx->materialStateWork(qpt);
        const auto need = s.state_offset_bytes + s.state_bytes;
        FE_THROW_IF(state.size() < need, InvalidArgumentException,
                    "Forms: nonlocal state_work buffer smaller than required (dual)");
        return state.subspan(s.state_offset_bytes, s.state_bytes);
    };
    nonlocal.physical_point = +[](const void* self, LocalIndex qpt) -> std::array<Real, 3> {
        const auto& s = *static_cast<const NonlocalDualCtx*>(self);
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal physical_point missing context (dual)");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal physical_point qpt out of range (dual)");
        return s.ctx->physicalPoint(qpt);
    };
    nonlocal.integration_weight = +[](const void* self, LocalIndex qpt) -> Real {
        const auto& s = *static_cast<const NonlocalDualCtx*>(self);
        FE_THROW_IF(s.ctx == nullptr, FEException,
                    "Forms: nonlocal integration_weight missing context (dual)");
        FE_THROW_IF(qpt >= s.ctx->numQuadraturePoints(), InvalidArgumentException,
                    "Forms: nonlocal integration_weight qpt out of range (dual)");
        return s.ctx->integrationWeight(qpt);
    };

    mctx.nonlocal = &nonlocal;

    std::vector<EvalValue<Dual>> inputs;
    inputs.reserve(kids.size());
    for (const auto& child : kids) {
        if (!child) {
            throw FEException("Forms: Constitutive node has a null child (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        inputs.push_back(evalDual(*child, env, side, q));
    }

    std::vector<EvalValue<Dual>> outputs(n_outputs);
    model->evaluateNaryOutputs({inputs.data(), inputs.size()}, mctx, *env.ws, {outputs.data(), outputs.size()});

    if (env.constitutive_cache != nullptr) {
        ConstitutiveCallKey key{&call_node, side, q, env.i, 0};
        auto [it, inserted] = env.constitutive_cache->values.emplace(key, std::move(outputs));
        (void)inserted;
        return it->second[output_index];
    }

    return outputs[output_index];
}

EvalValue<Dual> evalDual(const FormExprNode& node,
                         const EvalEnvDual& env,
                         Side side,
                         LocalIndex q)
{
    const auto& ctx = ctxForSide(env.minus, env.plus, side);
    const int dim = ctx.dimension();

    switch (node.type()) {
        case FormExprType::Constant: {
            const Real v = node.constantValue().value_or(0.0);
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(v, env.ws->alloc());
            return out;
        }
        case FormExprType::Time: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.time(), env.ws->alloc());
            return out;
        }
        case FormExprType::TimeStep: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.timeStep(), env.ws->alloc());
            return out;
        }
        case FormExprType::Coordinate: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Vector;
            out.vector_size = dim;
            const auto x = ctx.physicalPoint(q);
            for (int d = 0; d < 3; ++d) {
                out.v[static_cast<std::size_t>(d)] = makeDualConstant(x[static_cast<std::size_t>(d)], env.ws->alloc());
            }
            return out;
        }
        case FormExprType::ReferenceCoordinate: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Vector;
            out.vector_size = dim;
            const auto X = ctx.quadraturePoint(q);
            for (int d = 0; d < 3; ++d) {
                out.v[static_cast<std::size_t>(d)] = makeDualConstant(X[static_cast<std::size_t>(d)], env.ws->alloc());
            }
            return out;
        }
        case FormExprType::Identity: {
            const int idim = node.identityDim().value_or(dim);
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Matrix;
            if (idim > 0) {
                out.resizeMatrix(static_cast<std::size_t>(idim), static_cast<std::size_t>(idim));
                for (int r = 0; r < idim; ++r) {
                    for (int c = 0; c < idim; ++c) {
                        out.matrixAt(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) =
                            makeDualConstant((r == c) ? 1.0 : 0.0, env.ws->alloc());
                    }
                }
            }
            return out;
        }
        case FormExprType::Jacobian: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Matrix;
            out.matrix_rows = dim;
            out.matrix_cols = dim;
            const auto J = ctx.jacobian(q);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                        makeDualConstant(J[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                }
            }
            return out;
        }
        case FormExprType::JacobianInverse: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Matrix;
            out.matrix_rows = dim;
            out.matrix_cols = dim;
            const auto Jinv = ctx.inverseJacobian(q);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                        makeDualConstant(Jinv[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                }
            }
            return out;
        }
        case FormExprType::JacobianDeterminant: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.jacobianDet(q), env.ws->alloc());
            return out;
        }
        case FormExprType::CellDiameter: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.cellDiameter(), env.ws->alloc());
            return out;
        }
        case FormExprType::CellVolume: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.cellVolume(), env.ws->alloc());
            return out;
        }
        case FormExprType::FacetArea: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.facetArea(), env.ws->alloc());
            return out;
        }
        case FormExprType::CellDomainId: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(static_cast<Real>(ctx.cellDomainId()), env.ws->alloc());
            return out;
        }
        case FormExprType::TestFunction: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: TestFunction must be bound to a FunctionSpace",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                const Real v = (env.test_active == side) ? ctx.basisValue(env.i, q) : 0.0;
                out.s = makeDualConstant(v, env.ws->alloc());
                return out;
            }

            if (sig->field_type == FieldType::Vector) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = sig->value_dimension;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] = makeDualConstant(0.0, env.ws->alloc());
                }
                if (env.test_active != side) {
                    return out;
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: TestFunction vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: vector TestFunction DOF count is not divisible by value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: TestFunction vector DOF index out of range for its value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.v[static_cast<std::size_t>(comp)].value = ctx.basisValue(env.i, q);
                return out;
            }

            throw FEException("Forms: TestFunction field type not supported (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::TrialFunction: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: TrialFunction must be bound to a FunctionSpace",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                const Real val = ctx.solutionValue(q);
                Dual u = makeDualConstant(val, env.ws->alloc());
                if (env.trial_active == side) {
                    // Seed du/dU_j = phi_j at this quadrature point
                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                        u.deriv[j] = ctx.trialBasisValue(static_cast<LocalIndex>(j), q);
                    }
                }
                out.s = u;
                return out;
            }

            if (sig->field_type == FieldType::Vector) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: TrialFunction vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const LocalIndex n_trial = ctx.numTrialDofs();
                if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: vector TrialFunction DOF count is not divisible by value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));

                out.vector_size = vd;
                const auto u_val = ctx.solutionVectorValue(q);
                for (int c = 0; c < 3; ++c) {
                    out.v[static_cast<std::size_t>(c)] = makeDualConstant(u_val[static_cast<std::size_t>(c)], env.ws->alloc());
                }

                if (env.trial_active == side) {
                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                        const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                        if (comp_j < 0 || comp_j >= vd) {
                            throw FEException("Forms: TrialFunction derivative seeding index out of range (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        out.v[static_cast<std::size_t>(comp_j)].deriv[j] =
                            ctx.trialBasisValue(static_cast<LocalIndex>(j), q);
                    }
                }

                // Zero unused components above value_dimension.
                for (int c = vd; c < 3; ++c) {
                    out.v[static_cast<std::size_t>(c)].value = 0.0;
                }
                return out;
            }

            throw FEException("Forms: TrialFunction field type not supported (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw FEException("Forms: DiscreteField must be bound to a FunctionSpace (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto fid = node.fieldId();
            if (!fid || *fid == INVALID_FIELD_ID) {
                throw FEException("Forms: DiscreteField node missing a valid FieldId (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (sig->field_type == FieldType::Scalar) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant(ctx.fieldValue(*fid, q), env.ws->alloc());
                return out;
            }

            if (sig->field_type == FieldType::Vector) {
                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: DiscreteField vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto u_val = ctx.fieldVectorValue(*fid, q);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = vd;
                for (int c = 0; c < 3; ++c) {
                    out.v[static_cast<std::size_t>(c)] = makeDualConstant(u_val[static_cast<std::size_t>(c)], env.ws->alloc());
                }
                for (int c = vd; c < 3; ++c) {
                    out.v[static_cast<std::size_t>(c)].value = 0.0;
                }
                return out;
            }

            throw FEException("Forms: DiscreteField field type not supported (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::ParameterSymbol: {
            const auto nm = node.symbolName();
            FE_THROW_IF(!nm || nm->empty(), InvalidArgumentException,
                        "Forms: ParameterSymbol node missing name (dual)");
            const auto* get = ctx.realParameterGetter();
            FE_THROW_IF(get == nullptr || !static_cast<bool>(*get), InvalidArgumentException,
                        "Forms: ParameterSymbol requires a real parameter getter in AssemblyContext (dual)");
            const auto v = (*get)(*nm);
            FE_THROW_IF(!v.has_value(), InvalidArgumentException,
                        "Forms: missing required parameter '" + std::string(*nm) + "' (dual)");
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(*v, env.ws->alloc());
            return out;
        }
        case FormExprType::ParameterRef: {
            const auto slot = node.slotIndex().value_or(0u);
            const auto vals = ctx.jitConstants();
            FE_THROW_IF(vals.empty(), InvalidArgumentException,
                        "Forms: ParameterRef requires AssemblyContext::jitConstants() (dual)");
            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                        "Forms: ParameterRef slot out of range (dual)");
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(vals[slot], env.ws->alloc());
            return out;
        }
        case FormExprType::BoundaryIntegralRef: {
            const auto slot = node.slotIndex().value_or(0u);
            const auto vals = ctx.coupledIntegrals();
            FE_THROW_IF(vals.empty(), InvalidArgumentException,
                        "Forms: BoundaryIntegralRef requires coupled integrals in AssemblyContext (dual)");
            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                        "Forms: BoundaryIntegralRef slot out of range (dual)");
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(vals[slot], env.ws->alloc());
            if (env.coupled_integral_seed_slot.has_value() &&
                env.coupled_integral_seed_slot.value() == slot &&
                !out.s.deriv.empty()) {
                out.s.deriv[0] = 1.0;
            }
            return out;
        }
        case FormExprType::AuxiliaryStateRef: {
            const auto slot = node.slotIndex().value_or(0u);
            const auto vals = ctx.coupledAuxState();
            FE_THROW_IF(vals.empty(), InvalidArgumentException,
                        "Forms: AuxiliaryStateRef requires coupled auxiliary state in AssemblyContext (dual)");
            FE_THROW_IF(slot >= vals.size(), InvalidArgumentException,
                        "Forms: AuxiliaryStateRef slot out of range (dual)");
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(vals[slot], env.ws->alloc());
            if (!env.coupled_aux_dseed.empty() && env.coupled_integral_seed_slot.has_value()) {
                FE_THROW_IF(env.coupled_aux_dseed_cols == 0u, InvalidArgumentException,
                            "Forms: coupled auxiliary seed matrix missing column count (dual)");
                const auto k = static_cast<std::size_t>(env.coupled_integral_seed_slot.value());
                FE_THROW_IF(k >= env.coupled_aux_dseed_cols, InvalidArgumentException,
                            "Forms: coupled auxiliary seed column index out of range (dual)");
                const auto idx = static_cast<std::size_t>(slot) * env.coupled_aux_dseed_cols + k;
                FE_THROW_IF(idx >= env.coupled_aux_dseed.size(), InvalidArgumentException,
                            "Forms: coupled auxiliary seed matrix is too small (dual)");
                if (!out.s.deriv.empty()) {
                    out.s.deriv[0] = env.coupled_aux_dseed[idx];
                }
            }
            return out;
        }
        case FormExprType::MaterialStateOldRef: {
            const auto off = node.stateOffsetBytes();
            FE_THROW_IF(!off.has_value(), InvalidArgumentException,
                        "Forms: MaterialStateOldRef node missing offset (dual)");
            FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
                        "Forms: MaterialStateOldRef requires material state in AssemblyContext (dual)");
            const auto state = ctx.materialStateOld(q);
            const auto offset = static_cast<std::size_t>(*off);
            FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
                        "Forms: MaterialStateOldRef offset out of range (dual)");
            Real v = 0.0;
            std::memcpy(&v, state.data() + offset, sizeof(Real));
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(v, env.ws->alloc());
            return out;
        }
        case FormExprType::MaterialStateWorkRef: {
            const auto off = node.stateOffsetBytes();
            FE_THROW_IF(!off.has_value(), InvalidArgumentException,
                        "Forms: MaterialStateWorkRef node missing offset (dual)");
            FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
                        "Forms: MaterialStateWorkRef requires material state in AssemblyContext (dual)");
            const auto state = ctx.materialStateWork(q);
            const auto offset = static_cast<std::size_t>(*off);
            FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
                        "Forms: MaterialStateWorkRef offset out of range (dual)");
            Real v = 0.0;
            std::memcpy(&v, state.data() + offset, sizeof(Real));
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(v, env.ws->alloc());
            return out;
        }
        case FormExprType::PreviousSolutionRef: {
            const int k = node.historyIndex().value_or(1);
            FE_THROW_IF(k <= 0, InvalidArgumentException,
                        "Forms: PreviousSolutionRef requires k >= 1 (dual)");
            if (ctx.trialFieldType() == FieldType::Vector) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = ctx.trialValueDimension();
                const auto u_val = ctx.previousSolutionVectorValue(q, k);
                for (int c = 0; c < 3; ++c) {
                    out.v[static_cast<std::size_t>(c)] = makeDualConstant(u_val[static_cast<std::size_t>(c)], env.ws->alloc());
                }
                return out;
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = makeDualConstant(ctx.previousSolutionValue(q, k), env.ws->alloc());
            return out;
        }
        case FormExprType::Coefficient: {
            if (const auto* f = node.timeScalarCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant((*f)(x[0], x[1], x[2], ctx.time()), env.ws->alloc());
                return out;
            }
            if (const auto* f = node.scalarCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant((*f)(x[0], x[1], x[2]), env.ws->alloc());
                return out;
            }
            if (const auto* f = node.vectorCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto vv = (*f)(x[0], x[1], x[2]);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = dim;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] = makeDualConstant(vv[static_cast<std::size_t>(d)], env.ws->alloc());
                }
                return out;
            }
            if (const auto* f = node.matrixCoefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto mm = (*f)(x[0], x[1], x[2]);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeDualConstant(mm[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                    }
                }
                return out;
            }
            if (const auto* f = node.tensor3Coefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto tt = (*f)(x[0], x[1], x[2]);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Tensor3;
                out.tensor3_dim0 = dim;
                out.tensor3_dim1 = dim;
                out.tensor3_dim2 = dim;
                for (std::size_t i = 0; i < tt.size(); ++i) {
                    out.t3[i] = makeDualConstant(tt[i], env.ws->alloc());
                }
                return out;
            }
            if (const auto* f = node.tensor4Coefficient(); f) {
                const auto x = ctx.physicalPoint(q);
                const auto vv = (*f)(x[0], x[1], x[2]);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Tensor4;
                for (std::size_t i = 0; i < vv.size(); ++i) {
                    out.t4[i] = makeDualConstant(vv[i], env.ws->alloc());
                }
                return out;
            }
            throw FEException("Forms: coefficient node has no callable",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        case FormExprType::Constitutive:
            return evalConstitutiveOutputDual(node, env, side, q, 0u);
        case FormExprType::ConstitutiveOutput: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw FEException("Forms: ConstitutiveOutput node must have exactly 1 child (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto out_idx = node.constitutiveOutputIndex().value_or(0);
            FE_THROW_IF(out_idx < 0, InvalidArgumentException,
                        "Forms: ConstitutiveOutput node has negative output index (dual)");
            return evalConstitutiveOutputDual(*kids[0], env, side, q, static_cast<std::size_t>(out_idx));
        }
        case FormExprType::Normal: {
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Vector;
            out.vector_size = dim;
            const auto n = ctx.normal(q);
            for (int d = 0; d < 3; ++d) {
                out.v[static_cast<std::size_t>(d)] = makeDualConstant(n[static_cast<std::size_t>(d)], env.ws->alloc());
            }
            return out;
	        }
	        case FormExprType::Gradient: {
	            return evalSpatialJet<Dual>(node, env, side, q, 0).value;
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1 || !kids[0]) throw std::logic_error("grad must have 1 child");
	            const auto& child = *kids[0];

            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: grad(TestFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                if (sig->field_type == FieldType::Scalar) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.vector_size = dim;
                    const auto g = (env.test_active == side) ? ctx.physicalGradient(env.i, q)
                                                             : assembly::AssemblyContext::Vector3D{0.0, 0.0, 0.0};
                    for (int d = 0; d < 3; ++d) {
                        out.v[static_cast<std::size_t>(d)] = makeDualConstant(g[static_cast<std::size_t>(d)], env.ws->alloc());
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: grad(TestFunction) vector value_dimension must be 1..3 (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.matrix_rows = vd;
                    out.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeDualConstant(0.0, env.ws->alloc());
                        }
                    }
                    if (env.test_active != side) {
                        return out;
                    }
                    const LocalIndex n_test = ctx.numTestDofs();
                    if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: grad(TestFunction) DOF count is not divisible by value_dimension (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                    const int comp = static_cast<int>(env.i / dofs_per_component);
                    if (comp < 0 || comp >= vd) {
                        throw FEException("Forms: grad(TestFunction) vector DOF index out of range (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const auto g = ctx.physicalGradient(env.i, q);
                    for (int c = 0; c < dim; ++c) {
                        out.m[static_cast<std::size_t>(comp)][static_cast<std::size_t>(c)].value = g[static_cast<std::size_t>(c)];
                    }
                    return out;
                }

                throw FEException("Forms: grad(TestFunction) field type not supported (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (child.type() == FormExprType::TrialFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: grad(TrialFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                if (sig->field_type == FieldType::Scalar) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.vector_size = dim;
                    const auto gval = ctx.solutionGradient(q);

                    for (int d = 0; d < 3; ++d) {
                        Dual g = makeDualConstant(gval[static_cast<std::size_t>(d)], env.ws->alloc());
                        if (env.trial_active == side) {
                            for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);
                                g.deriv[j] = grad_j[static_cast<std::size_t>(d)];
                            }
                        }
                        out.v[static_cast<std::size_t>(d)] = g;
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: grad(TrialFunction) vector value_dimension must be 1..3 (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.matrix_rows = vd;
                    out.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeDualConstant(0.0, env.ws->alloc());
                        }
                    }
                    if (env.trial_active != side) {
                        return out;
                    }
                    const LocalIndex n_trial = ctx.numTrialDofs();
                    if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                        throw FEException("Forms: grad(TrialFunction) DOF count is not divisible by value_dimension (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const LocalIndex dofs_per_component =
                        static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));

                    const auto Jval = ctx.solutionJacobian(q);
                    for (int r = 0; r < vd; ++r) {
                        for (int c = 0; c < dim; ++c) {
                            Dual Jij = makeDualConstant(Jval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)],
                                                        env.ws->alloc());
                            for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                                if (comp_j == r) {
                                    const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);
                                    Jij.deriv[j] = grad_j[static_cast<std::size_t>(c)];
                                }
                            }
                            out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = Jij;
                        }
                    }
                    return out;
                }

                throw FEException("Forms: grad(TrialFunction) field type not supported (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: grad(DiscreteField) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto fid = child.fieldId();
                if (!fid || *fid == INVALID_FIELD_ID) {
                    throw FEException("Forms: grad(DiscreteField) missing a valid FieldId (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                if (sig->field_type == FieldType::Scalar) {
                    const auto gval = ctx.fieldGradient(*fid, q);
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.vector_size = dim;
                    for (int d = 0; d < 3; ++d) {
                        out.v[static_cast<std::size_t>(d)] = makeDualConstant(gval[static_cast<std::size_t>(d)], env.ws->alloc());
                    }
                    return out;
                }

                if (sig->field_type == FieldType::Vector) {
                    const int vd = sig->value_dimension;
                    if (vd <= 0 || vd > 3) {
                        throw FEException("Forms: grad(DiscreteField) vector value_dimension must be 1..3 (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    const auto Jval = ctx.fieldJacobian(*fid, q);
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.matrix_rows = vd;
                    out.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeDualConstant(Jval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                        }
                    }
                    return out;
                }

                throw FEException("Forms: grad(DiscreteField) field type not supported (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (child.type() == FormExprType::Constant) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = dim;
                for (int d = 0; d < 3; ++d) out.v[static_cast<std::size_t>(d)] = makeDualConstant(0.0, env.ws->alloc());
                return out;
            }

            if (child.type() == FormExprType::Coefficient && child.scalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto g = fdGradScalar(*child.scalarCoefficient(), x, dim);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = dim;
                for (int d = 0; d < 3; ++d) out.v[static_cast<std::size_t>(d)] = makeDualConstant(g[static_cast<std::size_t>(d)], env.ws->alloc());
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.timeScalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto g = fdGradScalarTime(*child.timeScalarCoefficient(), x, ctx.time(), dim);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = dim;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] = makeDualConstant(g[static_cast<std::size_t>(d)], env.ws->alloc());
                }
                return out;
            }

            throw FEException("Forms: grad() currently supports TestFunction, TrialFunction, Constant, and scalar Coefficient only",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
	        case FormExprType::Hessian: {
	            return evalSpatialJet<Dual>(node, env, side, q, 0).value;
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1 || !kids[0]) throw std::logic_error("H() must have 1 child (dual)");
	            const auto& child = *kids[0];

            if (child.type() == FormExprType::Component) {
                const auto ckids = child.childrenShared();
                if (ckids.size() != 1 || !ckids[0]) throw std::logic_error("component() must have 1 child (dual)");
                const auto& base = *ckids[0];
                const int comp = child.componentIndex0().value_or(0);
                const int col = child.componentIndex1().value_or(-1);
                if (col >= 0) {
                    throw FEException("Forms: H(component(A,i,j)) is not supported (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                }

                if (base.type() == FormExprType::TestFunction) {
                    const auto* sig = base.spaceSignature();
                    if (!sig) {
                        throw FEException("Forms: H(component(TestFunction,i)) requires a bound FunctionSpace (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }

                    assembly::AssemblyContext::Matrix3x3 H{};
                    if (sig->field_type == FieldType::Scalar) {
                        if (comp != 0) {
                            throw FEException("Forms: H(component(TestFunction,i)) invalid component index for scalar-valued TestFunction (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        if (env.test_active == side) {
                            H = ctx.physicalHessian(env.i, q);
                        }
                    } else if (sig->field_type == FieldType::Vector) {
                        const int vd = sig->value_dimension;
                        if (vd <= 0 || vd > 3) {
                            throw FEException("Forms: H(component(TestFunction,i)) vector value_dimension must be 1..3 (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        if (comp < 0 || comp >= vd) {
                            throw FEException("Forms: H(component(TestFunction,i)) component index out of range (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        if (env.test_active == side) {
                            const LocalIndex n_test = ctx.numTestDofs();
                            if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                                throw FEException("Forms: H(component(TestFunction,i)) DOF count is not divisible by value_dimension (dual)",
                                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                            }
                            const LocalIndex dofs_per_component =
                                static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                            const int comp_i = static_cast<int>(env.i / dofs_per_component);
                            if (comp_i == comp) {
                                H = ctx.physicalHessian(env.i, q);
                            }
                        }
                    } else {
                        throw FEException("Forms: H(component(TestFunction,i)) field type not supported (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
                    }

                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.matrix_rows = dim;
                    out.matrix_cols = dim;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                                makeDualConstant(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                        }
                    }
                    return out;
                }

                if (base.type() == FormExprType::TrialFunction) {
                    const auto* sig = base.spaceSignature();
                    if (!sig) {
                        throw FEException("Forms: H(component(TrialFunction,i)) requires a bound FunctionSpace (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }

                    if (sig->field_type == FieldType::Scalar) {
                        if (comp != 0) {
                            throw FEException("Forms: H(component(TrialFunction,i)) invalid component index for scalar-valued TrialFunction (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }

                        const auto Hval = ctx.solutionHessian(q);
                        EvalValue<Dual> out;
                        out.kind = EvalValue<Dual>::Kind::Matrix;
                        out.matrix_rows = dim;
                        out.matrix_cols = dim;
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                Dual h = makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                                if (env.trial_active == side) {
                                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                        const auto Hj = ctx.trialPhysicalHessian(static_cast<LocalIndex>(j), q);
                                        h.deriv[j] = Hj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                    }
                                }
                                out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = h;
                            }
                        }
                        return out;
                    }

                    if (sig->field_type == FieldType::Vector) {
                        const int vd = sig->value_dimension;
                        if (vd <= 0 || vd > 3) {
                            throw FEException("Forms: H(component(TrialFunction,i)) vector value_dimension must be 1..3 (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        if (comp < 0 || comp >= vd) {
                            throw FEException("Forms: H(component(TrialFunction,i)) component index out of range (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        const LocalIndex n_trial = ctx.numTrialDofs();
                        if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                            throw FEException("Forms: H(component(TrialFunction,i)) DOF count is not divisible by value_dimension (dual)",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        const LocalIndex dofs_per_component =
                            static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));

                        const auto Hval = ctx.solutionComponentHessian(q, comp);
                        EvalValue<Dual> out;
                        out.kind = EvalValue<Dual>::Kind::Matrix;
                        out.matrix_rows = dim;
                        out.matrix_cols = dim;
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                Dual h = makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                                if (env.trial_active == side) {
                                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                        const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                                        if (comp_j == comp) {
                                            const auto Hj = ctx.trialPhysicalHessian(static_cast<LocalIndex>(j), q);
                                            h.deriv[j] = Hj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                                        }
                                    }
                                }
                                out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = h;
                            }
                        }
                        return out;
                    }

	                    throw FEException("Forms: H(component(TrialFunction,i)) field type not supported (dual)",
	                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	                }

	                if (base.type() == FormExprType::DiscreteField || base.type() == FormExprType::StateField) {
	                    const auto* sig = base.spaceSignature();
	                    if (!sig) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) requires a bound FunctionSpace (dual)",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	                    const auto fid = base.fieldId();
	                    if (!fid || *fid == INVALID_FIELD_ID) {
	                        throw FEException("Forms: H(component(DiscreteField,i)) missing a valid FieldId (dual)",
	                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                    }
	
	                    if (sig->field_type == FieldType::Scalar) {
	                        if (comp != 0) {
	                            throw FEException("Forms: H(component(DiscreteField,i)) invalid component index for scalar-valued DiscreteField (dual)",
	                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                        }
	                        const auto Hval = ctx.fieldHessian(*fid, q);
	                        EvalValue<Dual> out;
	                        out.kind = EvalValue<Dual>::Kind::Matrix;
	                        out.matrix_rows = dim;
	                        out.matrix_cols = dim;
	                        for (int r = 0; r < 3; ++r) {
	                            for (int c = 0; c < 3; ++c) {
	                                out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
	                                    makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
	                            }
	                        }
	                        return out;
	                    }
	
	                    if (sig->field_type == FieldType::Vector) {
	                        const int vd = sig->value_dimension;
	                        if (vd <= 0 || vd > 3) {
	                            throw FEException("Forms: H(component(DiscreteField,i)) vector value_dimension must be 1..3 (dual)",
	                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                        }
	                        if (comp < 0 || comp >= vd) {
	                            throw FEException("Forms: H(component(DiscreteField,i)) component index out of range (dual)",
	                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                        }
	                        const auto Hval = ctx.fieldComponentHessian(*fid, q, comp);
	                        EvalValue<Dual> out;
	                        out.kind = EvalValue<Dual>::Kind::Matrix;
	                        out.matrix_rows = dim;
	                        out.matrix_cols = dim;
	                        for (int r = 0; r < 3; ++r) {
	                            for (int c = 0; c < 3; ++c) {
	                                out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
	                                    makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
	                            }
	                        }
	                        return out;
	                    }
	
	                    throw FEException("Forms: H(component(DiscreteField,i)) field type not supported (dual)",
	                                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	                }

	                throw FEException("Forms: H(component(...)) currently supports TestFunction, TrialFunction, and DiscreteField only (dual)",
	                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	            }

            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: H(TestFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Scalar) {
                    throw FEException("Forms: H(TestFunction) requires a scalar-valued TestFunction; use H(component(v,i)) for vector-valued spaces (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto H = (env.test_active == side) ? ctx.physicalHessian(env.i, q) : assembly::AssemblyContext::Matrix3x3{};
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeDualConstant(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                    }
                }
                return out;
            }

	            if (child.type() == FormExprType::TrialFunction) {
	                const auto* sig = child.spaceSignature();
	                if (!sig) {
	                    throw FEException("Forms: H(TrialFunction) requires a bound FunctionSpace (dual)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Scalar) {
                    throw FEException("Forms: H(TrialFunction) requires a scalar-valued TrialFunction; use H(component(u,i)) for vector-valued spaces (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto Hval = ctx.solutionHessian(q);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        Dual h = makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                        if (env.trial_active == side) {
                            for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                                const auto Hj = ctx.trialPhysicalHessian(static_cast<LocalIndex>(j), q);
                                h.deriv[j] = Hj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            }
                        }
                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = h;
                    }
	                }
	                return out;
	            }
	
	            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
	                const auto* sig = child.spaceSignature();
	                if (!sig) {
	                    throw FEException("Forms: H(DiscreteField) requires a bound FunctionSpace (dual)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                const auto fid = child.fieldId();
	                if (!fid || *fid == INVALID_FIELD_ID) {
	                    throw FEException("Forms: H(DiscreteField) missing a valid FieldId (dual)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                if (sig->field_type != FieldType::Scalar) {
	                    throw FEException("Forms: H(DiscreteField) requires a scalar-valued DiscreteField; use H(component(u,i)) for vector-valued spaces (dual)",
	                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	                }
	                const auto Hval = ctx.fieldHessian(*fid, q);
	                EvalValue<Dual> out;
	                out.kind = EvalValue<Dual>::Kind::Matrix;
	                out.matrix_rows = dim;
	                out.matrix_cols = dim;
	                for (int r = 0; r < 3; ++r) {
	                    for (int c = 0; c < 3; ++c) {
	                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
	                            makeDualConstant(Hval[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
	                    }
	                }
	                return out;
	            }

	            if (child.type() == FormExprType::Constant) {
	                EvalValue<Dual> out;
	                out.kind = EvalValue<Dual>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeDualConstant(0.0, env.ws->alloc());
                    }
                }
                return out;
            }

            if (child.type() == FormExprType::Coefficient && child.scalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto H = fdHessScalar(*child.scalarCoefficient(), x, dim);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeDualConstant(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                    }
                }
	                return out;
	            }
            if (child.type() == FormExprType::Coefficient && child.timeScalarCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto H = fdHessScalarTime(*child.timeScalarCoefficient(), x, ctx.time(), dim);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Matrix;
                out.matrix_rows = dim;
                out.matrix_cols = dim;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        out.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            makeDualConstant(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)], env.ws->alloc());
                    }
                }
                return out;
            }

	            throw FEException("Forms: H() currently supports TestFunction, TrialFunction, DiscreteField, Constant, and scalar Coefficient only (dual)",
	                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
	        }
        case FormExprType::TimeDerivative: {
            const int order = node.timeDerivativeOrder().value_or(1);
            if (order <= 0) {
                throw FEException("Forms: dt(·,k) requires k >= 1 (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            const auto* time_ctx = ctx.timeIntegrationContext();
            if (!time_ctx) {
                throw FEException("dt(...) operator requires a transient time-integration context",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            const auto* stencil = time_ctx->stencil(order);
            if (!stencil) {
                throw FEException("Forms: dt(·," + std::to_string(order) + ") is not supported by time integrator '" +
                                      time_ctx->integrator_name + "' (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            const auto current = evalDualUnary(node, env, side, q);
            if (current.kind == EvalValue<Dual>::Kind::Scalar) {
                Dual result = makeDualConstant(0.0, env.ws->alloc());
                result = mul(current.s, stencil->coeff(0), result);

                const int required = stencil->requiredHistoryStates();
                for (int k = 1; k <= required; ++k) {
                    Dual prev = makeDualConstant(ctx.previousSolutionValue(q, k), env.ws->alloc());
                    prev = mul(prev, stencil->coeff(k), prev);
                    result = add(result, prev, result);
                }

                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = result;
                return out;
            }

            if (current.kind == EvalValue<Dual>::Kind::Vector) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                out.vector_size = static_cast<int>(current.vectorSize());
                const int required = stencil->requiredHistoryStates();
                std::vector<assembly::AssemblyContext::Vector3D> prev_vals;
                prev_vals.reserve(static_cast<std::size_t>(std::max(0, required)));
                for (int k = 1; k <= required; ++k) {
                    prev_vals.push_back(ctx.previousSolutionVectorValue(q, k));
                }

                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    Dual result = makeDualConstant(0.0, env.ws->alloc());
                    result = mul(current.vectorAt(d), stencil->coeff(0), result);

                    for (int k = 1; k <= required; ++k) {
                        const auto prev_v = prev_vals[static_cast<std::size_t>(k - 1)];
                        Dual prev = makeDualConstant(prev_v[d], env.ws->alloc());
                        prev = mul(prev, stencil->coeff(k), prev);
                        result = add(result, prev, result);
                    }
                    out.v[d] = result;
                }
                return out;
            }

            throw FEException("Forms: dt() operand did not evaluate to a scalar or vector (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
	        }
	        case FormExprType::Divergence: {
	            return evalSpatialJet<Dual>(node, env, side, q, 0).value;
	            const auto kids = node.childrenShared();
	            if (kids.size() != 1 || !kids[0]) throw std::logic_error("div must have 1 child (dual)");
	            const auto& child = *kids[0];
            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: div(TestFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: div(TestFunction) requires a vector-valued TestFunction (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: div(TestFunction) vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: div(TestFunction) DOF count is not divisible by value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: div(TestFunction) vector DOF index out of range (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const Real div = (env.test_active == side)
                    ? ctx.physicalGradient(env.i, q)[static_cast<std::size_t>(comp)]
                    : 0.0;
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant(div, env.ws->alloc());
                return out;
            }
            if (child.type() == FormExprType::TrialFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: div(TrialFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: div(TrialFunction) requires a vector-valued TrialFunction (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: div(TrialFunction) vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_trial = ctx.numTrialDofs();
                if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: div(TrialFunction) DOF count is not divisible by value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));

                Dual divu = makeDualConstant(0.0, env.ws->alloc());
                if (env.trial_active == side) {
                    const auto J = ctx.solutionJacobian(q);
                    const int n = std::min(vd, dim);
                    for (int d = 0; d < n; ++d) {
                        divu.value += J[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                    }
                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                        const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                        if (comp_j >= 0 && comp_j < n) {
                            const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);
                            divu.deriv[j] = grad_j[static_cast<std::size_t>(comp_j)];
                        }
                    }
                }

                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = divu;
                return out;
            }
            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: div(DiscreteField) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: div(DiscreteField) requires a vector-valued DiscreteField (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto fid = child.fieldId();
                if (!fid || *fid == INVALID_FIELD_ID) {
                    throw FEException("Forms: div(DiscreteField) missing a valid FieldId (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: div(DiscreteField) vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto J = ctx.fieldJacobian(*fid, q);
                Real div = 0.0;
                const int n = std::min(vd, dim);
                for (int d = 0; d < n; ++d) {
                    div += J[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                }
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant(div, env.ws->alloc());
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.vectorCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const Real div = fdDivVector(*child.vectorCoefficient(), x, dim);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant(div, env.ws->alloc());
                return out;
            }
            if (child.type() == FormExprType::Constant) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant(0.0, env.ws->alloc());
                return out;
            }
            throw FEException("Forms: div() currently supports vector TestFunction/TrialFunction and vector Coefficient only (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::Curl: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("curl must have 1 child (dual)");
            const auto& child = *kids[0];
            if (child.type() == FormExprType::TestFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: curl(TestFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: curl(TestFunction) requires a vector-valued TestFunction (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] = makeDualConstant(0.0, env.ws->alloc());
                }
                if (env.test_active != side) {
                    return out;
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: curl(TestFunction) vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_test = ctx.numTestDofs();
                if ((n_test % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: curl(TestFunction) DOF count is not divisible by value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_test / static_cast<LocalIndex>(vd));
                const int comp = static_cast<int>(env.i / dofs_per_component);
                if (comp < 0 || comp >= vd) {
                    throw FEException("Forms: curl(TestFunction) vector DOF index out of range (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const auto g = ctx.physicalGradient(env.i, q);
                if (dim == 2) {
                    const Real cz = (comp == 0) ? -g[1] : (comp == 1) ? g[0] : 0.0;
                    out.v[2].value = cz;
                    return out;
                }

                if (comp == 0) {
                    out.v[1].value = g[2];
                    out.v[2].value = -g[1];
                } else if (comp == 1) {
                    out.v[0].value = -g[2];
                    out.v[2].value = g[0];
                } else if (comp == 2) {
                    out.v[0].value = g[1];
                    out.v[1].value = -g[0];
                }
                return out;
            }
            if (child.type() == FormExprType::TrialFunction) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: curl(TrialFunction) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: curl(TrialFunction) requires a vector-valued TrialFunction (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: curl(TrialFunction) vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex n_trial = ctx.numTrialDofs();
                if ((n_trial % static_cast<LocalIndex>(vd)) != 0) {
                    throw FEException("Forms: curl(TrialFunction) DOF count is not divisible by value_dimension (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const LocalIndex dofs_per_component =
                    static_cast<LocalIndex>(n_trial / static_cast<LocalIndex>(vd));

                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] = makeDualConstant(0.0, env.ws->alloc());
                }
                if (env.trial_active != side) {
                    return out;
                }

                const auto J = ctx.solutionJacobian(q);
                if (dim == 2) {
                    // curl_z = du_y/dx - du_x/dy
                    out.v[2].value =
                        J[1][0] - J[0][1];
                    for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                        const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                        const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);
                        if (comp_j == 1) {
                            out.v[2].deriv[j] = grad_j[0];
                        } else if (comp_j == 0) {
                            out.v[2].deriv[j] = -grad_j[1];
                        }
                    }
                    return out;
                }

                // 3D curl(u) = [dUz/dy - dUy/dz, dUx/dz - dUz/dx, dUy/dx - dUx/dy]
                out.v[0].value = J[2][1] - J[1][2];
                out.v[1].value = J[0][2] - J[2][0];
                out.v[2].value = J[1][0] - J[0][1];

                for (std::size_t j = 0; j < env.n_trial_dofs; ++j) {
                    const int comp_j = static_cast<int>(static_cast<LocalIndex>(j) / dofs_per_component);
                    const auto grad_j = ctx.trialPhysicalGradient(static_cast<LocalIndex>(j), q);

                    if (comp_j == 2) {
                        out.v[0].deriv[j] = grad_j[1];
                        out.v[1].deriv[j] = -grad_j[0];
                    } else if (comp_j == 1) {
                        out.v[0].deriv[j] = -grad_j[2];
                        out.v[2].deriv[j] = grad_j[0];
                    } else if (comp_j == 0) {
                        out.v[1].deriv[j] = grad_j[2];
                        out.v[2].deriv[j] = -grad_j[1];
                    }
                }
                return out;
            }
            if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                const auto* sig = child.spaceSignature();
                if (!sig) {
                    throw FEException("Forms: curl(DiscreteField) requires a bound FunctionSpace (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (sig->field_type != FieldType::Vector) {
                    throw FEException("Forms: curl(DiscreteField) requires a vector-valued DiscreteField (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto fid = child.fieldId();
                if (!fid || *fid == INVALID_FIELD_ID) {
                    throw FEException("Forms: curl(DiscreteField) missing a valid FieldId (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const int vd = sig->value_dimension;
                if (vd <= 0 || vd > 3) {
                    throw FEException("Forms: curl(DiscreteField) vector value_dimension must be 1..3 (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }

                const auto J = ctx.fieldJacobian(*fid, q);
                auto d = [&](int comp, int wrt) -> Real {
                    if (comp < 0 || comp >= vd) return 0.0;
                    if (wrt < 0 || wrt >= dim) return 0.0;
                    return J[static_cast<std::size_t>(comp)][static_cast<std::size_t>(wrt)];
                };

                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                for (int dcomp = 0; dcomp < 3; ++dcomp) {
                    out.v[static_cast<std::size_t>(dcomp)] = makeDualConstant(0.0, env.ws->alloc());
                }
                if (dim == 2) {
                    out.v[2].value = d(1, 0) - d(0, 1);
                    return out;
                }
                out.v[0].value = d(2, 1) - d(1, 2);
                out.v[1].value = d(0, 2) - d(2, 0);
                out.v[2].value = d(1, 0) - d(0, 1);
                return out;
            }
            if (child.type() == FormExprType::Coefficient && child.vectorCoefficient()) {
                const auto x = ctx.physicalPoint(q);
                const auto c = fdCurlVector(*child.vectorCoefficient(), x, dim);
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] =
                        makeDualConstant(c[static_cast<std::size_t>(d)], env.ws->alloc());
                }
                return out;
            }
            if (child.type() == FormExprType::Constant) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;
                for (int d = 0; d < 3; ++d) {
                    out.v[static_cast<std::size_t>(d)] = makeDualConstant(0.0, env.ws->alloc());
                }
                return out;
            }
            throw FEException("Forms: curl() currently supports vector TestFunction/TrialFunction and vector Coefficient only (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        case FormExprType::RestrictMinus: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("(-) must have 1 child");
            return evalDual(*kids[0], env, Side::Minus, q);
        }
        case FormExprType::RestrictPlus: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("(+) must have 1 child");
            return evalDual(*kids[0], env, Side::Plus, q);
        }
        case FormExprType::Jump: {
            if (!env.plus) {
                throw FEException("Forms: jump() used outside interior-face integral",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("jump must have 1 child");
            const auto& child = *kids[0];
            const auto a = evalDual(child, env, Side::Minus, q);
            const auto b = evalDual(child, env, Side::Plus, q);
            if (a.kind != b.kind) {
                throw FEException("Forms: jump() operand has inconsistent kinds across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isVectorKind<Dual>(a.kind) && a.vectorSize() != b.vectorSize()) {
                throw FEException("Forms: jump() operand has inconsistent vector sizes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isMatrixKind<Dual>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
                throw FEException("Forms: jump() operand has inconsistent matrix shapes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = a.kind;
            if (a.kind == EvalValue<Dual>::Kind::Scalar) {
                out.s = sub(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
            } else if (a.kind == EvalValue<Dual>::Kind::Vector) {
                out.resizeVector(a.vectorSize());
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    out.vectorAt(d) = sub(a.vectorAt(d), b.vectorAt(d), makeDualConstant(0.0, env.ws->alloc()));
                }
            } else if (isMatrixKind<Dual>(a.kind)) {
                out.resizeMatrix(a.matrixRows(), a.matrixCols());
                for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                        out.matrixAt(r, c) = sub(a.matrixAt(r, c), b.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()));
                    }
                }
            } else if (isTensor4Kind<Dual>(a.kind)) {
                for (std::size_t k = 0; k < out.t4.size(); ++k) {
                    out.t4[k] = sub(a.t4[k], b.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                }
            }
            return out;
        }
        case FormExprType::Average: {
            if (!env.plus) {
                throw FEException("Forms: avg() used outside interior-face integral",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("avg must have 1 child");
            const auto& child = *kids[0];
            const auto a = evalDual(child, env, Side::Minus, q);
            const auto b = evalDual(child, env, Side::Plus, q);
            if (a.kind != b.kind) {
                throw FEException("Forms: avg() operand has inconsistent kinds across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isVectorKind<Dual>(a.kind) && a.vectorSize() != b.vectorSize()) {
                throw FEException("Forms: avg() operand has inconsistent vector sizes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (isMatrixKind<Dual>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
                throw FEException("Forms: avg() operand has inconsistent matrix shapes across sides",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = a.kind;
            if (isScalarKind<Dual>(a.kind)) {
                auto tmp = add(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                out.s = mul(tmp, 0.5, makeDualConstant(0.0, env.ws->alloc()));
            } else if (isVectorKind<Dual>(a.kind)) {
                out.resizeVector(a.vectorSize());
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    auto tmp = add(a.vectorAt(d), b.vectorAt(d), makeDualConstant(0.0, env.ws->alloc()));
                    out.vectorAt(d) = mul(tmp, 0.5, makeDualConstant(0.0, env.ws->alloc()));
                }
            } else if (isMatrixKind<Dual>(a.kind)) {
                out.resizeMatrix(a.matrixRows(), a.matrixCols());
                for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                        auto tmp = add(a.matrixAt(r, c), b.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()));
                        out.matrixAt(r, c) = mul(tmp, 0.5, makeDualConstant(0.0, env.ws->alloc()));
                    }
                }
            } else {
                for (std::size_t k = 0; k < out.t4.size(); ++k) {
                    auto tmp = add(a.t4[k], b.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                    out.t4[k] = mul(tmp, 0.5, makeDualConstant(0.0, env.ws->alloc()));
                }
            }
            return out;
        }
        case FormExprType::Negate: {
            const auto a = evalDualUnary(node, env, side, q);
            EvalValue<Dual> out = a;
            if (isScalarKind<Dual>(a.kind)) out.s = neg(a.s, makeDualConstant(0.0, env.ws->alloc()));
            else if (isVectorKind<Dual>(a.kind)) {
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    out.vectorAt(d) = neg(a.vectorAt(d), makeDualConstant(0.0, env.ws->alloc()));
                }
            } else if (isMatrixKind<Dual>(a.kind)) {
                for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                        out.matrixAt(r, c) = neg(a.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()));
                    }
                }
            } else {
                for (std::size_t k = 0; k < out.t4.size(); ++k) {
                    out.t4[k] = neg(a.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                }
            }
            return out;
        }
        case FormExprType::Transpose: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: transpose() expects a matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            EvalValue<Dual> out;
            out.kind = (rows == cols) ? a.kind : EvalValue<Dual>::Kind::Matrix;
            out.resizeMatrix(cols, rows);
            for (std::size_t r = 0; r < cols; ++r) {
                for (std::size_t c = 0; c < rows; ++c) {
                    out.matrixAt(r, c) = a.matrixAt(c, r);
                }
            }
            return out;
        }
        case FormExprType::Trace: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: trace() expects a matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: trace() expects a square matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Dual tr = makeDualConstant(0.0, env.ws->alloc());
            for (std::size_t d = 0; d < rows; ++d) {
                tr = add(tr, a.matrixAt(d, d), tr);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = tr;
            return out;
        }
        case FormExprType::Determinant: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: det() expects a matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: det() expects a square matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (rows > 3u) {
                throw FEException("Forms: det() is implemented only for 1x1, 2x2, and 3x3 matrices (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            Dual detA = makeDualConstant(0.0, env.ws->alloc());
            if (rows == 1u) {
                detA = copy(a.matrixAt(0, 0), detA);
            } else if (rows == 2u) {
                auto t0 = mul(a.matrixAt(0, 0), a.matrixAt(1, 1), makeDualConstant(0.0, env.ws->alloc()));
                auto t1 = mul(a.matrixAt(0, 1), a.matrixAt(1, 0), makeDualConstant(0.0, env.ws->alloc()));
                detA = sub(t0, t1, detA);
            } else {
                auto m11m22 = mul(a.matrixAt(1, 1), a.matrixAt(2, 2), makeDualConstant(0.0, env.ws->alloc()));
                auto m12m21 = mul(a.matrixAt(1, 2), a.matrixAt(2, 1), makeDualConstant(0.0, env.ws->alloc()));
                auto minor0 = sub(m11m22, m12m21, makeDualConstant(0.0, env.ws->alloc()));
                auto t0 = mul(a.matrixAt(0, 0), minor0, makeDualConstant(0.0, env.ws->alloc()));

                auto m10m22 = mul(a.matrixAt(1, 0), a.matrixAt(2, 2), makeDualConstant(0.0, env.ws->alloc()));
                auto m12m20 = mul(a.matrixAt(1, 2), a.matrixAt(2, 0), makeDualConstant(0.0, env.ws->alloc()));
                auto minor1 = sub(m10m22, m12m20, makeDualConstant(0.0, env.ws->alloc()));
                auto t1 = mul(a.matrixAt(0, 1), minor1, makeDualConstant(0.0, env.ws->alloc()));

                auto m10m21 = mul(a.matrixAt(1, 0), a.matrixAt(2, 1), makeDualConstant(0.0, env.ws->alloc()));
                auto m11m20 = mul(a.matrixAt(1, 1), a.matrixAt(2, 0), makeDualConstant(0.0, env.ws->alloc()));
                auto minor2 = sub(m10m21, m11m20, makeDualConstant(0.0, env.ws->alloc()));
                auto t2 = mul(a.matrixAt(0, 2), minor2, makeDualConstant(0.0, env.ws->alloc()));

                auto tmp = sub(t0, t1, makeDualConstant(0.0, env.ws->alloc()));
                detA = add(tmp, t2, detA);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = detA;
            return out;
        }
        case FormExprType::Cofactor:
        case FormExprType::Inverse: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: inv()/cofactor() expects a matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: inv()/cofactor() expects a square matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (rows > 3u) {
                throw FEException("Forms: inv()/cofactor() is implemented only for 1x1, 2x2, and 3x3 matrices (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            EvalValue<Dual> cof;
            cof.kind = EvalValue<Dual>::Kind::Matrix;
            cof.resizeMatrix(rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    cof.matrixAt(r, c) = makeDualConstant(0.0, env.ws->alloc());
                }
            }

            if (rows == 1u) {
                cof.matrixAt(0, 0) = makeDualConstant(1.0, env.ws->alloc());
            } else if (rows == 2u) {
                cof.matrixAt(0, 0) = copy(a.matrixAt(1, 1), cof.matrixAt(0, 0));
                cof.matrixAt(0, 1) = neg(a.matrixAt(1, 0), cof.matrixAt(0, 1));
                cof.matrixAt(1, 0) = neg(a.matrixAt(0, 1), cof.matrixAt(1, 0));
                cof.matrixAt(1, 1) = copy(a.matrixAt(0, 0), cof.matrixAt(1, 1));
            } else {
                cof.matrixAt(0, 0) = sub(mul(a.matrixAt(1, 1), a.matrixAt(2, 2), makeDualConstant(0.0, env.ws->alloc())),
                                         mul(a.matrixAt(1, 2), a.matrixAt(2, 1), makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(0, 0));
                cof.matrixAt(0, 1) = neg(sub(mul(a.matrixAt(1, 0), a.matrixAt(2, 2), makeDualConstant(0.0, env.ws->alloc())),
                                             mul(a.matrixAt(1, 2), a.matrixAt(2, 0), makeDualConstant(0.0, env.ws->alloc())),
                                             makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(0, 1));
                cof.matrixAt(0, 2) = sub(mul(a.matrixAt(1, 0), a.matrixAt(2, 1), makeDualConstant(0.0, env.ws->alloc())),
                                         mul(a.matrixAt(1, 1), a.matrixAt(2, 0), makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(0, 2));

                cof.matrixAt(1, 0) = neg(sub(mul(a.matrixAt(0, 1), a.matrixAt(2, 2), makeDualConstant(0.0, env.ws->alloc())),
                                             mul(a.matrixAt(0, 2), a.matrixAt(2, 1), makeDualConstant(0.0, env.ws->alloc())),
                                             makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(1, 0));
                cof.matrixAt(1, 1) = sub(mul(a.matrixAt(0, 0), a.matrixAt(2, 2), makeDualConstant(0.0, env.ws->alloc())),
                                         mul(a.matrixAt(0, 2), a.matrixAt(2, 0), makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(1, 1));
                cof.matrixAt(1, 2) = neg(sub(mul(a.matrixAt(0, 0), a.matrixAt(2, 1), makeDualConstant(0.0, env.ws->alloc())),
                                             mul(a.matrixAt(0, 1), a.matrixAt(2, 0), makeDualConstant(0.0, env.ws->alloc())),
                                             makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(1, 2));

                cof.matrixAt(2, 0) = sub(mul(a.matrixAt(0, 1), a.matrixAt(1, 2), makeDualConstant(0.0, env.ws->alloc())),
                                         mul(a.matrixAt(0, 2), a.matrixAt(1, 1), makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(2, 0));
                cof.matrixAt(2, 1) = neg(sub(mul(a.matrixAt(0, 0), a.matrixAt(1, 2), makeDualConstant(0.0, env.ws->alloc())),
                                             mul(a.matrixAt(0, 2), a.matrixAt(1, 0), makeDualConstant(0.0, env.ws->alloc())),
                                             makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(2, 1));
                cof.matrixAt(2, 2) = sub(mul(a.matrixAt(0, 0), a.matrixAt(1, 1), makeDualConstant(0.0, env.ws->alloc())),
                                         mul(a.matrixAt(0, 1), a.matrixAt(1, 0), makeDualConstant(0.0, env.ws->alloc())),
                                         cof.matrixAt(2, 2));
            }

            if (node.type() == FormExprType::Cofactor) {
                return cof;
            }

            // det = first row dot cof first row
            Dual detA = makeDualConstant(0.0, env.ws->alloc());
            for (std::size_t c = 0; c < cols; ++c) {
                auto prod = mul(a.matrixAt(0, c), cof.matrixAt(0, c), makeDualConstant(0.0, env.ws->alloc()));
                detA = add(detA, prod, detA);
            }

            EvalValue<Dual> invA;
            invA.kind = EvalValue<Dual>::Kind::Matrix;
            invA.resizeMatrix(rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    invA.matrixAt(r, c) = makeDualConstant(0.0, env.ws->alloc());
                }
            }
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    invA.matrixAt(r, c) = div(cof.matrixAt(c, r), detA, makeDualConstant(0.0, env.ws->alloc()));
                }
            }
            return invA;
        }
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: sym()/skew() expects a matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols) {
                throw FEException("Forms: sym()/skew() expects a square matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = (node.type() == FormExprType::SymmetricPart)
                ? EvalValue<Dual>::Kind::SymmetricMatrix
                : EvalValue<Dual>::Kind::SkewMatrix;
            out.resizeMatrix(rows, cols);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    auto tmp = (node.type() == FormExprType::SymmetricPart)
                        ? add(a.matrixAt(r, c),
                              a.matrixAt(c, r),
                              makeDualConstant(0.0, env.ws->alloc()))
                        : sub(a.matrixAt(r, c),
                              a.matrixAt(c, r),
                              makeDualConstant(0.0, env.ws->alloc()));
                    out.matrixAt(r, c) =
                        mul(tmp, 0.5, makeDualConstant(0.0, env.ws->alloc()));
                }
            }
            return out;
        }
        case FormExprType::Deviator: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: dev() expects a matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (rows != cols || rows == 0u) {
                throw FEException("Forms: dev() expects a square matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Dual tr = makeDualConstant(0.0, env.ws->alloc());
            for (std::size_t d = 0; d < rows; ++d) tr = add(tr, a.matrixAt(d, d), tr);
            Dual mean = div(tr, static_cast<Real>(rows), makeDualConstant(0.0, env.ws->alloc()));

            EvalValue<Dual> out = a;
            out.kind = isMatrixKind<Dual>(a.kind) ? a.kind : EvalValue<Dual>::Kind::Matrix;
            for (std::size_t d = 0; d < rows; ++d) {
                out.matrixAt(d, d) = sub(a.matrixAt(d, d), mean, makeDualConstant(0.0, env.ws->alloc()));
            }
            return out;
        }
        case FormExprType::Norm: {
            const auto a = evalDualUnary(node, env, side, q);
            Dual sum = makeDualConstant(0.0, env.ws->alloc());
            if (isScalarKind<Dual>(a.kind)) {
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = abs(a.s, makeDualConstant(0.0, env.ws->alloc()));
                return out;
            }
            if (isVectorKind<Dual>(a.kind)) {
                for (std::size_t d = 0; d < a.vectorSize(); ++d) {
                    auto prod = mul(a.vectorAt(d),
                                    a.vectorAt(d),
                                    makeDualConstant(0.0, env.ws->alloc()));
                    sum = add(sum, prod, sum);
                }
            } else if (isMatrixKind<Dual>(a.kind)) {
                for (std::size_t r = 0; r < a.matrixRows(); ++r) {
                    for (std::size_t c = 0; c < a.matrixCols(); ++c) {
                        auto prod = mul(a.matrixAt(r, c),
                                        a.matrixAt(r, c),
                                        makeDualConstant(0.0, env.ws->alloc()));
                        sum = add(sum, prod, sum);
                    }
                }
            } else if (isTensor3Kind<Dual>(a.kind)) {
                for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                    for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                        for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                            auto prod = mul(a.tensor3At(i, j, k),
                                            a.tensor3At(i, j, k),
                                            makeDualConstant(0.0, env.ws->alloc()));
                            sum = add(sum, prod, sum);
                        }
                    }
                }
            } else {
                for (std::size_t k = 0; k < a.t4.size(); ++k) {
                    auto prod = mul(a.t4[k], a.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                    sum = add(sum, prod, sum);
                }
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = sqrt(sum, makeDualConstant(0.0, env.ws->alloc()));
            return out;
        }
        case FormExprType::Normalize: {
            const auto a = evalDualUnary(node, env, side, q);
            if (!isVectorKind<Dual>(a.kind)) {
                throw FEException("Forms: normalize() expects a vector (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            Dual sum = makeDualConstant(0.0, env.ws->alloc());
            for (std::size_t d = 0; d < a.vectorSize(); ++d) {
                auto prod = mul(a.vectorAt(d),
                                a.vectorAt(d),
                                makeDualConstant(0.0, env.ws->alloc()));
                sum = add(sum, prod, sum);
            }
            Dual nrm = sqrt(sum, makeDualConstant(0.0, env.ws->alloc()));
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Vector;
            out.resizeVector(a.vectorSize());
            if (nrm.value == 0.0) {
                for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                    out.vectorAt(d) = makeDualConstant(0.0, env.ws->alloc());
                }
                return out;
            }
            for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                out.vectorAt(d) = div(a.vectorAt(d), nrm, makeDualConstant(0.0, env.ws->alloc()));
            }
            return out;
        }
        case FormExprType::AbsoluteValue: {
            const auto a = evalDualUnary(node, env, side, q);
            if (a.kind != EvalValue<Dual>::Kind::Scalar) {
                throw FEException("Forms: abs() expects a scalar (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = abs(a.s, makeDualConstant(0.0, env.ws->alloc()));
            return out;
        }
        case FormExprType::Sign: {
            const auto a = evalDualUnary(node, env, side, q);
            if (a.kind != EvalValue<Dual>::Kind::Scalar) {
                throw FEException("Forms: sign() expects a scalar (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            out.s = sign(a.s, makeDualConstant(0.0, env.ws->alloc()));
            return out;
        }
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log: {
            const auto a = evalDualUnary(node, env, side, q);
            if (a.kind != EvalValue<Dual>::Kind::Scalar) {
                throw FEException("Forms: sqrt/exp/log expects a scalar (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            if (node.type() == FormExprType::Sqrt) out.s = sqrt(a.s, makeDualConstant(0.0, env.ws->alloc()));
            else if (node.type() == FormExprType::Exp) out.s = exp(a.s, makeDualConstant(0.0, env.ws->alloc()));
            else out.s = log(a.s, makeDualConstant(0.0, env.ws->alloc()));
            return out;
        }
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct:
        case FormExprType::Power:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2 || !kids[0] || !kids[1]) {
                throw std::logic_error("Forms: binary node must have 2 children");
            }
            const auto a = evalDual(*kids[0], env, side, q);
            const auto b = evalDual(*kids[1], env, side, q);

            if (node.type() == FormExprType::Add || node.type() == FormExprType::Subtract) {
                if (!sameCategory<Dual>(a.kind, b.kind)) {
                    throw FEException("Forms: add/sub kind mismatch",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isVectorKind<Dual>(a.kind) && a.vectorSize() != b.vectorSize()) {
                    throw FEException("Forms: add/sub vector size mismatch (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isMatrixKind<Dual>(a.kind) && (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols())) {
                    throw FEException("Forms: add/sub matrix shape mismatch (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isTensor3Kind<Dual>(a.kind) &&
                    (a.tensor3Dim0() != b.tensor3Dim0() ||
                     a.tensor3Dim1() != b.tensor3Dim1() ||
                     a.tensor3Dim2() != b.tensor3Dim2())) {
                    throw FEException("Forms: add/sub tensor3 shape mismatch (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Dual> out;
                out.kind = addSubResultKind<Dual>(a.kind, b.kind);
                if (isScalarKind<Dual>(a.kind)) {
                    out.s = (node.type() == FormExprType::Add)
                        ? add(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()))
                        : sub(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                } else if (isVectorKind<Dual>(a.kind)) {
                    out.resizeVector(a.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                        out.vectorAt(d) = (node.type() == FormExprType::Add)
                            ? add(a.vectorAt(d), b.vectorAt(d), makeDualConstant(0.0, env.ws->alloc()))
                            : sub(a.vectorAt(d), b.vectorAt(d), makeDualConstant(0.0, env.ws->alloc()));
                    }
                } else if (isMatrixKind<Dual>(a.kind)) {
                    out.resizeMatrix(a.matrixRows(), a.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = (node.type() == FormExprType::Add)
                                ? add(a.matrixAt(r, c), b.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()))
                                : sub(a.matrixAt(r, c), b.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()));
                        }
                    }
                } else if (isTensor3Kind<Dual>(a.kind)) {
                    out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = (node.type() == FormExprType::Add)
                                    ? add(a.tensor3At(i, j, k), b.tensor3At(i, j, k), makeDualConstant(0.0, env.ws->alloc()))
                                    : sub(a.tensor3At(i, j, k), b.tensor3At(i, j, k), makeDualConstant(0.0, env.ws->alloc()));
                            }
                        }
                    }
                } else {
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = (node.type() == FormExprType::Add)
                            ? add(a.t4[k], b.t4[k], makeDualConstant(0.0, env.ws->alloc()))
                            : sub(a.t4[k], b.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                    }
                }
                return out;
            }

            if (node.type() == FormExprType::Multiply) {
                if (isScalarKind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = mul(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                    return out;
                }
                if (isScalarKind<Dual>(a.kind) && isVectorKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.resizeVector(b.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                        out.vectorAt(d) = mul(a.s, b.vectorAt(d), makeDualConstant(0.0, env.ws->alloc()));
                    }
                    return out;
                }
                if (isVectorKind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.resizeVector(a.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                        out.vectorAt(d) = mul(a.vectorAt(d), b.s, makeDualConstant(0.0, env.ws->alloc()));
                    }
                    return out;
                }
                if (isScalarKind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = b.kind;
                    out.resizeMatrix(b.matrixRows(), b.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = mul(a.s, b.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()));
                        }
                    }
                    return out;
                }
                if (isMatrixKind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = a.kind;
                    out.resizeMatrix(a.matrixRows(), a.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = mul(a.matrixAt(r, c), b.s, makeDualConstant(0.0, env.ws->alloc()));
                        }
                    }
                    return out;
                }
                if (isMatrixKind<Dual>(a.kind) && isVectorKind<Dual>(b.kind)) {
                    const auto rows = a.matrixRows();
                    const auto cols = a.matrixCols();
                    if (b.vectorSize() != cols) {
                        throw FEException("Forms: matrix-vector multiplication shape mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.resizeVector(rows);
                    for (std::size_t r = 0; r < rows; ++r) {
                        Dual sum = makeDualConstant(0.0, env.ws->alloc());
                        for (std::size_t c = 0; c < cols; ++c) {
                            auto prod = mul(a.matrixAt(r, c), b.vectorAt(c), makeDualConstant(0.0, env.ws->alloc()));
                            sum = add(sum, prod, sum);
                        }
                        out.vectorAt(r) = sum;
                    }
                    return out;
                }
                if (isVectorKind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                    const auto rows = b.matrixRows();
                    const auto cols = b.matrixCols();
                    if (a.vectorSize() != rows) {
                        throw FEException("Forms: vector-matrix multiplication shape mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.resizeVector(cols);
                    for (std::size_t c = 0; c < cols; ++c) {
                        Dual sum = makeDualConstant(0.0, env.ws->alloc());
                        for (std::size_t r = 0; r < rows; ++r) {
                            auto prod = mul(a.vectorAt(r), b.matrixAt(r, c), makeDualConstant(0.0, env.ws->alloc()));
                            sum = add(sum, prod, sum);
                        }
                        out.vectorAt(c) = sum;
                    }
                    return out;
                }
                if (isMatrixKind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                    const auto rows = a.matrixRows();
                    const auto inner_dim = a.matrixCols();
                    const auto cols = b.matrixCols();
                    if (b.matrixRows() != inner_dim) {
                        throw FEException("Forms: matrix-matrix multiplication shape mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.resizeMatrix(rows, cols);
                    for (std::size_t r = 0; r < rows; ++r) {
                        for (std::size_t c = 0; c < cols; ++c) {
                            Dual sum = makeDualConstant(0.0, env.ws->alloc());
                            for (std::size_t k = 0; k < inner_dim; ++k) {
                                auto prod = mul(a.matrixAt(r, k), b.matrixAt(k, c), makeDualConstant(0.0, env.ws->alloc()));
                                sum = add(sum, prod, sum);
                            }
                            out.matrixAt(r, c) = sum;
                        }
                    }
                    return out;
                }
                if (isScalarKind<Dual>(a.kind) && isTensor3Kind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Tensor3;
                    out.resizeTensor3(b.tensor3Dim0(), b.tensor3Dim1(), b.tensor3Dim2());
                    for (std::size_t i = 0; i < b.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < b.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < b.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = mul(a.s, b.tensor3At(i, j, k), makeDualConstant(0.0, env.ws->alloc()));
                            }
                        }
                    }
                    return out;
                }
                if (isTensor3Kind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Tensor3;
                    out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = mul(a.tensor3At(i, j, k), b.s, makeDualConstant(0.0, env.ws->alloc()));
                            }
                        }
                    }
                    return out;
                }
                if (isScalarKind<Dual>(a.kind) && isTensor4Kind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Tensor4;
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = mul(a.s, b.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                    }
                    return out;
                }
                if (isTensor4Kind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Tensor4;
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = mul(a.t4[k], b.s, makeDualConstant(0.0, env.ws->alloc()));
                    }
                    return out;
                }
                throw FEException("Forms: unsupported multiplication kinds (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::Divide) {
                if (!isScalarKind<Dual>(b.kind)) {
                    throw FEException("Forms: division denominator must be scalar (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (isScalarKind<Dual>(a.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = div(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                    return out;
                }
                if (isVectorKind<Dual>(a.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Vector;
                    out.resizeVector(a.vectorSize());
                    for (std::size_t d = 0; d < out.vectorSize(); ++d) {
                        out.vectorAt(d) = div(a.vectorAt(d), b.s, makeDualConstant(0.0, env.ws->alloc()));
                    }
                    return out;
                }
                if (isMatrixKind<Dual>(a.kind)) {
                    EvalValue<Dual> out;
                    out.kind = a.kind;
                    out.resizeMatrix(a.matrixRows(), a.matrixCols());
                    for (std::size_t r = 0; r < out.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < out.matrixCols(); ++c) {
                            out.matrixAt(r, c) = div(a.matrixAt(r, c), b.s, makeDualConstant(0.0, env.ws->alloc()));
                        }
                    }
                    return out;
                }
                if (isTensor3Kind<Dual>(a.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Tensor3;
                    out.resizeTensor3(a.tensor3Dim0(), a.tensor3Dim1(), a.tensor3Dim2());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                out.tensor3At(i, j, k) = div(a.tensor3At(i, j, k), b.s, makeDualConstant(0.0, env.ws->alloc()));
                            }
                        }
                    }
                    return out;
                }
                if (isTensor4Kind<Dual>(a.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Tensor4;
                    for (std::size_t k = 0; k < out.t4.size(); ++k) {
                        out.t4[k] = div(a.t4[k], b.s, makeDualConstant(0.0, env.ws->alloc()));
                    }
                    return out;
                }
                throw FEException("Forms: unsupported division kinds (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::InnerProduct) {
                if (isScalarKind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = mul(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                    return out;
                }
                if (isVectorKind<Dual>(a.kind) && isVectorKind<Dual>(b.kind)) {
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    if (a.vectorSize() != b.vectorSize()) {
                        throw FEException("Forms: inner(vector,vector) size mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    for (std::size_t d = 0; d < a.vectorSize(); ++d) {
                        auto prod = mul(a.vectorAt(d),
                                        b.vectorAt(d),
                                        makeDualConstant(0.0, env.ws->alloc()));
                        sum = add(sum, prod, sum);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }
                if (isMatrixKind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    if (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols()) {
                        throw FEException("Forms: inner(matrix,matrix) shape mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    for (std::size_t r = 0; r < a.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < a.matrixCols(); ++c) {
                            auto prod = mul(a.matrixAt(r, c),
                                            b.matrixAt(r, c),
                                            makeDualConstant(0.0, env.ws->alloc()));
                            sum = add(sum, prod, sum);
                        }
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }
                if (isTensor3Kind<Dual>(a.kind) && isTensor3Kind<Dual>(b.kind)) {
                    if (a.tensor3Dim0() != b.tensor3Dim0() ||
                        a.tensor3Dim1() != b.tensor3Dim1() ||
                        a.tensor3Dim2() != b.tensor3Dim2()) {
                        throw FEException("Forms: inner(tensor3,tensor3) shape mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    for (std::size_t i = 0; i < a.tensor3Dim0(); ++i) {
                        for (std::size_t j = 0; j < a.tensor3Dim1(); ++j) {
                            for (std::size_t k = 0; k < a.tensor3Dim2(); ++k) {
                                auto prod = mul(a.tensor3At(i, j, k),
                                                b.tensor3At(i, j, k),
                                                makeDualConstant(0.0, env.ws->alloc()));
                                sum = add(sum, prod, sum);
                            }
                        }
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }
                if (isTensor4Kind<Dual>(a.kind) && isTensor4Kind<Dual>(b.kind)) {
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    for (std::size_t k = 0; k < a.t4.size(); ++k) {
                        auto prod = mul(a.t4[k], b.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                        sum = add(sum, prod, sum);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }
                throw FEException("Forms: unsupported inner() kinds (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::DoubleContraction) {
                if (isTensor4Kind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            Dual sum = makeDualConstant(0.0, env.ws->alloc());
                            for (int k = 0; k < dim; ++k) {
                                for (int l = 0; l < dim; ++l) {
                                    auto prod = mul(
                                        a.t4[idx4(i, j, k, l)],
                                        b.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(l)),
                                        makeDualConstant(0.0, env.ws->alloc()));
                                    sum = add(sum, prod, sum);
                                }
                            }
                            out.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = sum;
                        }
                    }
                    return out;
                }

                if (isMatrixKind<Dual>(a.kind) && isTensor4Kind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    out.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));
                    for (int k = 0; k < dim; ++k) {
                        for (int l = 0; l < dim; ++l) {
                            Dual sum = makeDualConstant(0.0, env.ws->alloc());
                            for (int i = 0; i < dim; ++i) {
                                for (int j = 0; j < dim; ++j) {
                                    auto prod = mul(
                                        a.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                                        b.t4[idx4(i, j, k, l)],
                                        makeDualConstant(0.0, env.ws->alloc()));
                                    sum = add(sum, prod, sum);
                                }
                            }
                            out.matrixAt(static_cast<std::size_t>(k), static_cast<std::size_t>(l)) = sum;
                        }
                    }
                    return out;
                }

                // Fall back to full contraction for matching shapes.
                if (isScalarKind<Dual>(a.kind) && isScalarKind<Dual>(b.kind)) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = mul(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                    return out;
                }
                if (isVectorKind<Dual>(a.kind) && isVectorKind<Dual>(b.kind)) {
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    if (a.vectorSize() != b.vectorSize()) {
                        throw FEException("Forms: doubleContraction(vector,vector) size mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    for (std::size_t d = 0; d < a.vectorSize(); ++d) {
                        auto prod = mul(a.vectorAt(d),
                                        b.vectorAt(d),
                                        makeDualConstant(0.0, env.ws->alloc()));
                        sum = add(sum, prod, sum);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }
                if (isMatrixKind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    if (a.matrixRows() != b.matrixRows() || a.matrixCols() != b.matrixCols()) {
                        throw FEException("Forms: doubleContraction(matrix,matrix) shape mismatch (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    for (std::size_t r = 0; r < a.matrixRows(); ++r) {
                        for (std::size_t c = 0; c < a.matrixCols(); ++c) {
                            auto prod = mul(a.matrixAt(r, c),
                                            b.matrixAt(r, c),
                                            makeDualConstant(0.0, env.ws->alloc()));
                            sum = add(sum, prod, sum);
                        }
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }
                if (isTensor4Kind<Dual>(a.kind) && isTensor4Kind<Dual>(b.kind)) {
                    Dual sum = makeDualConstant(0.0, env.ws->alloc());
                    for (std::size_t k = 0; k < a.t4.size(); ++k) {
                        auto prod = mul(a.t4[k], b.t4[k], makeDualConstant(0.0, env.ws->alloc()));
                        sum = add(sum, prod, sum);
                    }
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Scalar;
                    out.s = sum;
                    return out;
                }

                throw FEException("Forms: unsupported doubleContraction() kinds (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::OuterProduct) {
                if (a.kind == EvalValue<Dual>::Kind::Vector && b.kind == EvalValue<Dual>::Kind::Vector) {
                    EvalValue<Dual> out;
                    out.kind = EvalValue<Dual>::Kind::Matrix;
                    const auto rows = a.vectorSize();
                    const auto cols = b.vectorSize();
                    out.resizeMatrix(rows, cols);
                    for (std::size_t r = 0; r < rows; ++r) {
                        for (std::size_t c = 0; c < cols; ++c) {
                            out.matrixAt(r, c) = mul(a.vectorAt(r), b.vectorAt(c), makeDualConstant(0.0, env.ws->alloc()));
                        }
                    }
                    return out;
                }
                throw FEException("Forms: outer() currently supports vector-vector only",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }

            if (node.type() == FormExprType::CrossProduct) {
                if (a.kind != EvalValue<Dual>::Kind::Vector || b.kind != EvalValue<Dual>::Kind::Vector) {
                    throw FEException("Forms: cross() expects vector arguments (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (a.vectorSize() > 3u || b.vectorSize() > 3u) {
                    throw FEException("Forms: cross() supports vectors up to 3 components (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Vector;

                auto ax = (a.vectorSize() > 0u) ? a.vectorAt(0) : makeDualConstant(0.0, env.ws->alloc());
                auto ay = (a.vectorSize() > 1u) ? a.vectorAt(1) : makeDualConstant(0.0, env.ws->alloc());
                auto az = (a.vectorSize() > 2u) ? a.vectorAt(2) : makeDualConstant(0.0, env.ws->alloc());

                auto bx = (b.vectorSize() > 0u) ? b.vectorAt(0) : makeDualConstant(0.0, env.ws->alloc());
                auto by = (b.vectorSize() > 1u) ? b.vectorAt(1) : makeDualConstant(0.0, env.ws->alloc());
                auto bz = (b.vectorSize() > 2u) ? b.vectorAt(2) : makeDualConstant(0.0, env.ws->alloc());

                out.resizeVector(3u);
                auto aybz = mul(ay, bz, makeDualConstant(0.0, env.ws->alloc()));
                auto azby = mul(az, by, makeDualConstant(0.0, env.ws->alloc()));
                out.vectorAt(0) = sub(aybz, azby, makeDualConstant(0.0, env.ws->alloc()));

                auto azbx = mul(az, bx, makeDualConstant(0.0, env.ws->alloc()));
                auto axbz = mul(ax, bz, makeDualConstant(0.0, env.ws->alloc()));
                out.vectorAt(1) = sub(azbx, axbz, makeDualConstant(0.0, env.ws->alloc()));

                auto axby = mul(ax, by, makeDualConstant(0.0, env.ws->alloc()));
                auto aybx = mul(ay, bx, makeDualConstant(0.0, env.ws->alloc()));
                out.vectorAt(2) = sub(axby, aybx, makeDualConstant(0.0, env.ws->alloc()));
                return out;
            }

            if (node.type() == FormExprType::Power) {
                if (a.kind != EvalValue<Dual>::Kind::Scalar || b.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: pow() expects scalar arguments (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const bool exponent_is_constant = (kids[1] && kids[1]->type() == FormExprType::Constant);
                const Real exponent_value = exponent_is_constant ? kids[1]->constantValue().value_or(b.s.value) : b.s.value;
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = exponent_is_constant
                    ? pow(a.s, exponent_value, makeDualConstant(0.0, env.ws->alloc()))
                    : pow(a.s, b.s, makeDualConstant(0.0, env.ws->alloc()));
                return out;
            }

            if (node.type() == FormExprType::Minimum || node.type() == FormExprType::Maximum) {
                if (a.kind != EvalValue<Dual>::Kind::Scalar || b.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: min/max expects scalar arguments (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                return (node.type() == FormExprType::Minimum)
                    ? ((a.s.value <= b.s.value) ? a : b)
                    : ((a.s.value >= b.s.value) ? a : b);
            }

            if (node.type() == FormExprType::Less || node.type() == FormExprType::LessEqual ||
                node.type() == FormExprType::Greater || node.type() == FormExprType::GreaterEqual ||
                node.type() == FormExprType::Equal || node.type() == FormExprType::NotEqual) {
                if (a.kind != EvalValue<Dual>::Kind::Scalar || b.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: comparisons expect scalar arguments (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                bool truth = false;
                switch (node.type()) {
                    case FormExprType::Less: truth = (a.s.value < b.s.value); break;
                    case FormExprType::LessEqual: truth = (a.s.value <= b.s.value); break;
                    case FormExprType::Greater: truth = (a.s.value > b.s.value); break;
                    case FormExprType::GreaterEqual: truth = (a.s.value >= b.s.value); break;
                    case FormExprType::Equal: truth = (a.s.value == b.s.value); break;
                    case FormExprType::NotEqual: truth = (a.s.value != b.s.value); break;
                    default: break;
                }
                EvalValue<Dual> out;
                out.kind = EvalValue<Dual>::Kind::Scalar;
                out.s = makeDualConstant(truth ? 1.0 : 0.0, env.ws->alloc());
                return out;
            }

            throw FEException("Forms: unreachable binary operation (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::Unknown);
        }
        case FormExprType::Conditional: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3 || !kids[0] || !kids[1] || !kids[2]) {
                throw std::logic_error("Forms: conditional must have 3 children");
            }
            const auto cond = evalDual(*kids[0], env, side, q);
            if (cond.kind != EvalValue<Dual>::Kind::Scalar) {
                throw FEException("Forms: conditional condition must be scalar (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto a = evalDual(*kids[1], env, side, q);
            const auto b = evalDual(*kids[2], env, side, q);
            if (!sameCategory<Dual>(a.kind, b.kind)) {
                throw FEException("Forms: conditional branch kind mismatch (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            if (a.kind == b.kind) {
                return (cond.s.value > 0.0) ? a : b;
            }
            if (isMatrixKind<Dual>(a.kind) && isMatrixKind<Dual>(b.kind)) {
                auto out = (cond.s.value > 0.0) ? a : b;
                out.kind = EvalValue<Dual>::Kind::Matrix;
                return out;
            }
            return (cond.s.value > 0.0) ? a : b;
        }
        case FormExprType::AsVector: {
            const auto kids = node.childrenShared();
            if (kids.empty()) {
                throw FEException("Forms: as_vector expects at least 1 scalar component (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Vector;
            out.resizeVector(kids.size());
            for (std::size_t c = 0; c < kids.size(); ++c) {
                if (!kids[c]) throw std::logic_error("Forms: as_vector has null child (dual)");
                const auto v = evalDual(*kids[c], env, side, q);
                if (v.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: as_vector components must be scalar (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.vectorAt(c) = v.s;
            }
            return out;
        }
        case FormExprType::AsTensor: {
            const auto rows = node.tensorRows().value_or(0);
            const auto cols = node.tensorCols().value_or(0);
            if (rows <= 0 || cols <= 0) {
                throw FEException("Forms: as_tensor requires explicit shape with rows,cols >= 1 (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto kids = node.childrenShared();
            if (kids.size() != static_cast<std::size_t>(rows * cols)) {
                throw FEException("Forms: as_tensor child count does not match rows*cols (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Matrix;
            out.resizeMatrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const auto idx = static_cast<std::size_t>(r * cols + c);
                    if (!kids[idx]) throw std::logic_error("Forms: as_tensor has null child (dual)");
                    const auto v = evalDual(*kids[idx], env, side, q);
                    if (v.kind != EvalValue<Dual>::Kind::Scalar) {
                        throw FEException("Forms: as_tensor entries must be scalar (dual)",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    out.matrixAt(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) = v.s;
                }
            }
            return out;
        }
        case FormExprType::Component: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1 || !kids[0]) throw std::logic_error("component() must have 1 child");
            const auto a = evalDual(*kids[0], env, side, q);
            const int i = node.componentIndex0().value_or(0);
            const int j = node.componentIndex1().value_or(-1);

            EvalValue<Dual> out;
            out.kind = EvalValue<Dual>::Kind::Scalar;
            if (isScalarKind<Dual>(a.kind)) {
                if (i != 0 || j >= 0) {
                    throw FEException("Forms: component() invalid indices for scalar (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.s = a.s;
                return out;
            }
            if (isVectorKind<Dual>(a.kind)) {
                if (j >= 0) {
                    throw FEException("Forms: component(v,i,j) invalid for vector (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                const auto n = a.vectorSize();
                if (i < 0 || static_cast<std::size_t>(i) >= n) {
                    throw FEException("Forms: component(v,i) index out of range (dual)",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                out.s = a.vectorAt(static_cast<std::size_t>(i));
                return out;
            }
            if (!isMatrixKind<Dual>(a.kind)) {
                throw FEException("Forms: component() is not defined for this operand kind (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            if (j < 0) {
                throw FEException("Forms: component(A,i) missing column index for matrix (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            const auto rows = a.matrixRows();
            const auto cols = a.matrixCols();
            if (i < 0 || static_cast<std::size_t>(i) >= rows || j < 0 || static_cast<std::size_t>(j) >= cols) {
                throw FEException("Forms: component(A,i,j) index out of range (dual)",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            out.s = a.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            return out;
        }
        case FormExprType::IndexedAccess:
            throw FEException("Forms: indexed access must be lowered via forms::einsum(...) before compilation/evaluation (dual)",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
            break;
    }

    throw FEException("Forms: unsupported expression node in dual evaluation",
                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
}

void applyInlinedMaterialStateUpdatesReal(const assembly::AssemblyContext& ctx_minus,
                                          const assembly::AssemblyContext* ctx_plus,
                                          FormKind kind,
                                          const ConstitutiveStateLayout* constitutive_state,
                                          const std::vector<MaterialStateUpdate>& updates,
                                          Side side,
                                          LocalIndex q)
{
    if (updates.empty()) {
        return;
    }

    const auto& ctx = ctxForSide(ctx_minus, ctx_plus, side);
    FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
                "Forms: inlined material-state updates require AssemblyContext material state");
    auto state = ctx.materialStateWork(q);

    ConstitutiveCallCacheReal constitutive_cache;
    EvalEnvReal env{ctx_minus, ctx_plus, kind, Side::Minus, Side::Minus, 0, 0, constitutive_state, &constitutive_cache};

    for (const auto& op : updates) {
        FE_THROW_IF(!op.value.isValid() || op.value.node() == nullptr, InvalidArgumentException,
                    "Forms: inlined material-state update has invalid expression");
        const auto v = evalReal(*op.value.node(), env, side, q);
        FE_THROW_IF(v.kind != EvalValue<Real>::Kind::Scalar, InvalidArgumentException,
                    "Forms: inlined material-state update did not evaluate to scalar");

        const auto offset = static_cast<std::size_t>(op.offset_bytes);
        FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
                    "Forms: inlined material-state update offset out of bounds for bound state buffer");
        std::memcpy(state.data() + offset, &v.s, sizeof(Real));
    }
}

void applyInlinedMaterialStateUpdatesDual(const assembly::AssemblyContext& ctx_minus,
                                          const assembly::AssemblyContext* ctx_plus,
                                          const ConstitutiveStateLayout* constitutive_state,
                                          const std::vector<MaterialStateUpdate>& updates,
                                          Side side,
                                          LocalIndex q)
{
    if (updates.empty()) {
        return;
    }

    const auto& ctx = ctxForSide(ctx_minus, ctx_plus, side);
    FE_THROW_IF(!ctx.hasMaterialState(), InvalidArgumentException,
                "Forms: inlined material-state updates require AssemblyContext material state (dual)");
    auto state = ctx.materialStateWork(q);

    thread_local DualWorkspace ws;
    ws.reset(/*num_dofs=*/0u);

    ConstitutiveCallCacheDual constitutive_cache;
    EvalEnvDual env{ctx_minus, ctx_plus, Side::Minus, Side::Minus, /*i=*/0, /*n_trial_dofs=*/0u, &ws,
                    constitutive_state, &constitutive_cache};

    for (const auto& op : updates) {
        FE_THROW_IF(!op.value.isValid() || op.value.node() == nullptr, InvalidArgumentException,
                    "Forms: inlined material-state update has invalid expression (dual)");
        const auto v = evalDual(*op.value.node(), env, side, q);
        FE_THROW_IF(v.kind != EvalValue<Dual>::Kind::Scalar, InvalidArgumentException,
                    "Forms: inlined material-state update did not evaluate to scalar (dual)");

        const Real value = v.s.value;
        const auto offset = static_cast<std::size_t>(op.offset_bytes);
        FE_THROW_IF(offset + sizeof(Real) > state.size(), InvalidArgumentException,
                    "Forms: inlined material-state update offset out of bounds for bound state buffer (dual)");
        std::memcpy(state.data() + offset, &value, sizeof(Real));
    }
}

} // namespace

// ============================================================================
// FormKernel
// ============================================================================

FormKernel::FormKernel(FormIR ir)
    : ir_(std::move(ir))
{
    if (!ir_.isCompiled()) {
        throw std::invalid_argument("FormKernel: IR is not compiled");
    }
    if (ir_.kind() == FormKind::Residual) {
        throw std::invalid_argument("FormKernel: residual IR must be used with NonlinearFormKernel");
    }

    const FormIR* irs[] = {&ir_};
    parameter_specs_ = computeParameterSpecs(std::span<const FormIR* const>{irs});

    constitutive_state_ = buildConstitutiveStateLayout(ir_, material_state_spec_);
}

FormKernel::~FormKernel() = default;
FormKernel::FormKernel(FormKernel&&) noexcept = default;
FormKernel& FormKernel::operator=(FormKernel&&) noexcept = default;

assembly::RequiredData FormKernel::getRequiredData() const noexcept
{
    auto req = ir_.requiredData();
    if (material_state_spec_.bytes_per_qpt > 0u) {
        req |= assembly::RequiredData::MaterialState;
    }
    return req;
}

std::vector<assembly::FieldRequirement> FormKernel::fieldRequirements() const
{
    return ir_.fieldRequirements();
}

assembly::MaterialStateSpec FormKernel::materialStateSpec() const noexcept
{
    return material_state_spec_;
}

std::vector<params::Spec> FormKernel::parameterSpecs() const
{
    return parameter_specs_;
}

void FormKernel::resolveInlinableConstitutives()
{
    FormIR* irs[] = {&ir_};
    inlineInlinableConstitutives(std::span<FormIR*>{irs},
                                 constitutive_state_.get(),
                                 material_state_spec_,
                                 inlined_state_updates_);
}

void FormKernel::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    const auto transform = [&](const FormExprNode& n) -> std::optional<FormExpr> {
        if (n.type() != FormExprType::ParameterSymbol) {
            return std::nullopt;
        }
        const auto key = n.symbolName();
        FE_THROW_IF(!key || key->empty(), InvalidArgumentException,
                    "Forms::FormKernel: ParameterSymbol node missing name");
        const auto slot = slot_of_real_param(*key);
        FE_THROW_IF(!slot.has_value(), InvalidArgumentException,
                    "Forms::FormKernel: could not resolve parameter slot for '" + std::string(*key) + "'");
        return FormExpr::parameterRef(*slot);
    };

    ir_.transformIntegrands(transform);
    const auto rewrite_updates = [&](std::vector<MaterialStateUpdate>& updates) {
        for (auto& op : updates) {
            op.value = op.value.transformNodes(transform);
        }
    };
    rewrite_updates(inlined_state_updates_.cell);
    rewrite_updates(inlined_state_updates_.boundary_all);
    for (auto& kv : inlined_state_updates_.boundary_by_marker) {
        rewrite_updates(kv.second);
    }
    rewrite_updates(inlined_state_updates_.interior_face);
    rewrite_updates(inlined_state_updates_.interface_face);
}

bool FormKernel::hasCell() const noexcept { return ir_.hasCellTerms(); }
bool FormKernel::hasBoundaryFace() const noexcept { return ir_.hasBoundaryTerms(); }
bool FormKernel::hasInteriorFace() const noexcept { return ir_.hasInteriorFaceTerms(); }
bool FormKernel::hasInterfaceFace() const noexcept { return ir_.hasInterfaceFaceTerms(); }

void FormKernel::computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    const bool want_matrix = (ir_.kind() == FormKind::Bilinear);
    const bool want_vector = (ir_.kind() == FormKind::Linear);

    output.reserve(n_test, n_trial, want_matrix, want_vector);
    output.clear();

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    if (!inlined_state_updates_.cell.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesReal(ctx, nullptr, ir_.kind(),
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.cell,
                                                 Side::Minus, q);
        }
    }

    ConstitutiveCallCacheReal constitutive_cache;
    EvalEnvReal env{ctx, nullptr, ir_.kind(), Side::Minus, Side::Minus, 0, 0, constitutive_state_.get(), &constitutive_cache};

    for (const auto& term : ir_.terms()) {
        if (term.domain != IntegralDomain::Cell) continue;
        Real term_weight = 1.0;
        if (time_ctx) {
            if (term.time_derivative_order == 1) {
                term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
            } else if (term.time_derivative_order == 2) {
                term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
            } else if (term.time_derivative_order > 0) {
                term_weight = time_ctx->time_derivative_term_weight;
            } else {
                term_weight = time_ctx->non_time_derivative_term_weight;
            }
        }
        if (term_weight == 0.0) continue;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);

            if (want_matrix) {
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        env.j = j;
                        const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: cell bilinear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        output.matrixEntry(i, j) += (term_weight * w) * val.s;
                    }
                }
            } else if (want_vector) {
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                    if (val.kind != EvalValue<Real>::Kind::Scalar) {
                        throw FEException("Forms: cell linear integrand did not evaluate to scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    output.vectorEntry(i) += (term_weight * w) * val.s;
                }
            }
        }
    }

    output.has_matrix = want_matrix;
    output.has_vector = want_vector;
}

void FormKernel::computeBoundaryFace(
    const assembly::AssemblyContext& ctx,
    int boundary_marker,
    assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    const bool want_matrix = (ir_.kind() == FormKind::Bilinear);
    const bool want_vector = (ir_.kind() == FormKind::Linear);

    output.reserve(n_test, n_trial, want_matrix, want_vector);
    output.clear();

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    const auto* boundary_marker_updates = [&]() -> const std::vector<MaterialStateUpdate>* {
        const auto it = inlined_state_updates_.boundary_by_marker.find(boundary_marker);
        if (it == inlined_state_updates_.boundary_by_marker.end()) {
            return nullptr;
        }
        return &it->second;
    }();

    if (!inlined_state_updates_.boundary_all.empty() || boundary_marker_updates != nullptr) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            if (!inlined_state_updates_.boundary_all.empty()) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, ir_.kind(),
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.boundary_all,
                                                     Side::Minus, q);
            }
            if (boundary_marker_updates != nullptr && !boundary_marker_updates->empty()) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, ir_.kind(),
                                                     constitutive_state_.get(),
                                                     *boundary_marker_updates,
                                                     Side::Minus, q);
            }
        }
    }

    ConstitutiveCallCacheReal constitutive_cache;
    EvalEnvReal env{ctx, nullptr, ir_.kind(), Side::Minus, Side::Minus, 0, 0, constitutive_state_.get(), &constitutive_cache};

    for (const auto& term : ir_.terms()) {
        if (term.domain != IntegralDomain::Boundary) continue;
        if (term.boundary_marker >= 0 && term.boundary_marker != boundary_marker) continue;
        Real term_weight = 1.0;
        if (time_ctx) {
            if (term.time_derivative_order == 1) {
                term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
            } else if (term.time_derivative_order == 2) {
                term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
            } else if (term.time_derivative_order > 0) {
                term_weight = time_ctx->time_derivative_term_weight;
            } else {
                term_weight = time_ctx->non_time_derivative_term_weight;
            }
        }
        if (term_weight == 0.0) continue;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);

            if (want_matrix) {
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        env.j = j;
                        const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: boundary bilinear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        output.matrixEntry(i, j) += (term_weight * w) * val.s;
                    }
                }
            } else if (want_vector) {
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                    if (val.kind != EvalValue<Real>::Kind::Scalar) {
                        throw FEException("Forms: boundary linear integrand did not evaluate to scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    output.vectorEntry(i) += (term_weight * w) * val.s;
                }
            }
        }
    }

    output.has_matrix = want_matrix;
    output.has_vector = want_vector;
}

void FormKernel::computeInteriorFace(
    const assembly::AssemblyContext& ctx_minus,
    const assembly::AssemblyContext& ctx_plus,
    assembly::KernelOutput& output_minus,
    assembly::KernelOutput& output_plus,
    assembly::KernelOutput& coupling_mp,
    assembly::KernelOutput& coupling_pm)
{
    if (ir_.kind() != FormKind::Bilinear) {
        // Interior face assembly is currently defined only for bilinear forms in FormKernel.
        output_minus.clear();
        output_plus.clear();
        coupling_mp.clear();
        coupling_pm.clear();
        return;
    }

    const auto n_test_minus = ctx_minus.numTestDofs();
    const auto n_trial_minus = ctx_minus.numTrialDofs();
    const auto n_test_plus = ctx_plus.numTestDofs();
    const auto n_trial_plus = ctx_plus.numTrialDofs();

    output_minus.reserve(n_test_minus, n_trial_minus, true, false);
    output_plus.reserve(n_test_plus, n_trial_plus, true, false);
    coupling_mp.reserve(n_test_minus, n_trial_plus, true, false);
    coupling_pm.reserve(n_test_plus, n_trial_minus, true, false);

    output_minus.clear();
    output_plus.clear();
    coupling_mp.clear();
    coupling_pm.clear();

    const auto n_qpts = ctx_minus.numQuadraturePoints();

    if (!inlined_state_updates_.interior_face.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.interior_face,
                                                 Side::Minus, q);

            const auto* base_minus = ctx_minus.materialStateWorkBase();
            const auto* base_plus = ctx_plus.materialStateWorkBase();
            if (base_plus != nullptr && base_plus != base_minus) {
                applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.interior_face,
                                                     Side::Plus, q);
            }
        }
    }

    auto assembleBlock = [&](Side eval_side,
                             Side test_active,
                             Side trial_active,
                             assembly::KernelOutput& out,
                             LocalIndex n_test,
                             LocalIndex n_trial) {
        ConstitutiveCallCacheReal constitutive_cache;
        EvalEnvReal env{ctx_minus, &ctx_plus, FormKind::Bilinear, test_active, trial_active, 0, 0,
                        constitutive_state_.get(), &constitutive_cache};

        for (const auto& term : ir_.terms()) {
            if (term.domain != IntegralDomain::InteriorFace) continue;
            const auto& ctx_eval = ctxForSide(ctx_minus, &ctx_plus, eval_side);
            const auto* time_ctx = ctx_eval.timeIntegrationContext();
            Real term_weight = 1.0;
            if (time_ctx) {
                if (term.time_derivative_order == 1) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                } else if (term.time_derivative_order == 2) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                } else if (term.time_derivative_order > 0) {
                    term_weight = time_ctx->time_derivative_term_weight;
                } else {
                    term_weight = time_ctx->non_time_derivative_term_weight;
                }
            }
            if (term_weight == 0.0) continue;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx_eval.integrationWeight(q);
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        env.j = j;
                        const auto val = evalReal(*term.integrand.node(), env, eval_side, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: interior-face bilinear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        out.matrixEntry(i, j) += (term_weight * w) * val.s;
                    }
                }
            }
        }
        out.has_matrix = true;
        out.has_vector = false;
    };

    // minus-minus (evaluate on minus side)
    assembleBlock(Side::Minus, Side::Minus, Side::Minus,
                  output_minus, n_test_minus, n_trial_minus);

    // plus-plus (evaluate on plus side)
    assembleBlock(Side::Plus, Side::Plus, Side::Plus,
                  output_plus, n_test_plus, n_trial_plus);

    // minus-plus coupling (evaluate on minus side; trial active on plus)
    assembleBlock(Side::Minus, Side::Minus, Side::Plus,
                  coupling_mp, n_test_minus, n_trial_plus);

    // plus-minus coupling (evaluate on plus side; trial active on minus)
    assembleBlock(Side::Plus, Side::Plus, Side::Minus,
                  coupling_pm, n_test_plus, n_trial_minus);
}

void FormKernel::computeInterfaceFace(
    const assembly::AssemblyContext& ctx_minus,
    const assembly::AssemblyContext& ctx_plus,
    int interface_marker,
    assembly::KernelOutput& output_minus,
    assembly::KernelOutput& output_plus,
    assembly::KernelOutput& coupling_mp,
    assembly::KernelOutput& coupling_pm)
{
    if (ir_.kind() != FormKind::Bilinear) {
        // Interface face assembly is currently defined only for bilinear forms in FormKernel.
        output_minus.clear();
        output_plus.clear();
        coupling_mp.clear();
        coupling_pm.clear();
        return;
    }

    const auto n_test_minus = ctx_minus.numTestDofs();
    const auto n_trial_minus = ctx_minus.numTrialDofs();
    const auto n_test_plus = ctx_plus.numTestDofs();
    const auto n_trial_plus = ctx_plus.numTrialDofs();

    output_minus.reserve(n_test_minus, n_trial_minus, true, false);
    output_plus.reserve(n_test_plus, n_trial_plus, true, false);
    coupling_mp.reserve(n_test_minus, n_trial_plus, true, false);
    coupling_pm.reserve(n_test_plus, n_trial_minus, true, false);

    output_minus.clear();
    output_plus.clear();
    coupling_mp.clear();
    coupling_pm.clear();

    const auto n_qpts = ctx_minus.numQuadraturePoints();

    if (!inlined_state_updates_.interface_face.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.interface_face,
                                                 Side::Minus, q);

            const auto* base_minus = ctx_minus.materialStateWorkBase();
            const auto* base_plus = ctx_plus.materialStateWorkBase();
            if (base_plus != nullptr && base_plus != base_minus) {
                applyInlinedMaterialStateUpdatesReal(ctx_minus, &ctx_plus, FormKind::Bilinear,
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.interface_face,
                                                     Side::Plus, q);
            }
        }
    }

    auto assembleBlock = [&](Side eval_side,
                             Side test_active,
                             Side trial_active,
                             assembly::KernelOutput& out,
                             LocalIndex n_test,
                             LocalIndex n_trial) {
        ConstitutiveCallCacheReal constitutive_cache;
        EvalEnvReal env{ctx_minus, &ctx_plus, FormKind::Bilinear, test_active, trial_active, 0, 0,
                        constitutive_state_.get(), &constitutive_cache};

        for (const auto& term : ir_.terms()) {
            if (term.domain != IntegralDomain::InterfaceFace) continue;
            if (term.interface_marker >= 0 && term.interface_marker != interface_marker) continue;

            const auto& ctx_eval = ctxForSide(ctx_minus, &ctx_plus, eval_side);
            const auto* time_ctx = ctx_eval.timeIntegrationContext();
            Real term_weight = 1.0;
            if (time_ctx) {
                if (term.time_derivative_order == 1) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                } else if (term.time_derivative_order == 2) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                } else if (term.time_derivative_order > 0) {
                    term_weight = time_ctx->time_derivative_term_weight;
                } else {
                    term_weight = time_ctx->non_time_derivative_term_weight;
                }
            }
            if (term_weight == 0.0) continue;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx_eval.integrationWeight(q);
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        env.j = j;
                        const auto val = evalReal(*term.integrand.node(), env, eval_side, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: interface-face bilinear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        out.matrixEntry(i, j) += (term_weight * w) * val.s;
                    }
                }
            }
        }
        out.has_matrix = true;
        out.has_vector = false;
    };

    // minus-minus (evaluate on minus side)
    assembleBlock(Side::Minus, Side::Minus, Side::Minus,
                  output_minus, n_test_minus, n_trial_minus);

    // plus-plus (evaluate on plus side)
    assembleBlock(Side::Plus, Side::Plus, Side::Plus,
                  output_plus, n_test_plus, n_trial_plus);

    // minus-plus coupling (evaluate on minus side; trial active on plus)
    assembleBlock(Side::Minus, Side::Minus, Side::Plus,
                  coupling_mp, n_test_minus, n_trial_plus);

    // plus-minus coupling (evaluate on plus side; trial active on minus)
    assembleBlock(Side::Plus, Side::Plus, Side::Minus,
                  coupling_pm, n_test_plus, n_trial_minus);
}

// ============================================================================
// LinearFormKernel (affine residuals)
// ============================================================================

namespace {

[[nodiscard]] std::vector<assembly::FieldRequirement> mergeFieldRequirements(
    const std::vector<assembly::FieldRequirement>& a,
    const std::vector<assembly::FieldRequirement>& b)
{
    std::vector<assembly::FieldRequirement> out;
    out.reserve(a.size() + b.size());

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i].field == b[j].field) {
            out.push_back(assembly::FieldRequirement{
                a[i].field,
                a[i].required | b[j].required,
            });
            ++i;
            ++j;
        } else if (a[i].field < b[j].field) {
            out.push_back(a[i]);
            ++i;
        } else {
            out.push_back(b[j]);
            ++j;
        }
    }

    for (; i < a.size(); ++i) out.push_back(a[i]);
    for (; j < b.size(); ++j) out.push_back(b[j]);
    return out;
}

} // namespace

LinearFormKernel::LinearFormKernel(FormIR bilinear_ir,
                                   std::optional<FormIR> linear_ir,
                                   LinearKernelOutput output)
    : bilinear_ir_(std::move(bilinear_ir))
    , linear_ir_(std::move(linear_ir))
    , output_(output)
{
    if (!bilinear_ir_.isCompiled()) {
        throw std::invalid_argument("LinearFormKernel: bilinear IR is not compiled");
    }
    if (bilinear_ir_.kind() != FormKind::Bilinear) {
        throw std::invalid_argument("LinearFormKernel: bilinear IR kind must be Bilinear");
    }

    if (linear_ir_.has_value()) {
        if (!linear_ir_->isCompiled()) {
            throw std::invalid_argument("LinearFormKernel: linear IR is not compiled");
        }
        if (linear_ir_->kind() != FormKind::Linear) {
            throw std::invalid_argument("LinearFormKernel: linear IR kind must be Linear");
        }

        field_requirements_ = mergeFieldRequirements(bilinear_ir_.fieldRequirements(),
                                                     linear_ir_->fieldRequirements());
    } else {
        field_requirements_ = bilinear_ir_.fieldRequirements();
    }

    const FormIR* irs[2] = {&bilinear_ir_, linear_ir_.has_value() ? &(*linear_ir_) : nullptr};
    parameter_specs_ = computeParameterSpecs(std::span<const FormIR* const>{irs});

    // NOTE: Constitutive calls are currently not expected in affine residuals (the affine splitter
    // rejects Constitutive nodes). We still build state layout from the bilinear part for safety.
    constitutive_state_ = buildConstitutiveStateLayout(bilinear_ir_, material_state_spec_);
}

LinearFormKernel::~LinearFormKernel() = default;
LinearFormKernel::LinearFormKernel(LinearFormKernel&&) noexcept = default;
LinearFormKernel& LinearFormKernel::operator=(LinearFormKernel&&) noexcept = default;

assembly::RequiredData LinearFormKernel::getRequiredData() const noexcept
{
    auto req = bilinear_ir_.requiredData();
    if (linear_ir_.has_value()) {
        req |= linear_ir_->requiredData();
    }

    // Residual vector requires access to element-local DOF coefficients (but not u_h values/gradients).
    if (output_ != LinearKernelOutput::MatrixOnly) {
        req |= assembly::RequiredData::SolutionCoefficients;
    }

    if (material_state_spec_.bytes_per_qpt > 0u) {
        req |= assembly::RequiredData::MaterialState;
    }
    return req;
}

std::vector<assembly::FieldRequirement> LinearFormKernel::fieldRequirements() const
{
    return field_requirements_;
}

assembly::MaterialStateSpec LinearFormKernel::materialStateSpec() const noexcept
{
    return material_state_spec_;
}

std::vector<params::Spec> LinearFormKernel::parameterSpecs() const
{
    return parameter_specs_;
}

void LinearFormKernel::resolveInlinableConstitutives()
{
    FormIR* irs[2] = {&bilinear_ir_, linear_ir_.has_value() ? &(*linear_ir_) : nullptr};
    inlineInlinableConstitutives(std::span<FormIR*>{irs},
                                 constitutive_state_.get(),
                                 material_state_spec_,
                                 inlined_state_updates_);
}

void LinearFormKernel::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    auto rewrite = [&](FormIR& ir, std::string_view where) {
        const auto transform = [&](const FormExprNode& n) -> std::optional<FormExpr> {
            if (n.type() != FormExprType::ParameterSymbol) {
                return std::nullopt;
            }
            const auto key = n.symbolName();
            FE_THROW_IF(!key || key->empty(), InvalidArgumentException,
                        std::string(where) + ": ParameterSymbol node missing name");
            const auto slot = slot_of_real_param(*key);
            FE_THROW_IF(!slot.has_value(), InvalidArgumentException,
                        std::string(where) + ": could not resolve parameter slot for '" + std::string(*key) + "'");
            return FormExpr::parameterRef(*slot);
        };
        ir.transformIntegrands(transform);
        const auto rewrite_updates = [&](std::vector<MaterialStateUpdate>& updates) {
            for (auto& op : updates) {
                op.value = op.value.transformNodes(transform);
            }
        };
        rewrite_updates(inlined_state_updates_.cell);
        rewrite_updates(inlined_state_updates_.boundary_all);
        for (auto& kv : inlined_state_updates_.boundary_by_marker) {
            rewrite_updates(kv.second);
        }
        rewrite_updates(inlined_state_updates_.interior_face);
        rewrite_updates(inlined_state_updates_.interface_face);
    };

    rewrite(bilinear_ir_, "Forms::LinearFormKernel(bilinear)");
    if (linear_ir_.has_value()) {
        rewrite(*linear_ir_, "Forms::LinearFormKernel(linear)");
    }
}

int LinearFormKernel::maxTemporalDerivativeOrder() const noexcept
{
    const int a = bilinear_ir_.maxTimeDerivativeOrder();
    const int b = linear_ir_.has_value() ? linear_ir_->maxTimeDerivativeOrder() : 0;
    return std::max(a, b);
}

bool LinearFormKernel::hasCell() const noexcept
{
    return bilinear_ir_.hasCellTerms() || (linear_ir_.has_value() && linear_ir_->hasCellTerms());
}

bool LinearFormKernel::hasBoundaryFace() const noexcept
{
    return bilinear_ir_.hasBoundaryTerms() || (linear_ir_.has_value() && linear_ir_->hasBoundaryTerms());
}

bool LinearFormKernel::hasInteriorFace() const noexcept
{
    return bilinear_ir_.hasInteriorFaceTerms() || (linear_ir_.has_value() && linear_ir_->hasInteriorFaceTerms());
}

bool LinearFormKernel::hasInterfaceFace() const noexcept
{
    return bilinear_ir_.hasInterfaceFaceTerms() || (linear_ir_.has_value() && linear_ir_->hasInterfaceFaceTerms());
}

void LinearFormKernel::computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    const bool want_matrix = (output_ != LinearKernelOutput::VectorOnly);
    const bool want_vector = (output_ != LinearKernelOutput::MatrixOnly);
    output.reserve(n_test, n_trial, want_matrix, want_vector);
    output.clear();

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    if (!inlined_state_updates_.cell.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Bilinear,
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.cell,
                                                 Side::Minus, q);
        }
    }

    ConstitutiveCallCacheReal constitutive_cache;

    // 1) Assemble Jacobian (bilinear part) if requested.
    if (want_matrix) {
        EvalEnvReal env{ctx, nullptr, FormKind::Bilinear, Side::Minus, Side::Minus, 0, 0,
                        constitutive_state_.get(), &constitutive_cache};

        for (const auto& term : bilinear_ir_.terms()) {
            if (term.domain != IntegralDomain::Cell) continue;

            Real term_weight = 1.0;
            if (time_ctx) {
                if (term.time_derivative_order == 1) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                } else if (term.time_derivative_order == 2) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                } else if (term.time_derivative_order > 0) {
                    term_weight = time_ctx->time_derivative_term_weight;
                } else {
                    term_weight = time_ctx->non_time_derivative_term_weight;
                }
            }
            if (term_weight == 0.0) continue;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx.integrationWeight(q);
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        env.j = j;
                        const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: cell bilinear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        output.matrixEntry(i, j) += (term_weight * w) * val.s;
                    }
                }
            }
        }
    }

    // 2) Assemble residual vector if requested.
    if (want_vector) {
        const auto coeffs = ctx.solutionCoefficients();
        FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(n_trial), InvalidArgumentException,
                    "LinearFormKernel::computeCell: missing solution coefficients (need SolutionCoefficients)");

        // 2a) Add linear (trial-independent) contributions.
        if (linear_ir_.has_value()) {
            EvalEnvReal env{ctx, nullptr, FormKind::Linear, Side::Minus, Side::Minus, 0, 0,
                            constitutive_state_.get(), &constitutive_cache};

            for (const auto& term : linear_ir_->terms()) {
                if (term.domain != IntegralDomain::Cell) continue;

                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;

                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const Real w = ctx.integrationWeight(q);
                    for (LocalIndex i = 0; i < n_test; ++i) {
                        env.i = i;
                        const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: cell linear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        output.vectorEntry(i) += (term_weight * w) * val.s;
                    }
                }
            }
        }

        // 2b) Add bilinear contribution applied to coefficients (K*u).
        if (want_matrix) {
            for (LocalIndex i = 0; i < n_test; ++i) {
                Real sum = 0.0;
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    sum += output.matrixEntry(i, j) * coeffs[static_cast<std::size_t>(j)];
                }
                output.vectorEntry(i) += sum;
            }
        } else {
            EvalEnvReal env{ctx, nullptr, FormKind::Bilinear, Side::Minus, Side::Minus, 0, 0,
                            constitutive_state_.get(), &constitutive_cache};

            for (const auto& term : bilinear_ir_.terms()) {
                if (term.domain != IntegralDomain::Cell) continue;

                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;

                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const Real w = ctx.integrationWeight(q);
                    for (LocalIndex i = 0; i < n_test; ++i) {
                        env.i = i;
                        Real sum = 0.0;
                        for (LocalIndex j = 0; j < n_trial; ++j) {
                            env.j = j;
                            const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                            if (val.kind != EvalValue<Real>::Kind::Scalar) {
                                throw FEException("Forms: cell bilinear integrand did not evaluate to scalar",
                                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                            }
                            sum += val.s * coeffs[static_cast<std::size_t>(j)];
                        }
                        output.vectorEntry(i) += (term_weight * w) * sum;
                    }
                }
            }
        }
    }
}

void LinearFormKernel::computeBoundaryFace(
    const assembly::AssemblyContext& ctx,
    int boundary_marker,
    assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    const bool want_matrix = (output_ != LinearKernelOutput::VectorOnly);
    const bool want_vector = (output_ != LinearKernelOutput::MatrixOnly);
    output.reserve(n_test, n_trial, want_matrix, want_vector);
    output.clear();

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    const auto* boundary_marker_updates = [&]() -> const std::vector<MaterialStateUpdate>* {
        const auto it = inlined_state_updates_.boundary_by_marker.find(boundary_marker);
        if (it == inlined_state_updates_.boundary_by_marker.end()) {
            return nullptr;
        }
        return &it->second;
    }();

    if (!inlined_state_updates_.boundary_all.empty() || boundary_marker_updates != nullptr) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            if (!inlined_state_updates_.boundary_all.empty()) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Bilinear,
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.boundary_all,
                                                     Side::Minus, q);
            }
            if (boundary_marker_updates != nullptr && !boundary_marker_updates->empty()) {
                applyInlinedMaterialStateUpdatesReal(ctx, nullptr, FormKind::Bilinear,
                                                     constitutive_state_.get(),
                                                     *boundary_marker_updates,
                                                     Side::Minus, q);
            }
        }
    }

    ConstitutiveCallCacheReal constitutive_cache;

    // 1) Assemble Jacobian (bilinear boundary part) if requested.
    if (want_matrix) {
        EvalEnvReal env{ctx, nullptr, FormKind::Bilinear, Side::Minus, Side::Minus, 0, 0,
                        constitutive_state_.get(), &constitutive_cache};

        for (const auto& term : bilinear_ir_.terms()) {
            if (term.domain != IntegralDomain::Boundary) continue;
            if (term.boundary_marker >= 0 && term.boundary_marker != boundary_marker) continue;

            Real term_weight = 1.0;
            if (time_ctx) {
                if (term.time_derivative_order == 1) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                } else if (term.time_derivative_order == 2) {
                    term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                } else if (term.time_derivative_order > 0) {
                    term_weight = time_ctx->time_derivative_term_weight;
                } else {
                    term_weight = time_ctx->non_time_derivative_term_weight;
                }
            }
            if (term_weight == 0.0) continue;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx.integrationWeight(q);
                for (LocalIndex i = 0; i < n_test; ++i) {
                    env.i = i;
                    for (LocalIndex j = 0; j < n_trial; ++j) {
                        env.j = j;
                        const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: boundary bilinear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        output.matrixEntry(i, j) += (term_weight * w) * val.s;
                    }
                }
            }
        }
    }

    // 2) Assemble residual vector if requested.
    if (want_vector) {
        const auto coeffs = ctx.solutionCoefficients();
        FE_THROW_IF(coeffs.size() < static_cast<std::size_t>(n_trial), InvalidArgumentException,
                    "LinearFormKernel::computeBoundaryFace: missing solution coefficients (need SolutionCoefficients)");

        // 2a) Add linear (trial-independent) boundary contributions.
        if (linear_ir_.has_value()) {
            EvalEnvReal env{ctx, nullptr, FormKind::Linear, Side::Minus, Side::Minus, 0, 0,
                            constitutive_state_.get(), &constitutive_cache};

            for (const auto& term : linear_ir_->terms()) {
                if (term.domain != IntegralDomain::Boundary) continue;
                if (term.boundary_marker >= 0 && term.boundary_marker != boundary_marker) continue;

                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;

                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const Real w = ctx.integrationWeight(q);
                    for (LocalIndex i = 0; i < n_test; ++i) {
                        env.i = i;
                        const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                        if (val.kind != EvalValue<Real>::Kind::Scalar) {
                            throw FEException("Forms: boundary linear integrand did not evaluate to scalar",
                                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                        }
                        output.vectorEntry(i) += (term_weight * w) * val.s;
                    }
                }
            }
        }

        // 2b) Add bilinear boundary contribution applied to coefficients (K*u).
        if (want_matrix) {
            for (LocalIndex i = 0; i < n_test; ++i) {
                Real sum = 0.0;
                for (LocalIndex j = 0; j < n_trial; ++j) {
                    sum += output.matrixEntry(i, j) * coeffs[static_cast<std::size_t>(j)];
                }
                output.vectorEntry(i) += sum;
            }
        } else {
            EvalEnvReal env{ctx, nullptr, FormKind::Bilinear, Side::Minus, Side::Minus, 0, 0,
                            constitutive_state_.get(), &constitutive_cache};

            for (const auto& term : bilinear_ir_.terms()) {
                if (term.domain != IntegralDomain::Boundary) continue;
                if (term.boundary_marker >= 0 && term.boundary_marker != boundary_marker) continue;

                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;

                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    const Real w = ctx.integrationWeight(q);
                    for (LocalIndex i = 0; i < n_test; ++i) {
                        env.i = i;
                        Real sum = 0.0;
                        for (LocalIndex j = 0; j < n_trial; ++j) {
                            env.j = j;
                            const auto val = evalReal(*term.integrand.node(), env, Side::Minus, q);
                            if (val.kind != EvalValue<Real>::Kind::Scalar) {
                                throw FEException("Forms: boundary bilinear integrand did not evaluate to scalar",
                                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                            }
                            sum += val.s * coeffs[static_cast<std::size_t>(j)];
                        }
                        output.vectorEntry(i) += (term_weight * w) * sum;
                    }
                }
            }
        }
    }
}

void LinearFormKernel::computeInteriorFace(
    const assembly::AssemblyContext& /*ctx_minus*/,
    const assembly::AssemblyContext& /*ctx_plus*/,
    assembly::KernelOutput& output_minus,
    assembly::KernelOutput& output_plus,
    assembly::KernelOutput& coupling_mp,
    assembly::KernelOutput& coupling_pm)
{
    // Affine optimization currently does not cover interior-face (DG) residual assembly.
    output_minus.reserve(0, 0, false, false);
    output_plus.reserve(0, 0, false, false);
    coupling_mp.reserve(0, 0, false, false);
    coupling_pm.reserve(0, 0, false, false);
}

void LinearFormKernel::computeInterfaceFace(
    const assembly::AssemblyContext& /*ctx_minus*/,
    const assembly::AssemblyContext& /*ctx_plus*/,
    int /*interface_marker*/,
    assembly::KernelOutput& output_minus,
    assembly::KernelOutput& output_plus,
    assembly::KernelOutput& coupling_mp,
    assembly::KernelOutput& coupling_pm)
{
    // Affine optimization currently does not cover interface-face residual assembly.
    output_minus.reserve(0, 0, false, false);
    output_plus.reserve(0, 0, false, false);
    coupling_mp.reserve(0, 0, false, false);
    coupling_pm.reserve(0, 0, false, false);
}

// ============================================================================
// NonlinearFormKernel
// ============================================================================

NonlinearFormKernel::NonlinearFormKernel(FormIR residual_ir, ADMode ad_mode, NonlinearKernelOutput output)
    : residual_ir_(std::move(residual_ir))
    , ad_mode_(ad_mode)
    , output_(output)
{
    if (!residual_ir_.isCompiled()) {
        throw std::invalid_argument("NonlinearFormKernel: residual IR is not compiled");
    }
    if (residual_ir_.kind() != FormKind::Residual) {
        throw std::invalid_argument("NonlinearFormKernel: IR kind must be Residual");
    }

    const FormIR* irs[] = {&residual_ir_};
    parameter_specs_ = computeParameterSpecs(std::span<const FormIR* const>{irs});

    constitutive_state_ = buildConstitutiveStateLayout(residual_ir_, material_state_spec_);
}

NonlinearFormKernel::~NonlinearFormKernel() = default;
NonlinearFormKernel::NonlinearFormKernel(NonlinearFormKernel&&) noexcept = default;
NonlinearFormKernel& NonlinearFormKernel::operator=(NonlinearFormKernel&&) noexcept = default;

assembly::RequiredData NonlinearFormKernel::getRequiredData() const noexcept
{
    // Require solution data so the assembler sets per-cell coefficients and caches u_h, grad(u_h).
    auto req = residual_ir_.requiredData() |
               assembly::RequiredData::SolutionValues |
               assembly::RequiredData::SolutionGradients;
    if (material_state_spec_.bytes_per_qpt > 0u) {
        req |= assembly::RequiredData::MaterialState;
    }
    return req;
}

std::vector<assembly::FieldRequirement> NonlinearFormKernel::fieldRequirements() const
{
    return residual_ir_.fieldRequirements();
}

assembly::MaterialStateSpec NonlinearFormKernel::materialStateSpec() const noexcept
{
    return material_state_spec_;
}

std::vector<params::Spec> NonlinearFormKernel::parameterSpecs() const
{
    return parameter_specs_;
}

void NonlinearFormKernel::resolveInlinableConstitutives()
{
    FormIR* irs[] = {&residual_ir_};
    inlineInlinableConstitutives(std::span<FormIR*>{irs},
                                 constitutive_state_.get(),
                                 material_state_spec_,
                                 inlined_state_updates_);
}

void NonlinearFormKernel::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    const auto transform = [&](const FormExprNode& n) -> std::optional<FormExpr> {
        if (n.type() != FormExprType::ParameterSymbol) {
            return std::nullopt;
        }
        const auto key = n.symbolName();
        FE_THROW_IF(!key || key->empty(), InvalidArgumentException,
                    "Forms::NonlinearFormKernel: ParameterSymbol node missing name");
        const auto slot = slot_of_real_param(*key);
        FE_THROW_IF(!slot.has_value(), InvalidArgumentException,
                    "Forms::NonlinearFormKernel: could not resolve parameter slot for '" + std::string(*key) + "'");
        return FormExpr::parameterRef(*slot);
    };
    residual_ir_.transformIntegrands(transform);
    const auto rewrite_updates = [&](std::vector<MaterialStateUpdate>& updates) {
        for (auto& op : updates) {
            op.value = op.value.transformNodes(transform);
        }
    };
    rewrite_updates(inlined_state_updates_.cell);
    rewrite_updates(inlined_state_updates_.boundary_all);
    for (auto& kv : inlined_state_updates_.boundary_by_marker) {
        rewrite_updates(kv.second);
    }
    rewrite_updates(inlined_state_updates_.interior_face);
    rewrite_updates(inlined_state_updates_.interface_face);
}

bool NonlinearFormKernel::hasCell() const noexcept { return residual_ir_.hasCellTerms(); }
bool NonlinearFormKernel::hasBoundaryFace() const noexcept { return residual_ir_.hasBoundaryTerms(); }
bool NonlinearFormKernel::hasInteriorFace() const noexcept { return residual_ir_.hasInteriorFaceTerms(); }
bool NonlinearFormKernel::hasInterfaceFace() const noexcept { return residual_ir_.hasInterfaceFaceTerms(); }

void NonlinearFormKernel::computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    const bool want_matrix = (output_ != NonlinearKernelOutput::VectorOnly);
    const bool want_vector = (output_ != NonlinearKernelOutput::MatrixOnly);
    output.reserve(n_test, n_trial, want_matrix, want_vector);

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    if (!inlined_state_updates_.cell.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesDual(ctx, nullptr,
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.cell,
                                                 Side::Minus, q);
        }
    }

    thread_local DualWorkspace ws;

    for (LocalIndex i = 0; i < n_test; ++i) {
        std::span<Real> row{};
        if (want_matrix) {
            row = std::span<Real>(
                output.local_matrix.data() + static_cast<std::size_t>(i * n_trial),
                static_cast<std::size_t>(n_trial));
        }

        Dual residual_i;
        residual_i.value = 0.0;
        residual_i.deriv = row;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);

            // Reset workspace for this quadrature evaluation.
            const std::size_t n_trial_ad = want_matrix ? static_cast<std::size_t>(n_trial) : 0u;
            ws.reset(n_trial_ad);

            ConstitutiveCallCacheDual constitutive_cache;
            EvalEnvDual env{ctx, nullptr, Side::Minus, Side::Minus, i, n_trial_ad, &ws,
                            constitutive_state_.get(), &constitutive_cache};

            Dual sum_q = makeDualConstant(0.0, ws.alloc());

            for (const auto& term : residual_ir_.terms()) {
                if (term.domain != IntegralDomain::Cell) continue;
                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;
                const auto val = evalDual(*term.integrand.node(), env, Side::Minus, q);
                if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: residual cell integrand did not evaluate to scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                sum_q.value += term_weight * val.s.value;
                for (std::size_t j = 0; j < sum_q.deriv.size(); ++j) {
                    sum_q.deriv[j] += term_weight * val.s.deriv[j];
                }
            }

            residual_i.value += w * sum_q.value;
            for (std::size_t j = 0; j < residual_i.deriv.size(); ++j) {
                residual_i.deriv[j] += w * sum_q.deriv[j];
            }
        }

        if (want_vector) {
            output.local_vector[static_cast<std::size_t>(i)] = residual_i.value;
        }
    }
}

void NonlinearFormKernel::computeBoundaryFace(
    const assembly::AssemblyContext& ctx,
    int boundary_marker,
    assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();

    const bool want_matrix = (output_ != NonlinearKernelOutput::VectorOnly);
    const bool want_vector = (output_ != NonlinearKernelOutput::MatrixOnly);
    output.reserve(n_test, n_trial, want_matrix, want_vector);

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    const auto* boundary_marker_updates = [&]() -> const std::vector<MaterialStateUpdate>* {
        const auto it = inlined_state_updates_.boundary_by_marker.find(boundary_marker);
        if (it == inlined_state_updates_.boundary_by_marker.end()) {
            return nullptr;
        }
        return &it->second;
    }();

    if (!inlined_state_updates_.boundary_all.empty() || boundary_marker_updates != nullptr) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            if (!inlined_state_updates_.boundary_all.empty()) {
                applyInlinedMaterialStateUpdatesDual(ctx, nullptr,
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.boundary_all,
                                                     Side::Minus, q);
            }
            if (boundary_marker_updates != nullptr && !boundary_marker_updates->empty()) {
                applyInlinedMaterialStateUpdatesDual(ctx, nullptr,
                                                     constitutive_state_.get(),
                                                     *boundary_marker_updates,
                                                     Side::Minus, q);
            }
        }
    }

    thread_local DualWorkspace ws;

    for (LocalIndex i = 0; i < n_test; ++i) {
        std::span<Real> row{};
        if (want_matrix) {
            row = std::span<Real>(
                output.local_matrix.data() + static_cast<std::size_t>(i * n_trial),
                static_cast<std::size_t>(n_trial));
        }

        Dual residual_i;
        residual_i.value = 0.0;
        residual_i.deriv = row;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);
            const std::size_t n_trial_ad = want_matrix ? static_cast<std::size_t>(n_trial) : 0u;
            ws.reset(n_trial_ad);

            ConstitutiveCallCacheDual constitutive_cache;
            EvalEnvDual env{ctx, nullptr, Side::Minus, Side::Minus, i, n_trial_ad, &ws,
                            constitutive_state_.get(), &constitutive_cache};
            Dual sum_q = makeDualConstant(0.0, ws.alloc());

            for (const auto& term : residual_ir_.terms()) {
                if (term.domain != IntegralDomain::Boundary) continue;
                if (term.boundary_marker >= 0 && term.boundary_marker != boundary_marker) continue;
                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;
                const auto val = evalDual(*term.integrand.node(), env, Side::Minus, q);
                if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: residual boundary integrand did not evaluate to scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                sum_q.value += term_weight * val.s.value;
                for (std::size_t j = 0; j < sum_q.deriv.size(); ++j) {
                    sum_q.deriv[j] += term_weight * val.s.deriv[j];
                }
            }

            residual_i.value += w * sum_q.value;
            for (std::size_t j = 0; j < residual_i.deriv.size(); ++j) {
                residual_i.deriv[j] += w * sum_q.deriv[j];
            }
        }

        if (want_vector) {
            output.local_vector[static_cast<std::size_t>(i)] = residual_i.value;
        }
    }
}

void NonlinearFormKernel::computeInteriorFace(
    const assembly::AssemblyContext& ctx_minus,
    const assembly::AssemblyContext& ctx_plus,
    assembly::KernelOutput& output_minus,
    assembly::KernelOutput& output_plus,
    assembly::KernelOutput& coupling_mp,
    assembly::KernelOutput& coupling_pm)
{
    const auto n_test_minus = ctx_minus.numTestDofs();
    const auto n_trial_minus = ctx_minus.numTrialDofs();
    const auto n_test_plus = ctx_plus.numTestDofs();
    const auto n_trial_plus = ctx_plus.numTrialDofs();

    const bool want_matrix = (output_ != NonlinearKernelOutput::VectorOnly);
    const bool want_vector = (output_ != NonlinearKernelOutput::MatrixOnly);

    output_minus.reserve(n_test_minus, n_trial_minus, want_matrix, want_vector);
    output_plus.reserve(n_test_plus, n_trial_plus, want_matrix, want_vector);
    coupling_mp.reserve(n_test_minus, n_trial_plus, want_matrix, /*need_vector=*/false);
    coupling_pm.reserve(n_test_plus, n_trial_minus, want_matrix, /*need_vector=*/false);

    const bool want_minus_matrix = output_minus.has_matrix;
    const bool want_minus_vector = output_minus.has_vector;
    const bool want_plus_matrix = output_plus.has_matrix;
    const bool want_plus_vector = output_plus.has_vector;
    const bool want_mp_matrix = coupling_mp.has_matrix;
    const bool want_pm_matrix = coupling_pm.has_matrix;

    const auto n_qpts = ctx_minus.numQuadraturePoints();

    if (!inlined_state_updates_.interior_face.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.interior_face,
                                                 Side::Minus, q);

            const auto* base_minus = ctx_minus.materialStateWorkBase();
            const auto* base_plus = ctx_plus.materialStateWorkBase();
            if (base_plus != nullptr && base_plus != base_minus) {
                applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.interior_face,
                                                     Side::Plus, q);
            }
        }
    }

    thread_local DualWorkspace ws;

    auto assembleResidualJacBlock = [&](Side eval_side,
                                        Side test_active,
                                        Side trial_active,
                                        const assembly::AssemblyContext& ctx_eval,
                                        const assembly::AssemblyContext& ctx_other,
                                        assembly::KernelOutput& out,
                                        LocalIndex n_test,
                                        LocalIndex n_trial) {
        const bool want_matrix = out.has_matrix;
        const bool want_vector = out.has_vector;
        if (!want_matrix && !want_vector) {
            return;
        }
        const auto* time_ctx = ctx_eval.timeIntegrationContext();
        for (LocalIndex i = 0; i < n_test; ++i) {
            std::span<Real> row{};
            if (want_matrix) {
                row = std::span<Real>(
                    out.local_matrix.data() + static_cast<std::size_t>(i * n_trial),
                    static_cast<std::size_t>(n_trial));
                std::fill(row.begin(), row.end(), 0.0);
            }

            Dual residual_i;
            residual_i.value = 0.0;
            residual_i.deriv = row;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx_eval.integrationWeight(q);
                const std::size_t n_trial_ad = want_matrix ? static_cast<std::size_t>(n_trial) : 0u;
                ws.reset(n_trial_ad);

                ConstitutiveCallCacheDual constitutive_cache;
                EvalEnvDual env{ctx_eval, &ctx_other, test_active, trial_active, i,
                                n_trial_ad, &ws, constitutive_state_.get(), &constitutive_cache};

                Dual sum_q = makeDualConstant(0.0, ws.alloc());
                for (const auto& term : residual_ir_.terms()) {
                    if (term.domain != IntegralDomain::InteriorFace) continue;
                    Real term_weight = 1.0;
                    if (time_ctx) {
                        if (term.time_derivative_order == 1) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                        } else if (term.time_derivative_order == 2) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                        } else if (term.time_derivative_order > 0) {
                            term_weight = time_ctx->time_derivative_term_weight;
                        } else {
                            term_weight = time_ctx->non_time_derivative_term_weight;
                        }
                    }
                    if (term_weight == 0.0) continue;
                    const auto val = evalDual(*term.integrand.node(), env, eval_side, q);
                    if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                        throw FEException("Forms: residual interior-face integrand did not evaluate to scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    sum_q.value += term_weight * val.s.value;
                    for (std::size_t j = 0; j < sum_q.deriv.size(); ++j) {
                        sum_q.deriv[j] += term_weight * val.s.deriv[j];
                    }
                }

                residual_i.value += w * sum_q.value;
                for (std::size_t j = 0; j < residual_i.deriv.size(); ++j) {
                    residual_i.deriv[j] += w * sum_q.deriv[j];
                }
            }

            if (want_vector) {
                out.local_vector[static_cast<std::size_t>(i)] = residual_i.value;
            }
        }

    };

    if (want_minus_matrix || want_minus_vector) {
        // minus equations, derivatives w.r.t minus dofs (residual + minus-minus block)
        assembleResidualJacBlock(Side::Minus, Side::Minus, Side::Minus,
                                 ctx_minus, ctx_plus,
                                 output_minus, n_test_minus, n_trial_minus);
    }

    if (want_mp_matrix) {
        // minus equations, derivatives w.r.t plus dofs (minus-plus coupling)
        assembleResidualJacBlock(Side::Minus, Side::Minus, Side::Plus,
                                 ctx_minus, ctx_plus,
                                 coupling_mp, n_test_minus, n_trial_plus);
    }

    if (want_plus_matrix || want_plus_vector) {
        // plus equations, derivatives w.r.t plus dofs (residual + plus-plus block)
        assembleResidualJacBlock(Side::Plus, Side::Plus, Side::Plus,
                                 ctx_plus, ctx_minus,
                                 output_plus, n_test_plus, n_trial_plus);
    }

    if (want_pm_matrix) {
        // plus equations, derivatives w.r.t minus dofs (plus-minus coupling)
        assembleResidualJacBlock(Side::Plus, Side::Plus, Side::Minus,
                                 ctx_plus, ctx_minus,
                                 coupling_pm, n_test_plus, n_trial_minus);
    }
}

void NonlinearFormKernel::computeInterfaceFace(
    const assembly::AssemblyContext& ctx_minus,
    const assembly::AssemblyContext& ctx_plus,
    int interface_marker,
    assembly::KernelOutput& output_minus,
    assembly::KernelOutput& output_plus,
    assembly::KernelOutput& coupling_mp,
    assembly::KernelOutput& coupling_pm)
{
    const auto n_test_minus = ctx_minus.numTestDofs();
    const auto n_trial_minus = ctx_minus.numTrialDofs();
    const auto n_test_plus = ctx_plus.numTestDofs();
    const auto n_trial_plus = ctx_plus.numTrialDofs();

    const bool want_matrix = (output_ != NonlinearKernelOutput::VectorOnly);
    const bool want_vector = (output_ != NonlinearKernelOutput::MatrixOnly);

    output_minus.reserve(n_test_minus, n_trial_minus, want_matrix, want_vector);
    output_plus.reserve(n_test_plus, n_trial_plus, want_matrix, want_vector);
    coupling_mp.reserve(n_test_minus, n_trial_plus, want_matrix, /*need_vector=*/false);
    coupling_pm.reserve(n_test_plus, n_trial_minus, want_matrix, /*need_vector=*/false);

    const bool want_minus_matrix = output_minus.has_matrix;
    const bool want_minus_vector = output_minus.has_vector;
    const bool want_plus_matrix = output_plus.has_matrix;
    const bool want_plus_vector = output_plus.has_vector;
    const bool want_mp_matrix = coupling_mp.has_matrix;
    const bool want_pm_matrix = coupling_pm.has_matrix;

    const auto n_qpts = ctx_minus.numQuadraturePoints();

    if (!inlined_state_updates_.interface_face.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                 constitutive_state_.get(),
                                                 inlined_state_updates_.interface_face,
                                                 Side::Minus, q);

            const auto* base_minus = ctx_minus.materialStateWorkBase();
            const auto* base_plus = ctx_plus.materialStateWorkBase();
            if (base_plus != nullptr && base_plus != base_minus) {
                applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                     constitutive_state_.get(),
                                                     inlined_state_updates_.interface_face,
                                                     Side::Plus, q);
            }
        }
    }

    thread_local DualWorkspace ws;

    auto assembleResidualJacBlock = [&](Side eval_side,
                                        Side test_active,
                                        Side trial_active,
                                        const assembly::AssemblyContext& ctx_eval,
                                        const assembly::AssemblyContext& ctx_other,
                                        assembly::KernelOutput& out,
                                        LocalIndex n_test,
                                        LocalIndex n_trial) {
        const bool want_matrix = out.has_matrix;
        const bool want_vector = out.has_vector;
        if (!want_matrix && !want_vector) {
            return;
        }
        const auto* time_ctx = ctx_eval.timeIntegrationContext();
        for (LocalIndex i = 0; i < n_test; ++i) {
            std::span<Real> row{};
            if (want_matrix) {
                row = std::span<Real>(
                    out.local_matrix.data() + static_cast<std::size_t>(i * n_trial),
                    static_cast<std::size_t>(n_trial));
                std::fill(row.begin(), row.end(), 0.0);
            }

            Dual residual_i;
            residual_i.value = 0.0;
            residual_i.deriv = row;

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx_eval.integrationWeight(q);
                const std::size_t n_trial_ad = want_matrix ? static_cast<std::size_t>(n_trial) : 0u;
                ws.reset(n_trial_ad);

                ConstitutiveCallCacheDual constitutive_cache;
                EvalEnvDual env{ctx_eval, &ctx_other, test_active, trial_active, i,
                                n_trial_ad, &ws, constitutive_state_.get(), &constitutive_cache};

                Dual sum_q = makeDualConstant(0.0, ws.alloc());
                for (const auto& term : residual_ir_.terms()) {
                    if (term.domain != IntegralDomain::InterfaceFace) continue;
                    if (term.interface_marker >= 0 && term.interface_marker != interface_marker) continue;
                    Real term_weight = 1.0;
                    if (time_ctx) {
                        if (term.time_derivative_order == 1) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                        } else if (term.time_derivative_order == 2) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                        } else if (term.time_derivative_order > 0) {
                            term_weight = time_ctx->time_derivative_term_weight;
                        } else {
                            term_weight = time_ctx->non_time_derivative_term_weight;
                        }
                    }
                    if (term_weight == 0.0) continue;
                    const auto val = evalDual(*term.integrand.node(), env, eval_side, q);
                    if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                        throw FEException("Forms: residual interface-face integrand did not evaluate to scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    sum_q.value += term_weight * val.s.value;
                    for (std::size_t j = 0; j < sum_q.deriv.size(); ++j) {
                        sum_q.deriv[j] += term_weight * val.s.deriv[j];
                    }
                }

                residual_i.value += w * sum_q.value;
                for (std::size_t j = 0; j < residual_i.deriv.size(); ++j) {
                    residual_i.deriv[j] += w * sum_q.deriv[j];
                }
            }

            if (want_vector) {
                out.local_vector[static_cast<std::size_t>(i)] = residual_i.value;
            }
        }
    };

    if (want_minus_matrix || want_minus_vector) {
        // minus equations, derivatives w.r.t minus dofs (residual + minus-minus block)
        assembleResidualJacBlock(Side::Minus, Side::Minus, Side::Minus,
                                 ctx_minus, ctx_plus,
                                 output_minus, n_test_minus, n_trial_minus);
    }

    if (want_mp_matrix) {
        // minus equations, derivatives w.r.t plus dofs (minus-plus coupling)
        assembleResidualJacBlock(Side::Minus, Side::Minus, Side::Plus,
                                 ctx_minus, ctx_plus,
                                 coupling_mp, n_test_minus, n_trial_plus);
    }

    if (want_plus_matrix || want_plus_vector) {
        // plus equations, derivatives w.r.t plus dofs (residual + plus-plus block)
        assembleResidualJacBlock(Side::Plus, Side::Plus, Side::Plus,
                                 ctx_plus, ctx_minus,
                                 output_plus, n_test_plus, n_trial_plus);
    }

    if (want_pm_matrix) {
        // plus equations, derivatives w.r.t minus dofs (plus-minus coupling)
        assembleResidualJacBlock(Side::Plus, Side::Plus, Side::Minus,
                                 ctx_plus, ctx_minus,
                                 coupling_pm, n_test_plus, n_trial_minus);
    }
}

// ============================================================================
// CoupledResidualSensitivityKernel (dR/dQ for coupled-BC chain rule)
// ============================================================================

CoupledResidualSensitivityKernel::CoupledResidualSensitivityKernel(const NonlinearFormKernel& base,
                                                                   std::uint32_t coupled_integral_slot,
                                                                   std::span<const Real> daux_dintegrals,
                                                                   std::size_t num_integrals)
    : base_(&base)
    , coupled_integral_slot_(coupled_integral_slot)
    , daux_dintegrals_(daux_dintegrals)
    , num_integrals_(num_integrals)
{
    FE_CHECK_NOT_NULL(base_, "CoupledResidualSensitivityKernel: base kernel");
}

assembly::RequiredData CoupledResidualSensitivityKernel::getRequiredData() const noexcept
{
    // Mirror the base residual kernel requirements.
    return base_->getRequiredData();
}

std::vector<assembly::FieldRequirement> CoupledResidualSensitivityKernel::fieldRequirements() const
{
    return base_->fieldRequirements();
}

assembly::MaterialStateSpec CoupledResidualSensitivityKernel::materialStateSpec() const noexcept
{
    return base_->materialStateSpec();
}

std::vector<params::Spec> CoupledResidualSensitivityKernel::parameterSpecs() const
{
    return base_->parameterSpecs();
}

int CoupledResidualSensitivityKernel::maxTemporalDerivativeOrder() const noexcept
{
    return base_->maxTemporalDerivativeOrder();
}

bool CoupledResidualSensitivityKernel::hasCell() const noexcept
{
    return base_->residualIR().hasCellTerms();
}

bool CoupledResidualSensitivityKernel::hasBoundaryFace() const noexcept
{
    return base_->residualIR().hasBoundaryTerms();
}

bool CoupledResidualSensitivityKernel::hasInteriorFace() const noexcept
{
    return base_->residualIR().hasInteriorFaceTerms();
}

bool CoupledResidualSensitivityKernel::hasInterfaceFace() const noexcept
{
    return base_->residualIR().hasInterfaceFaceTerms();
}

void CoupledResidualSensitivityKernel::computeCell(const assembly::AssemblyContext& ctx,
                                                   assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();
    output.reserve(n_test, n_trial, /*need_matrix=*/false, /*need_vector=*/true);
    std::fill(output.local_vector.begin(), output.local_vector.end(), 0.0);

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    const auto* constitutive_state = base_->constitutiveStateLayout();
    const auto& updates = base_->inlinedStateUpdates().cell;
    if (!updates.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesDual(ctx, nullptr,
                                                 constitutive_state,
                                                 updates,
                                                 Side::Minus, q);
        }
    }

    thread_local DualWorkspace ws;

    for (LocalIndex i = 0; i < n_test; ++i) {
        Real dres_i = 0.0;

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);
            ws.reset(/*num_dofs=*/1u);

            ConstitutiveCallCacheDual constitutive_cache;
            EvalEnvDual env{ctx, nullptr, Side::Minus, Side::Minus, i, /*n_trial_dofs=*/0u, &ws,
                            constitutive_state, &constitutive_cache};
            env.coupled_integral_seed_slot = coupled_integral_slot_;
            env.coupled_aux_dseed = daux_dintegrals_;
            env.coupled_aux_dseed_cols = num_integrals_;

            Real sum_dq = 0.0;
            for (const auto& term : base_->residualIR().terms()) {
                if (term.domain != IntegralDomain::Cell) continue;
                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;

                const auto val = evalDual(*term.integrand.node(), env, Side::Minus, q);
                if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: coupled residual sensitivity cell integrand did not evaluate to scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (!val.s.deriv.empty()) {
                    sum_dq += term_weight * val.s.deriv[0];
                }
            }

            dres_i += w * sum_dq;
        }

        output.local_vector[static_cast<std::size_t>(i)] = dres_i;
    }
}

void CoupledResidualSensitivityKernel::computeBoundaryFace(const assembly::AssemblyContext& ctx,
                                                           int boundary_marker,
                                                           assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();
    output.reserve(n_test, n_trial, /*need_matrix=*/false, /*need_vector=*/true);
    std::fill(output.local_vector.begin(), output.local_vector.end(), 0.0);

    const auto n_qpts = ctx.numQuadraturePoints();
    const auto* time_ctx = ctx.timeIntegrationContext();

    const auto* constitutive_state = base_->constitutiveStateLayout();
    const auto& updates_all = base_->inlinedStateUpdates().boundary_all;
    const auto* updates_marker = [&]() -> const std::vector<MaterialStateUpdate>* {
        const auto it = base_->inlinedStateUpdates().boundary_by_marker.find(boundary_marker);
        if (it == base_->inlinedStateUpdates().boundary_by_marker.end()) {
            return nullptr;
        }
        return &it->second;
    }();

    if (!updates_all.empty() || (updates_marker != nullptr && !updates_marker->empty())) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            if (!updates_all.empty()) {
                applyInlinedMaterialStateUpdatesDual(ctx, nullptr,
                                                     constitutive_state,
                                                     updates_all,
                                                     Side::Minus, q);
            }
            if (updates_marker != nullptr && !updates_marker->empty()) {
                applyInlinedMaterialStateUpdatesDual(ctx, nullptr,
                                                     constitutive_state,
                                                     *updates_marker,
                                                     Side::Minus, q);
            }
        }
    }

    thread_local DualWorkspace ws;

    for (LocalIndex i = 0; i < n_test; ++i) {
        Real dres_i = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const Real w = ctx.integrationWeight(q);
            ws.reset(/*num_dofs=*/1u);

            ConstitutiveCallCacheDual constitutive_cache;
            EvalEnvDual env{ctx, nullptr, Side::Minus, Side::Minus, i, /*n_trial_dofs=*/0u, &ws,
                            constitutive_state, &constitutive_cache};
            env.coupled_integral_seed_slot = coupled_integral_slot_;
            env.coupled_aux_dseed = daux_dintegrals_;
            env.coupled_aux_dseed_cols = num_integrals_;

            Real sum_dq = 0.0;
            for (const auto& term : base_->residualIR().terms()) {
                if (term.domain != IntegralDomain::Boundary) continue;
                if (term.boundary_marker >= 0 && term.boundary_marker != boundary_marker) continue;
                Real term_weight = 1.0;
                if (time_ctx) {
                    if (term.time_derivative_order == 1) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                    } else if (term.time_derivative_order == 2) {
                        term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                    } else if (term.time_derivative_order > 0) {
                        term_weight = time_ctx->time_derivative_term_weight;
                    } else {
                        term_weight = time_ctx->non_time_derivative_term_weight;
                    }
                }
                if (term_weight == 0.0) continue;

                const auto val = evalDual(*term.integrand.node(), env, Side::Minus, q);
                if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                    throw FEException("Forms: coupled residual sensitivity boundary integrand did not evaluate to scalar",
                                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                }
                if (!val.s.deriv.empty()) {
                    sum_dq += term_weight * val.s.deriv[0];
                }
            }

            dres_i += w * sum_dq;
        }

        output.local_vector[static_cast<std::size_t>(i)] = dres_i;
    }
}

void CoupledResidualSensitivityKernel::computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                                                           const assembly::AssemblyContext& ctx_plus,
                                                           assembly::KernelOutput& output_minus,
                                                           assembly::KernelOutput& output_plus,
                                                           assembly::KernelOutput& coupling_mp,
                                                           assembly::KernelOutput& coupling_pm)
{
    const auto n_test_minus = ctx_minus.numTestDofs();
    const auto n_trial_minus = ctx_minus.numTrialDofs();
    const auto n_test_plus = ctx_plus.numTestDofs();
    const auto n_trial_plus = ctx_plus.numTrialDofs();

    output_minus.reserve(n_test_minus, n_trial_minus, /*need_matrix=*/false, /*need_vector=*/true);
    output_plus.reserve(n_test_plus, n_trial_plus, /*need_matrix=*/false, /*need_vector=*/true);
    coupling_mp.reserve(0, 0, false, false);
    coupling_pm.reserve(0, 0, false, false);

    std::fill(output_minus.local_vector.begin(), output_minus.local_vector.end(), 0.0);
    std::fill(output_plus.local_vector.begin(), output_plus.local_vector.end(), 0.0);

    const auto n_qpts = ctx_minus.numQuadraturePoints();
    const auto* constitutive_state = base_->constitutiveStateLayout();
    const auto& updates = base_->inlinedStateUpdates().interior_face;
    if (!updates.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                 constitutive_state,
                                                 updates,
                                                 Side::Minus, q);

            const auto* base_minus = ctx_minus.materialStateWorkBase();
            const auto* base_plus = ctx_plus.materialStateWorkBase();
            if (base_plus != nullptr && base_plus != base_minus) {
                applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                     constitutive_state,
                                                     updates,
                                                     Side::Plus, q);
            }
        }
    }

    thread_local DualWorkspace ws;

    auto assembleResidualBlock = [&](Side eval_side,
                                     Side test_active,
                                     Side trial_active,
                                     const assembly::AssemblyContext& ctx_eval,
                                     const assembly::AssemblyContext& ctx_other,
                                     assembly::KernelOutput& out,
                                     LocalIndex n_test) {
        const auto* time_ctx = ctx_eval.timeIntegrationContext();
        for (LocalIndex i = 0; i < n_test; ++i) {
            Real dres_i = 0.0;
            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx_eval.integrationWeight(q);
                ws.reset(/*num_dofs=*/1u);

                ConstitutiveCallCacheDual constitutive_cache;
                EvalEnvDual env{ctx_eval, &ctx_other, test_active, trial_active, i,
                                /*n_trial_dofs=*/0u, &ws, constitutive_state, &constitutive_cache};
                env.coupled_integral_seed_slot = coupled_integral_slot_;
                env.coupled_aux_dseed = daux_dintegrals_;
                env.coupled_aux_dseed_cols = num_integrals_;

                Real sum_dq = 0.0;
                for (const auto& term : base_->residualIR().terms()) {
                    if (term.domain != IntegralDomain::InteriorFace) continue;
                    Real term_weight = 1.0;
                    if (time_ctx) {
                        if (term.time_derivative_order == 1) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                        } else if (term.time_derivative_order == 2) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                        } else if (term.time_derivative_order > 0) {
                            term_weight = time_ctx->time_derivative_term_weight;
                        } else {
                            term_weight = time_ctx->non_time_derivative_term_weight;
                        }
                    }
                    if (term_weight == 0.0) continue;
                    const auto val = evalDual(*term.integrand.node(), env, eval_side, q);
                    if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                        throw FEException("Forms: coupled residual sensitivity interior-face integrand did not evaluate to scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    if (!val.s.deriv.empty()) {
                        sum_dq += term_weight * val.s.deriv[0];
                    }
                }

                dres_i += w * sum_dq;
            }
            out.local_vector[static_cast<std::size_t>(i)] = dres_i;
        }
    };

    // Mirror the base kernel's convention for interior-face residual assembly.
    assembleResidualBlock(Side::Minus, Side::Minus, Side::Minus,
                          ctx_minus, ctx_plus,
                          output_minus, n_test_minus);
    assembleResidualBlock(Side::Plus, Side::Plus, Side::Plus,
                          ctx_plus, ctx_minus,
                          output_plus, n_test_plus);
}

void CoupledResidualSensitivityKernel::computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                                                            const assembly::AssemblyContext& ctx_plus,
                                                            int interface_marker,
                                                            assembly::KernelOutput& output_minus,
                                                            assembly::KernelOutput& output_plus,
                                                            assembly::KernelOutput& coupling_mp,
                                                            assembly::KernelOutput& coupling_pm)
{
    const auto n_test_minus = ctx_minus.numTestDofs();
    const auto n_trial_minus = ctx_minus.numTrialDofs();
    const auto n_test_plus = ctx_plus.numTestDofs();
    const auto n_trial_plus = ctx_plus.numTrialDofs();

    output_minus.reserve(n_test_minus, n_trial_minus, /*need_matrix=*/false, /*need_vector=*/true);
    output_plus.reserve(n_test_plus, n_trial_plus, /*need_matrix=*/false, /*need_vector=*/true);
    coupling_mp.reserve(0, 0, false, false);
    coupling_pm.reserve(0, 0, false, false);

    std::fill(output_minus.local_vector.begin(), output_minus.local_vector.end(), 0.0);
    std::fill(output_plus.local_vector.begin(), output_plus.local_vector.end(), 0.0);

    const auto n_qpts = ctx_minus.numQuadraturePoints();
    const auto* constitutive_state = base_->constitutiveStateLayout();
    const auto& updates = base_->inlinedStateUpdates().interface_face;
    if (!updates.empty()) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                 constitutive_state,
                                                 updates,
                                                 Side::Minus, q);

            const auto* base_minus = ctx_minus.materialStateWorkBase();
            const auto* base_plus = ctx_plus.materialStateWorkBase();
            if (base_plus != nullptr && base_plus != base_minus) {
                applyInlinedMaterialStateUpdatesDual(ctx_minus, &ctx_plus,
                                                     constitutive_state,
                                                     updates,
                                                     Side::Plus, q);
            }
        }
    }

    thread_local DualWorkspace ws;

    auto assembleResidualBlock = [&](Side eval_side,
                                     Side test_active,
                                     Side trial_active,
                                     const assembly::AssemblyContext& ctx_eval,
                                     const assembly::AssemblyContext& ctx_other,
                                     assembly::KernelOutput& out,
                                     LocalIndex n_test) {
        const auto* time_ctx = ctx_eval.timeIntegrationContext();
        for (LocalIndex i = 0; i < n_test; ++i) {
            Real dres_i = 0.0;
            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const Real w = ctx_eval.integrationWeight(q);
                ws.reset(/*num_dofs=*/1u);

                ConstitutiveCallCacheDual constitutive_cache;
                EvalEnvDual env{ctx_eval, &ctx_other, test_active, trial_active, i,
                                /*n_trial_dofs=*/0u, &ws, constitutive_state, &constitutive_cache};
                env.coupled_integral_seed_slot = coupled_integral_slot_;
                env.coupled_aux_dseed = daux_dintegrals_;
                env.coupled_aux_dseed_cols = num_integrals_;

                Real sum_dq = 0.0;
                for (const auto& term : base_->residualIR().terms()) {
                    if (term.domain != IntegralDomain::InterfaceFace) continue;
                    if (term.interface_marker >= 0 && term.interface_marker != interface_marker) continue;
                    Real term_weight = 1.0;
                    if (time_ctx) {
                        if (term.time_derivative_order == 1) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt1_term_weight;
                        } else if (term.time_derivative_order == 2) {
                            term_weight = time_ctx->time_derivative_term_weight * time_ctx->dt2_term_weight;
                        } else if (term.time_derivative_order > 0) {
                            term_weight = time_ctx->time_derivative_term_weight;
                        } else {
                            term_weight = time_ctx->non_time_derivative_term_weight;
                        }
                    }
                    if (term_weight == 0.0) continue;
                    const auto val = evalDual(*term.integrand.node(), env, eval_side, q);
                    if (val.kind != EvalValue<Dual>::Kind::Scalar) {
                        throw FEException("Forms: coupled residual sensitivity interface-face integrand did not evaluate to scalar",
                                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
                    }
                    if (!val.s.deriv.empty()) {
                        sum_dq += term_weight * val.s.deriv[0];
                    }
                }

                dres_i += w * sum_dq;
            }
            out.local_vector[static_cast<std::size_t>(i)] = dres_i;
        }
    };

    // Mirror the base kernel's convention for interface-face residual assembly.
    assembleResidualBlock(Side::Minus, Side::Minus, Side::Minus,
                          ctx_minus, ctx_plus,
                          output_minus, n_test_minus);
    assembleResidualBlock(Side::Plus, Side::Plus, Side::Plus,
                          ctx_plus, ctx_minus,
                          output_plus, n_test_plus);
}

// ============================================================================
// FunctionalFormKernel (FE/Forms -> FE/Assembly functionals)
// ============================================================================

FunctionalFormKernel::FunctionalFormKernel(
    FormExpr integrand,
    Domain domain,
    assembly::RequiredData required,
    std::vector<assembly::FieldRequirement> field_requirements)
    : integrand_(std::move(integrand))
    , domain_(domain)
    , required_data_(required)
    , field_requirements_(std::move(field_requirements))
{
    FE_THROW_IF(!integrand_.isValid(), InvalidArgumentException,
                "Forms::FunctionalFormKernel: invalid integrand");
    FE_THROW_IF(containsTestOrTrial(*integrand_.node()), InvalidArgumentException,
                "Forms::FunctionalFormKernel: integrand must not contain TestFunction/TrialFunction (use DiscreteField/StateField instead)");
}

Real FunctionalFormKernel::evaluateCell(const assembly::AssemblyContext& ctx, LocalIndex q)
{
    if (domain_ != Domain::Cell) {
        return 0.0;
    }

    EvalEnvReal env{
        /*minus=*/ctx,
        /*plus=*/nullptr,
        /*kind=*/FormKind::Linear,
        /*test_active=*/Side::Minus,
        /*trial_active=*/Side::Minus,
        /*i=*/0,
        /*j=*/0,
        /*constitutive_state=*/nullptr,
        /*constitutive_cache=*/nullptr,
    };

    const auto v = evalReal(*integrand_.node(), env, Side::Minus, q);
    FE_THROW_IF(v.kind != EvalValue<Real>::Kind::Scalar, InvalidArgumentException,
                "Forms::FunctionalFormKernel: integrand did not evaluate to a scalar");
    return v.s;
}

Real FunctionalFormKernel::evaluateBoundaryFace(const assembly::AssemblyContext& ctx,
                                                LocalIndex q,
                                                int /*boundary_marker*/)
{
    if (domain_ != Domain::BoundaryFace) {
        return 0.0;
    }

    EvalEnvReal env{
        /*minus=*/ctx,
        /*plus=*/nullptr,
        /*kind=*/FormKind::Linear,
        /*test_active=*/Side::Minus,
        /*trial_active=*/Side::Minus,
        /*i=*/0,
        /*j=*/0,
        /*constitutive_state=*/nullptr,
        /*constitutive_cache=*/nullptr,
    };

    const auto v = evalReal(*integrand_.node(), env, Side::Minus, q);
    FE_THROW_IF(v.kind != EvalValue<Real>::Kind::Scalar, InvalidArgumentException,
                "Forms::FunctionalFormKernel: integrand did not evaluate to a scalar");
    return v.s;
}

namespace detail {

assembly::RequiredData analyzeRequiredData(const FormExprNode& node, FormKind kind);
std::vector<assembly::FieldRequirement> analyzeFieldRequirements(const FormExprNode& node);

} // namespace detail

// ============================================================================
// BoundaryFunctionalGradientKernel (dQ/du for coupled BCs)
// ============================================================================

BoundaryFunctionalGradientKernel::BoundaryFunctionalGradientKernel(FormExpr integrand, int boundary_marker)
    : integrand_(std::move(integrand))
    , boundary_marker_(boundary_marker)
{
    FE_THROW_IF(!integrand_.isValid(), InvalidArgumentException,
                "Forms::BoundaryFunctionalGradientKernel: invalid integrand");
    FE_THROW_IF(boundary_marker_ < 0, InvalidArgumentException,
                "Forms::BoundaryFunctionalGradientKernel: boundary_marker must be >= 0");
    FE_CHECK_NOT_NULL(integrand_.node(), "Forms::BoundaryFunctionalGradientKernel: integrand node");
    FE_THROW_IF(containsTestFunction(*integrand_.node()), InvalidArgumentException,
                "Forms::BoundaryFunctionalGradientKernel: integrand must not contain TestFunction");

    field_requirements_ = detail::analyzeFieldRequirements(*integrand_.node());
    required_data_ = detail::analyzeRequiredData(*integrand_.node(), FormKind::Residual);
    for (const auto& fr : field_requirements_) {
        required_data_ |= fr.required;
    }
    required_data_ |= assembly::RequiredData::Normals;
}

void BoundaryFunctionalGradientKernel::computeBoundaryFace(const assembly::AssemblyContext& ctx,
                                                           int boundary_marker,
                                                           assembly::KernelOutput& output)
{
    const auto n_test = ctx.numTestDofs();
    const auto n_trial = ctx.numTrialDofs();
    output.reserve(n_test, n_trial, /*need_matrix=*/false, /*need_vector=*/true);

    std::fill(output.local_vector.begin(), output.local_vector.end(), 0.0);

    if (boundary_marker != boundary_marker_) {
        return;
    }

    FE_THROW_IF(n_test != n_trial, InvalidArgumentException,
                "Forms::BoundaryFunctionalGradientKernel: expected square context (test == trial)");

    thread_local DualWorkspace ws;
    ws.reset(static_cast<std::size_t>(n_trial));

    EvalEnvDual env{
        /*minus=*/ctx,
        /*plus=*/nullptr,
        /*test_active=*/Side::Minus,
        /*trial_active=*/Side::Minus,
        /*i=*/0,
        /*n_trial_dofs=*/static_cast<std::size_t>(n_trial),
        /*ws=*/&ws,
        /*constitutive_state=*/nullptr,
        /*constitutive_cache=*/nullptr,
    };

    const auto n_qpts = ctx.numQuadraturePoints();
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto val = evalDual(*integrand_.node(), env, Side::Minus, q);
        FE_THROW_IF(val.kind != EvalValue<Dual>::Kind::Scalar, InvalidArgumentException,
                    "Forms::BoundaryFunctionalGradientKernel: integrand did not evaluate to scalar (dual)");

        const Real w = ctx.integrationWeight(q);
        for (LocalIndex j = 0; j < n_trial; ++j) {
            output.local_vector[static_cast<std::size_t>(j)] += w * val.s.deriv[static_cast<std::size_t>(j)];
        }
    }
}

} // namespace forms
} // namespace FE
} // namespace svmp
