/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/KernelIR.h"

#include "Core/FEException.h"
#include "Forms/Tensor/TensorIndex.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

constexpr std::uint64_t kFNVOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFNVPrime = 1099511628211ULL;

inline void hashMix(std::uint64_t& h, std::uint64_t v) noexcept
{
    h ^= v;
    h *= kFNVPrime;
}

[[nodiscard]] bool isCommutative(FormExprType t) noexcept
{
    switch (t) {
        case FormExprType::Add:
        case FormExprType::Multiply:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> packSpaceSignature(const FormExprNode::SpaceSignature& sig)
{
    // Pack into two 64-bit words. The exact bit layout is an internal detail,
    // but must remain deterministic for hashing/caching.
    const auto st = static_cast<std::uint64_t>(sig.space_type);
    const auto ft = static_cast<std::uint64_t>(sig.field_type);
    const auto cont = static_cast<std::uint64_t>(sig.continuity);
    const auto vd = static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig.value_dimension));
    const auto td = static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig.topological_dimension));
    const auto po = static_cast<std::uint64_t>(static_cast<std::uint32_t>(sig.polynomial_order));
    const auto et = static_cast<std::uint64_t>(sig.element_type);

    std::uint64_t imm0 = 0;
    imm0 |= (st & 0xffULL) << 0;
    imm0 |= (ft & 0xffULL) << 8;
    imm0 |= (cont & 0xffULL) << 16;
    imm0 |= (vd & 0xffffULL) << 24;
    imm0 |= (td & 0xffULL) << 40;
    imm0 |= (po & 0xffffULL) << 48;

    std::uint64_t imm1 = 0;
    imm1 |= (et & 0xffffULL) << 0;
    return {imm0, imm1};
}

struct ImmPayload {
    std::uint64_t imm0{0};
    std::uint64_t imm1{0};
    bool cacheable{true};
};

[[nodiscard]] ImmPayload extractImmediate(const FormExprNode& node)
{
    ImmPayload out{};
    switch (node.type()) {
        case FormExprType::Constant: {
            const auto v = node.constantValue().value_or(0.0);
            out.imm0 = std::bit_cast<std::uint64_t>(v);
            return out;
        }

        case FormExprType::ParameterRef:
        case FormExprType::BoundaryIntegralRef:
        case FormExprType::AuxiliaryStateRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("KernelIR: slot-ref node missing slot index");
            }
            out.imm0 = static_cast<std::uint64_t>(*slot);
            return out;
        }

        case FormExprType::MaterialStateOldRef:
        case FormExprType::MaterialStateWorkRef: {
            const auto off = node.stateOffsetBytes();
            if (!off) {
                throw std::invalid_argument("KernelIR: material-state ref node missing offset");
            }
            out.imm0 = static_cast<std::uint64_t>(*off);
            return out;
        }

        case FormExprType::PreviousSolutionRef: {
            const auto k = node.historyIndex();
            if (!k) {
                throw std::invalid_argument("KernelIR: PreviousSolutionRef missing history index");
            }
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::int64_t>(*k));
            return out;
        }

        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::TestFunction:
        case FormExprType::TrialFunction: {
            const auto* sig = node.spaceSignature();
            if (!sig) {
                throw std::invalid_argument("KernelIR: field/test/trial node missing SpaceSignature");
            }
            auto [imm0, imm1] = packSpaceSignature(*sig);
            out.imm0 = imm0;

            // For DiscreteField/StateField, also pack FieldId.
            if (node.type() == FormExprType::DiscreteField || node.type() == FormExprType::StateField) {
                const auto fid = node.fieldId();
                if (!fid) {
                    throw std::invalid_argument("KernelIR: DiscreteField/StateField missing FieldId");
                }
                imm1 |= (static_cast<std::uint64_t>(*fid) & 0xffffULL) << 16;
            }

            out.imm1 = imm1;
            return out;
        }

        case FormExprType::Identity: {
            const auto dim = node.identityDim().value_or(0);
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::int64_t>(dim));
            return out;
        }

        case FormExprType::Component: {
            const std::uint32_t i = static_cast<std::uint32_t>(node.componentIndex0().value_or(0));
            const std::uint32_t j = static_cast<std::uint32_t>(node.componentIndex1().value_or(-1));
            out.imm0 = static_cast<std::uint64_t>(i) | (static_cast<std::uint64_t>(j) << 32);
            return out;
        }

        case FormExprType::AsVector:
            // Shape inference uses child_count; no immediate payload needed.
            return out;

        case FormExprType::AsTensor: {
            const auto rows = node.tensorRows().value_or(0);
            const auto cols = node.tensorCols().value_or(0);
            if (rows <= 0 || cols <= 0) {
                throw std::invalid_argument("KernelIR: AsTensor missing (rows,cols)");
            }
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::uint32_t>(rows)) |
                       (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cols)) << 32);
            return out;
        }

        case FormExprType::SymmetricEigenvalue:
        case FormExprType::SymmetricEigenvalueDirectionalDerivative:
        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
            const auto which = node.eigenIndex();
            if (!which) {
                throw std::invalid_argument("KernelIR: symmetric-eigen node missing eigen index");
            }
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::int64_t>(*which));
            return out;
        }

        case FormExprType::TimeDerivative: {
            const auto order = node.timeDerivativeOrder().value_or(1);
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::int64_t>(order));
            return out;
        }

        case FormExprType::ConstitutiveOutput: {
            const auto idx = node.constitutiveOutputIndex().value_or(0);
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::int64_t>(idx));
            return out;
        }

        case FormExprType::Coefficient:
        case FormExprType::Constitutive:
            // These are JIT-compatible only via external calls/trampolines and are not
            // considered cacheable across runs. Keep them distinct by node address.
            out.cacheable = false;
            out.imm0 = reinterpret_cast<std::uintptr_t>(&node);
            return out;

        default:
            return out;
    }
}

struct Builder {
    KernelIRBuildOptions options{};
    KernelIRBuildResult result{};

    std::vector<KernelIROp> ops{};
    std::vector<std::uint32_t> children{};

    // hash -> candidate op indices (collision-resolved by structural equality).
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> hash_to_ops{};

    // Canonical renumbering for IndexedAccess index ids (Index::id() is not stable across runs).
    std::unordered_map<int, std::uint16_t> indexed_id_renaming{};
    std::uint16_t next_indexed_id{0};

    [[nodiscard]] std::uint16_t canonicalizeIndexedId(int id)
    {
        if (id < 0) {
            throw std::invalid_argument("KernelIR: IndexedAccess has negative index id");
        }
        if (auto it = indexed_id_renaming.find(id); it != indexed_id_renaming.end()) {
            return it->second;
        }
        const std::uint16_t cid = next_indexed_id++;
        indexed_id_renaming.emplace(id, cid);
        return cid;
    }

    [[nodiscard]] std::uint64_t opHash(FormExprType type,
                                       std::uint64_t imm0,
                                       std::uint64_t imm1,
                                       std::span<const std::uint32_t> kids) const noexcept
    {
        std::uint64_t h = kFNVOffset;
        hashMix(h, static_cast<std::uint64_t>(type));
        hashMix(h, imm0);
        hashMix(h, imm1);
        hashMix(h, static_cast<std::uint64_t>(kids.size()));
        for (const auto k : kids) {
            hashMix(h, static_cast<std::uint64_t>(k));
        }
        return h;
    }

    [[nodiscard]] bool opEquals(std::uint32_t idx,
                                FormExprType type,
                                std::uint64_t imm0,
                                std::uint64_t imm1,
                                std::span<const std::uint32_t> kids) const noexcept
    {
        if (idx >= ops.size()) return false;
        const auto& op = ops[idx];
        if (op.type != type || op.imm0 != imm0 || op.imm1 != imm1) return false;
        if (op.child_count != kids.size()) return false;
        for (std::size_t i = 0; i < kids.size(); ++i) {
            const auto k = children[static_cast<std::size_t>(op.first_child) + i];
            if (k != kids[i]) return false;
        }
        return true;
    }

    [[nodiscard]] std::uint32_t lower(const FormExprNode& node)
    {
        std::vector<std::uint32_t> kid_indices;
        const auto kids = node.childrenShared();
        kid_indices.reserve(kids.size());
        for (const auto& child : kids) {
            if (!child) {
                throw std::invalid_argument("KernelIR: null child pointer in FormExprNode");
            }
            kid_indices.push_back(lower(*child));
        }

        if (options.canonicalize_commutative && kid_indices.size() == 2u && isCommutative(node.type())) {
            if (kid_indices[0] > kid_indices[1]) {
                std::swap(kid_indices[0], kid_indices[1]);
            }
        }

        ImmPayload imm{};
        if (node.type() == FormExprType::IndexedAccess) {
            const int rank = node.indexRank().value_or(0);
            const auto ids_opt = node.indexIds();
            const auto ext_opt = node.indexExtents();
            const auto var_opt = node.indexVariances();
            if (rank <= 0 || !ids_opt || !ext_opt) {
                throw std::invalid_argument("KernelIR: IndexedAccess missing index metadata");
            }
            if (rank > 4) {
                throw std::invalid_argument("KernelIR: IndexedAccess rank > 4 is not supported");
            }

            const auto ids = *ids_opt;
            const auto ext = *ext_opt;
            std::array<tensor::IndexVariance, 4> vars{};
            vars.fill(tensor::IndexVariance::None);
            if (var_opt) {
                vars = *var_opt;
            }

            std::uint64_t packed_ids = 0;
            std::uint64_t packed_ext = 0;
            for (int k = 0; k < rank; ++k) {
                const int id = ids[static_cast<std::size_t>(k)];
                const int e = ext[static_cast<std::size_t>(k)];
                if (e <= 0 || e > 255) {
                    throw std::invalid_argument("KernelIR: IndexedAccess index extent out of supported range (1..255)");
                }
                const auto cid = canonicalizeIndexedId(id);
                packed_ids |= (static_cast<std::uint64_t>(cid) & 0xffffULL) << (16u * static_cast<std::uint32_t>(k));
                packed_ext |= (static_cast<std::uint64_t>(static_cast<std::uint8_t>(e)) & 0xffULL)
                              << (8u * static_cast<std::uint32_t>(k));

                // Pack index variances (2 bits per index) in imm1[40..47].
                std::uint64_t vbits = 0;
                switch (vars[static_cast<std::size_t>(k)]) {
                    case tensor::IndexVariance::None:
                        vbits = 0;
                        break;
                    case tensor::IndexVariance::Lower:
                        vbits = 1;
                        break;
                    case tensor::IndexVariance::Upper:
                        vbits = 2;
                        break;
                }
                packed_ext |= (vbits & 0x3ULL) << (40u + 2u * static_cast<std::uint32_t>(k));
            }
            packed_ext |= (static_cast<std::uint64_t>(static_cast<std::uint8_t>(rank)) & 0xffULL) << 32u;

            imm.imm0 = packed_ids;
            imm.imm1 = packed_ext;
            imm.cacheable = true;
        } else {
            imm = extractImmediate(node);
        }
        result.cacheable = result.cacheable && imm.cacheable;

        const std::uint64_t h = opHash(node.type(), imm.imm0, imm.imm1, kid_indices);
        if (options.cse) {
            auto it = hash_to_ops.find(h);
            if (it != hash_to_ops.end()) {
                for (const auto cand : it->second) {
                    if (opEquals(cand, node.type(), imm.imm0, imm.imm1, kid_indices)) {
                        return cand;
                    }
                }
            }
        }

        const auto first = static_cast<std::uint32_t>(children.size());
        children.insert(children.end(), kid_indices.begin(), kid_indices.end());

        const auto idx = static_cast<std::uint32_t>(ops.size());
        ops.push_back(KernelIROp{
            .type = node.type(),
            .first_child = first,
            .child_count = static_cast<std::uint32_t>(kid_indices.size()),
            .imm0 = imm.imm0,
            .imm1 = imm.imm1,
        });

        hash_to_ops[h].push_back(idx);
        return idx;
    }
};

} // namespace

std::uint64_t KernelIR::stableHash64() const
{
    if (ops.empty()) {
        return 0;
    }
    if (root >= ops.size()) {
        throw std::out_of_range("KernelIR::stableHash64: root index out of range");
    }

    // Structural hash per op (children < parent due to post-order lowering).
    std::vector<std::uint64_t> op_hash;
    op_hash.resize(ops.size(), 0);

    for (std::size_t i = 0; i < ops.size(); ++i) {
        const auto& op = ops[i];

        std::uint64_t h = kFNVOffset;
        hashMix(h, 0x4b49525f5631ULL); // "KIR_V1" tag
        hashMix(h, static_cast<std::uint64_t>(op.type));
        hashMix(h, op.imm0);
        hashMix(h, op.imm1);
        hashMix(h, static_cast<std::uint64_t>(op.child_count));

        if (op.child_count == 2u && isCommutative(op.type)) {
            const auto a_idx = children[static_cast<std::size_t>(op.first_child) + 0];
            const auto b_idx = children[static_cast<std::size_t>(op.first_child) + 1];
            if (a_idx >= i || b_idx >= i) {
                throw std::logic_error("KernelIR::stableHash64: non-topological child index");
            }
            const auto ha = op_hash[a_idx];
            const auto hb = op_hash[b_idx];
            const auto lo = std::min(ha, hb);
            const auto hi = std::max(ha, hb);
            hashMix(h, lo);
            hashMix(h, hi);
        } else {
            for (std::size_t c = 0; c < op.child_count; ++c) {
                const auto child_idx = children[static_cast<std::size_t>(op.first_child) + c];
                if (child_idx >= i) {
                    throw std::logic_error("KernelIR::stableHash64: non-topological child index");
                }
                hashMix(h, op_hash[child_idx]);
            }
        }

        op_hash[i] = h;
    }

    return op_hash[root];
}

KernelIRBuildResult lowerToKernelIR(const FormExprNode& root,
                                    const KernelIRBuildOptions& options)
{
    Builder b;
    b.options = options;
    b.result.cacheable = true;

    const auto root_idx = b.lower(root);
    b.result.ir.ops = std::move(b.ops);
    b.result.ir.children = std::move(b.children);
    b.result.ir.root = root_idx;
    return b.result;
}

KernelIRBuildResult lowerToKernelIR(const FormExpr& integrand,
                                    const KernelIRBuildOptions& options)
{
    if (!integrand.isValid()) {
        throw std::invalid_argument("lowerToKernelIR: invalid expression");
    }
    const auto* root = integrand.node();
    FE_CHECK_NOT_NULL(root, "lowerToKernelIR: integrand node");
    return lowerToKernelIR(*root, options);
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
