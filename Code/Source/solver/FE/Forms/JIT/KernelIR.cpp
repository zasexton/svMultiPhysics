/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/KernelIR.h"

#include "Core/FEException.h"
#include "Forms/IndexExtent.h"
#include "Forms/Tensor/TensorIndex.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <sstream>
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

[[nodiscard]] const char* opName(FormExprType t) noexcept
{
    switch (t) {
        case FormExprType::TestFunction: return "TestFunction";
        case FormExprType::TrialFunction: return "TrialFunction";
        case FormExprType::DiscreteField: return "DiscreteField";
        case FormExprType::StateField: return "StateField";
        case FormExprType::Coefficient: return "Coefficient";
        case FormExprType::ParameterSymbol: return "ParameterSymbol";
        case FormExprType::ParameterRef: return "ParameterRef";
        case FormExprType::Constant: return "Constant";
        case FormExprType::TypedZero: return "TypedZero";
        case FormExprType::BoundaryFunctionalSymbol: return "BoundaryFunctionalSymbol";
        case FormExprType::BoundaryIntegralSymbol: return "BoundaryIntegralSymbol";
        case FormExprType::BoundaryIntegralRef: return "BoundaryIntegralRef";
        case FormExprType::AuxiliaryStateSymbol: return "AuxiliaryStateSymbol";
        case FormExprType::AuxiliaryStateRef: return "AuxiliaryStateRef";
        case FormExprType::MaterialStateOldRef: return "MaterialStateOldRef";
        case FormExprType::MaterialStateWorkRef: return "MaterialStateWorkRef";
        case FormExprType::PreviousSolutionRef: return "PreviousSolutionRef";
        case FormExprType::Coordinate: return "Coordinate";
        case FormExprType::ReferenceCoordinate: return "ReferenceCoordinate";
        case FormExprType::MeshDisplacement: return "MeshDisplacement";
        case FormExprType::MeshVelocity: return "MeshVelocity";
        case FormExprType::MeshAcceleration: return "MeshAcceleration";
        case FormExprType::CurrentCoordinate: return "CurrentCoordinate";
        case FormExprType::PreviousCoordinate: return "PreviousCoordinate";
        case FormExprType::ReferencePhysicalCoordinate: return "ReferencePhysicalCoordinate";
        case FormExprType::PreviousMeshVelocity: return "PreviousMeshVelocity";
        case FormExprType::PredictedMeshVelocity: return "PredictedMeshVelocity";
        case FormExprType::CurrentJacobian: return "CurrentJacobian";
        case FormExprType::ReferenceJacobian: return "ReferenceJacobian";
        case FormExprType::CurrentJacobianDeterminant: return "CurrentJacobianDeterminant";
        case FormExprType::ReferenceJacobianDeterminant: return "ReferenceJacobianDeterminant";
        case FormExprType::CurrentNormal: return "CurrentNormal";
        case FormExprType::ReferenceNormal: return "ReferenceNormal";
        case FormExprType::CurrentMeasure: return "CurrentMeasure";
        case FormExprType::ReferenceMeasure: return "ReferenceMeasure";
        case FormExprType::SurfaceJacobian: return "SurfaceJacobian";
        case FormExprType::GeometryTrialVectorVariation: return "GeometryTrialVectorVariation";
        case FormExprType::GeometryTrialJacobianVariation: return "GeometryTrialJacobianVariation";
        case FormExprType::MeshVelocityVariation: return "MeshVelocityVariation";
        case FormExprType::CurrentMeasureVariation: return "CurrentMeasureVariation";
        case FormExprType::CurrentNormalVariation: return "CurrentNormalVariation";
        case FormExprType::SurfaceJacobianVariation: return "SurfaceJacobianVariation";
        case FormExprType::Time: return "Time";
        case FormExprType::TimeStep: return "TimeStep";
        case FormExprType::EffectiveTimeStep: return "EffectiveTimeStep";
        case FormExprType::Identity: return "Identity";
        case FormExprType::Jacobian: return "Jacobian";
        case FormExprType::JacobianInverse: return "JacobianInverse";
        case FormExprType::JacobianDeterminant: return "JacobianDeterminant";
        case FormExprType::Normal: return "Normal";
        case FormExprType::CellDiameter: return "CellDiameter";
        case FormExprType::CellVolume: return "CellVolume";
        case FormExprType::FacetArea: return "FacetArea";
        case FormExprType::CellDomainId: return "CellDomainId";

        case FormExprType::Gradient: return "Gradient";
        case FormExprType::Divergence: return "Divergence";
        case FormExprType::Curl: return "Curl";
        case FormExprType::Hessian: return "Hessian";
        case FormExprType::TimeDerivative: return "TimeDerivative";

        case FormExprType::RestrictMinus: return "RestrictMinus";
        case FormExprType::RestrictPlus: return "RestrictPlus";
        case FormExprType::Jump: return "Jump";
        case FormExprType::Average: return "Average";

        case FormExprType::Negate: return "Negate";
        case FormExprType::Transpose: return "Transpose";
        case FormExprType::Trace: return "Trace";
        case FormExprType::Determinant: return "Determinant";
        case FormExprType::Inverse: return "Inverse";
        case FormExprType::Cofactor: return "Cofactor";
        case FormExprType::Deviator: return "Deviator";
        case FormExprType::SymmetricPart: return "SymmetricPart";
        case FormExprType::SkewPart: return "SkewPart";
        case FormExprType::Norm: return "Norm";
        case FormExprType::Normalize: return "Normalize";
        case FormExprType::AbsoluteValue: return "AbsoluteValue";
        case FormExprType::Sign: return "Sign";
        case FormExprType::Sqrt: return "Sqrt";
        case FormExprType::Exp: return "Exp";
        case FormExprType::Log: return "Log";

        case FormExprType::SymmetricEigenvalue: return "SymmetricEigenvalue";
        case FormExprType::SymmetricEigenvalueDirectionalDerivative: return "SymmetricEigenvalueDirectionalDerivative";
        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: return "SymmetricEigenvalueDirectionalDerivativeWrtA";
        case FormExprType::MatrixExponential: return "MatrixExponential";
        case FormExprType::MatrixLogarithm: return "MatrixLogarithm";
        case FormExprType::MatrixSqrt: return "MatrixSqrt";
        case FormExprType::MatrixPower: return "MatrixPower";
        case FormExprType::MatrixExponentialDirectionalDerivative: return "MatrixExponentialDirectionalDerivative";
        case FormExprType::MatrixLogarithmDirectionalDerivative: return "MatrixLogarithmDirectionalDerivative";
        case FormExprType::MatrixSqrtDirectionalDerivative: return "MatrixSqrtDirectionalDerivative";
        case FormExprType::MatrixPowerDirectionalDerivative: return "MatrixPowerDirectionalDerivative";
        case FormExprType::SmoothHeaviside: return "SmoothHeaviside";
        case FormExprType::SmoothAbsoluteValue: return "SmoothAbsoluteValue";
        case FormExprType::SmoothMin: return "SmoothMin";
        case FormExprType::SmoothMax: return "SmoothMax";
        case FormExprType::SmoothSign: return "SmoothSign";
        case FormExprType::Eigenvalue: return "Eigenvalue";
        case FormExprType::SymmetricEigenvector: return "SymmetricEigenvector";
        case FormExprType::SpectralDecomposition: return "SpectralDecomposition";
        case FormExprType::SymmetricEigenvectorDirectionalDerivative: return "SymmetricEigenvectorDirectionalDerivative";
        case FormExprType::SpectralDecompositionDirectionalDerivative: return "SpectralDecompositionDirectionalDerivative";
        case FormExprType::HistoryWeightedSum: return "HistoryWeightedSum";
        case FormExprType::HistoryConvolution: return "HistoryConvolution";

        case FormExprType::Add: return "Add";
        case FormExprType::Subtract: return "Subtract";
        case FormExprType::Multiply: return "Multiply";
        case FormExprType::Divide: return "Divide";
        case FormExprType::InnerProduct: return "InnerProduct";
        case FormExprType::DoubleContraction: return "DoubleContraction";
        case FormExprType::OuterProduct: return "OuterProduct";
        case FormExprType::CrossProduct: return "CrossProduct";
        case FormExprType::Pullback: return "Pullback";
        case FormExprType::Pushforward: return "Pushforward";
        case FormExprType::Power: return "Power";
        case FormExprType::Minimum: return "Minimum";
        case FormExprType::Maximum: return "Maximum";

        case FormExprType::Less: return "Less";
        case FormExprType::LessEqual: return "LessEqual";
        case FormExprType::Greater: return "Greater";
        case FormExprType::GreaterEqual: return "GreaterEqual";
        case FormExprType::Equal: return "Equal";
        case FormExprType::NotEqual: return "NotEqual";
        case FormExprType::Conditional: return "Conditional";

        case FormExprType::Component: return "Component";
        case FormExprType::AsVector: return "AsVector";
        case FormExprType::AsTensor: return "AsTensor";
        case FormExprType::IndexedAccess: return "IndexedAccess";

        case FormExprType::ConstitutiveOutput: return "ConstitutiveOutput";
        case FormExprType::Constitutive: return "Constitutive";

        default:
            return "Unknown";
    }
}

[[nodiscard]] bool isCommutative(FormExprType t) noexcept
{
    // NOTE: FormExprType::Multiply is intentionally not treated as commutative.
    // In the FE form language it can represent non-commutative tensor products
    // (e.g. matrix-vector multiplication), so operand reordering can change the
    // numerical result and break residual/Jacobian consistency.
    switch (t) {
        case FormExprType::Add:
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
        case FormExprType::TypedZero: {
            // TypedZero lowers to Constant(0.0) in KernelIR
            const double zero = 0.0;
            out.imm0 = std::bit_cast<std::uint64_t>(zero);
            return out;
        }

        case FormExprType::Constant: {
            const auto v = node.constantValue().value_or(0.0);
            out.imm0 = std::bit_cast<std::uint64_t>(v);
            return out;
        }

        case FormExprType::ParameterRef:
        case FormExprType::BoundaryIntegralRef:
        case FormExprType::AuxiliaryStateRef:
        case FormExprType::AuxiliaryInputRef:
        case FormExprType::AuxiliaryOutputRef: {
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
        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA:
        case FormExprType::Eigenvalue:
        case FormExprType::SymmetricEigenvector:
        case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
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

        case FormExprType::Pullback:
        case FormExprType::Pushforward: {
            const auto from = node.fromConfiguration().value_or(GeometryConfiguration::Reference);
            const auto to = node.toConfiguration().value_or(GeometryConfiguration::Current);
            out.imm0 = static_cast<std::uint64_t>(static_cast<std::uint8_t>(from)) |
                       (static_cast<std::uint64_t>(static_cast<std::uint8_t>(to)) << 8u);
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
    int auto_index_extent{3};

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
                const int raw = ext[static_cast<std::size_t>(k)];
                const int e = (raw == 0) ? auto_index_extent : raw;
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

        const auto ir_type = node.type();

        const std::uint64_t h = opHash(ir_type, imm.imm0, imm.imm1, kid_indices);
        if (options.cse) {
            auto it = hash_to_ops.find(h);
            if (it != hash_to_ops.end()) {
                for (const auto cand : it->second) {
                    if (opEquals(cand, ir_type, imm.imm0, imm.imm1, kid_indices)) {
                        return cand;
                    }
                }
            }
        }

        const auto first = static_cast<std::uint32_t>(children.size());
        children.insert(children.end(), kid_indices.begin(), kid_indices.end());

        const auto idx = static_cast<std::uint32_t>(ops.size());
        ops.push_back(KernelIROp{
            .type = ir_type,
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

std::vector<std::uint64_t> KernelIR::perOpStructuralHashes() const
{
    std::vector<std::uint64_t> op_hash(ops.size(), 0);
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const auto& op = ops[i];
        std::uint64_t h = kFNVOffset;
        hashMix(h, 0x4b49525f5631ULL);
        hashMix(h, static_cast<std::uint64_t>(op.type));
        hashMix(h, op.imm0);
        hashMix(h, op.imm1);
        hashMix(h, static_cast<std::uint64_t>(op.child_count));

        if (op.child_count == 2u && isCommutative(op.type)) {
            const auto a_idx = children[static_cast<std::size_t>(op.first_child) + 0];
            const auto b_idx = children[static_cast<std::size_t>(op.first_child) + 1];
            const auto ha = op_hash[a_idx];
            const auto hb = op_hash[b_idx];
            hashMix(h, std::min(ha, hb));
            hashMix(h, std::max(ha, hb));
        } else {
            for (std::size_t c = 0; c < op.child_count; ++c) {
                const auto child_idx = children[static_cast<std::size_t>(op.first_child) + c];
                hashMix(h, op_hash[child_idx]);
            }
        }
        op_hash[i] = h;
    }
    return op_hash;
}

std::string KernelIR::dump() const
{
    std::ostringstream oss;
    oss << "KernelIR(root=" << root
        << ", ops=" << ops.size()
        << ", children=" << children.size()
        << ", hash=0x" << std::hex << stableHash64() << std::dec << ")\n";

    for (std::size_t i = 0; i < ops.size(); ++i) {
        const auto& op = ops[i];
        oss << "  [" << i << "] " << opName(op.type)
            << " first_child=" << op.first_child
            << " child_count=" << op.child_count
            << " imm0=0x" << std::hex << op.imm0
            << " imm1=0x" << op.imm1 << std::dec;

        if (op.child_count != 0u) {
            oss << " kids=[";
            for (std::size_t k = 0; k < op.child_count; ++k) {
                if (k != 0u) {
                    oss << ",";
                }
                const auto idx = children[static_cast<std::size_t>(op.first_child) + k];
                oss << idx;
            }
            oss << "]";
        }
        oss << "\n";
    }
    return oss.str();
}

namespace {

[[nodiscard]] bool isZeroOp(const KernelIROp& op) noexcept
{
    if (op.type == FormExprType::TypedZero) return true;
    if (op.type != FormExprType::Constant) return false;
    const double v = std::bit_cast<double>(op.imm0);
    return v == 0.0;
}

void makeTypedZero(KernelIROp& op) noexcept
{
    op.type = FormExprType::TypedZero;
    op.first_child = 0;
    op.child_count = 0;
    op.imm0 = 0;
    op.imm1 = 0;
}

[[nodiscard]] bool isConstantOne(const KernelIROp& op) noexcept
{
    if (op.type != FormExprType::Constant) return false;
    const double v = std::bit_cast<double>(op.imm0);
    return v == 1.0;
}

[[nodiscard]] bool isConstant(const KernelIROp& op) noexcept
{
    return op.type == FormExprType::Constant;
}

[[nodiscard]] double constantVal(const KernelIROp& op) noexcept
{
    return std::bit_cast<double>(op.imm0);
}

void makeConstant(KernelIROp& op, double value) noexcept
{
    op.type = FormExprType::Constant;
    op.first_child = 0;
    op.child_count = 0;
    op.imm0 = std::bit_cast<std::uint64_t>(value);
    op.imm1 = 0;
}

} // namespace

std::size_t KernelIR::optimize()
{
    if (ops.empty()) return 0;

    const std::size_t original_count = ops.size();

    // Pass 1: Zero propagation and constant folding (bottom-up, single pass).
    // Since ops are in post-order (children before parents), a single forward
    // scan propagates zeros/constants through the entire DAG.
    {
    const std::size_t n = ops.size();
    for (std::size_t i = 0; i < n; ++i) {
        auto& op = ops[i];
        if (op.child_count == 0u) continue;

        // Get children indices
        auto kidIdx = [&](std::size_t k) -> std::uint32_t {
            return children[static_cast<std::size_t>(op.first_child) + k];
        };

        // Resolve through forwarding (an op that was folded to a constant
        // may have children pointing to the original op index; after folding,
        // the op AT that index is now a constant).
        auto& kid0 = ops[kidIdx(0)];

        if (op.child_count == 1u) {
            // Unary ops with zero child: only collapse to TypedZero for
            // shape-PRESERVING ops (output shape == input shape).
            // Shape-CHANGING ops (Gradient, Curl, Hessian, Transpose, etc.)
            // must keep their operation node so that LLVMGen::inferShapes()
            // can derive the correct output shape from the op type + spatial
            // dimension.  Collapsing them to shapeless TypedZero loses shape
            // information and causes downstream shape errors (e.g. outer(0,v)
            // reconstructed as square instead of rectangular).
            if (isZeroOp(kid0)) {
                switch (op.type) {
                    // Shape-preserving AND rank-preserving: safe to collapse
                    // to TypedZero only for ops where TypedZero (inferred as
                    // scalar in inferShapes) matches the actual output rank.
                    // Negate and TimeDerivative are truly shape-preserving
                    // scalar→scalar ops.
                    case FormExprType::Negate:
                    case FormExprType::TimeDerivative:
                        makeTypedZero(op);
                        break;

                    // Always-scalar results from zero input
                    case FormExprType::Component:
                    case FormExprType::Norm:
                        makeConstant(op, 0.0);
                        break;

                    // DG restriction/averaging ops: these ARE shape-preserving,
                    // but TypedZero in inferShapes maps to scalar.  If the
                    // input was a vector/matrix that collapsed to TypedZero
                    // upstream, collapsing these further loses the rank
                    // context needed by downstream ops (e.g. inner, outer).
                    // Keep the node so inferShapes can propagate shape from
                    // the child's op type.
                    case FormExprType::RestrictMinus:
                    case FormExprType::RestrictPlus:
                    case FormExprType::Jump:
                    case FormExprType::Average:
                        break;

                    // Always-scalar results from zero input (contraction ops):
                    // Divergence(0) = 0 (sum of derivatives of zero),
                    // Trace(0) = 0 (sum of diagonal of zero matrix),
                    // Determinant(0) = 0 (det of zero matrix).
                    case FormExprType::Divergence:
                    case FormExprType::Trace:
                    case FormExprType::Determinant:
                        makeConstant(op, 0.0);
                        break;

                    // Shape-changing ops: keep the node with zero child so
                    // inferShapes() can compute the correct output shape.
                    // Gradient(0)→zero-vector, Hessian(0)→zero-matrix, etc.
                    case FormExprType::Gradient:
                    case FormExprType::Curl:
                    case FormExprType::Hessian:
                    case FormExprType::Transpose:
                    case FormExprType::SymmetricPart:
                    case FormExprType::SkewPart:
                    case FormExprType::Deviator:
                    case FormExprType::Normalize:
                        // Leave op unchanged — child is zero but shape is
                        // determined by the op type in inferShapes().
                        break;

                    default:
                        break;
                }
            }

            // Unary constant folding (scalar→scalar ops)
            if (isConstant(kid0)) {
                const double v = constantVal(kid0);
                switch (op.type) {
                    case FormExprType::Negate:
                        makeConstant(op, -v);
                        break;
                    case FormExprType::Exp:
                        makeConstant(op, std::exp(v));
                        break;
                    case FormExprType::Log:
                        if (v > 0.0) makeConstant(op, std::log(v));
                        break;
                    case FormExprType::Sqrt:
                        if (v >= 0.0) makeConstant(op, std::sqrt(v));
                        break;
                    case FormExprType::AbsoluteValue:
                        makeConstant(op, std::fabs(v));
                        break;
                    default:
                        break;
                }
            }

            // Double negate: --X → X (redirect to grandchild, shape-preserving)
            if (op.type == FormExprType::Negate && kid0.type == FormExprType::Negate && kid0.child_count == 1u) {
                const auto grandchild = children[static_cast<std::size_t>(kid0.first_child)];
                op = ops[grandchild]; // copy the grandchild op
            }

            continue;
        }

        if (op.child_count == 2u) {
            auto& kid1 = ops[kidIdx(1)];
            const bool k0_zero = isZeroOp(kid0);
            const bool k1_zero = isZeroOp(kid1);
            const bool k0_one = isConstantOne(kid0);
            const bool k1_one = isConstantOne(kid1);
            const bool k0_const = isConstant(kid0);
            const bool k1_const = isConstant(kid1);

            switch (op.type) {
                case FormExprType::Multiply:
                    // 0*X or X*0: keep the Multiply node — collapsing to
                    // TypedZero loses the output shape (scalar*vector=vector,
                    // scalar*matrix=matrix).  inferShapes handles TypedZero
                    // (scalar) children correctly for Multiply.  LLVM's own
                    // optimizer will fold fmul(0.0, x) → 0.0 at IR level.
                    // 1*X → X or X*1 → X (shape-preserving alias)
                    if (k0_one) { op = kid1; break; }
                    if (k1_one) { op = kid0; break; }
                    // const * const → const
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) * constantVal(kid1)); break; }
                    break;

                case FormExprType::Add:
                    // 0+X → X or X+0 → X (shape-preserving alias)
                    if (k0_zero) { op = kid1; break; }
                    if (k1_zero) { op = kid0; break; }
                    // const + const → const
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) + constantVal(kid1)); break; }
                    break;

                case FormExprType::Subtract:
                    // X-0 → X (shape-preserving alias)
                    if (k1_zero) { op = kid0; break; }
                    // 0-X → keep as subtract (LLVM handles it)
                    // const - const → const
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) - constantVal(kid1)); break; }
                    break;

                case FormExprType::Divide:
                    // 0/X: keep — numerator shape determines output shape.
                    // inferShapes requires scalar denominator, so output =
                    // numerator shape.  LLVM folds fdiv(0.0, x) → 0.0.
                    // X/1 → X (shape-preserving alias)
                    if (k1_one) { op = kid0; break; }
                    // const / const → const
                    if (k0_const && k1_const && constantVal(kid1) != 0.0) {
                        makeConstant(op, constantVal(kid0) / constantVal(kid1));
                        break;
                    }
                    break;

                case FormExprType::Power:
                    // X^0 → 1 (Power is always scalar^scalar)
                    if (k1_zero) { makeConstant(op, 1.0); break; }
                    // X^1 → X (shape-preserving alias)
                    if (k1_one) { op = kid0; break; }
                    // const ^ const → const
                    if (k0_const && k1_const) { makeConstant(op, std::pow(constantVal(kid0), constantVal(kid1))); break; }
                    break;

                case FormExprType::Minimum:
                    if (k0_const && k1_const) { makeConstant(op, std::fmin(constantVal(kid0), constantVal(kid1))); break; }
                    break;

                case FormExprType::Maximum:
                    if (k0_const && k1_const) { makeConstant(op, std::fmax(constantVal(kid0), constantVal(kid1))); break; }
                    break;

                // Comparisons: const op const → 1.0 or 0.0
                case FormExprType::Less:
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) < constantVal(kid1) ? 1.0 : 0.0); break; }
                    break;
                case FormExprType::LessEqual:
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) <= constantVal(kid1) ? 1.0 : 0.0); break; }
                    break;
                case FormExprType::Greater:
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) > constantVal(kid1) ? 1.0 : 0.0); break; }
                    break;
                case FormExprType::GreaterEqual:
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) >= constantVal(kid1) ? 1.0 : 0.0); break; }
                    break;
                case FormExprType::Equal:
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) == constantVal(kid1) ? 1.0 : 0.0); break; }
                    break;
                case FormExprType::NotEqual:
                    if (k0_const && k1_const) { makeConstant(op, constantVal(kid0) != constantVal(kid1) ? 1.0 : 0.0); break; }
                    break;

                case FormExprType::InnerProduct:
                case FormExprType::DoubleContraction:
                    // inner(0,X) or inner(X,0) → 0 (always scalar)
                    if (k0_zero || k1_zero) { makeConstant(op, 0.0); break; }
                    break;

                case FormExprType::OuterProduct:
                case FormExprType::CrossProduct:
                    // Keep the node — collapsing outer(0,v) to TypedZero
                    // loses the rectangular shape (m×n → shapeless).
                    // inferShapes handles TypedZero children and uses the
                    // spatial dimension for the unknown operand.
                    break;

                default:
                    break;
            }
        }

        // Conditional(const_cond, a, b): if condition is a constant,
        // select the appropriate branch.  Semantics: cond > 0.0 → a, else → b.
        if (op.child_count == 3u && op.type == FormExprType::Conditional) {
            if (isConstant(kid0)) {
                const double cond_val = constantVal(kid0);
                // cond > 0.0 → take "then" branch (child 1), else → "else" branch (child 2)
                const auto selected = kidIdx(cond_val > 0.0 ? 1u : 2u);
                op = ops[selected];
            }
        }

        // Multi-child: AsVector/AsTensor with all-zero children.
        // Do NOT collapse to TypedZero — AsVector(0,0,0) is a 3-vector zero,
        // not a scalar zero.  TypedZero in inferShapes maps to scalarShape(),
        // so collapsing loses the vector/matrix rank and causes shape errors
        // in downstream ops (e.g. inner(AsVector(0,0,0), v) expects vector).
        // Keep the AsVector/AsTensor node; LLVM will fold fmul(0,x)→0 at IR level.
    }
    } // end Pass 1 scope

    // Pass 1.3: Algebraic strength reduction.
    // Specializes expensive operations to cheaper equivalents.
    //
    // Safety policy:
    //   - Rewrites that are algebraic identities for all finite values
    //     (Transpose², sym idempotence, Negate(Sub)) are always safe.
    //   - Power specializations (pow→mul/sqrt/div) are safe because
    //     llvm.pow already handles NaN/inf/zero the same way as the
    //     replacement ops for these specific exponents.
    //   - Rewrites that change behavior for NaN, inf, or zero inputs
    //     (Sub(X,X)→0, Div(X,X)→1, Inv(Inv(X))→X, Log(Exp)→X,
    //     Exp(Log)→X, Sqrt(X*X)→|X|) are NOT applied because the JIT
    //     FP contract does not include no-nans or no-infs.
    //
    // The pass runs in a fixed-point loop (max 2 sweeps) so that
    // compound rewrites are fully collapsed: e.g. x/-1 → x*(-1) →
    // Negate(x).
    {
        bool needs_sort = false;

        auto kidIdx_sr = [&](const KernelIROp& op, std::size_t k) -> std::uint32_t {
            return children[static_cast<std::size_t>(op.first_child) + k];
        };

        auto setKid_sr = [&](KernelIROp& op, std::size_t k, std::uint32_t val) {
            children[static_cast<std::size_t>(op.first_child) + k] = val;
        };

        // Append helpers for compound rewrites (may break post-order).
        auto appendConstOp = [&](double v) -> std::uint32_t {
            const auto idx = static_cast<std::uint32_t>(ops.size());
            KernelIROp nop{};
            nop.type = FormExprType::Constant;
            nop.imm0 = std::bit_cast<std::uint64_t>(v);
            ops.push_back(nop);
            needs_sort = true;
            return idx;
        };

        auto appendUnaryOp = [&](FormExprType type, std::uint32_t child) -> std::uint32_t {
            const auto idx = static_cast<std::uint32_t>(ops.size());
            KernelIROp nop{};
            nop.type = type;
            nop.first_child = static_cast<std::uint32_t>(children.size());
            nop.child_count = 1;
            ops.push_back(nop);
            children.push_back(child);
            needs_sort = true;
            return idx;
        };

        auto appendBinaryOp = [&](FormExprType type, std::uint32_t c0, std::uint32_t c1) -> std::uint32_t {
            const auto idx = static_cast<std::uint32_t>(ops.size());
            KernelIROp nop{};
            nop.type = type;
            nop.first_child = static_cast<std::uint32_t>(children.size());
            nop.child_count = 2;
            ops.push_back(nop);
            children.push_back(c0);
            children.push_back(c1);
            needs_sort = true;
            return idx;
        };

        constexpr int kMaxSweeps = 2;
        for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
            bool changed = false;
            const std::size_t n_before_sr = ops.size();

            // Pre-reserve capacity so push_back in append helpers never
            // reallocates, keeping op/kid references valid.
            // Worst case: each binary op spawns ~4 new ops + ~4 children.
            ops.reserve(n_before_sr + n_before_sr * 4);
            children.reserve(children.size() + n_before_sr * 8);

            for (std::size_t i = 0; i < n_before_sr; ++i) {
                auto& op = ops[i];

                // ============================================================
                // Unary reductions (child_count == 1)
                // ============================================================
                if (op.child_count == 1u) {
                    const auto child_idx = kidIdx_sr(op, 0);
                    const auto& kid = ops[child_idx];

                    // Transpose(Transpose(X)) → X  [algebraic identity]
                    if (op.type == FormExprType::Transpose &&
                        kid.type == FormExprType::Transpose && kid.child_count == 1u)
                    {
                        op = ops[children[static_cast<std::size_t>(kid.first_child)]];
                        changed = true; continue;
                    }

                    // SymmetricPart(SymmetricPart(X)) → SymmetricPart(X) [idempotent]
                    if (op.type == FormExprType::SymmetricPart &&
                        kid.type == FormExprType::SymmetricPart)
                    {
                        op = kid;
                        changed = true; continue;
                    }

                    // SymmetricPart(Transpose(X)) → SymmetricPart(X) [sym(A^T)=sym(A)]
                    if (op.type == FormExprType::SymmetricPart &&
                        kid.type == FormExprType::Transpose && kid.child_count == 1u)
                    {
                        setKid_sr(op, 0, children[static_cast<std::size_t>(kid.first_child)]);
                        changed = true; continue;
                    }

                    // Trace(Transpose(X)) → Trace(X)  [tr(A^T)=tr(A)]
                    if (op.type == FormExprType::Trace &&
                        kid.type == FormExprType::Transpose && kid.child_count == 1u)
                    {
                        setKid_sr(op, 0, children[static_cast<std::size_t>(kid.first_child)]);
                        changed = true; continue;
                    }

                    // Trace(SymmetricPart(X)) → Trace(X)  [tr(sym(A))=tr(A)]
                    if (op.type == FormExprType::Trace &&
                        kid.type == FormExprType::SymmetricPart && kid.child_count == 1u)
                    {
                        setKid_sr(op, 0, children[static_cast<std::size_t>(kid.first_child)]);
                        changed = true; continue;
                    }

                    // Determinant(Transpose(X)) → Determinant(X)  [det(A^T)=det(A)]
                    if (op.type == FormExprType::Determinant &&
                        kid.type == FormExprType::Transpose && kid.child_count == 1u)
                    {
                        setKid_sr(op, 0, children[static_cast<std::size_t>(kid.first_child)]);
                        changed = true; continue;
                    }

                    // Negate(Subtract(A, B)) → Subtract(B, A)  [-(a-b)=b-a]
                    if (op.type == FormExprType::Negate &&
                        kid.type == FormExprType::Subtract && kid.child_count == 2u)
                    {
                        const auto a = children[static_cast<std::size_t>(kid.first_child)];
                        const auto b = children[static_cast<std::size_t>(kid.first_child) + 1];
                        op.type = FormExprType::Subtract;
                        op.child_count = 2;
                        op.first_child = static_cast<std::uint32_t>(children.size());
                        children.push_back(b);
                        children.push_back(a);
                        changed = true; continue;
                    }

                    continue;
                }

                // ============================================================
                // Binary reductions (child_count == 2)
                // ============================================================
                if (op.child_count != 2u) continue;

                const auto kid0_idx = kidIdx_sr(op, 0);
                const auto kid1_idx = kidIdx_sr(op, 1);
                const auto& kid0 = ops[kid0_idx];
                const auto& kid1 = ops[kid1_idx];

                // ---- Power specialization ----
                // These are safe because llvm.pow(x, n) for integer/half-integer
                // n handles NaN/inf/zero consistently with the replacement ops.
                if (op.type == FormExprType::Power && isConstant(kid1)) {
                    const double exp = constantVal(kid1);

                    if (exp == 2.0) {
                        op.type = FormExprType::Multiply;
                        setKid_sr(op, 0, kid0_idx);
                        setKid_sr(op, 1, kid0_idx);
                        changed = true; continue;
                    }
                    if (exp == 0.5) {
                        op.type = FormExprType::Sqrt;
                        op.child_count = 1;
                        changed = true; continue;
                    }
                    if (exp == -1.0) {
                        const auto one = appendConstOp(1.0);
                        op.type = FormExprType::Divide;
                        setKid_sr(op, 0, one);
                        setKid_sr(op, 1, kid0_idx);
                        changed = true; continue;
                    }
                    if (exp == -0.5) {
                        const auto one = appendConstOp(1.0);
                        const auto sq = appendUnaryOp(FormExprType::Sqrt, kid0_idx);
                        op.type = FormExprType::Divide;
                        setKid_sr(op, 0, one);
                        setKid_sr(op, 1, sq);
                        changed = true; continue;
                    }
                    if (exp == 3.0) {
                        const auto sq = appendBinaryOp(FormExprType::Multiply, kid0_idx, kid0_idx);
                        op.type = FormExprType::Multiply;
                        setKid_sr(op, 0, sq);
                        setKid_sr(op, 1, kid0_idx);
                        changed = true; continue;
                    }
                    if (exp == 4.0) {
                        const auto sq = appendBinaryOp(FormExprType::Multiply, kid0_idx, kid0_idx);
                        op.type = FormExprType::Multiply;
                        setKid_sr(op, 0, sq);
                        setKid_sr(op, 1, sq);
                        changed = true; continue;
                    }
                    if (exp == -2.0) {
                        const auto one = appendConstOp(1.0);
                        const auto sq = appendBinaryOp(FormExprType::Multiply, kid0_idx, kid0_idx);
                        op.type = FormExprType::Divide;
                        setKid_sr(op, 0, one);
                        setKid_sr(op, 1, sq);
                        changed = true; continue;
                    }
                    if (exp == -3.0) {
                        const auto one = appendConstOp(1.0);
                        const auto sq = appendBinaryOp(FormExprType::Multiply, kid0_idx, kid0_idx);
                        const auto cu = appendBinaryOp(FormExprType::Multiply, sq, kid0_idx);
                        op.type = FormExprType::Divide;
                        setKid_sr(op, 0, one);
                        setKid_sr(op, 1, cu);
                        changed = true; continue;
                    }
                    if (exp == -4.0) {
                        const auto one = appendConstOp(1.0);
                        const auto sq = appendBinaryOp(FormExprType::Multiply, kid0_idx, kid0_idx);
                        const auto q4 = appendBinaryOp(FormExprType::Multiply, sq, sq);
                        op.type = FormExprType::Divide;
                        setKid_sr(op, 0, one);
                        setKid_sr(op, 1, q4);
                        changed = true; continue;
                    }
                }

                // ---- Multiply by -1 → Negate ----
                if (op.type == FormExprType::Multiply) {
                    if (isConstant(kid0) && constantVal(kid0) == -1.0) {
                        op.type = FormExprType::Negate;
                        op.child_count = 1;
                        setKid_sr(op, 0, kid1_idx);
                        changed = true; continue;
                    }
                    if (isConstant(kid1) && constantVal(kid1) == -1.0) {
                        op.type = FormExprType::Negate;
                        op.child_count = 1;
                        changed = true; continue;
                    }
                }

                // ---- Divide by -1 → Negate ----
                if (op.type == FormExprType::Divide &&
                    isConstant(kid1) && constantVal(kid1) == -1.0)
                {
                    op.type = FormExprType::Negate;
                    op.child_count = 1;
                    // first_child already points to kid0_idx (numerator)
                    changed = true; continue;
                }

                // ---- Divide by exact-reciprocal constant → Multiply ----
                // Only when c * (1/c) == 1.0 in double precision, ensuring
                // the rewrite is FP-identical (no rounding change).
                if (op.type == FormExprType::Divide && isConstant(kid1)) {
                    const double c = constantVal(kid1);
                    if (c != 0.0 && c != -1.0) { // -1 already handled above
                        const double recip = 1.0 / c;
                        if (c * recip == 1.0) {
                            const auto recip_idx = appendConstOp(recip);
                            op.type = FormExprType::Multiply;
                            setKid_sr(op, 1, recip_idx);
                            changed = true; continue;
                        }
                    }
                }

                // ---- Add(X, Negate(X)) ----
                // NOT collapsed. The KernelIR has no shape information, so we
                // cannot produce a correctly-shaped zero (TypedZero defaults to
                // scalar in inferShapes, losing tensor rank). For tensors,
                // Add(A, Negate(A)) where A is a vector/matrix would produce a
                // scalar zero, breaking downstream shape inference. LLVM will
                // fold fadd(x, fneg(x)) → 0 per element at IR level.
            }

            if (!changed) break;
        }

        // If compound rewrites appended new ops, restore post-order via
        // topological sort (DFS post-order from root).
        if (needs_sort) {
            const std::size_t total_n = ops.size();
            std::vector<std::uint32_t> order;
            order.reserve(total_n);
            std::vector<std::uint8_t> state(total_n, 0);

            std::vector<std::pair<std::uint32_t, std::uint32_t>> dfs_stack;
            dfs_stack.push_back({root, 0});
            state[root] = 1;
            while (!dfs_stack.empty()) {
                auto& [node, next_c] = dfs_stack.back();
                const auto& node_op = ops[node];
                if (next_c < node_op.child_count) {
                    const auto child = children[static_cast<std::size_t>(node_op.first_child) + next_c];
                    ++next_c;
                    if (child < total_n && state[child] == 0) {
                        state[child] = 1;
                        dfs_stack.push_back({child, 0});
                    }
                } else {
                    state[node] = 2;
                    order.push_back(node);
                    dfs_stack.pop_back();
                }
            }

            std::vector<std::uint32_t> remap(total_n, UINT32_MAX);
            std::vector<KernelIROp> sorted_ops;
            sorted_ops.reserve(order.size());
            std::vector<std::uint32_t> sorted_children;
            sorted_children.reserve(children.size());

            for (auto old_idx : order) {
                remap[old_idx] = static_cast<std::uint32_t>(sorted_ops.size());
                const auto& old_op = ops[old_idx];
                KernelIROp nop = old_op;
                nop.first_child = static_cast<std::uint32_t>(sorted_children.size());
                for (std::uint32_t c = 0; c < old_op.child_count; ++c) {
                    const auto old_child = children[static_cast<std::size_t>(old_op.first_child) + c];
                    sorted_children.push_back(remap[old_child]);
                }
                sorted_ops.push_back(nop);
            }

            root = remap[root];
            ops = std::move(sorted_ops);
            children = std::move(sorted_children);
        }
    }

    // Pass 1.5: Algebraic strength reduction — constant factor extraction.
    // Rewrites Add(Mul(C,B), Mul(C,D)) → Mul(C, Add(B,D)) when:
    //   - C is a Constant or ParameterRef (compile-time invariant),
    //   - both Multiply ops are single-use (only used by this Add/Sub).
    // Also handles Subtract(Mul(C,B), Mul(C,D)) → Mul(C, Sub(B,D)).
    //
    // Only factors out compile-time constants (Constant, ParameterRef) to
    // minimize floating-point perturbation.  Factoring variable-dependent
    // subexpressions (grad(u), solution fields) can change FP evaluation
    // order enough to destabilize Newton convergence on sensitive problems.
    //
    // Applied iteratively (max 3 passes) since factorization can expose
    // new opportunities: C*B + C*D + C*E → C*(B+D) + C*E → C*(B+D+E).
    {
        // Compute use counts: how many ops reference each op as a child.
        auto computeUseCounts = [&]() {
            std::vector<std::uint32_t> use_count(ops.size(), 0);
            for (std::size_t i = 0; i < ops.size(); ++i) {
                const auto& op = ops[i];
                for (std::uint32_t c = 0; c < op.child_count; ++c) {
                    const auto child = children[static_cast<std::size_t>(op.first_child) + c];
                    if (child < use_count.size()) ++use_count[child];
                }
            }
            // Root is also "used"
            if (root < use_count.size()) ++use_count[root];
            return use_count;
        };

        auto getKid = [&](const KernelIROp& op, std::size_t k) -> std::uint32_t {
            return children[static_cast<std::size_t>(op.first_child) + k];
        };

        // Only factor out compile-time-constant ops to avoid FP perturbation.
        auto isFactorable = [&](std::uint32_t idx) -> bool {
            const auto t = ops[idx].type;
            return t == FormExprType::Constant || t == FormExprType::ParameterRef;
        };

        constexpr int kMaxFactorPasses = 3;
        for (int pass = 0; pass < kMaxFactorPasses; ++pass) {
            auto use_count = computeUseCounts();
            bool changed = false;

            for (std::size_t i = 0; i < ops.size(); ++i) {
                auto& op = ops[i];
                // Look for Add(Mul(A,B), Mul(A,C)) or Sub(Mul(A,B), Mul(A,C))
                if (op.child_count != 2u) continue;
                if (op.type != FormExprType::Add && op.type != FormExprType::Subtract) continue;

                const auto lhs_idx = getKid(op, 0);
                const auto rhs_idx = getKid(op, 1);
                auto& lhs = ops[lhs_idx];
                auto& rhs = ops[rhs_idx];

                if (lhs.type != FormExprType::Multiply || lhs.child_count != 2u) continue;
                if (rhs.type != FormExprType::Multiply || rhs.child_count != 2u) continue;

                // Both Multiply ops must be single-use (only used by this Add/Sub)
                // to allow safe in-place rewriting.
                if (use_count[lhs_idx] != 1u || use_count[rhs_idx] != 1u) continue;

                const auto la = getKid(lhs, 0), lb = getKid(lhs, 1);
                const auto ra = getKid(rhs, 0), rb = getKid(rhs, 1);

                std::uint32_t common = UINT32_MAX, other_l = UINT32_MAX, other_r = UINT32_MAX;
                if      (la == ra && isFactorable(la)) { common = la; other_l = lb; other_r = rb; }
                else if (la == rb && isFactorable(la)) { common = la; other_l = lb; other_r = ra; }
                else if (lb == ra && isFactorable(lb)) { common = lb; other_l = la; other_r = rb; }
                else if (lb == rb && isFactorable(lb)) { common = lb; other_l = la; other_r = ra; }

                if (common == UINT32_MAX) continue;

                // Rewrite: repurpose lhs Multiply as Add/Sub(other_l, other_r),
                // then change the outer op to Multiply(common, lhs_idx).
                lhs.type = op.type;  // Add or Subtract
                children[static_cast<std::size_t>(lhs.first_child)]     = other_l;
                children[static_cast<std::size_t>(lhs.first_child) + 1] = other_r;

                op.type = FormExprType::Multiply;
                children[static_cast<std::size_t>(op.first_child)]     = common;
                children[static_cast<std::size_t>(op.first_child) + 1] = lhs_idx;

                changed = true;
            }

            if (!changed) break;
        }
    }

    // Pass 2: Post-rewrite structural CSE.
    // After zero propagation/constant folding, some previously-distinct
    // subexpressions may now be structurally identical (e.g., two subtrees
    // that both simplified to Constant(0.0)).  Merge them by remapping
    // child indices to the first occurrence of each structural pattern.
    //
    // NOTE: Commutative canonicalization (merging a+b with b+a) is
    // intentionally NOT done here.  While mathematically equivalent,
    // reusing one for the other perturbs the Jacobian by ULPs (different
    // floating-point summation order), which can destabilize iterative
    // linear solvers on ill-conditioned systems.  Commutative
    // canonicalization is already applied at lowering time (line 424)
    // before any rewrites occur, which handles the common case.
    {
        const std::size_t n = ops.size();
        // Compute structural hash per op (children-aware, bottom-up).
        std::vector<std::uint64_t> op_hash(n, 0);
        // Map from structural hash → list of candidate op indices
        std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> hash_to_idx;
        // CSE remap: old index → canonical index (identity initially)
        std::vector<std::uint32_t> cse_remap(n);
        for (std::uint32_t i = 0; i < n; ++i) cse_remap[i] = i;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& op = ops[i];
            std::uint64_t h = kFNVOffset;
            hashMix(h, static_cast<std::uint64_t>(op.type));
            hashMix(h, op.imm0);
            hashMix(h, op.imm1);
            hashMix(h, static_cast<std::uint64_t>(op.child_count));
            for (std::uint32_t c = 0; c < op.child_count; ++c) {
                // Use the canonical (CSE-remapped) child index for hashing
                auto raw_child = children[static_cast<std::size_t>(op.first_child) + c];
                hashMix(h, static_cast<std::uint64_t>(cse_remap[raw_child]));
            }
            op_hash[i] = h;

            // Check for structural match with existing ops of same hash
            bool merged = false;
            auto& candidates = hash_to_idx[h];
            for (const auto cand : candidates) {
                const auto& cop = ops[cand];
                if (cop.type != op.type || cop.imm0 != op.imm0 ||
                    cop.imm1 != op.imm1 || cop.child_count != op.child_count)
                    continue;
                bool match = true;
                for (std::uint32_t c = 0; c < op.child_count; ++c) {
                    const auto c1 = cse_remap[children[static_cast<std::size_t>(cop.first_child) + c]];
                    const auto c2 = cse_remap[children[static_cast<std::size_t>(op.first_child) + c]];
                    if (c1 != c2) { match = false; break; }
                }
                if (match) {
                    cse_remap[i] = cse_remap[cand];
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                candidates.push_back(static_cast<std::uint32_t>(i));
            }
        }

        // Apply CSE remapping to all child references
        for (auto& c : children) {
            c = cse_remap[c];
        }
        root = cse_remap[root];
    }

    // Pass 3: Dead code elimination.
    // Mark reachable ops from root, then compact.
    const std::size_t n = ops.size();
    std::vector<bool> reachable(n, false);
    {
        // BFS/DFS from root
        std::vector<std::uint32_t> stack;
        stack.push_back(root);
        while (!stack.empty()) {
            const auto idx = stack.back();
            stack.pop_back();
            if (idx >= n || reachable[idx]) continue;
            reachable[idx] = true;
            const auto& op = ops[idx];
            for (std::uint32_t c = 0; c < op.child_count; ++c) {
                const auto child = children[static_cast<std::size_t>(op.first_child) + c];
                if (!reachable[child]) {
                    stack.push_back(child);
                }
            }
        }
    }

    // Count reachable ops
    std::size_t live_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (reachable[i]) ++live_count;
    }

    if (live_count == n) {
        return 0; // nothing eliminated
    }

    // Build remapping: old index → new index
    std::vector<std::uint32_t> remap(n, UINT32_MAX);
    std::vector<KernelIROp> new_ops;
    new_ops.reserve(live_count);
    std::vector<std::uint32_t> new_children;
    new_children.reserve(children.size());

    for (std::size_t i = 0; i < n; ++i) {
        if (!reachable[i]) continue;
        const auto& old_op = ops[i];
        const auto new_idx = static_cast<std::uint32_t>(new_ops.size());
        remap[i] = new_idx;

        KernelIROp new_op = old_op;
        new_op.first_child = static_cast<std::uint32_t>(new_children.size());
        for (std::uint32_t c = 0; c < old_op.child_count; ++c) {
            const auto old_child = children[static_cast<std::size_t>(old_op.first_child) + c];
            const auto remapped = remap[old_child];
            new_children.push_back(remapped);
        }
        new_ops.push_back(new_op);
    }

    root = remap[root];
    ops = std::move(new_ops);
    children = std::move(new_children);

    return original_count - live_count;
}

std::vector<std::uint32_t> KernelIR::subtreeCosts() const
{
    const auto n = ops.size();
    std::vector<std::uint32_t> cost(n, 1u);

    // Bottom-up (post-order guarantees children processed first).
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = ops[i];
        if (op.child_count == 0u) {
            // Leaf cost: 1 for cheap ops, higher for reduce-sum ops
            // that expand to DOF loops in codegen.
            switch (op.type) {
                case FormExprType::StateField:
                case FormExprType::DiscreteField:
                case FormExprType::PreviousSolutionRef:
                    // These expand to N_dof-length reduce-sums in codegen.
                    // Use a multiplier to reflect that.
                    cost[i] = 16;
                    break;
                default:
                    cost[i] = 1;
                    break;
            }
            continue;
        }

        std::uint32_t c = 1; // op's own cost
        for (std::uint32_t k = 0; k < op.child_count; ++k) {
            const auto child = children[static_cast<std::size_t>(op.first_child) + k];
            c += cost[child];
        }

        // Gradient/Hessian of a field involves J_inv transforms
        if (op.child_count == 1u) {
            const auto child = children[static_cast<std::size_t>(op.first_child)];
            const auto ct = ops[child].type;
            if ((op.type == FormExprType::Gradient || op.type == FormExprType::Hessian) &&
                (ct == FormExprType::StateField || ct == FormExprType::DiscreteField ||
                 ct == FormExprType::PreviousSolutionRef || ct == FormExprType::TrialFunction ||
                 ct == FormExprType::TestFunction)) {
                c += 8; // J_inv transform overhead
            }
        }

        // Matrix spectral ops: O(d^3) work for d×d matrices
        switch (op.type) {
            case FormExprType::Inverse:
            case FormExprType::Cofactor:
            case FormExprType::Determinant:
                c += 16;  // LU decomposition / cofactor expansion
                break;
            case FormExprType::MatrixExponential:
            case FormExprType::MatrixSqrt:
            case FormExprType::MatrixLogarithm:
            case FormExprType::MatrixPower:
                c += 32;  // series expansion / eigendecomposition
                break;
            case FormExprType::Eigenvalue:
            case FormExprType::SymmetricEigenvalue:
                c += 24;  // eigenvalue computation
                break;
            // Directional derivatives of matrix spectral ops
            // (involve the base spectral op plus Fréchet derivative)
            case FormExprType::MatrixExponentialDirectionalDerivative:
            case FormExprType::MatrixLogarithmDirectionalDerivative:
            case FormExprType::MatrixSqrtDirectionalDerivative:
            case FormExprType::MatrixPowerDirectionalDerivative:
            case FormExprType::SymmetricEigenvalueDirectionalDerivative:
            case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA:
            case FormExprType::SymmetricEigenvectorDirectionalDerivative:
            case FormExprType::SpectralDecompositionDirectionalDerivative:
                c += 40;  // spectral op + Fréchet derivative
                break;
            // Transcendental scalar ops (libm calls)
            case FormExprType::Exp:
            case FormExprType::Log:
            case FormExprType::Sqrt:
            case FormExprType::AbsoluteValue:
            case FormExprType::Power:
                c += 4;   // libm call overhead
                break;
            // History operators (weighted sum over time steps)
            case FormExprType::HistoryWeightedSum:
            case FormExprType::HistoryConvolution:
                c += 8;   // loop over history steps
                break;
            // Constitutive model evaluation
            case FormExprType::Constitutive:
            case FormExprType::ConstitutiveOutput:
                c += 16;  // typically involves multiple matrix ops
                break;
            default:
                break;
        }

        cost[i] = c;
    }

    return cost;
}

KernelIRBuildResult lowerToKernelIR(const FormExprNode& root,
                                    const KernelIRBuildOptions& options)
{
    Builder b;
    b.options = options;
    b.result.cacheable = true;
    b.auto_index_extent = forms::inferAutoIndexExtent(root);

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

// ============================================================================
// Term-group planning for micro-kernel splitting
// ============================================================================

BlockSplitPlan planTermGroups(
    const std::vector<std::size_t>& term_op_counts,
    std::uint64_t budget_bytes,
    std::uint64_t bytes_per_op)
{
    BlockSplitPlan plan;

    if (term_op_counts.empty()) {
        plan.needs_split = false;
        return plan;
    }

    // Calculate total estimated bytes for this block.
    std::uint64_t total_bytes = 0;
    for (const auto ops : term_op_counts) {
        total_bytes += static_cast<std::uint64_t>(ops) * bytes_per_op;
    }
    plan.total_estimated_bytes = total_bytes;

    // If total fits in budget, no splitting needed.
    if (budget_bytes == 0 || total_bytes <= budget_bytes) {
        plan.needs_split = false;
        TermGroupPlan g;
        g.first_term = 0;
        g.num_terms = term_op_counts.size();
        g.estimated_text_bytes = total_bytes;
        plan.groups.push_back(g);
        return plan;
    }

    // Greedy contiguous packing.
    plan.needs_split = true;
    std::size_t current_first = 0;
    std::uint64_t current_bytes = 0;

    for (std::size_t i = 0; i < term_op_counts.size(); ++i) {
        const std::uint64_t term_bytes =
            static_cast<std::uint64_t>(term_op_counts[i]) * bytes_per_op;

        if (current_bytes + term_bytes > budget_bytes && i > current_first) {
            // Close current group.
            TermGroupPlan g;
            g.first_term = current_first;
            g.num_terms = i - current_first;
            g.estimated_text_bytes = current_bytes;
            plan.groups.push_back(g);
            current_first = i;
            current_bytes = 0;
        }
        current_bytes += term_bytes;
    }

    // Close last group.
    if (current_first < term_op_counts.size()) {
        TermGroupPlan g;
        g.first_term = current_first;
        g.num_terms = term_op_counts.size() - current_first;
        g.estimated_text_bytes = current_bytes;
        plan.groups.push_back(g);
    }

    return plan;
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
