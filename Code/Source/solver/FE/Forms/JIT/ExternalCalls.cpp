/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/ExternalCalls.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/FormExpr.h"
#include "Forms/Value.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

using SideArgs = assembly::jit::KernelSideArgsV3;

[[nodiscard]] const external::ExternalCallTableV1* tryGetExternalCallTable(const SideArgs* side) noexcept
{
    if (side == nullptr) {
        return nullptr;
    }

    const auto* table = static_cast<const external::ExternalCallTableV1*>(side->user_data);
    if (table == nullptr) {
        return nullptr;
    }
    if (table->magic != external::kExternalCallsMagicV1) {
        return nullptr;
    }
    if (table->abi_version != external::kExternalCallsABIVersionV1) {
        return nullptr;
    }
    return table;
}

[[nodiscard]] std::array<Real, 3> physicalPoint(const SideArgs& side, std::uint32_t q) noexcept
{
    if (side.physical_points_xyz == nullptr) {
        return {Real(0.0), Real(0.0), Real(0.0)};
    }
    const std::size_t base = static_cast<std::size_t>(q) * 3u;
    return {side.physical_points_xyz[base + 0u],
            side.physical_points_xyz[base + 1u],
            side.physical_points_xyz[base + 2u]};
}

[[nodiscard]] std::array<Real, 3> physicalPointAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.physical_points_xyz == nullptr || qpt < 0) {
        return {Real(0.0), Real(0.0), Real(0.0)};
    }
    const std::size_t base = static_cast<std::size_t>(qpt) * 3u;
    return {side.physical_points_xyz[base + 0u],
            side.physical_points_xyz[base + 1u],
            side.physical_points_xyz[base + 2u]};
}

[[nodiscard]] Real integrationWeightAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.integration_weights == nullptr || qpt < 0) {
        return Real(0.0);
    }
    return side.integration_weights[static_cast<std::size_t>(qpt)];
}

struct NonlocalV1 {
    const SideArgs* side{nullptr};
};

const std::byte* stateOldPtrAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.material_state_old_base == nullptr || qpt < 0) {
        return nullptr;
    }
    if (side.material_state_stride_bytes == 0u) {
        return nullptr;
    }
    const std::size_t byte_off =
        static_cast<std::size_t>(qpt) * static_cast<std::size_t>(side.material_state_stride_bytes);
    return side.material_state_old_base + byte_off;
}

std::byte* stateWorkPtrAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.material_state_work_base == nullptr || qpt < 0) {
        return nullptr;
    }
    if (side.material_state_stride_bytes == 0u) {
        return nullptr;
    }
    const std::size_t byte_off =
        static_cast<std::size_t>(qpt) * static_cast<std::size_t>(side.material_state_stride_bytes);
    return side.material_state_work_base + byte_off;
}

[[nodiscard]] std::span<const std::byte> stateOldSpanAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    const auto* p = stateOldPtrAt(side, qpt);
    if (p == nullptr) return {};
    return {p, static_cast<std::size_t>(side.material_state_bytes_per_qpt)};
}

[[nodiscard]] std::span<std::byte> stateWorkSpanAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    auto* p = stateWorkPtrAt(side, qpt);
    if (p == nullptr) return {};
    return {p, static_cast<std::size_t>(side.material_state_bytes_per_qpt)};
}

[[nodiscard]] forms::Value<Real> toFormsValue(const external::ValueViewV1& view) noexcept
{
    forms::Value<Real> v;

    const auto kind = static_cast<external::ValueKindV1>(view.kind);
    const auto* data = view.data;
    const auto n = view.length;

    switch (kind) {
        case external::ValueKindV1::Scalar:
            v.kind = forms::Value<Real>::Kind::Scalar;
            v.s = (data != nullptr && n >= 1u) ? static_cast<Real>(data[0]) : Real(0.0);
            return v;

        case external::ValueKindV1::Vector:
            v.kind = forms::Value<Real>::Kind::Vector;
            v.vector_size = static_cast<int>(std::min<std::uint32_t>(n, 3u));
            v.v = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 3u); ++i) {
                    v.v[static_cast<std::size_t>(i)] = static_cast<Real>(data[i]);
                }
            }
            return v;

        case external::ValueKindV1::Matrix:
            v.kind = forms::Value<Real>::Kind::Matrix;
            v.matrix_rows = 3;
            v.matrix_cols = 3;
            v.m = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 9u); ++i) {
                    v.m[static_cast<std::size_t>(i / 3u)][static_cast<std::size_t>(i % 3u)] = static_cast<Real>(data[i]);
                }
            }
            return v;

        case external::ValueKindV1::Tensor3:
            v.kind = forms::Value<Real>::Kind::Tensor3;
            v.tensor3_dim0 = 3;
            v.tensor3_dim1 = 3;
            v.tensor3_dim2 = 3;
            v.t3 = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 27u); ++i) {
                    v.t3[static_cast<std::size_t>(i)] = static_cast<Real>(data[i]);
                }
            }
            return v;

        case external::ValueKindV1::Tensor4:
            v.kind = forms::Value<Real>::Kind::Tensor4;
            v.t4 = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 81u); ++i) {
                    v.t4[static_cast<std::size_t>(i)] = static_cast<Real>(data[i]);
                }
            }
            return v;
    }

    v.kind = forms::Value<Real>::Kind::Scalar;
    v.s = Real(0.0);
    return v;
}

[[nodiscard]] bool isMatrixKind(forms::Value<Real>::Kind k) noexcept
{
    return k == forms::Value<Real>::Kind::Matrix ||
           k == forms::Value<Real>::Kind::SymmetricMatrix ||
           k == forms::Value<Real>::Kind::SkewMatrix;
}

void zeroOut(double* out_values, std::uint32_t out_len) noexcept
{
    if (out_values == nullptr) return;
    for (std::uint32_t i = 0; i < out_len; ++i) {
        out_values[i] = 0.0;
    }
}

void writeFormsValueToFlat(const forms::Value<Real>& v,
                           std::uint32_t out_kind_u32,
                           double* out_values,
                           std::uint32_t out_len) noexcept
{
    zeroOut(out_values, out_len);
    if (out_values == nullptr || out_len == 0u) {
        return;
    }

    const auto out_kind = static_cast<external::ValueKindV1>(out_kind_u32);

    switch (out_kind) {
        case external::ValueKindV1::Scalar:
            if (v.kind == forms::Value<Real>::Kind::Scalar) {
                out_values[0] = static_cast<double>(v.s);
            }
            return;

        case external::ValueKindV1::Vector:
            if (v.kind != forms::Value<Real>::Kind::Vector) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 3u); ++i) {
                out_values[i] = static_cast<double>(v.v[static_cast<std::size_t>(i)]);
            }
            return;

        case external::ValueKindV1::Matrix:
            if (!isMatrixKind(v.kind)) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 9u); ++i) {
                out_values[i] = static_cast<double>(v.m[static_cast<std::size_t>(i / 3u)][static_cast<std::size_t>(i % 3u)]);
            }
            return;

        case external::ValueKindV1::Tensor3:
            if (v.kind != forms::Value<Real>::Kind::Tensor3) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 27u); ++i) {
                out_values[i] = static_cast<double>(v.t3[static_cast<std::size_t>(i)]);
            }
            return;

        case external::ValueKindV1::Tensor4:
            if (v.kind != forms::Value<Real>::Kind::Tensor4) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 81u); ++i) {
                out_values[i] = static_cast<double>(v.t4[static_cast<std::size_t>(i)]);
            }
            return;
    }
}

} // namespace

extern "C" double svmp_fe_jit_coeff_eval_scalar_v1(const void* side_ptr,
                                                   std::uint32_t q,
                                                   const void* coeff_node_ptr) noexcept
{
    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return 0.0;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            double out = 0.0;
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Scalar),
                                                    &out,
                                                    1u);
            if (ok) {
                return out;
            }
        }

        const auto x = physicalPoint(*side, q);

        if (const auto* f = node->timeScalarCoefficient(); f != nullptr) {
            return static_cast<double>((*f)(x[0], x[1], x[2], side->time));
        }
        if (const auto* f = node->scalarCoefficient(); f != nullptr) {
            return static_cast<double>((*f)(x[0], x[1], x[2]));
        }
        return 0.0;
    } catch (...) {
        return 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_vector_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 const void* coeff_node_ptr,
                                                 double* out3) noexcept
{
    if (out3 == nullptr) return;
    out3[0] = out3[1] = out3[2] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Vector),
                                                    out3,
                                                    3u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->vectorCoefficient(); f != nullptr) {
            const auto v = (*f)(x[0], x[1], x[2]);
            out3[0] = static_cast<double>(v[0]);
            out3[1] = static_cast<double>(v[1]);
            out3[2] = static_cast<double>(v[2]);
        }
    } catch (...) {
        out3[0] = out3[1] = out3[2] = 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_matrix_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 const void* coeff_node_ptr,
                                                 double* out9) noexcept
{
    if (out9 == nullptr) return;
    for (int i = 0; i < 9; ++i) out9[i] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Matrix),
                                                    out9,
                                                    9u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->matrixCoefficient(); f != nullptr) {
            const auto m = (*f)(x[0], x[1], x[2]);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out9[r * 3 + c] = static_cast<double>(m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]);
                }
            }
        }
    } catch (...) {
        for (int i = 0; i < 9; ++i) out9[i] = 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_tensor3_v1(const void* side_ptr,
                                                  std::uint32_t q,
                                                  const void* coeff_node_ptr,
                                                  double* out27) noexcept
{
    if (out27 == nullptr) return;
    for (int i = 0; i < 27; ++i) out27[i] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Tensor3),
                                                    out27,
                                                    27u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->tensor3Coefficient(); f != nullptr) {
            const auto t3 = (*f)(x[0], x[1], x[2]);
            for (int i = 0; i < 27; ++i) {
                out27[i] = static_cast<double>(t3[static_cast<std::size_t>(i)]);
            }
        }
    } catch (...) {
        for (int i = 0; i < 27; ++i) out27[i] = 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_tensor4_v1(const void* side_ptr,
                                                  std::uint32_t q,
                                                  const void* coeff_node_ptr,
                                                  double* out81) noexcept
{
    if (out81 == nullptr) return;
    for (int i = 0; i < 81; ++i) out81[i] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Tensor4),
                                                    out81,
                                                    81u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->tensor4Coefficient(); f != nullptr) {
            const auto t4 = (*f)(x[0], x[1], x[2]);
            for (int i = 0; i < 81; ++i) {
                out81[i] = static_cast<double>(t4[static_cast<std::size_t>(i)]);
            }
        }
    } catch (...) {
        for (int i = 0; i < 81; ++i) out81[i] = 0.0;
    }
}

extern "C" void svmp_fe_jit_constitutive_eval_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 std::uint32_t trace_side,
                                                 const void* constitutive_node_ptr,
                                                 std::uint32_t output_index,
                                                 const external::ValueViewV1* inputs,
                                                 std::uint32_t num_inputs,
                                                 std::uint32_t out_kind,
                                                 double* out_values,
                                                 std::uint32_t out_len) noexcept
{
    zeroOut(out_values, out_len);

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(constitutive_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_constitutive) {
            const bool ok = table->eval_constitutive(table->context,
                                                     side_ptr,
                                                     q,
                                                     trace_side,
                                                     constitutive_node_ptr,
                                                     output_index,
                                                     inputs,
                                                     num_inputs,
                                                     out_kind,
                                                     out_values,
                                                     out_len);
            if (ok) {
                return;
            }
        }

        const auto* model = node->constitutiveModel();
        if (model == nullptr) {
            return;
        }

        ConstitutiveEvalContext ctx;
        ctx.domain = [&]() {
            const auto ct = static_cast<assembly::ContextType>(side->context_type);
            switch (ct) {
                case assembly::ContextType::Cell:
                    return ConstitutiveEvalContext::Domain::Cell;
                case assembly::ContextType::BoundaryFace:
                    return ConstitutiveEvalContext::Domain::BoundaryFace;
                case assembly::ContextType::InteriorFace:
                default:
                    return ConstitutiveEvalContext::Domain::InteriorFace;
            }
        }();
        ctx.side = (trace_side == static_cast<std::uint32_t>(external::TraceSideV1::Plus))
                       ? ConstitutiveEvalContext::TraceSide::Plus
                       : ConstitutiveEvalContext::TraceSide::Minus;
        ctx.dim = static_cast<int>(side->dim);
        ctx.x = physicalPoint(*side, q);
        ctx.time = side->time;
        ctx.dt = side->dt;
        ctx.cell_id = side->cell_id;
        ctx.face_id = side->face_id;
        ctx.local_face_id = static_cast<LocalIndex>(side->local_face_id);
        ctx.boundary_marker = static_cast<int>(side->boundary_marker);
        ctx.q = static_cast<LocalIndex>(q);
        ctx.num_qpts = static_cast<LocalIndex>(side->n_qpts);
        ctx.user_data = side->user_data;

        NonlocalV1 nonlocal_state{side};
        ConstitutiveEvalContext::NonlocalAccess nonlocal;
        nonlocal.self = &nonlocal_state;
        nonlocal.state_old = +[](const void* self, LocalIndex qpt) -> std::span<const std::byte> {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return {};
            return stateOldSpanAt(*s.side, qpt);
        };
        nonlocal.state_work = +[](const void* self, LocalIndex qpt) -> std::span<std::byte> {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return {};
            return stateWorkSpanAt(*s.side, qpt);
        };
        nonlocal.physical_point = +[](const void* self, LocalIndex qpt) -> std::array<Real, 3> {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return {Real(0.0), Real(0.0), Real(0.0)};
            return physicalPointAt(*s.side, qpt);
        };
        nonlocal.integration_weight = +[](const void* self, LocalIndex qpt) -> Real {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return Real(0.0);
            return integrationWeightAt(*s.side, qpt);
        };
        ctx.nonlocal = &nonlocal;
        ctx.state_old = stateOldSpanAt(*side, static_cast<LocalIndex>(q));
        ctx.state_work = stateWorkSpanAt(*side, static_cast<LocalIndex>(q));

        std::vector<forms::Value<Real>> input_values;
        input_values.reserve(num_inputs);
        for (std::uint32_t i = 0; i < num_inputs; ++i) {
            input_values.push_back(toFormsValue(inputs[i]));
        }

        const auto n_outputs = model->outputCount();
        if (output_index >= n_outputs) {
            return;
        }

        std::vector<forms::Value<Real>> output_values;
        output_values.resize(n_outputs);

        model->evaluateNaryOutputs({input_values.data(), input_values.size()},
                                   ctx,
                                   {output_values.data(), output_values.size()});

        writeFormsValueToFlat(output_values[output_index], out_kind, out_values, out_len);
    } catch (...) {
        zeroOut(out_values, out_len);
    }
}

extern "C" void svmp_fe_jit_constitutive_eval_batch_v1(const void* side_ptr,
                                                       const std::uint32_t* q_indices,
                                                       std::uint32_t num_q,
                                                       std::uint32_t trace_side,
                                                       const void* constitutive_node_ptr,
                                                       std::uint32_t output_index,
                                                       const external::ValueBatchViewV1* inputs,
                                                       std::uint32_t num_inputs,
                                                       std::uint32_t out_kind,
                                                       double* out_values,
                                                       std::uint32_t out_len,
                                                       std::uint32_t out_stride) noexcept
{
    if (out_values == nullptr) {
        return;
    }

    try {
        std::vector<external::ValueViewV1> views;
        views.resize(num_inputs);

        for (std::uint32_t qi = 0; qi < num_q; ++qi) {
            const std::uint32_t q = (q_indices != nullptr) ? q_indices[qi] : qi;

            for (std::uint32_t i = 0; i < num_inputs; ++i) {
                external::ValueViewV1 v;
                v.kind = (inputs != nullptr) ? inputs[i].kind : 0u;
                v.length = (inputs != nullptr) ? inputs[i].length : 0u;
                v.data = nullptr;
                if (inputs != nullptr && inputs[i].data != nullptr) {
                    const auto stride = static_cast<std::size_t>(inputs[i].stride);
                    v.data = inputs[i].data + stride * static_cast<std::size_t>(qi);
                }
                views[i] = v;
            }

            auto* out_q = out_values + static_cast<std::size_t>(out_stride) * static_cast<std::size_t>(qi);
            svmp_fe_jit_constitutive_eval_v1(side_ptr,
                                             q,
                                             trace_side,
                                             constitutive_node_ptr,
                                             output_index,
                                             views.data(),
                                             num_inputs,
                                             out_kind,
                                             out_q,
                                             out_len);
        }
    } catch (...) {
        // Best-effort: leave outputs as-is.
    }
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
