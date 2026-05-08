/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/JIT/KernelArgs.h"
#include "Forms/FormIR.h"
#include "Forms/JIT/JITCacheKey.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] JITOptions baseJITOptions()
{
    JITOptions opt;
    opt.enable = true;
    opt.optimization_level = 2;
    opt.cache_kernels = true;
    opt.vectorize = true;
    opt.simd_batch = true;
    opt.cache_directory = "ignored-cache-directory";
    opt.cache_diagnostics = false;
    opt.max_in_memory_kernels = 64;
    opt.dump_kernel_ir = false;
    opt.dump_llvm_ir = false;
    opt.dump_llvm_ir_optimized = false;
    opt.debug_info = false;
    opt.dump_directory = "ignored-dump-directory";
    return opt;
}

[[nodiscard]] jit::KernelCacheKeyInputs baseInputs()
{
    jit::KernelCacheKeyInputs in;
    in.abi_version = 6;
    in.abi_layout_revision = 17;
    in.form_kind = FormKind::Bilinear;
    in.domain = IntegralDomain::Cell;
    in.boundary_marker = 3;
    in.interface_marker = -1;
    in.combined_ir_hash = 0x1020'3040'5060'7080ULL;
    in.test_space_hash = 0x0102'0304'0506'0708ULL;
    in.trial_space_hash = 0x8877'6655'4433'2211ULL;
    in.jit_options = baseJITOptions();
    in.target_triple = "x86_64-pc-linux-gnu";
    in.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64";
    in.cpu_name = "znver4";
    in.cpu_features = "+sse2,+avx,+avx2";
    in.llvm_version = "LLVM 18.1.8";
    in.hardware_profile_hash = 0x1234'5678'9abc'def0ULL;
    return in;
}

[[nodiscard]] jit::JITCompileSpecialization matchedSpecialization()
{
    jit::JITCompileSpecialization spec;
    spec.domain = IntegralDomain::Cell;
    spec.n_qpts_minus = 4u;
    spec.n_test_dofs_minus = 8u;
    spec.n_trial_dofs_minus = 9u;
    spec.n_qpts_plus = 5u;
    spec.n_test_dofs_plus = 10u;
    spec.n_trial_dofs_plus = 11u;
    spec.is_affine = true;
    return spec;
}

[[nodiscard]] std::uint64_t keyFor(const jit::KernelCacheKeyInputs& in)
{
    return jit::computeKernelCacheKey(in);
}

[[nodiscard]] std::string lowerHex(std::uint64_t value)
{
    std::ostringstream os;
    os << std::hex << value;
    return os.str();
}

template <typename Mutator>
void expectKeyChanges(std::string_view label, Mutator&& mutator)
{
    const auto base = baseInputs();
    auto changed = base;
    mutator(changed);
    EXPECT_NE(keyFor(changed), keyFor(base)) << std::string(label);
}

template <typename Mutator>
void expectKeyUnchanged(std::string_view label, Mutator&& mutator)
{
    const auto base = baseInputs();
    auto changed = base;
    mutator(changed);
    EXPECT_EQ(keyFor(changed), keyFor(base)) << std::string(label);
}

template <typename Mutator>
void expectMatchedSpecializationKeyChanges(std::string_view label, Mutator&& mutator)
{
    auto spec = matchedSpecialization();
    auto base = baseInputs();
    base.specialization = &spec;

    auto changed_spec = spec;
    auto changed = base;
    changed.specialization = &changed_spec;
    mutator(changed, changed_spec);
    EXPECT_NE(keyFor(changed), keyFor(base)) << std::string(label);
}

} // namespace

TEST(JITCacheKey, DeterministicForIdenticalInputs)
{
    const auto in = baseInputs();
    EXPECT_EQ(keyFor(in), keyFor(in));
    EXPECT_NE(keyFor(in), 0u);
}

TEST(JITCacheKey, StableSymbolUsesLowerHexCacheKey)
{
    constexpr std::uint64_t cache_key = 0x012a'bcde'ff10'9001ULL;
    EXPECT_EQ(jit::cacheKeyToHex(cache_key), lowerHex(cache_key));
    EXPECT_EQ(jit::stableSymbolForKernel(cache_key),
              "svmp_fe_jit_kernel_" + lowerHex(cache_key));
}

TEST(JITCacheKey, StructuralInputsChangeKey)
{
    expectKeyChanges("schema version", [](auto& in) { in.cache_key_schema_version += 1u; });
    expectKeyChanges("ABI version", [](auto& in) { in.abi_version += 1u; });
    expectKeyChanges("ABI layout revision", [](auto& in) { in.abi_layout_revision += 1u; });
    expectKeyChanges("form kind", [](auto& in) { in.form_kind = FormKind::Residual; });
    expectKeyChanges("domain", [](auto& in) { in.domain = IntegralDomain::Boundary; });
    expectKeyChanges("boundary marker", [](auto& in) { in.boundary_marker += 1; });
    expectKeyChanges("interface marker", [](auto& in) { in.interface_marker = 42; });
    expectKeyChanges("IR hash", [](auto& in) { in.combined_ir_hash ^= 0x10ULL; });
    expectKeyChanges("test space hash", [](auto& in) { in.test_space_hash ^= 0x20ULL; });
    expectKeyChanges("trial space hash", [](auto& in) { in.trial_space_hash ^= 0x40ULL; });
}

TEST(JITCacheKey, CellBatchAbiDoesNotAliasScalarAbi)
{
    auto scalar = baseInputs();
    scalar.abi_version = 6u;

    auto batch = scalar;
    batch.abi_version = assembly::jit::kCellKernelBatchArgsABIV1;

    EXPECT_NE(keyFor(batch), keyFor(scalar));
}

TEST(JITCacheKey, TargetAndHardwareInputsChangeKey)
{
    expectKeyChanges("target triple", [](auto& in) { in.target_triple = "aarch64-unknown-linux-gnu"; });
    expectKeyChanges("data layout", [](auto& in) { in.data_layout = "e-m:e-p:64:64"; });
    expectKeyChanges("CPU name", [](auto& in) { in.cpu_name = "generic"; });
    expectKeyChanges("CPU features", [](auto& in) { in.cpu_features = "+sse2"; });
    expectKeyChanges("LLVM version", [](auto& in) { in.llvm_version = "LLVM 19.0.0"; });
    expectKeyChanges("hardware profile", [](auto& in) { in.hardware_profile_hash ^= 0x80ULL; });
}

TEST(JITCacheKey, GenericCodegenOptionsChangeKey)
{
    expectKeyChanges("optimization level", [](auto& in) { in.jit_options.optimization_level = 3; });
    expectKeyChanges("vectorize", [](auto& in) { in.jit_options.vectorize = false; });
    expectKeyChanges("SIMD batch", [](auto& in) { in.jit_options.simd_batch = false; });
    expectKeyChanges("debug info", [](auto& in) { in.jit_options.debug_info = true; });

    expectKeyChanges("tensor mode", [](auto& in) { in.jit_options.tensor.mode = TensorLoweringMode::Auto; });
    expectKeyChanges("tensor force loop nest", [](auto& in) { in.jit_options.tensor.force_loop_nest = true; });
    expectKeyChanges("tensor symmetry lowering",
                     [](auto& in) { in.jit_options.tensor.enable_symmetry_lowering = false; });
    expectKeyChanges("tensor contraction order",
                     [](auto& in) { in.jit_options.tensor.enable_optimal_contraction_order = false; });
    expectKeyChanges("tensor vector hints",
                     [](auto& in) { in.jit_options.tensor.enable_vectorization_hints = false; });
    expectKeyChanges("tensor delta shortcuts",
                     [](auto& in) { in.jit_options.tensor.enable_delta_shortcuts = false; });
    expectKeyChanges("tensor scalar expansion threshold",
                     [](auto& in) { in.jit_options.tensor.scalar_expansion_term_threshold += 1u; });
    expectKeyChanges("tensor stack entries",
                     [](auto& in) { in.jit_options.tensor.temp_stack_max_entries += 1u; });
    expectKeyChanges("tensor temp alignment",
                     [](auto& in) { in.jit_options.tensor.temp_alignment_bytes += 64u; });
    expectKeyChanges("tensor temp reuse", [](auto& in) { in.jit_options.tensor.temp_enable_reuse = false; });
    expectKeyChanges("tensor polly", [](auto& in) { in.jit_options.tensor.enable_polly = true; });
    expectKeyChanges("tensor loop tiling", [](auto& in) { in.jit_options.tensor.enable_loop_tiling = false; });
    expectKeyChanges("tensor tile size", [](auto& in) { in.jit_options.tensor.tile_size = 8u; });
    expectKeyChanges("tensor min tiling extent",
                     [](auto& in) { in.jit_options.tensor.min_tiling_extent += 1u; });

    expectKeyChanges("unroll metadata",
                     [](auto& in) { in.jit_options.specialization.enable_loop_unroll_metadata = false; });
    expectKeyChanges("max unroll trip count",
                     [](auto& in) { in.jit_options.specialization.max_unroll_trip_count = 12u; });
    expectKeyChanges("bytes per op",
                     [](auto& in) { in.jit_options.specialization.bytes_per_op_estimate += 1u; });
    expectKeyChanges("raw bytes per op",
                     [](auto& in) { in.jit_options.specialization.raw_bytes_per_op_estimate += 1u; });

    expectKeyChanges("basis baking enable", [](auto& in) { in.jit_options.basis_baking.enable = false; });
    expectKeyChanges("basis baking dof specialization",
                     [](auto& in) { in.jit_options.basis_baking.force_dof_specialization = false; });
    expectKeyChanges("basis baking max qpts",
                     [](auto& in) { in.jit_options.basis_baking.max_baked_qpts += 1u; });
    expectKeyChanges("basis baking max dofs",
                     [](auto& in) { in.jit_options.basis_baking.max_baked_dofs += 1u; });
    expectKeyChanges("basis baking max entries",
                     [](auto& in) { in.jit_options.basis_baking.max_baked_entries += 1u; });
}

TEST(JITCacheKey, NonCodegenOptionsDoNotChangeKernelKey)
{
    expectKeyUnchanged("JIT enable", [](auto& in) { in.jit_options.enable = !in.jit_options.enable; });
    expectKeyUnchanged("cache kernels", [](auto& in) { in.jit_options.cache_kernels = !in.jit_options.cache_kernels; });
    expectKeyUnchanged("cache directory", [](auto& in) { in.jit_options.cache_directory = "different-cache"; });
    expectKeyUnchanged("cache diagnostics", [](auto& in) { in.jit_options.cache_diagnostics = true; });
    expectKeyUnchanged("in-memory cache size", [](auto& in) { in.jit_options.max_in_memory_kernels += 1u; });
    expectKeyUnchanged("dump KernelIR", [](auto& in) { in.jit_options.dump_kernel_ir = true; });
    expectKeyUnchanged("dump LLVM IR", [](auto& in) { in.jit_options.dump_llvm_ir = true; });
    expectKeyUnchanged("dump optimized LLVM IR", [](auto& in) { in.jit_options.dump_llvm_ir_optimized = true; });
    expectKeyUnchanged("dump directory", [](auto& in) { in.jit_options.dump_directory = "different-dump"; });
    expectKeyUnchanged("tensor logging", [](auto& in) { in.jit_options.tensor.log_decisions = true; });
    expectKeyUnchanged("specialization enable", [](auto& in) { in.jit_options.specialization.enable = true; });
    expectKeyUnchanged("specialize qpts", [](auto& in) { in.jit_options.specialization.specialize_n_qpts = false; });
    expectKeyUnchanged("specialize dofs", [](auto& in) { in.jit_options.specialization.specialize_dofs = true; });
    expectKeyUnchanged("max specialized qpts",
                       [](auto& in) { in.jit_options.specialization.max_specialized_n_qpts += 1u; });
    expectKeyUnchanged("max specialized dofs",
                       [](auto& in) { in.jit_options.specialization.max_specialized_dofs += 1u; });
    expectKeyUnchanged("max specialization variants",
                       [](auto& in) { in.jit_options.specialization.max_variants_per_kernel += 1u; });
    expectKeyUnchanged("unmatched text budget",
                       [](auto& in) { in.jit_options.specialization.text_budget_bytes = 999u; });
}

TEST(JITCacheKey, MatchedSpecializationInputsChangeKey)
{
    auto base = baseInputs();
    auto spec = matchedSpecialization();
    base.specialization = &spec;
    EXPECT_NE(keyFor(base), keyFor(baseInputs()));

    expectMatchedSpecializationKeyChanges("n qpts minus", [](auto&, auto& spec) { spec.n_qpts_minus = 6u; });
    expectMatchedSpecializationKeyChanges("test dofs minus",
                                          [](auto&, auto& spec) { spec.n_test_dofs_minus = 12u; });
    expectMatchedSpecializationKeyChanges("trial dofs minus",
                                          [](auto&, auto& spec) { spec.n_trial_dofs_minus = 13u; });
    expectMatchedSpecializationKeyChanges("n qpts plus", [](auto&, auto& spec) { spec.n_qpts_plus = 7u; });
    expectMatchedSpecializationKeyChanges("test dofs plus",
                                          [](auto&, auto& spec) { spec.n_test_dofs_plus = 14u; });
    expectMatchedSpecializationKeyChanges("trial dofs plus",
                                          [](auto&, auto& spec) { spec.n_trial_dofs_plus = 15u; });
    expectMatchedSpecializationKeyChanges("affine flag", [](auto&, auto& spec) { spec.is_affine = false; });
    expectMatchedSpecializationKeyChanges("baked basis presence", [](auto&, auto& spec) {
        spec.baked_basis.enabled = true;
        spec.baked_basis.geometry_affine = true;
        spec.baked_basis.hash = 0x1234'5678'9abc'def0ULL;
    });
    expectMatchedSpecializationKeyChanges("baked basis hash", [](auto&, auto& spec) {
        spec.baked_basis.enabled = true;
        spec.baked_basis.geometry_affine = true;
        spec.baked_basis.hash = 0x1234'5678'9abc'def1ULL;
    });
    expectMatchedSpecializationKeyChanges("baked basis geometry mode", [](auto&, auto& spec) {
        spec.baked_basis.enabled = true;
        spec.baked_basis.geometry_affine = false;
        spec.baked_basis.hash = 0x1234'5678'9abc'def0ULL;
    });
    expectMatchedSpecializationKeyChanges("optional presence",
                                          [](auto&, auto& spec) { spec.n_qpts_minus.reset(); });
    expectMatchedSpecializationKeyChanges("matched text budget",
                                          [](auto& in, auto&) {
                                              in.jit_options.specialization.text_budget_bytes = 4096u;
                                          });
}

TEST(JITCacheKey, UnmatchedSpecializationInputsAreIgnored)
{
    const auto no_spec = baseInputs();

    auto unmatched_spec = matchedSpecialization();
    unmatched_spec.domain = IntegralDomain::Boundary;

    auto unmatched = no_spec;
    unmatched.specialization = &unmatched_spec;
    EXPECT_EQ(keyFor(unmatched), keyFor(no_spec));

    auto changed_spec = unmatched_spec;
    changed_spec.n_qpts_minus = 99u;
    changed_spec.n_test_dofs_minus.reset();
    changed_spec.is_affine = !changed_spec.is_affine;
    changed_spec.baked_basis.enabled = true;
    changed_spec.baked_basis.geometry_affine = true;
    changed_spec.baked_basis.hash = 0xfeed'face'1234'5678ULL;

    auto changed = no_spec;
    changed.specialization = &changed_spec;
    changed.jit_options.specialization.text_budget_bytes = 7777u;
    EXPECT_EQ(keyFor(changed), keyFor(no_spec));
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
