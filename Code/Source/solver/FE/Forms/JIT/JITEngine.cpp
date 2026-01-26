/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITEngine.h"

#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Forms/JIT/ExternalCalls.h"
#include "Forms/JIT/LLVMJITBuildInfo.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>

#if __has_include(<llvm/ExecutionEngine/Orc/Debugging/PerfJITEventListener.h>)
#include <llvm/ExecutionEngine/Orc/Debugging/PerfJITEventListener.h>
#define SVMP_FE_LLVM_HAS_PERF_LISTENER 1
#else
#define SVMP_FE_LLVM_HAS_PERF_LISTENER 0
#endif
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

#if SVMP_FE_ENABLE_LLVM_JIT
[[nodiscard]] int sanitizeOptLevel(int level) noexcept
{
    return std::clamp(level, 0, 3);
}

[[nodiscard]] std::string llvmErrorToString(llvm::Error err)
{
    return llvm::toString(std::move(err));
}

[[nodiscard]] llvm::OptimizationLevel toLLVMOptLevel(int opt_level) noexcept
{
    switch (sanitizeOptLevel(opt_level)) {
        case 0:
            return llvm::OptimizationLevel::O0;
        case 1:
            return llvm::OptimizationLevel::O1;
        case 2:
            return llvm::OptimizationLevel::O2;
        case 3:
        default:
            return llvm::OptimizationLevel::O3;
    }
}

class InMemoryObjectCache final : public llvm::ObjectCache {
public:
    InMemoryObjectCache() = default;
    ~InMemoryObjectCache() override = default;

    InMemoryObjectCache(const InMemoryObjectCache&) = delete;
    InMemoryObjectCache& operator=(const InMemoryObjectCache&) = delete;

    void notifyObjectCompiled(const llvm::Module* module, llvm::MemoryBufferRef obj_buffer) override
    {
        if (module == nullptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto copy = llvm::MemoryBuffer::getMemBufferCopy(obj_buffer.getBuffer(), obj_buffer.getBufferIdentifier());
        objects_by_module_id_[module->getModuleIdentifier()] = std::move(copy);
    }

    [[nodiscard]] std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module* module) override
    {
        if (module == nullptr) {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = objects_by_module_id_.find(module->getModuleIdentifier());
        if (it == objects_by_module_id_.end() || !it->second) {
            return nullptr;
        }
        return llvm::MemoryBuffer::getMemBufferCopy(it->second->getBuffer(), it->second->getBufferIdentifier());
    }

private:
    std::mutex mutex_{};
    std::unordered_map<std::string, std::unique_ptr<llvm::MemoryBuffer>> objects_by_module_id_{};
};

class FileSystemObjectCache final : public llvm::ObjectCache {
public:
    explicit FileSystemObjectCache(std::filesystem::path directory)
        : directory_(std::move(directory))
    {
    }

    ~FileSystemObjectCache() override = default;

    FileSystemObjectCache(const FileSystemObjectCache&) = delete;
    FileSystemObjectCache& operator=(const FileSystemObjectCache&) = delete;

    void notifyObjectCompiled(const llvm::Module* module, llvm::MemoryBufferRef obj_buffer) override
    {
        if (module == nullptr) {
            return;
        }

        const std::string module_id = module->getModuleIdentifier();
        auto copy =
            llvm::MemoryBuffer::getMemBufferCopy(obj_buffer.getBuffer(), obj_buffer.getBufferIdentifier());

        {
            std::lock_guard<std::mutex> lock(mutex_);
            objects_by_module_id_[module_id] = std::move(copy);
        }

        if (directory_.empty()) {
            return;
        }

        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        if (ec) {
            return;
        }

        const auto final_path = cachePathForModuleId(module_id);
        if (std::filesystem::exists(final_path, ec) && !ec) {
            return;
        }

        const auto tmp_path = tempPathFor(final_path);

        try {
            {
                std::ofstream os(tmp_path, std::ios::binary | std::ios::trunc);
                if (!os.good()) {
                    return;
                }
                os.write(obj_buffer.getBufferStart(), static_cast<std::streamsize>(obj_buffer.getBufferSize()));
                os.flush();
                if (!os.good()) {
                    return;
                }
            }

            std::filesystem::rename(tmp_path, final_path, ec);
            if (ec) {
                std::filesystem::remove(tmp_path, ec);
            }
        } catch (...) {
            std::filesystem::remove(tmp_path, ec);
        }
    }

    [[nodiscard]] std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module* module) override
    {
        if (module == nullptr) {
            return nullptr;
        }

        const std::string module_id = module->getModuleIdentifier();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = objects_by_module_id_.find(module_id);
            if (it != objects_by_module_id_.end() && it->second) {
                return llvm::MemoryBuffer::getMemBufferCopy(it->second->getBuffer(), it->second->getBufferIdentifier());
            }
        }

        if (directory_.empty()) {
            return nullptr;
        }

        std::error_code ec;
        const auto path = cachePathForModuleId(module_id);
        if (!std::filesystem::exists(path, ec) || ec) {
            return nullptr;
        }

        auto buf_or_err = llvm::MemoryBuffer::getFile(path.string(), /*IsText=*/false);
        if (!buf_or_err) {
            return nullptr;
        }

        return std::move(*buf_or_err);
    }

private:
    [[nodiscard]] static std::string sanitizeFilename(std::string_view s)
    {
        std::string out;
        out.reserve(s.size());
        for (const char ch : s) {
            const bool ok =
                (ch >= 'a' && ch <= 'z') ||
                (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') ||
                (ch == '_' || ch == '-' || ch == '.');
            out.push_back(ok ? ch : '_');
        }
        return out;
    }

    [[nodiscard]] std::filesystem::path cachePathForModuleId(std::string_view module_id) const
    {
        return directory_ / (sanitizeFilename(module_id) + ".objcache");
    }

    [[nodiscard]] std::filesystem::path tempPathFor(const std::filesystem::path& final_path) noexcept
    {
        const auto now =
            static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const auto salt = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this));
        const auto ctr = tmp_counter_.fetch_add(1u, std::memory_order_relaxed);
        const auto nonce = now ^ salt ^ ctr;
        return std::filesystem::path(final_path.string() + ".tmp." + std::to_string(nonce));
    }

    std::filesystem::path directory_{};
    std::mutex mutex_{};
    std::unordered_map<std::string, std::unique_ptr<llvm::MemoryBuffer>> objects_by_module_id_{};
    std::atomic<std::uint64_t> tmp_counter_{0u};
};

[[nodiscard]] std::string hostCPUName()
{
    return llvm::sys::getHostCPUName().str();
}

[[nodiscard]] std::string hostCPUFeaturesString()
{
    llvm::StringMap<bool> feats;
    if (!llvm::sys::getHostCPUFeatures(feats)) {
        return {};
    }

    std::vector<std::string> enabled;
    enabled.reserve(feats.size());
    for (const auto& kv : feats) {
        if (kv.getValue()) {
            enabled.emplace_back(kv.getKey().str());
        }
    }
    std::sort(enabled.begin(), enabled.end());

    std::string out;
    for (std::size_t i = 0; i < enabled.size(); ++i) {
        if (i != 0u) out += ",";
        out += enabled[i];
    }
    return out;
}

void initializeLLVMOnce()
{
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        if (llvm::InitializeNativeTarget()) {
            FE_THROW(FEException, "LLVM JIT: InitializeNativeTarget failed");
        }
        if (llvm::InitializeNativeTargetAsmPrinter()) {
            FE_THROW(FEException, "LLVM JIT: InitializeNativeTargetAsmPrinter failed");
        }
        if (llvm::InitializeNativeTargetAsmParser()) {
            FE_THROW(FEException, "LLVM JIT: InitializeNativeTargetAsmParser failed");
        }
    });
}

void configureObjectCache(llvm::orc::IRCompileLayer& compile_layer,
                          llvm::ObjectCache* cache)
{
    if (cache == nullptr) {
        return;
    }

    // LLVM 14: IRCompileLayer has no setObjectCache; configure via the compiler.
    auto& compiler = compile_layer.getCompiler();
    if (auto* simple = dynamic_cast<llvm::orc::SimpleCompiler*>(&compiler)) {
        simple->setObjectCache(cache);
        return;
    }
    if (auto* concurrent = dynamic_cast<llvm::orc::ConcurrentIRCompiler*>(&compiler)) {
        concurrent->setObjectCache(cache);
        return;
    }
}

void configureTransformLayer(llvm::orc::IRTransformLayer& transform_layer,
                             int opt_level,
                             bool vectorize)
{
    const int sanitized_opt_level = sanitizeOptLevel(opt_level);
    if (sanitized_opt_level == 0) {
        return;
    }

    const llvm::OptimizationLevel llvm_opt_level = toLLVMOptLevel(sanitized_opt_level);

    llvm::orc::IRTransformLayer::TransformFunction transform =
        [llvm_opt_level, vectorize](llvm::orc::ThreadSafeModule tsm,
                                    llvm::orc::MaterializationResponsibility& /*responsibility*/)
            -> llvm::Expected<llvm::orc::ThreadSafeModule> {
        tsm.withModuleDo([&](llvm::Module& module) {
            llvm::LoopAnalysisManager loop_analysis_manager;
            llvm::FunctionAnalysisManager function_analysis_manager;
            llvm::CGSCCAnalysisManager cgscc_analysis_manager;
            llvm::ModuleAnalysisManager module_analysis_manager;

            llvm::PipelineTuningOptions tuning_options;
            tuning_options.LoopVectorization = vectorize;
            tuning_options.SLPVectorization = vectorize;

            if constexpr (requires(llvm::PipelineTuningOptions opts) { llvm::PassBuilder(nullptr, opts); }) {
                llvm::PassBuilder pass_builder(nullptr, tuning_options);

                pass_builder.registerModuleAnalyses(module_analysis_manager);
                pass_builder.registerCGSCCAnalyses(cgscc_analysis_manager);
                pass_builder.registerFunctionAnalyses(function_analysis_manager);
                pass_builder.registerLoopAnalyses(loop_analysis_manager);
                pass_builder.crossRegisterProxies(loop_analysis_manager,
                                                 function_analysis_manager,
                                                 cgscc_analysis_manager,
                                                 module_analysis_manager);

                llvm::ModulePassManager module_pass_manager;
                if (llvm_opt_level == llvm::OptimizationLevel::O0) {
                    module_pass_manager = pass_builder.buildO0DefaultPipeline(llvm_opt_level);
                } else {
                    module_pass_manager = pass_builder.buildPerModuleDefaultPipeline(llvm_opt_level);
                }
                module_pass_manager.run(module, module_analysis_manager);
                return;
            }

            llvm::PassBuilder pass_builder;
            pass_builder.registerModuleAnalyses(module_analysis_manager);
            pass_builder.registerCGSCCAnalyses(cgscc_analysis_manager);
            pass_builder.registerFunctionAnalyses(function_analysis_manager);
            pass_builder.registerLoopAnalyses(loop_analysis_manager);
            pass_builder.crossRegisterProxies(loop_analysis_manager,
                                             function_analysis_manager,
                                             cgscc_analysis_manager,
                                             module_analysis_manager);

            llvm::ModulePassManager module_pass_manager;
            if (llvm_opt_level == llvm::OptimizationLevel::O0) {
                module_pass_manager = pass_builder.buildO0DefaultPipeline(llvm_opt_level);
            } else {
                module_pass_manager = pass_builder.buildPerModuleDefaultPipeline(llvm_opt_level);
            }
            module_pass_manager.run(module, module_analysis_manager);
        });

        return std::move(tsm);
    };

    transform_layer.setTransform(std::move(transform));
}

void configureProcessSymbolResolution(llvm::orc::LLJIT& jit)
{
    auto generator_expected =
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit.getDataLayout().getGlobalPrefix());

    if (!generator_expected) {
        FE_THROW(FEException, "LLVM JIT: failed to create DynamicLibrarySearchGenerator: " +
                                  llvmErrorToString(generator_expected.takeError()));
    }

    // LLVM 14: addGenerator returns a reference and does not report errors.
    jit.getMainJITDylib().addGenerator(std::move(*generator_expected));
}

void registerExternalCallSymbols(llvm::orc::LLJIT& jit)
{
    llvm::orc::MangleAndInterner mangle(jit.getExecutionSession(), jit.getDataLayout());
    llvm::orc::SymbolMap symbols;

    auto add = [&](const char* name, auto fn_ptr) {
        symbols[mangle(name)] =
            llvm::JITEvaluatedSymbol(llvm::pointerToJITTargetAddress(fn_ptr),
                                     llvm::JITSymbolFlags::Exported);
    };

    add("svmp_fe_jit_coeff_eval_scalar_v1", &svmp_fe_jit_coeff_eval_scalar_v1);
    add("svmp_fe_jit_coeff_eval_vector_v1", &svmp_fe_jit_coeff_eval_vector_v1);
    add("svmp_fe_jit_coeff_eval_matrix_v1", &svmp_fe_jit_coeff_eval_matrix_v1);
    add("svmp_fe_jit_coeff_eval_tensor3_v1", &svmp_fe_jit_coeff_eval_tensor3_v1);
    add("svmp_fe_jit_coeff_eval_tensor4_v1", &svmp_fe_jit_coeff_eval_tensor4_v1);
    add("svmp_fe_jit_constitutive_eval_v1", &svmp_fe_jit_constitutive_eval_v1);
    add("svmp_fe_jit_constitutive_eval_batch_v1", &svmp_fe_jit_constitutive_eval_batch_v1);

    auto err = jit.getMainJITDylib().define(llvm::orc::absoluteSymbols(std::move(symbols)));
    if (err) {
        FE_THROW(FEException, "LLVM JIT: failed to register external-call symbols: " +
                                  llvmErrorToString(std::move(err)));
    }
}

void configureEventListeners(llvm::orc::LLJIT& jit,
                             std::unique_ptr<llvm::JITEventListener>& gdb_listener,
                             std::unique_ptr<llvm::JITEventListener>& perf_listener)
{
    if (!gdb_listener) {
        gdb_listener.reset(llvm::JITEventListener::createGDBRegistrationListener());
    }
    if (gdb_listener) {
        if (auto* layer = dynamic_cast<llvm::orc::RTDyldObjectLinkingLayer*>(&jit.getObjLinkingLayer())) {
            layer->registerJITEventListener(*gdb_listener);
        }
    }

#if SVMP_FE_LLVM_HAS_PERF_LISTENER
    if (!perf_listener) {
        llvm::JITEventListener* listener = nullptr;
        if constexpr (requires { llvm::JITEventListener::createPerfJITEventListener(); }) {
            listener = llvm::JITEventListener::createPerfJITEventListener();
        } else if constexpr (requires { llvm::createPerfJITEventListener(); }) {
            listener = llvm::createPerfJITEventListener();
        } else if constexpr (requires { llvm::orc::createPerfJITEventListener(); }) {
            listener = llvm::orc::createPerfJITEventListener();
        }
        perf_listener.reset(listener);
    }
    if (perf_listener) {
        if (auto* layer = dynamic_cast<llvm::orc::RTDyldObjectLinkingLayer*>(&jit.getObjLinkingLayer())) {
            layer->registerJITEventListener(*perf_listener);
        }
    }
#else
    (void)perf_listener;
#endif
}

[[nodiscard]] std::unique_ptr<llvm::orc::LLJIT> createLLJIT(const JITOptions& options,
                                                           std::string& out_target_triple,
                                                           std::string& out_data_layout)
{
    auto jtmb_expected = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb_expected) {
        FE_THROW(FEException, "LLVM JIT: failed to detect host target machine: " +
                                  llvmErrorToString(jtmb_expected.takeError()));
    }

    auto jtmb = std::move(*jtmb_expected);
    switch (sanitizeOptLevel(options.optimization_level)) {
        case 0:
            jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::None);
            break;
        case 1:
            jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::Less);
            break;
        case 2:
            jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::Default);
            break;
        case 3:
        default:
            jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);
            break;
    }
    out_target_triple = jtmb.getTargetTriple().str();

    llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(jtmb));

    auto jit_expected = builder.create();
    if (!jit_expected) {
        FE_THROW(FEException, "LLVM JIT: LLJITBuilder::create failed: " +
                                  llvmErrorToString(jit_expected.takeError()));
    }

    auto jit = std::move(*jit_expected);
    out_data_layout = jit->getDataLayout().getStringRepresentation();

    configureProcessSymbolResolution(*jit);
    registerExternalCallSymbols(*jit);

    configureTransformLayer(jit->getIRTransformLayer(),
                            sanitizeOptLevel(options.optimization_level),
                            options.vectorize);

    return jit;
}
#endif

} // namespace

struct JITEngine::Impl {
#if SVMP_FE_ENABLE_LLVM_JIT
    JITOptions options{};
    std::unique_ptr<llvm::JITEventListener> gdb_listener{};
    std::unique_ptr<llvm::JITEventListener> perf_listener{};
    std::unique_ptr<llvm::ObjectCache> object_cache{};
    std::unique_ptr<llvm::orc::LLJIT> jit{};
    std::string target_triple{};
    std::string data_layout{};
    std::string cpu_name{};
    std::string cpu_features{};
#endif
};

std::unique_ptr<JITEngine> JITEngine::create(const JITOptions& options)
{
#if SVMP_FE_ENABLE_LLVM_JIT
    try {
        initializeLLVMOnce();

        auto engine = std::unique_ptr<JITEngine>(new JITEngine());
        engine->impl_ = std::make_unique<Impl>();
        engine->impl_->options = options;

        std::string triple;
        std::string data_layout;
        auto jit = createLLJIT(options, triple, data_layout);

        engine->impl_->jit = std::move(jit);
        engine->impl_->target_triple = std::move(triple);
        engine->impl_->data_layout = std::move(data_layout);
        engine->impl_->cpu_name = hostCPUName();
        engine->impl_->cpu_features = hostCPUFeaturesString();

        configureEventListeners(*engine->impl_->jit,
                                engine->impl_->gdb_listener,
                                engine->impl_->perf_listener);

        if (!options.cache_directory.empty()) {
            engine->impl_->object_cache = std::make_unique<FileSystemObjectCache>(options.cache_directory);
            configureObjectCache(engine->impl_->jit->getIRCompileLayer(), engine->impl_->object_cache.get());
        } else if (options.cache_kernels) {
            engine->impl_->object_cache = std::make_unique<InMemoryObjectCache>();
            configureObjectCache(engine->impl_->jit->getIRCompileLayer(), engine->impl_->object_cache.get());
        }

        FE_LOG_INFO("LLVM JIT: OrcJIT initialized (LLVM " + llvmVersionString() +
                    ", triple=" + engine->impl_->target_triple +
                    ", opt=" + std::to_string(sanitizeOptLevel(options.optimization_level)) +
                    ", vectorize=" + std::string(options.vectorize ? "true" : "false") + ")");

        return engine;
    } catch (const std::exception& e) {
        FE_LOG_WARNING(std::string("LLVM JIT: failed to initialize OrcJIT: ") + e.what());
        return {};
    }
#else
    (void)options;
    return {};
#endif
}

JITEngine::~JITEngine() = default;

bool JITEngine::available() const noexcept
{
    return impl_ != nullptr;
}

void JITEngine::addModule(llvm::orc::ThreadSafeModule&& module)
{
#if SVMP_FE_ENABLE_LLVM_JIT
    std::lock_guard<std::mutex> lock(mutex_);
    FE_THROW_IF(impl_ == nullptr || impl_->jit == nullptr, "JITEngine::addModule: engine is not available");

    auto err = impl_->jit->addIRModule(std::move(module));
    if (err) {
        FE_THROW(FEException, "LLVM JIT: addIRModule failed: " + llvmErrorToString(std::move(err)));
    }
#else
    (void)module;
    FE_THROW(FEException, "JITEngine::addModule: FE was built without LLVM JIT support");
#endif
}

JITEngine::SymbolAddress JITEngine::lookup(std::string_view name)
{
#if SVMP_FE_ENABLE_LLVM_JIT
    std::lock_guard<std::mutex> lock(mutex_);
    FE_THROW_IF(impl_ == nullptr || impl_->jit == nullptr, "JITEngine::lookup: engine is not available");

    auto sym_expected = impl_->jit->lookup(std::string(name));
    if (!sym_expected) {
        FE_THROW(FEException, "LLVM JIT: symbol lookup failed for '" + std::string(name) + "': " +
                                  llvmErrorToString(sym_expected.takeError()));
    }
    return static_cast<SymbolAddress>(sym_expected->getAddress());
#else
    (void)name;
    FE_THROW(FEException, "JITEngine::lookup: FE was built without LLVM JIT support");
#endif
}

std::string JITEngine::targetTriple() const
{
#if SVMP_FE_ENABLE_LLVM_JIT
    std::lock_guard<std::mutex> lock(mutex_);
    if (impl_ == nullptr) return {};
    return impl_->target_triple;
#else
    return {};
#endif
}

std::string JITEngine::dataLayoutString() const
{
#if SVMP_FE_ENABLE_LLVM_JIT
    std::lock_guard<std::mutex> lock(mutex_);
    if (impl_ == nullptr) return {};
    return impl_->data_layout;
#else
    return {};
#endif
}

std::string JITEngine::cpuName() const
{
#if SVMP_FE_ENABLE_LLVM_JIT
    std::lock_guard<std::mutex> lock(mutex_);
    if (impl_ == nullptr) return {};
    return impl_->cpu_name;
#else
    return {};
#endif
}

std::string JITEngine::cpuFeaturesString() const
{
#if SVMP_FE_ENABLE_LLVM_JIT
    std::lock_guard<std::mutex> lock(mutex_);
    if (impl_ == nullptr) return {};
    return impl_->cpu_features;
#else
    return {};
#endif
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
