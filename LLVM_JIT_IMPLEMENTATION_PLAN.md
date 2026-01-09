# LLVM OrcJIT Implementation Plan for FE/Forms

## 1. Objective
Replace the current interpreter-based `FormKernel` execution model with a high-performance, Just-In-Time (JIT) compiled backend using **LLVM OrcJIT**. This infrastructure aims to match or exceed the performance of the legacy hand-written C++ solver while retaining the flexibility of the symbolic `Forms` library.

---

## 2. Directory Structure
A new subdirectory will be established to encapsulate the JIT logic and LLVM dependencies:

`Code/Source/solver/FE/Forms/JIT/`

---

## 3. Core Components & New Files

### A. JIT Engine Wrapper
**Files:**
*   `Code/Source/solver/FE/Forms/JIT/JITEngine.h`
*   `Code/Source/solver/FE/Forms/JIT/JITEngine.cpp`

**Responsibilities:**
*   **Encapsulation:** Wraps the complex `llvm::orc::LLJIT` class to provide a simplified API.
*   **Initialization:** Handles `llvm::InitializeNativeTarget` and target machine setup.
*   **Module Management:** Provides `addModule(std::unique_ptr<llvm::Module>)` to submit generated IR for compilation.
*   **Symbol Resolution:** Provides `lookup(std::string name)` to retrieve the memory address (function pointer) of compiled kernels.
*   **Optimization:** Configures the `llvm::FunctionPassManager` for optimization levels (O2/O3).

### B. Code Generator (AST Visitor)
**Files:**
*   `Code/Source/solver/FE/Forms/JIT/LLVMGen.h`
*   `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`

**Responsibilities:**
*   **Translation:** Recursively visits the `FormExpr` AST nodes and emits corresponding LLVM IR instructions.
*   **Intermediate Optimization:** Implements a lightweight DAG pass before LLVM emission to perform CSE (Common Subexpression Elimination) at the tensor level and hoist loop invariants.
*   **Tensor Scheduling:** Manages the orchestration of nested loops for tensor contractions, prioritizing SIMD-friendly access patterns and data reuse.
*   **Context Management:** Owns the `llvm::LLVMContext`, `llvm::Module`, and `llvm::IRBuilder`.
*   **Type System Mapping:** Maps `Forms::Value<T>` types (Scalar, Vector, Tensor) to LLVM types (double, `<n x double>` vectors, or flat arrays). **Shape tracking** is performed at compile-time to determine the IR structure.
*   **Kernel ABI:** Uses a versioned, packed struct to enforce alignment and aliasing guarantees (improving optimization):
    ```c
    struct KernelArgs {
        int n_qpts;
        double* element_matrix;             // Output: flattened element matrix/vector
        const double* const* solution_fields;// Input: pointers to DOF values (restrict)
        const double* geometry_data;         // Input: Jacobians, dets, points (restrict)
        const double* constants;             // Input: Time step, physical constants
        void* user_context;                  // Input: Pointer for callbacks
    };
    // Signature: void kernel(const KernelArgs* __restrict args);
    ```
*   **Specialization:** Generates specialized kernels for common quadrature counts (e.g., `kernel_NQ4`, `kernel_NQ8`) to allow loop unrolling and hard-coded array sizes, falling back to a generic loop only when necessary.
*   **Vectorization Hints:** Emits LLVM metadata (`llvm::MDNode`, `llvm.assume`) to enforce pointer alignment and encourage SIMD generation.

#### Node Lowering Strategy (Mapping `FormExprType`)

| Category | FormExpr Types | Lowering Strategy |
| :--- | :--- | :--- |
| **Terminals** | `Constant`, `Identity` | Emit `llvm::ConstantFP` or constant arrays. |
| **Coefficients** | `Coefficient` | Prefer `Load` from `KernelArgs.constants` when the coefficient is a known scalar parameter; otherwise emit a call to a C ABI “coefficient trampoline” using `KernelArgs.user_context` to dispatch to the stored callable(s). |
| **Geometry** | `Coordinate`, `Jacobian`, `Normal` | `Load` from the `geometry_data` pointer passed in arguments. |
| **Fields** | `TrialFunction`, `DiscreteField` | `Load` from `solution_fields` based on pre-calculated offsets. |
| **Basis** | `TestFunction` | `Load` basis function values/gradients from pre-computed tables in `geometry_data` or a dedicated basis pointer. |
| **Algebra** | `Add`, `Multiply`, `Power` | Emit `fadd`, `fmul`; call `llvm.pow` intrinsic. Vector/Tensor ops are expanded loops or SIMD vectors. |
| **Calculus** | `Gradient`, `Divergence` | Requires basis derivatives. Maps to `Load` of `GradPhi` from basis tables * combined with field coefficients. |
| **Logic** | `Conditional`, `Less`, `Greater` | Emit `fcmp` instructions and `Select` instructions (avoid branching for SIMD). |
| **Tensors** | `Det`, `Inv`, `Trace` | Inlined implementation of 3x3 determinant/inverse using standard formulas (Cramer's rule or similar). |

### C. JIT Execution Kernel
**Files:**
*   `Code/Source/solver/FE/Forms/JIT/JITFormKernel.h`

**Responsibilities:**
*   **Container:** Acts as the executable object created by the compiler.
*   **Function Pointer:** Stores the raw function pointer `KernelFn` resolved by the `JITEngine`.
*   **Execution:** Implements the `execute(AssemblyContext&)` method. This method marshals the high-level C++ data structures (Vectors, Arrays) from the `AssemblyContext` into the raw pointer arrays expected by the generated kernel ABI.

### D. JIT Compiler Facade
**Files:**
*   `Code/Source/solver/FE/Forms/JIT/JITCompiler.h`
*   `Code/Source/solver/FE/Forms/JIT/JITCompiler.cpp`

**Responsibilities:**
*   **Entry Point:** Serves as the high-level API called by `FormCompiler`.
*   **Workflow:**
    1.  Receives `FormIR`.
    2.  Instantiates `LLVMGen`.
    3.  Generates the `llvm::Module`.
    4.  Submits module to `JITEngine`.
    5.  Resolves symbol address.
    6.  Returns a configured `JITFormKernel`.

---

## 4. Build System Updates
**File:** `Code/Source/solver/FE/Forms/CMakeLists.txt`

**Required Changes:**
1.  **Find LLVM:** Use `find_package(LLVM REQUIRED CONFIG)`.
2.  **Component Mapping:** Map required components to libraries:
    *   `Core`, `OrcJIT`, `Support`, `Native` (Target codegen), `Analysis`, `Passes`.
3.  **Linking:** Link the `Forms` library against these LLVM components.

---

## 5. Integration Strategy

**File:** `Code/Source/solver/FE/Forms/FormCompiler.cpp`

**Logic:**
Update `compileImpl` to dispatch based on options:
```cpp
if (options.jit.enable) {
    return JITCompiler::compile(ir, options);
} else {
    return InterpreterCompiler::compile(ir); // Existing logic
}
```

---

## 6. Parallelism & MPI Compatibility

The LLVM JIT infrastructure is designed to be fully compatible with the distributed nature of the solver:

*   **MPI (Distributed Memory):** "Shared-Nothing" architecture. Each MPI rank initializes its own `JITEngine` and compiles the kernels locally at startup. Since the math is identical, every rank generates the same machine code in its own memory space. No inter-process communication is required for compilation.
*   **OpenMP (Shared Memory):** Thread-safe execution. The generated kernels are pure functions (re-entrant) operating only on stack variables and passed arguments. Multiple threads within a single MPI rank can safely call the generated kernel function pointer simultaneously for different elements.

---

## 7. Advanced Features Implementation

### A. Automatic Differentiation (AD) Strategy
Since `NonlinearFormKernel` relies on Forward Mode AD, the JIT must support computing Jacobians.
**Strategy:** **Symbolic Differentiation pre-lowering.**
*   Instead of implementing `Dual<double>` in LLVM IR (which is complex and bloated), the `FormCompiler` will perform symbolic differentiation on the `FormExpr` tree *before* passing it to `LLVMGen`.
*   The `derivative(Expr, var)` function will generate a new `FormExpr` representing the linearized form (Jacobian).
*   The JIT then compiles this explicit derivative expression like any other form.
*   *Fallback:* If symbolic differentiation is too complex for certain nodes, we can implement Dual numbers as a struct `{double primal, double tangent}` in IR, but symbolic is preferred for performance.

### B. Constitutive Model Integration
Constitutive models are complex C++ classes (`Code/Source/solver/FE/Forms/ConstitutiveModel.h`). Re-implementing them in IR is duplicated effort.
**Strategy:** **Function Callbacks (Trampolines).**
1.  The `kernel_signature` accepts a `void* user_context` (pointer to `ConstitutiveModel`).
2.  `LLVMGen` emits a `CallInst` to an external symbol `wrapper_constitutive_eval`.
3.  The wrapper (written in C++) casts the context and calls `model->evaluate(...)`.
4.  *Performance Note:* This introduces overhead. For critical models (Newtonian), we should implementing "Intrinsic" versions directly in IR to inline them.

### C. Coefficient Callbacks
Similar to Constitutive Models, `std::function` coefficients cannot be inlined easily.
**Strategy:** Pass a table of function pointers and context pointers to the kernel. `LLVMGen` emits an indirect call to the function pointer.

### D. Einsum Integration
`Forms::einsum(expr)` handles index contraction.
**Strategy:** Run `einsum` **before** JIT compilation. The JIT receives the lowered, fully contracted scalar expressions. 

### E. Tensor Scheduling & Loop Optimizations
To match the performance of domain-specific compilers (like FEniCS/TSFC), the JIT backend implements several scheduling passes:
*   **Loop Tiling & Batching:** Instead of a single large loop over all quadrature points, the kernel processes points in small **SIMD-aligned chunks** (e.g., batches of 8 or 16). This keeps intermediate tensor values in L1 cache/registers and maximizes vector instruction throughput.
*   **Tensor-Level CSE:** Identifies repeated high-level operations (e.g., multiple uses of `grad(u)` or `inv(J)`) and ensures they are computed exactly once per quadrature point, storing results in stack-allocated arrays.
*   **Strength Reduction (Algebraic Elision):** Explicitly identifies `Identity`, `Zero`, and `Symmetric` tensor properties.
    *   `A * Identity` is elided.
    *   Symmetric contractions only calculate the upper/lower triangle.
*   **Explicit SIMD Emission:** For fixed-size 2D/3D tensor math, `LLVMGen` emits explicit LLVM vector instructions (`<4 x double>`) rather than scalar loops, ensuring the backend produces optimal code for small matrix operations without relying on the autovectorizer's heuristics.

### F. Expanded Node Lowering Table

| Category | FormExpr Types | Lowering Strategy |
| :--- | :--- | :--- |
| **Terminals** | `Constant`, `Identity` | Emit `llvm::ConstantFP` or constant arrays. |
| **Coefficients** | `Coefficient` | Prefer `Load` from `KernelArgs.constants`; otherwise call an external C ABI callback/trampoline via `KernelArgs.user_context`. |
| **Geometry** | `Coordinate`, `Jacobian`, `Normal` | `Load` from the `geometry_data` pointer passed in arguments. |
| **Fields** | `TrialFunction`, `DiscreteField` | `Load` from `solution_fields` based on pre-calculated offsets. |
| **Basis** | `TestFunction` | `Load` basis function values/gradients from pre-computed tables in `geometry_data` or a dedicated basis pointer. |
| **Algebra** | `Add`, `Multiply`, `Power` | Emit `fadd`, `fmul`; call `llvm.pow` intrinsic. Vector/Tensor ops are expanded loops or SIMD vectors. |
| **Calculus** | `Gradient`, `Divergence` | Requires basis derivatives. Maps to `Load` of `GradPhi` from basis tables * combined with field coefficients. |
| **Logic** | `Conditional`, `Less`, `Greater` | Emit `fcmp` instructions and `Select` instructions (avoid branching for SIMD). |
| **Tensors** | `Det`, `Inv`, `Trace` | Inlined implementation of 3x3 determinant/inverse using standard formulas (Cramer's rule or similar). |
| **DG Ops** | `Jump`, `Avg`, `Restrict±` | Emit logic to load from **two** contexts (`ctx_minus`, `ctx_plus`) passed via arguments. Requires generating a `kernel_dg` signature with double inputs. |
| **Time Deriv** | `TimeDerivative` | Load from time-history arrays. The kernel receives `u_n`, `u_dot_n` pointers. Implementation maps `dt(u)` to `u_dot` value loaded from memory. |
| **High Order** | `Tensor3`, `Tensor4` | Mapped to flat arrays. Indexing is computed at compile-time (e.g., `idx = i*27 + j*9 + k*3 + l`). |
| **Hessian** | `Hessian` | Requires second derivatives of basis functions. `Load` `HessPhi` from pre-computed basis tables. |
| **Special** | `Norm`, `Abs`, `Sign` | Map to `llvm.fabs`, `llvm.sqrt`. `Sign` implemented as `Select(val < 0 ? -1 : 1)`. |

---

## 8. Robustness & Quality Assurance

### A. Testing Strategy
*   **Unit Tests:** New test suite `Code/Source/solver/FE/Tests/Unit/JIT/` to compare JIT outputs against the interpreter.
    *   *Bitwise Check:* For integer/logic ops.
    *   *Tolerance Check:* `EXPECT_NEAR` for floating point math (JIT might optimize differently than C++ standard library).
*   **Regression Tests:** Run the full `svMultiPhysics` test suite (fluid, struct, FSI) with `jit.enable = true` to verify end-to-end correctness.
*   **Benchmarks:** Compare execution time of `assembleVector` using Interpreter vs. JIT for a range of mesh sizes.

### B. Error Handling & Fallback
The JIT compiler should never crash the simulation.
*   **Graceful Fallback:** Wrap the compilation process in a `try-catch` block. If `LLVMGen` encounters an unsupported node or LLVM throws an error, catch it, log a warning: *"JIT Compilation failed for form X. Falling back to interpreter."*, and return the legacy `FormKernel`.
*   **Validation:** Implement a `canCompile(FormIR)` pre-check pass to identify unsupported features early.

### C. Kernel Caching
To avoid recompiling identical forms (e.g., across time steps or different blocks with same physics):
*   **Hash Key:** Compute a robust hash of the `FormIR` (integrand structure + coefficients).
*   **Cache:** `std::unordered_map<size_t, JITFormKernel>` stored in `JITEngine`.
*   **Persistence:** Start with in-memory caching. Future work: On-disk caching (like `~/.cache/svmp/kernels`) using LLVM's `ObjectCache` API.

### D. Debugging & Profiling
*   **GDB Support:** Enable `llvm::JITEventListener::createGDBRegistrationListener()` so GDB can see JIT-ed function names.
*   **Profiling:** Enable `createPerfJITEventListener()` for Linux `perf` tool support.
*   **Debug Info:** Add a `debug_info` flag to `JITOptions`. If true, `LLVMGen` emits DWARF debug metadata (line numbers mapped to AST nodes) for stepping through generated code.

---

## 9. Phased Implementation Plan

1.  **Phase 1: Foundation**
    *   Setup CMake and `JITEngine`.
    *   Implement `LLVMGen` for basic arithmetic (`Add`, `Mul`, `Constant`) and Terminals (`Trial`, `Test`).
    *   Verify with a simple Poisson equation ($ \nabla u \cdot \nabla v $).
2.  **Phase 2: Geometry & Algebra**
    *   Add support for `Gradient`, `Jacobian`, `Det`, `Inv`.
    *   Implement tensor algebra lowering.
3.  **Phase 3: Nonlinearity**
    *   Implement symbolic differentiation pipeline or Dual number support.
    *   Add support for `Pow`, `Exp`, `Log`.
4.  **Phase 4: Constitutive & Callbacks**
    *   Implement the callback/trampoline mechanism for coefficients and material models.
5.  **Phase 5: Optimization**
    *   Enable vectorization (SIMD loop generation).
    *   Benchmark against legacy solver.

---

## 10. References & Resources

*   **LLVM Documentation:**
    *   [Building a JIT (Tutorial)](https://llvm.org/docs/tutorial/BuildingAJIT1.html)
    *   [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html)
*   **Similar Projects:**
    *   [FEniCS Project (UFL/FFCx)](https://fenicsproject.org) - Source-to-source compilation of variational forms.
    *   [Firedrake / TSFC](https://firedrakeproject.org/) - Two-Stage Form Compiler using SSA IR.
    *   [libCEED](https://libceed.org/) - Efficient high-order operator evaluation.
*   **Papers:**
    *   Kirby & Logg (2006): *"A Compiler for Variational Forms"* (Foundations of UFL).
    *   Homolya et al. (2018): *"TSFC: A Structure-Preserving Form Compiler"*.
