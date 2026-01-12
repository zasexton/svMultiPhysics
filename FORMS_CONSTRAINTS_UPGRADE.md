# Plan: Unified Weak Form & Seamless Constraints Upgrade

This plan outlines the refactoring of the Finite Element (FE) library to introduce a `WeakForm` abstraction. This unifies the definition of variational residuals and their associated strong constraints (Dirichlet BCs), preventing common integration errors (like missing constraints on Jacobians) and simplifying the Physics API.

## Goals

1.  **Unified Abstraction:** Create `forms::WeakForm` to bundle a residual `FormExpr` with its `StrongDirichlet` constraints.
2.  **Safety & Correctness:** Ensure strong constraints are automatically applied whenever a form is installed, fixing the "unconstrained Jacobian" bug.
3.  **Seamless API:** Simplify `PoissonModule` and future physics modules to use a single installation step.
4.  **Architectural Integrity:** Respect existing dependency layers (`Forms` depends on `Core`, `Systems` depends on `Forms` + `Constraints` + `Assembly`).

---

## 1. FE/Forms: Introduce `WeakForm`

**File:** `Code/Source/solver/FE/Forms/WeakForm.h` (New)

We will define a lightweight struct that acts as a container for the complete variational statement.

```cpp
namespace svmp::FE::forms {

/**
 * @brief Represents a complete variational problem (Weak Form + Strong Constraints).
 *
 * This container bundles the residual form R(u,v) with the set of strong (essential)
 * boundary conditions that must be enforced algebraically on the system.
 */
struct WeakForm {
    /// The weak form expression, e.g., (k*grad(u), grad(v)) * dx
    FormExpr residual;

    /// List of strong Dirichlet constraints (u = g on boundary) to be enforced.
    std::vector<bc::StrongDirichlet> strong_constraints;

    /// Helper to add a constraint
    void addDirichlet(bc::StrongDirichlet bc) {
        strong_constraints.push_back(std::move(bc));
    }
};

} // namespace svmp::FE::forms
```

**Actions:**
- [ ] Create `Code/Source/solver/FE/Forms/WeakForm.h`.
- [ ] Update `Code/Source/solver/FE/CMakeLists.txt` to include this new header.

---

## 2. FE/Systems: Update `FormsInstaller`

**File:** `Code/Source/solver/FE/Systems/FormsInstaller.h` & `.cpp`

We will add a new `installWeakForm` overload that accepts the `WeakForm` bundle. This function will serve as the "single source of truth" for how to lower a problem description into the system.

**New Signature:**
```cpp
KernelPtr installWeakForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::WeakForm& form,
    const FormInstallOptions& options = {});
```

**Implementation Logic:**
1.  **Install Constraints:** Iterate through `form.strong_constraints` and add them to the system via `system.addSystemConstraint(...)` (wrapping them in `StrongDirichletConstraint`).
    *   *Note:* Constraints should only be added *once* per system if they are identical, but `FESystem` handles duplicate constraint definitions gracefully (or we can add a check). For `residual` vs `jacobian`, the constraints are the same.
2.  **Install Residual:** Call the existing `installResidualForm` with `form.residual`.

**Actions:**
- [ ] Modify `FormsInstaller.h` to include `Forms/WeakForm.h` and declare `installWeakForm`.
- [ ] Modify `FormsInstaller.cpp` to implement `installWeakForm`.
    - Reuse existing `installStrongDirichlet` logic.
    - Forward to `installResidualForm`.

---

## 3. FE/Constraints: Review & Cleanup (Low Impact)

The `Constraints` folder remains the "Algebraic" layer. `DirichletBC` operates on indices. `Systems/StrongDirichletConstraint` remains the bridge between `Forms` (markers/expressions) and `Constraints` (indices/values).

**Actions:**
- [ ] Confirm `Code/Source/solver/FE/Systems/StrongDirichletConstraint.h` is working as intended (it appeared correct in review). No changes needed here, just usage.

---

## 4. Physics: Refactor `PoissonModule`

**File:** `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`

Update the module to use the new API. This fixes the bug where constraints were missing from the Jacobian.

**Refactoring Steps:**
1.  Instantiate `forms::WeakForm problem;`.
2.  Build the residual expression as before (`integrand.dx()`, `applyNeumann`, etc.) and assign it to `problem.residual`.
3.  Generate the `StrongDirichlet` list using `forms::bc::makeStrongDirichletListValue` and assign it to `problem.strong_constraints`.
4.  Replace the `installResidualForm` calls:

```cpp
// OLD
// FE::systems::installResidualForm(system, "residual", u_id, u_id, residual, dirichlet_bcs);
// FE::systems::installResidualForm(system, "jacobian", u_id, u_id, residual); // BUG FIXED

// NEW
FE::systems::installWeakForm(system, "residual", u_id, u_id, problem);
FE::systems::installWeakForm(system, "jacobian", u_id, u_id, problem);
```

**Actions:**
- [ ] Update `PoissonModule.cpp`.

---

## 5. Verification

**Tests:**
1.  **Compilation:** Ensure `WeakForm` compiles and `FormsInstaller` links.
2.  **Unit Tests:** Run `test_PoissonModule`.
    - *Crucial:* The current test `AssembledJacobianMatchesFiniteDifference` uses *Neumann* BCs. We should add a case with *Strong Dirichlet* BCs to verify the fix. If the constraints are missing from the Jacobian, the Jacobian rows for boundary DOFs will contain PDE entries instead of Identity/Constraint entries, and the test should fail (or we can assert on the matrix structure).

**Action Items:**
- [ ] Update `Code/Source/solver/Physics/Tests/Unit/test_PoissonModule.cpp` to include a test case with Strong Dirichlet BCs.

---

## Summary of Checklist

- [ ] Create `Code/Source/solver/FE/Forms/WeakForm.h`.
- [ ] Update `Code/Source/solver/FE/Systems/FormsInstaller.{h,cpp}`.
- [ ] Update `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`.
- [ ] Add Strong Dirichlet test case to `Code/Source/solver/Physics/Tests/Unit/test_PoissonModule.cpp`.
- [ ] Verify build and tests.
