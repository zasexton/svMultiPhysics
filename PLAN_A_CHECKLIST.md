# Plan A Checklist — Low-Effort / High-Value Form API Improvements

Goal: Remove explicit boundary-condition loops from weak-form formulation files (e.g., `PoissonModule.cpp`) by moving the looping/validation boilerplate into FE/Forms helpers while keeping the weak-form notation (`dx()`, `ds(marker)`).

## Checklist

- [x] Add FE/Forms helper APIs to apply weak BC lists without loops (`Code/Source/solver/FE/Forms/BoundaryConditions.h`).
- [x] Add unit tests for the new helper APIs (compile to `FormIR`, validate boundary markers/term dispatch) (`Code/Source/solver/FE/Tests/Unit/Forms/test_BoundaryConditionHelpers.cpp`).
- [x] Wire the new unit test into the FE test targets (`Code/Source/solver/FE/CMakeLists.txt`).
- [x] Refactor the Poisson formulation to use the helper APIs (no explicit loops in `PoissonModule::registerOn`) (`Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`).
- [x] Run `test_fe_forms` and `test_fe_systems` and ensure green.
