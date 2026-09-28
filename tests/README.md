# Testing Guide

[Integration testing](https://en.wikipedia.org/wiki/Integration_testing) is an essential part of software development. It is performed when integrating code changes into the main development branch to verify that the code works as expected. The following sections describe how to run and add integration tests used to the svMultiPhysics program.

Running a test case requires 
- Build svMultiPhysics
- Install Git LFS used to download test data
- Build svZeroDSolver (only required from certain tests)

All the integration testing steps described below are automatically performed upon pull requests to the main branch, using GitHub Actions. See [WORKFLOW.md](../.github/WORKFLOW.md) for more details.

# Build svMultiPhysics
svMultiPhysics can be built following these [instructions](../README.md).

To automatically run test cases using `pytest` you must build svMultiPhysics in a folder named `build` located at the svMultiPhysics repository  root directory (i.e. the svMultiPhysics directory created when doing a git clone of the svMultiPhysics repository).

# Install Git LFS
The svMultiPhysics tests require finite element mesh data stored in VTK-format VTP and VTU files. These large files are managed using the [Git Large File Storage (LFS)](https://git-lfs.com/) extension. *Git LFS* stores files as text pointers inside git until the file contents are explicitly pulled from a remote server.  

The file extensions currently tracked with *Git LFS* are stored [in this file](../.gitattributes).

*Git LFS* is install following [this guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

To set up *Git LFS* for the svMultiPhysics repository run the following commands to activate *Git lfs*
```
git lfs install
```
    
and download file data
```
git lfs pull
```
    
These steps need to be performed only once. All large files are handled automatically during all Git operations, like `push`, `pull`, or `commit`.

# Running tests using pytest
You can run an individual test by navigating to the `./tests/cases/<physics>/<test>` folder you want to run and execute `svMultiPhysics` with the `svFSI.xml` input file as an argument. A more elegant way, e.g., to run a whole group of tests, is using [`pytest`](https://docs.pytest.org/). By default, it will run all tests defined in the `test_*.py` files in the [./tests](https://github.com/SimVascular/svMultiPhysics/tree/main/tests) folder. Tests and input files in [./tests/cases](https://github.com/SimVascular/svMultiPhysics/tree/main/tests/cases) are grouped by physics type, e.g., [struct](https://github.com/SimVascular/svMultiPhysics/tree/main/tests/cases/struct), [fluid](https://github.com/SimVascular/svMultiPhysics/tree/main/tests/cases/fluid), or [fsi](https://github.com/SimVascular/svMultiPhysics/tree/main/tests/cases/fsi) (using the naming convention from `EquationType`). Here are a couple of useful `Pytest` commands:

- Run only tests matching a pattern (can be physics or test case name):
    ```
    pytest -k ustruct
    pytest -k block_compression
    ```
- List individual test cases that were run:
    ```
    pytest -v
    ```
- Print the simulation output of all tests:
    ```
    pytest -rP
    ```

For more options, simply call `pytest -h`.

## Interior-node Dirichlet example

The `interior_node` tests generate meshes and input files in pytest temporary
directories. No downloaded mesh fixtures are needed for these tests. From the
repository root, after building its solver and installing the usual test dependencies:

```sh
python -m pytest tests/test_heats.py tests/test_linear_elasticity.py -k interior_node -v
```

To generate and run a standalone 3D Darcy example, use the existing test helper:

```sh
python - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET
from tests.conftest import make_interior_node_case, run_by_name

folder = Path("build/interior-node-example").resolve()
folder.mkdir(parents=True, exist_ok=True)
root, points, cells, ids = make_interior_node_case(folder, nsd=3)
ET.ElementTree(root).write(folder / "solver.xml")
run_by_name(folder, "solver.xml", t_max=2, n_proc=1)
print(f"Input: {folder / 'solver.xml'}; prescribed input point IDs: {ids}")
PY
```

The generated unit-cube tetrahedral mesh has no faces. Two disconnected interior
nodes, input point indices 43 and 22, have prescribed pressures 2 and 7. The
whitespace-separated `values.dat` file uses the existing general BC format:

```text
1 2 2
0 1
22 7 7
43 2 2
```

The header gives components, times and nodes. After the time vector, each record
gives the one-based input point index followed by the values for every time. Two
identical samples hold each pressure constant. The complete XML, volume mesh and
data remain in `build/interior-node-example`; VTK results are in its `1-procs`
subdirectory. Change `n_proc` to 3 or 4 to exercise MPI with the same inputs.

The scalar tests independently assemble the diffusion matrix, eliminate prescribed
degrees of freedom, and compare the entire field and free-node residual. Other
cases cover selected vector components, displacement/velocity histories, MPI ranks
with empty selections, restart, multiple meshes/equations, overlap errors and input
validation. The [solver input contract](../Code/Source/solver/README.md#dirichlet-values-at-arbitrary-mesh-nodes)
describes masks, ID files, temporal data and unsupported options.

## Code coverage
We expect that new code is fully covered with at least one integration test. We also strive to increase our coverage of existing code. You can have a look at our current code coverage [with Codecov](https://codecov.io/github/SimVascular/svMultiPhysics). It analyzes every pull request and checks the change of coverage (ideally increasing) and if any non-covered lines have been modified. We avoid modifying untested lines of codeas there is no guarantee that the code will still do the same thing as before.

## Create a new test
Here are some steps you can follow to create a new test for the code you implemented. This will satisfy the coverage requirement (see above) and help other people who want to run your code. A test case is a great way to show what your code can do! Ideally, you do this early in your development. Then you can keep running your test case as you are refactoring and optimizing your code.

1. Create an **example** (mesh and input file) that showcases what your code can do. If it doesn't cover everything, consider adding other tests.
2. **Verify** the results: Since an analytical solution would be ideal but is rarely available, find other ways to ensure that your code is doing the right thing (reference solutions from other codes, manufactured solutions, convergence analysis, ...).
3. Crank up the **tolerances**: Be as strict in your linear and nonlinear solver as you can be with the test still converging. This will avoid problems when running the test on different machines or with multiple processors.
4. **Reduce** the computational effort: Try to make the test run in a few seconds by coarsening spatial and temporal discretization (but keeping the core function of the test).
5. Set the **maximum number** of nonlinear iterations to the one where it currently achieves the set tolerance. This way, the test will fail if the linearization gets broken in the future (but the test still slowly converges to the correct solution).
6. **Test the test**: Does it fail if you change parts of your input file that the test should be sensitive to (e.g., material parameters if you implemented a new solid material)?
7. Put your files in the appropriate **folder** under `./tests/cases` and append the test to the Python `test_*.py` file. If you created a new physics type, create new ones for both.
8. Check that the test is **executed** correctly in `GitHub Actions` when opening your pull request. You should see in your coverage report that your new code is covered.

If you want to parameterize values in your test case, you can use `@pytest` [fixtures](https://docs.pytest.org/en/6.2.x/fixture.html). We currently use them to automatically loop different numbers of processors, meshes, or input files.
