import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyvista as pv


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_pressure_row_support.py"
    )
    spec = importlib.util.spec_from_file_location("audit_pressure_row_support", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result(path: Path) -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.asarray(
        [
            4,
            0,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
        ],
        dtype=np.int64,
    )
    cell_types = np.asarray([pv.CellType.TETRA, pv.CellType.TETRA], dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["Pressure"] = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0])
    grid.point_data["phi"] = np.asarray([-1.0, -1.0, -1.0, -1.0, 0.1])
    grid.point_data["ActiveFluid"] = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0])
    grid.point_data["Velocity"] = np.zeros((5, 3), dtype=float)
    grid.cell_data["WetVolumeFraction"] = np.asarray([1.0, 5.0e-5])
    grid.save(path)


def test_pressure_row_support_audit_classifies_logged_zero_rows(tmp_path):
    audit = _load_audit_module()
    result = tmp_path / "result_090.vtu"
    log = tmp_path / "run.log"
    _write_result(result)
    log.write_text(
        "[svMultiPhysics::FE] Eigen direct factorization diagnostic "
        "phase=factorize info=numerical_issue rows=15 cols=15 zero_rows=2 "
        "zero_cols=2 block_summaries="
        "phi{begin=0,end=5,zero_rows=0,zero_cols=0,"
        "zero_rows_first_local=none,zero_cols_first_local=none,"
        "zero_row_runs_local=none,zero_col_runs_local=none};"
        "Pressure{begin=10,end=15,zero_rows=2,zero_cols=2,"
        "zero_rows_first_local=0|4,zero_cols_first_local=0|4,"
        "zero_row_runs_local=0|4,zero_col_runs_local=0|4,"
        "missing_diag=0,zero_diag=2,identity_rows=0,"
        "min_positive_row_sum=1,max_row_sum=2}\n",
        encoding="utf-8",
    )

    report = audit.audit_pressure_row_support(
        source_result=result,
        solver_log=log,
    )

    assert report["field_block"]["begin"] == 10
    assert report["reported_zero_row_count"] == 2
    assert report["classified_zero_row_count"] == 2
    assert report["row_list_complete"]
    assert report["zero_rows_match_zero_cols"]
    assert report["support_class_counts"] == {
        "full_wet_supported": 1,
        "tiny_cut_supported": 1,
    }
    assert report["zero_pressure_rows"][0]["point_index"] == 0
    assert report["zero_pressure_rows"][1]["point_index"] == 4


def test_expand_index_runs_handles_ranges_and_none():
    audit = _load_audit_module()

    assert audit.expand_index_runs("none") == []
    assert audit.expand_index_runs("1-3|5|7-6") == [1, 2, 3, 5, 7, 6]
