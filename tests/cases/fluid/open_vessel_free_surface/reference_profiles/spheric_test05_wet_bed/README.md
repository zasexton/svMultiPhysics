# SPHERIC Test05 Wet-Bed Reference Profiles

These files are the two-column digitized free-surface profiles distributed from
the SPHERIC Test Case 05 data archive:

- Source page: `https://www.spheric-sph.org/tests/test-05`
- Archive file name on the source page: `SPHERIC_TestCase5.zip`
- Archive directory name: `SPHERIC_TestCase6`

The archive stores coordinates in centimeters. The comparison utility
`tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py`
converts the values to meters when loading the tables.

File groups:

- `d18_*.dat`: wet-bed depth `d = 18 mm`
- `d38_*.dat`: wet-bed depth `d = 38 mm`

## Frame times

The suffix is the photograph row in the corresponding wet-bed montage, counted
from the top. Rows 1–7 are labelled 0.156, 0.219, 0.281, 0.343, 0.406, 0.468
and 0.531 s after release. D38 has an eighth row at 0.593 s. The M1 test uses
the printed labels for the first three rows. [Benchmark data and photographs](https://www.spheric-sph.org/tests/test-05) · [Original photographs, Fig. 2b–c](https://doi.org/10.1007/s00348-004-0804-4).

The benchmark authors reported an approximately 43 ms discrepancy and attributed
it to difficulty identifying when the gate began to move. Published comparisons
for both depths use simulation time equal to the photo label minus 0.043 s.
For example, the 0.219 s photograph is compared with a simulation at 0.176 s.
The accessible wet-run records give no synchronized timestamp for the first
gate motion. Report this comparison separately from scores at the printed times.
[SPHERIC intercomparison, p. 5](https://9449af45-2363-44f6-8b03-03b6b7c2aee5.usrfiles.com/ugd/9449af_4162080ba58044b7a5d8e5b9e0ad25be.pdf#page=5) · [Violeau, §7.3.2, pp. 490–491](https://academic.oup.com/book/8904/chapter/155238510).

## Surface comparison

The scorer connects consecutive measured points in file order so that an
overturning wave retains its folds. At each distinct measured x in the selected
window, it takes the highest vertical intersection on each curve. Each station
has equal weight in the height RMSE. A gap in the simulated contour reduces
the reported coverage. All measured stations must be covered for the full-frame
comparison. Supplying `--sample-radius` selects the historical nearby-vertex
method for reproducing earlier reports. [Profile scorer](../../compare_test05_profiles.py).
