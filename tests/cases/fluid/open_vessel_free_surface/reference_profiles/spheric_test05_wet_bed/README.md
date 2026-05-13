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

The numbered files preserve the source archive order.
