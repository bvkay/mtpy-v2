# BC87 LITHOPROBE MT data fixtures

These fixtures back the McNeice-Jones validation test in
`tests/test_mj_bc87_validation.py`. **The data files themselves are
not committed to git** (see [.gitignore](.gitignore)) because the
upstream BC87 README requests that the dataset not be redistributed
person-to-person — every user is asked to obtain it from the canonical
source so all participants work from the same files.

## How to populate this directory

Place the 27 BC87 LITHOPROBE LIT-line EDI files
(`lit000.edi`–`lit024.edi`, `lit901.edi`, `lit902.edi`) directly under
`tests/data/bc87/`. Two practical sources:

1. **Public download** from the MTNet archive:
   <https://www.mtnet.info/data/bc87/bc87.html>. Pick either the EDI
   archive (`bc87.zip` / `bc87.edi.Z`) or the J-format archive
   (`bc87j.zip` / `bc87j.tar.Z`); the EDI files are the path of least
   resistance for mtpy-v2. Unpack in place.
2. **Locally cached copy** (this project only): set the environment
   variable `MTPY_BC87_DATA_DIR` to a directory containing the EDI
   files; the test will fall back to that location.

If neither path produces files, the validation test skips with a
clear message rather than failing.

## Citations

When using these data, cite the providers as the BC87 README directs:

- Jones, A. G. (1993). The BC87 dataset: tectonic setting, previous EM
  results, and recorded MT data. *Journal of Geomagnetism and
  Geoelectricity*, 45(9), 1089–1105.
- Jones, A. G., Groom, R. W., & Kurtz, R. D. (1993). Decomposition and
  modelling of the BC87 dataset. *Journal of Geomagnetism and
  Geoelectricity*, 45(9), 1127–1150.
- McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
  tensor decomposition of magnetotelluric data. *Geophysics*, 66(1),
  158–173.

The data are credited to LITHOPROBE (D. Oldenburg, UBC) and the
Geological Survey of Canada (A. Jones).
