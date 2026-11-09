# Case 3 Input Data Specification and Preparation Guide

## 1. Purpose and data-availability statement

This document describes the input format required by the published Case 3 implementation of SP-PINN. It is intended to let researchers prepare an equivalent input dataset for another river reach without access to the restricted Lower Yellow River field observations used in the study.

The original field dataset is not distributed because permission for public disclosure was not granted by the third-party data provider. No observation values, coordinates, or derived result tables are included here. The source code, field definitions, array construction rules, scaling operations, and consistency checks are provided so that the implementation can be reused with appropriately licensed hydrodynamic and sediment data.

This guide documents the behavior of the released code, especially `data3.py`, `data4.py`, `data5.py`, `data7.py`, `test.py`, `scalePINN.py`. Where old comments in the source still mention an earlier 30 h/146-step configuration, the executable implementation and the current Case 3 files take precedence: the released Case 3 configuration uses 12 h sampling, 367 time levels, 54 modeled sections, and 19,818 space-time samples.

## 2. Required directory layout

Run the scripts from the `src` directory and prepare the following private data tree locally:

```text
src/
├── output77/
│   ├── 每12h参数新.xlsx
│   └── excel/
│       ├── LYR_SediCS  1.xlsx
│       ├── LYR_SediCS  2.xlsx
│       ├── ...
│       └── LYR_SediCS 54.xlsx
├── data3.py
├── data4.py
├── data5.py
├── data7.py
├── test.py
└── ...
```

The strict loader accepts any section-workbook prefix as long as the filename ends in an integer followed by `.xlsx`; files are sorted by that trailing integer. Exactly 54 section workbooks are expected unless the constants and indexing logic are deliberately adapted.

## 3. Dataset dimensions and ordering

The restricted dataset used for Case 3 has the following structural properties. These are metadata only; no field values are disclosed.

| Item | Case 3 value |
|---|---:|
| Sections present in the parameter workbook | 93 |
| Sections used by the model | first 54 in workbook spatial order |
| Time levels per section | 367 |
| Sampling interval | 12 h |
| Time range | 0–4392 h, inclusive |
| Model grid size | 54 × 367 = 19,818 points |
| Observation sections | zero-based indices 10, 19, and 37 (the 11th, 20th, and 38th modeled sections) |
| Observation samples | 3 × 367 = 1,101 points |

Two orderings are used during preparation:

1. The valid numeric records in the parameter workbook are time-major: each time block contains all 93 sections in a fixed spatial order, and this 93-section order repeats 367 times.
2. Model arrays are section-major: all 367 times of section 1, followed by all 367 times of section 2, and so on through section 54.

For a modeled section index `i` and time index `j`, the flattened row index is

```text
row = i * 367 + j
```

Every scalar or 10-class vector supplied to the network must use this same pointwise ordering.

## 4. Parameter workbook: `每12h参数新.xlsx`

### 4.1 General format

- File type: Excel `.xlsx`.
- Worksheet used: first worksheet (`sheet_name=0`).
- Header handling: read with `header=None`; columns are positional.
- Required width: exactly 64 columns, Excel columns A through BL.
- Numeric-row rule: non-data/header rows are discarded. All fields used by the model must be finite numeric values.
- Spatial structure: 93 distinct `DistLG(km)` values, each with 367 valid records.
- Case 3 selection: group by `DistLG(km)` in first-appearance order and retain the first 54 groups.

In the restricted source workbook, the used sheet range contains extra nonnumeric rows. The loader intentionally keeps only numeric records, producing exactly `93 × 367 = 34,131` valid rows. A replacement dataset does not need to reproduce the extra nonnumeric rows; a clean 34,131-row numeric table is preferable.

### 4.2 Positional column dictionary

| Excel column | Zero-based index | Code name | Meaning / unit | Used by released Case 3 loader |
|---|---:|---|---|---|
| A | 0 | `Ncs` | Cross-section identifier/index | No |
| B | 1 | `DistLG(km)` / `x` | Longitudinal section coordinate, km | Yes |
| C | 2 | `Hav(m)` / `H` | Cross-section-averaged water depth, m | Yes |
| D | 3 | `Uav(m/s)` | Cross-section-averaged velocity, m s⁻¹ | No; taken from section workbooks |
| E | 4 | `Area(m2)` / `A` | Wetted cross-sectional area, m² | Yes |
| F | 5 | `BW(m)` / `B` | Water-surface/top width, m | Yes |
| G | 6 | `QD(m3/s)` | Discharge, m³ s⁻¹ | No; taken from section workbooks |
| H | 7 | `ZW(m)` | Water-surface elevation, m | Not in the strict ablation loader |
| I | 8 | `ZBav(m)` | Mean bed elevation, m | No |
| J | 9 | `Froude` | Froude number, dimensionless | No |
| K | 10 | `RN` / `n` | Manning roughness coefficient, s m⁻¹ᐟ³ | Yes |
| L | 11 | `J(%%)` | Source hydraulic-gradient field | No; `Jf` is recomputed |
| M | 12 | `Qmc%` | Source sediment-transport percentage field | No |
| N:W | 13:22 | `挟沙力级配1…10` / `delta_S1` | Ten fractions of sediment-carrying-capacity gradation; dimensionless fractions | Yes |
| X:AG | 23:32 | `悬沙级配1…10` / `delta_S` | Ten fractions of suspended-sediment gradation; dimensionless fractions | Yes |
| AH:AQ | 33:42 | `床沙级配1…10` | Ten fractions of bed-material gradation | No in the current network input |
| AR:BA | 43:52 | `浑水沉速1…10` / `w` | Settling velocities for ten sediment classes; use one consistent velocity unit, matching the governing-equation implementation | Yes |
| BB:BK | 53:62 | `恢复饱和系数1…10` / `a` | Recovery-to-saturation coefficients for ten sediment classes | Yes |
| BL | 63 | `形心水深hc` / `hc` | Submergence depth of the wetted-section centroid, m | Yes |

The ten gradation components must use one consistent grain-size-bin definition in all three places: carrying-capacity gradation, suspended-sediment gradation, and the corresponding `w` and `a` vectors. The current implementation assumes ten classes and does not infer class boundaries from the workbook.

## 5. Section workbooks: `excel/*.xlsx`

### 5.1 General format

- Number of files: exactly 54.
- File order: numeric suffix order, 1 through 54.
- Worksheet used: first worksheet.
- Header: row 1 (`header=0`); leading/trailing spaces in names are stripped.
- Time selection: rows satisfying `Time(h) % 12 == 0`.
- Required selected count: exactly 367 rows in every file.
- All files must have identical selected time values and ordering.
- Required values must be finite; `NaN` and infinity are rejected.

The original section workbooks contain these 20 headers:

```text
K, [blank], Time(h), ZW, QD, Hav, Uav, FR, Rn, SP(%%),
Ust, ZBav(f), DZmc, Qm/QD(%), S*, S, SD50(mm), BD50(mm),
dA0/dt(10^-3), dAf/dt(10^-3)
```

### 5.2 Fields required by the released implementation

| Column name | Code symbol | Meaning / unit | Processing |
|---|---|---|---|
| `Time(h)` | `t` | Elapsed time, h | Keep multiples of 12; no unit conversion |
| `QD` | `Q_physical`, `Q` | Discharge, m³ s⁻¹ | Store physical value; network target/input uses `round(QD / 50, 4)` |
| `ZW` | `Z` | Water-surface elevation, m | Needed by the legacy/original data path; not used as the second target by the strict Manning-ablation loader |
| `Hav` | `H` | Mean water depth, m | Used without scaling; must match parameter-workbook column C pointwise |
| `Uav` | `U` | Mean velocity, m s⁻¹ | `round(Uav, 4)` |
| `Rn` | `n` | Manning roughness, s m⁻¹ᐟ³ | Must match parameter-workbook column K pointwise |
| `S*` | `S_1` | Total sediment-carrying capacity, in the same concentration unit as `S` | Used without scaling |
| `S` | `S` | Suspended-sediment concentration, kg m⁻³ in the Case 3 plotting/output labels | Used without scaling |
| `BD50(mm)` | `D50` | Median bed-material diameter, mm | Loaded for checking/output; not a network feature |
| `dA0/dt(10^-3)` | `A0_raw`, `A0` | Source bed-area-change-rate field | Network target is `round(raw_value × 43.2, 4)` |

The factor 43.2 is fixed in the released code. It equals `12 h × 3600 s h⁻¹ × 10⁻³` and converts the coded `10^-3` rate field to the 12 h model-step scale. Researchers should preserve this transformation only if their replacement field uses the same source convention; otherwise they should revise the factor and document the new units.

The remaining columns are not consumed by the released Case 3 training path. Their labels indicate additional source metadata or diagnostics (`FR`, `SP`, `Ust`, bed elevation/change, transport ratio, suspended `D50`, and `dAf/dt`). Because they are unused and their provider-specific definitions are not recoverable from the model code alone, users should consult their own data documentation rather than infer their semantics from these abbreviated labels.

## 6. Cross-file alignment requirements

The parameter workbook and section workbooks are not independent tables. The following conditions are mandatory:

1. Section workbook 1 corresponds to the first retained `DistLG(km)` group, workbook 2 to the second, and so on.
2. Each retained parameter group has exactly 367 records.
3. Every section workbook has the same 367 selected times in the same order.
4. Parameter `Hav(m)` equals section `Hav` point by point.
5. Parameter `RN` equals section `Rn` point by point.
6. All scalar and ten-class vectors are flattened in section-major order.
7. Adjacent spatial coordinates must be distinct, because `A_x` uses finite differences.
8. `H > 0` and `A != 0` everywhere, because the coded `Jf` calculation would otherwise be invalid.

The strict loader stops on an inconsistency rather than attempting to interpolate or guess an alignment.

## 7. Derived variables and exact preparation formulas

Let `N = 54 × 367 = 19,818`. After alignment, all scalar variables have shape `(N, 1)` and all class-resolved variables have shape `(N, 10)`.

### 7.1 Discharge target

```text
Q = round(QD / 50, 4)
```

`QD` remains available as `Q_physical` for physical-unit reporting.

### 7.2 Water depth, velocity, concentration, and bed-change target

```text
H  = Hav
U  = round(Uav, 4)
S  = suspended-sediment concentration
A0 = round([dA0/dt(10^-3)] * 43.2, 4)
```

### 7.3 Muddy-water density

The released code uses water density 1000 kg m⁻³, sediment density 2650 kg m⁻³, and submerged density difference 1650 kg m⁻³:

```text
pm = 1000 + (1650 / 2650) * S
```

### 7.4 Friction slope used by the code

The released implementation recomputes `Jf` and does not use parameter column `J(%%)`:

```text
Jf = (Q / A)^2 * n^2 / H^(4/3)
```

Here `Q` is the coded discharge `QD / 50`, not the unscaled physical discharge. This detail must be retained to reproduce the published implementation exactly.

### 7.5 Longitudinal area derivative

`A_x` is derived from all 93 parameter-workbook sections before the first 54 are retained:

1. Reshape valid numeric parameter rows to `(367 times, 93 sections, 64 columns)`.
2. Verify that the 93-section coordinate order is identical at every time.
3. At each time, calculate forward differences:

   ```text
   difference_j = (A_(j+1) - A_j) / (x_(j+1) - x_j)
   ```

4. Assign the first forward difference to the first section and the `j-1 -> j` difference to each later section. This produces 93 derivative values per time.
5. Retain the first 54 sections, transpose to section-major order, and flatten to `(19,818, 1)`.

No km-to-m conversion is applied by the released code. With `A` in m² and `x` numerically stored in km, `A_x` is numerically expressed per km. A new application should either preserve this convention for exact code compatibility or consistently revise the coordinate, derivatives, and equation scaling.

### 7.6 Grain-class quantities

For class `k = 1,...,10`, the governing-equation code constructs

```text
s_k  = S   * delta_S[k]
s*_k = S_1 * delta_S1[k]
```

The class-specific exchange term uses `B * w[k] * a[k] * (s*_k - s_k)`. Consequently, the ten positions in `delta_S`, `delta_S1`, `w`, and `a` must refer to the same ordered grain-size classes.

## 8. Network inputs and outputs

### 8.1 Baseline 49-dimensional input

The exact concatenation order is

```text
[x, t, A, B, S_1,
 delta_S1_1 ... delta_S1_10,
 w_1 ... w_10,
 a_1 ... a_10,
 Jf, hc,
 delta_S_1 ... delta_S_10,
 pm, Ax]
```

| Group | Width |
|---|---:|
| `x`, `t`, `A`, `B`, `S_1` | 5 |
| `delta_S1` | 10 |
| `w` | 10 |
| `a` | 10 |
| `Jf`, `hc` | 2 |
| `delta_S` | 10 |
| `pm`, `Ax` | 2 |
| Total | 49 |

Every input column is min-max mapped to `[-1, 1]` using bounds computed over the complete 19,818-point grid:

```text
X_scaled = 2 * (X - lower_bound) / (upper_bound - lower_bound) - 1
```

For a constant feature, the implementation adds a small epsilon to the upper bound to avoid division by zero.

### 8.3 Five model targets

```text
[Q, H, U, S, A0]
```

The original model class retains some legacy `Z` variable names, but the released Case 3 runners deliberately pass `Hav`/`H` into that second output/constraint slot. This guide therefore calls the second Case 3 target `H`.

## 9. Boundary, initial-profile, observation, and physics samples

The complete 19,818-point grid is used for physics residuals.

| Sample set | Exact indexing in section-major arrays | Size | Targets used |
|---|---|---:|---|
| Inlet time series | rows `0:367` (first modeled section, all times) | 367 | `Q`, `S` |
| Initial spatial profile | rows `0, 367, 734, ...` (all 54 sections at first time) | 54 | `H` |
| Interior observations | all times at zero-based sections 10, 19, and 37 | 1,101 | `Q`, `H`, `U`, `S`, `A0` |
| Physics/collocation set | every grid row | 19,818 | PDE residuals |

Some source variables and comments call the 54-point first-time set an "outlet" or `Z_BC` set. The actual index operation is `[::367]`, so it is an initial spatial profile, not a downstream-section time series. The table above describes the executable behavior needed for reproducibility.

When transferring the implementation to another reach, the three observation-section indices may be changed, but each selected section must supply all requested observation variables at the selected times.

## 10. Recommended preparation workflow for another river reach

1. **Define the model domain.** Select ordered cross sections and a common time grid. Either retain 54 sections and 367 12-hourly times or update all relevant constants and index rules.
2. **Choose ten sediment classes.** Fix class boundaries and use the same order for suspended gradation, carrying-capacity gradation, settling velocity, and recovery coefficient.
3. **Prepare section workbooks.** Create one workbook per modeled section, with a common header and identical `Time(h)` values. At minimum provide the fields listed in Section 5.2; include `ZW` if the legacy/original pipeline will be used.
4. **Prepare the 64-column parameter workbook.** Write fields in the exact positional order in Section 4.2. A clean numeric table is recommended. For direct reuse of the current `A_x` routine, arrange records as repeated time blocks containing every available section in fixed spatial order.
5. **Check cross-file alignment.** Verify section order, time order, `Hav`, and `Rn` before flattening.
6. **Filter the common time grid.** Apply the same time selection to every section; the current code uses `Time(h) % 12 == 0`.
7. **Reorder to section-major arrays.** Produce `N` scalar rows and `N × 10` class-resolved matrices.
8. **Compute derived features.** Apply the exact formulas for `Q`, `A0`, `pm`, `Jf`, and `A_x` above.
9. **Construct sample subsets.** Build the inlet series, initial spatial profile, observation sections, and full physics grid using Section 9.
10. **Normalize network features.** Compute lower and upper bounds over the full grid and map inputs to `[-1, 1]`.
11. **Run strict validation before training.** Reject missing values, inconsistent dimensions, duplicated adjacent coordinates, mismatched times, and cross-file differences in `H` or `n`.
12. **Document any adaptation.** If units, sampling, section count, class count, scaling factors, or observation locations differ, update both the code constants and this specification.

## 11. Validation checklist

Before training, all answers should be **yes**:

- [ ] Is the parameter workbook exactly 64 columns wide?
- [ ] Does every modeled section contain the same number of valid times?
- [ ] Are the selected times identical and identically ordered across section files?
- [ ] Does the numeric filename order match the parameter-workbook spatial order?
- [ ] Are `Hav(m)`/`Hav` and `RN`/`Rn` equal point by point across the two sources?
- [ ] Are all required values finite?
- [ ] Are `H` values positive and `A` values nonzero?
- [ ] Are adjacent `x` coordinates distinct and ordered consistently at every time?
- [ ] Do the ten entries of each class-resolved vector use the same class order?
- [ ] After flattening, do all scalar arrays have shape `(19,818, 1)` and class arrays `(19,818, 10)` for the unchanged Case 3 configuration?
- [ ] Does the baseline concatenated input have width 49?
- [ ] Do the inlet, initial-profile, and observation sets have 367, 54, and 1,101 rows, respectively?

## 12. Minimal structural pseudocode

The following pseudocode summarizes the preparation logic without containing restricted values:

```python
parameter = read_first_sheet("output77/每12h参数新.xlsx", header=None)
parameter = keep_numeric_rows(parameter)
assert parameter.shape[1] == 64

# Parameter file is time-major; retain first 54 spatial groups and reorder.
groups = group_by_first_appearance(parameter, column=1)
groups = groups[:54]
assert all(len(group) == 367 for group in groups)
parameter_section_major = concatenate(groups)

section_frames = []
for workbook in sort_by_trailing_integer("output77/excel/*.xlsx"):
    frame = read_first_sheet(workbook, header=0)
    strip_column_names(frame)
    frame = frame[frame["Time(h)"] % 12 == 0]
    assert len(frame) == 367
    section_frames.append(frame)

assert len(section_frames) == 54
assert all_time_vectors_equal(section_frames)

# Flatten by section, then time.
Q_physical = concatenate(frame["QD"] for frame in section_frames)[:, None]
H = concatenate(frame["Hav"] for frame in section_frames)[:, None]
U = concatenate(frame["Uav"] for frame in section_frames)[:, None]
S = concatenate(frame["S"] for frame in section_frames)[:, None]
S_1 = concatenate(frame["S*"] for frame in section_frames)[:, None]
n_check = concatenate(frame["Rn"] for frame in section_frames)[:, None]

Q = round(Q_physical / 50.0, 4)
A0 = round(concatenate(frame["dA0/dt(10^-3)"] for frame in section_frames)[:, None] * 43.2, 4)
pm = 1000.0 + (1650.0 / 2650.0) * S
Jf = (Q / A) ** 2 * n ** 2 / H ** (4.0 / 3.0)

X = concatenate([x, t, A, B, S_1, delta_S1, w, a,
                 Jf, hc, delta_S, pm, A_x], axis=1)
assert X.shape == (54 * 367, 49)
```

## 13. Scope and limitations

This specification provides the information needed to reconstruct the data interface and reproduce the released preprocessing behavior. It does not grant rights to the restricted Lower Yellow River observations, replace provider metadata, or imply that a different river dataset is scientifically interchangeable without calibration and validation. Users remain responsible for data licensing, unit consistency, quality control, selection of representative observation sections, and documenting modifications made for a new study reach.
