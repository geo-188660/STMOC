# STMOC spatial optimization: pMOLU code and processed data

## Overview

This directory contains the processed inputs and core implementation of the probabilistic multi-objective land-use model (pMOLU), which forms the spatial-allocation component of the Spatiotemporal Multi-Objective Consolidation (STMOC) framework.

pMOLU represents three possible states for each potentially releasable rural-homestead patch—retention in its current use, cropland reclamation, and reforestation—as continuous allocation probabilities. It then uses gradient-based optimization to reconcile five objectives:

1. food production;
2. carbon sequestration;
3. biodiversity conservation capacity;
4. one-off net quota revenue; and
5. annual operating income from agricultural and forestry products.

The files in this directory cover data preparation, objective construction, probability-based spatial optimization, policy-weight experiments, and categorical output. Upstream candidate identification using nighttime-light and Tencent activity data, aggregate population-demand assessment, detailed benefit accounting, and phased temporal sequencing are described in the accompanying manuscript but are not included in this directory.

## Repository structure

```text
code1/
├── README.md
├── LUloader.py
├── model.py
├── objs.py
├── preprocess.ipynb
├── vGDA.ipynb
└── data2/
    ├── data-1.xlsx
    └── multiob-1.xlsx
```

### File descriptions

| File | Role |
| --- | --- |
| `preprocess.ipynb` | Reads the patch- and county-level Excel inputs, checks county-code matching, creates point geometries, assigns the initial land-use state, defines the 50%, 65%, and 80% implementation-feasibility subsets, and exports GeoFeather inputs for pMOLU. |
| `LUloader.py` | Loads vector land-use data and converts categorical states to probability matrices and back. It also contains optional export and plotting helpers; the present workflow relies on its vector/GeoFeather loading and state-conversion functions. |
| `objs.py` | Defines the five benefit objectives and the conversion term in TensorFlow, together with NumPy functions for evaluating a completed allocation. Benefits to be maximized enter the minimization problem with a negative sign. |
| `model.py` | Implements the `GDAmodel` pMOLU optimizer: softmax probability representation, single-objective attainable benchmarks, normalized minimax loss, Adam updates, and optional logging/checkpoint interfaces. |
| `vGDA.ipynb` | Provides the model-running workflow: assembling patch-level objective coefficients, initializing pMOLU, calculating single-objective benchmarks, applying an objective-emphasis setting, optimizing the allocation, and joining the final classes back to spatial patches. |
| `data2/data-1.xlsx` | Patch-level input table for 357,714 potentially releasable rural-homestead patches, covering approximately 5.29 million ha. |
| `data2/multiob-1.xlsx` | County-level coefficients used to calculate the five objectives for 2,662 county codes. |

## Input data

### Patch-level data: `data2/data-1.xlsx`

The main worksheet is `Sheet1`. Each row represents one candidate patch.

| Field | Meaning |
| --- | --- |
| `pointid` | Unique patch identifier. |
| `grid_code` | County code used to join county-level objective coefficients. |
| `food` | Cropland-reclamation suitability score; larger values indicate greater suitability. |
| `tree` | Reforestation suitability score; larger values indicate greater suitability. |
| `centroid_x`, `centroid_y` | Patch-centroid coordinates. `preprocess.ipynb` currently assigns EPSG:28413 to these coordinates. |
| `area` | Patch area in m². |
| `priority2` | Relative implementation-feasibility score; larger values indicate higher priority for admission to optimization. |

The workbook also contains Chinese-language worksheets documenting the field definitions and scenario design. The scenario note refers to cumulative area, whereas the current notebook uses `priority2` quantiles and therefore selects the highest-ranked 50%, 65%, and 80% of patches by patch count. This difference is documented under “Version and reproducibility notes” below.

### County-level data: `data2/multiob-1.xlsx`

The main worksheet is `zong`. `PAC` is the county identifier and matches `grid_code` in the patch table.

| Field | Model meaning | Unit in the supplied workbook |
| --- | --- | --- |
| `PAC` | County code | — |
| `FOOD` | Food-production capacity | g m⁻² |
| `NEP` | Net ecosystem productivity used for carbon sequestration | g m⁻² |
| `BIO` | Biodiversity conservation-capacity coefficient | index per m² |
| `FOODVALUEC` | Reclamation-related annual agricultural return | CNY m⁻² |
| `FOODVALUED` | Reclamation-related one-off quota return | CNY m⁻² |
| `TREEVALUEC` | Reforestation-related annual forestry return | CNY m⁻² |
| `TREEVALUED` | Reforestation-related one-off quota return | CNY m⁻² |

The suffixes `C` and `D` are retained from the original code. They correspond, respectively, to recurrent operating income (`ecoc`) and one-off quota revenue (`ecod`) in the current manuscript terminology.

## Model states and objective names

`vGDA.ipynb` defines the land-use labels as `Restrict`, `U`, `C`, and `T`. `Restrict` is an internal placeholder, leaving three optimized states:

| Code label | State |
| --- | --- |
| `U` | Retention in current use |
| `C` | Cropland reclamation |
| `T` | Reforestation |

Within `objs.py`, the objective names map to the manuscript as follows:

| Internal name | Manuscript objective |
| --- | --- |
| `food` | Food production |
| `nep` | Carbon sequestration |
| `bio` | Biodiversity conservation capacity |
| `ecoc` | Annual operating income (Obj. 5) |
| `ecod` | One-off net quota revenue (Obj. 4) |
| `Cov` | Conversion term used by the optimization routine |

The internal code order places annual operating income before one-off quota revenue. This is the reverse of their Obj. 4/Obj. 5 numbering in the manuscript; consequently, `objX = 3` denotes AOI-ES and `objX = 4` denotes OQR-ES.

`LUloader` stores categorical states as 1–3, whereas `argmax()` in the final notebook cell returns 0–2. In the exported `LU` field, `0`, `1`, and `2` therefore denote retention in current use, cropland reclamation, and reforestation, respectively.

## Computational workflow

Start Jupyter from the `code1/` directory so that the relative paths in both notebooks resolve correctly.

1. **Create working directories.** Create empty `CL/` and `sensitivity/` directories under `code1/`.
2. **Prepare the spatial inputs.** Run `preprocess.ipynb` to retain patch records whose county codes have available coefficients and generate `CL/Ws.csv`, `CL/luX.feather`, `CL/lu50.feather`, `CL/lu65.feather`, and `CL/lu80.feather`.
3. **Select an implementation-feasibility set.** In `vGDA.ipynb`, load `lu50`, `lu65`, or `lu80` for the conservative, intermediate, or high implementation-feasibility scenario. The notebook currently loads `luX`, which contains all candidate patches, unless this setting is changed.
4. **Assemble objective coefficients.** The notebook combines patch-level suitability and feasibility information with the county-level food, ecosystem-service, and economic coefficients.
5. **Calculate attainable benchmarks.** `GDAmodel.init_train()` independently optimizes each objective to establish the normalization benchmarks used in the multi-objective loss.
6. **Set policy preferences.** The equal-weight setting requires `pre = [1] * 5` with the line `pre[objX] = 1.2` skipped. For an emphasis scenario, retain that line and set `objX` from `0` to `4` for FP-ES, CS-ES, BIO-ES, AOI-ES, or OQR-ES, respectively. The supplied notebook uses `objX = 4` by default and therefore runs OQR-ES, not EWS.
7. **Run pMOLU.** `GDAmodel.train()` updates patch-level probabilities using Adam. After the final iteration, each patch is assigned to the state with the greatest probability.
8. **Export the allocation.** The last notebook cell joins the categorical result to a polygon layer through `pointid`/`PatchUID` and writes the spatial output to `sensitivity/`.

The notebook metadata record Python 3.9.5. The main dependencies are:

```text
numpy
pandas
geopandas
matplotlib
tensorflow (TensorFlow 1.x-compatible graph mode)
openpyxl
pyarrow
jupyter
Fiona or pyogrio for vector spatial I/O
```

The supplied notebook sets `GPU = 1`. Change this argument to `GPU = 0` when a compatible TensorFlow GPU is unavailable.

## Additional files required for a complete run

The current directory is a compact model-and-data snapshot rather than a one-command reproduction package. The following auxiliary inputs referenced by `vGDA.ipynb` are not included here:

- `CL/CCM.csv`, a 3 × 3 land-use conversion matrix whose rows and columns follow the `U`, `C`, and `T` state order;
- the source polygon layer `data2/dixiao-xiugai-xiaochu-HOUXUAN7501-shai_with_elderly-qutaihong`, whose `PatchUID` field must match `pointid`;
- the output directories `CL/` and `sensitivity/`, which should be created before running the notebooks.

## Generated outputs

- `CL/inp/Norm0.npy`, `Norm1.npy`, `Norm2.npy`, `Org0.npy`, and `Org1.npy` store the normalization inputs created by `save_inp()`.
- `CL/plot/LU/` receives diagnostic allocation plots when plotting is enabled.
- `sensitivity/objpre120_<objective>` receives the final spatial layer produced by the supplied notebook.

The final spatial layer contains `PatchUID`, categorical `LU`, and geometry. Patch-level probability matrices remain available in memory through `Model.get_opt()` but are not written by the current notebook.

## Version and reproducibility notes

- The scenario note in `data-1.xlsx` describes cumulative-area selection, but the current preprocessing code uses `priority2` quantiles by patch count. It selects 286,171, 232,514, and 178,857 patches for the 80%, 65%, and 50% sets; these represent approximately 71.84%, 55.23%, and 40.98% of candidate area. An area-based rule therefore requires explicit cumulative-area ranking.
- `vGDA.ipynb` retains the legacy `thC` conversion-threshold option and loads `luX` by default. With `thStop = 1`, the saved example run stops when this threshold is reached (at epoch 85) rather than completing all 1,000 requested iterations. Reproduction of the revised manuscript should use the appropriate scenario-specific feasible set and numerical settings stated in the final Methods.
- In the current notebook, `priority2` is used both to create scenario subsets and when assembling objective coefficients. Its intended role should be kept consistent with the accompanying manuscript when the release is finalized.
- The current notebook constructs each patch coefficient as `priority2 × suitability / 10 × county coefficient` and does not explicitly multiply by patch area. This step should be synchronized with the area-weighted objective equations in the final manuscript before archival release.
- The function `perference()` implements an emphasis setting by scaling the single-objective benchmark vector. Its equivalence to the explicit objective-weight formulation reported in the manuscript should be confirmed in the synchronized release.
- `vGDA.ipynb` creates a large county-membership matrix named `p` that is not used later. For the supplied data, it contains approximately 952 million elements and should be removed before rerunning the national model to avoid unnecessary memory use.
- The code is written in TensorFlow graph mode and records a TensorFlow random seed of `1486`. No environment lock file is presently included, so package versions should be recorded before archival release.
- Raster helper calls in `LUloader.py` are not active in this snapshot; the documented workflow should therefore be treated as vector/GeoFeather-based.
