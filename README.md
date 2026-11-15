# STMOC -- Spatiotemporal Multi-Objective Consolidation

A Python implementation of the probabilistic multi-objective land-use optimization (pMOLU) framework for the STMOC application, adapted from [pSO/pMOLU](https://github.com/codeRimoe/pSO/tree/main/pMOLU) (Luo & Huang, 2023).

## Overview

This code implements a gradient-based optimizer that assigns each land-use patch to a probability distribution over land-use categories (retention, reclamation, reforestation), then optimizes those probabilities to maximize five sustainability objectives simultaneously.

### Core algorithm

1. **Softmax parameterization** -- raw logits are transformed into valid allocation probabilities via `softmax(theta * sigP)`, ensuring every patch's category probabilities sum to one.
2. **Single-objective warm-up** -- each objective is optimized independently to establish per-objective benchmarks.
3. **Weighted minimax loss** -- normalized shortfalls across all objectives are combined into a single scalar that the Adam optimizer minimizes.
4. **Oscillation switching** -- when any constrained objective exceeds threshold `thC`, the optimizer alternates between minimizing the maximin loss and maximizing constraint satisfaction to escape local optima.

## Project structure

| File                        | Description                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `model.py`                | GDA model core -- softmax variables, loss construction, training loop ([reference](https://github.com/codeRimoe/pSO/tree/main/pMOLU)) |
| `objs.py`                 | Objective function definitions -- five sustainability objectives + conversion-cost constraint                                        |
| `LUloader.py`             | Land-use data loader for vectors and rasters ([reference](https://github.com/codeRimoe/pSO/tree/main/pMOLU))                          |
| `vGDA.ipynb`              | Minimal Jupyter notebook demo -- end-to-end walk-through of a single optimization run with commented steps                           |
| `sensitivity_analysis.py` | Monte Carlo sensitivity experiments for score perturbation and mask contraction                                                      |

## Dependencies

```
numpy >= 1.20
pandas >= 1.3
geopandas >= 0.9
tensorflow == 1.15   # TF v1 API required
matplotlib >= 3.3
tqdm >= 4.60
```

Install:

```bash
pip install numpy pandas geopandas matplotlib tqdm
# For TensorFlow 1.x:
pip install tensorflow==1.15
```

## Quick start

### Running the demo (recommended first step)

Open `vGDA.ipynb` in Jupyter Lab or VS Code. It walks through a complete
optimization in seven annotated cells:

1. Import dependencies and set parameters (`thC`, `objX`)
2. Load land-use data and build weighted objective arrays
3. Build conversion-cost matrix and feasibility mask
4. Create the GDAmodel, run single-objective warm-up, save normalisation constants
5. Set objective preferences and run the main multi-objective training loop
6. Extract the final categorical assignment (`argmax` over probabilities)
7. Attach the result to the original geometry for export

```bash
jupyter lab vGDA.ipynb   # or: jupyter notebook vGDA.ipynb
```

### Running the optimization

```python
from LUloader import LUloader
from model import GDAmodel
import objs

# Load land-use data
CL = LUloader(
    LUs=['Restrict', 'U', 'C', 'T'], asProb=True, LUType='V',
    LUname='luMC_0', LUcol='lu', driver='feather', cmap=Func_cmap, path='CL'
)

# Build weights and constraints (see CL/Ws.csv and CL/CCM.csv)
maskR = np.ones_like(CL.var.LU)
CCM = np.loadtxt('CL/CCM.csv', delimiter=',')
CCM = 1 - CCM / CCM.max()
con = {...}  # see objs.py or sensitivity_analysis.py for full dict

M = {'LU': CL}
Model = GDAmodel(M, con, maskR, objs, thC=1.0, GPU=1, path='.')

# Warm-up -> optimize -> inspect results
Model.init_train(500, lr=0.1)
Model.train(n_iter=1000, thStop=1, nprint=100)
res = Model.log()
```

### Running sensitivity analysis

```bash
python sensitivity_analysis.py
```

Three experiment suites run automatically:

1. **Score sensitivity** -- binary / convex / concave suitability schedules (100 repeats each)
2. **Baseline** -- original unelevated scores (100 repeats)
3. **Mask contraction** -- random removal rates from 0.1% to 5% (100 repeats each)

Results are saved incrementally under `sensitivity/MC_rate{rate}_rs{seed}/`. Interrupted runs resume automatically by skipping completed seeds.

## Data requirements

All input files live in the `CL/` directory (relative to where you run the optimizer).
Below is a complete schema for each required file. Processed datasets used in the
manuscript are also available at https://github.com/geo-188660/STMOC.

### 1. `CL/luX.feather` -- land-use patch inventory

A GeoPandas Feather file (`.feather`) containing one row per candidate rural-homestead
patch. The column `lu` holds the current land-use code (1 = Retention,
2 = Cropland, 3 = Tree/forest). The optimizer treats every non-restricted patch as
a candidate for re-allocation.

| Column        | Type  | Description                                                            | Example                |
| ------------- | ----- | ---------------------------------------------------------------------- | ---------------------- |
| `pointid`   | int   | Unique patch identifier                                                | `42`                 |
| `grid_code` | int   | County/prefecture code (`PAC`), used to look up county-level weights | `350102`             |
| `priority2` | float | Implementation-feasibility score (0--1); higher = easier to act on     | `0.73`               |
| `lu`        | int   | Current land-use category code (1=Retention, 2=Cropland, 3=Forest)     | `1`                  |
| `area`      | float | Patch area in square metres                                            | `12500.0`            |
| `food`      | float | Raw suitability score for cropland reclamation (0–10 scale)           | `7.2`                |
| `tree`      | float | Raw suitability score for reforestation (0–10 scale)                  | `8.5`                |
| `biod`      | float | Biodiversity-conservation capacity index                               | `0.64`               |
| `geometry`  | WKT   | Patch geometry (Point or Polygon)                                      | `POINT (119.3 26.1)` |

**Minimal example** (three rows):

```python
import geopandas as gpd
sample = gpd.GeoDataFrame({
    'pointid': [101, 102, 103],
    'grid_code': [350102, 350102, 350103],
    'priority2': [0.85, 0.42, 0.91],
    'lu':        [1, 1, 1],           # all current usage
    'area':      [15000.0, 8200.0, 21000.0],
    'food':      [6.0, 3.0, 8.0],
    'tree':      [4.0, 7.0, 5.0],
    'biod':      [0.3, 0.8, 0.5],
}, geometry=gpd.points_from_xy([119.3, 119.3, 119.4], [26.1, 26.1, 26.2]))
# save as: sample.to_feather('CL/luX.feather')
```

The feather format is recommended because it preserves GeoPandas geometry types
and writes ~10× faster than CSV for large spatial datasets. Shapefiles and GeoJSON
are also supported by `LUloader`.

---

### 2. `CL/Ws.csv` -- county-level objective weight tables

A CSV indexed by `PAC` (county administrative code). Each row supplies the
county-specific coefficient used to scale the five primary objectives for every
patch in that county. These coefficients come from official statistics (yield,
NEP, biodiversity capacity, operating income, quota revenue).

| Column         | Type  | Meaning                                                      | Unit / Source        |
| -------------- | ----- | ------------------------------------------------------------ | -------------------- |
| `PAC`        | int   | County code (matches `grid_code` in `luX.feather`)       | administrative code  |
| `FOOD`       | float | Grain yield of locally dominant crop                         | kg ha⁻¹            |
| `NEP`        | float | Net ecosystem productivity (carbon-sequestration proxy)      | t C ha⁻¹ yr⁻¹    |
| `BIO`        | float | Biodiversity conservation capacity index                     | dimensionless [0, 1] |
| `FOODVALUEC` | float | Composite ecosystem-value score for cropland reclamation     | index                |
| `FOODVALUED` | float | Disaggregated ecosystem-value score for cropland reclamation | index                |
| `TREEVALUEC` | float | Composite ecosystem-value score for reforestation            | index                |
| `TREEVALUED` | float | Disaggregated ecosystem-value score for reforestation        | index                |

**Minimal example**:

```csv
PAC,FOOD,NEP,BIO,FOODVALUEC,FOODVALUED,TREEVALUEC,TREEVALUED
350102,5200,4.2,0.71,0.65,0.58,0.82,0.74
350103,3100,6.8,0.88,0.41,0.35,0.91,0.85
```

In the code these weights are joined to patches via `wx['grid_code'].apply(lambda x: w[col][x])`.

---

### 3. `CL/CCM.csv` -- conversion-cost matrix

A symmetric numeric matrix loaded with `np.loadtxt(..., delimiter=',')`. Entry
`CCM[i, j]` represents the relative cost (or difficulty) of converting land from
category `i` to category `j`. Values are normalised to `[0, 1]` inside the
optimizer (`CCM = 1 - CCM / CCM.max()`), so that:

- **0** = free / trivial conversion (e.g. retaining current use: `CCM[1,1] = 0`)
- **1** = prohibitively costly (hard constraint; allocation probability stays near zero)

For a three-category problem (Retention → U, Reclamation → C, Reforestation → T)
the raw matrix is typically 3 × 3 or 4 × 4 if Restrict is included:

```csv
0.0,0.9,0.7,0.0
0.9,0.0,0.5,0.0
0.7,0.5,0.0,0.0
0.0,0.0,0.0,0.0
```

After `1 - CCM / max`, entry `(U→C) = 0.1` means switching current to cropland is
cheap; entry `(U→T) = 0.3` means reforestation is more costly but still feasible.

---

### How the three inputs connect in the model

```
luX.feather          Ws.csv              CCM.csv
   │                   │                    │
   │  grid_code join   │                    │
   ▼                   ▼                    │
  wx (per-patch      (county weights        │
   objective         broadcast to           │
   arrays)          every patch)            │
   │                   │                    │
   └───────────────────┴────────────────────┘
                         │
                         ▼
              GDAmodel(M, con, maskR, objs)
                         │
            softmax(θ · sigP) over LU categories
                         │
              Adam-optimize weighted minimax loss
                         │
                         ▼
              argmax → final categorical allocation
```

## References

- Luo, H. & Huang, B. (2023). A probabilistic framework with the gradient-based method for multi-objective land use optimization. *Int. J. Geogr. Inf. Sci.*, 37, 1128-1156.
- Original code: https://github.com/codeRimoe/pSO/tree/main/pMOLU
