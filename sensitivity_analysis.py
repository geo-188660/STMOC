"""
Sensitivity Analysis: Monte Carlo experiments for the pMOLU model.

Reference:
    Core functions adapted from https://github.com/codeRimoe/pSO/tree/main/pMOLU
    (Luo & Huang, 2023 — A probabilistic framework with the gradient-based
     method for multi-objective land use optimization. Int. J. Geogr. Inf. Sci.)

Runs sensitivity analyses to assess result robustness against:
  1. Suitability-score perturbation (binary / convex / concave schedules)
  2. Mask contraction (random removal of candidate patches at various rates)
Each experiment repeats 100 times with different seeds and saves results incrementally.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
from multiprocessing import Process, Queue
from tqdm import tqdm

# ============================================================
# Global configuration (passed to child processes via arguments)
# ============================================================
Func_types = ['Restrict', 'U', 'C', 'T']
Func_cmap = ['white', 'lightgray', 'springgreen', 'forestgreen']
thC = 100

def power_scores(gamma, S_min=2, S_max=10, R=5):
    """Generate suitability scores using a power-function schedule."""
    r = np.arange(1, R + 1)
    return S_min + (S_max - S_min) * ((r - 1) / (R - 1)) ** gamma

suitX = dict(
    convex=power_scores(0.5),   # Convex: optimistic -- middle ranks score relatively high
    concave=power_scores(2.0),  # Concave: conservative -- only top ranks score highly
    binary=[0, 0, 0, 10, 10],   # Binary: only highest two ranks receive positive scores
)


# ============================================================
# Single-run worker (executed in an isolated subprocess)
# ============================================================
def run_single(path, rseed, frac, suit, objX, status_queue):
    """Run one complete pMOLU optimization in an independent subprocess."""
    try:
        # Re-establish environment in child process
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import sys
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)

        import numpy as np
        import pandas as pd
        import geopandas as gpd
        from LUloader import LUloader
        from model import GDAmodel
        import objs

        Func_types = ['Restrict', 'U', 'C', 'T']
        Func_cmap = ['white', 'lightgray', 'springgreen', 'forestgreen']
        thC = 100

        suitX_local = dict(
            convex=[2, 5, 7, 9, 10],
            concave=[2, 3, 5, 7, 10],
            binary=[0, 0, 0, 10, 10],
        )

        # Load base land-use data
        CLraw = gpd.read_feather('CL/luX.feather')
        if frac < 1.0:
            CLraw = CLraw.sample(frac=frac, random_state=rseed)

        mc_feather = f'CL/luMC_{rseed}.feather'
        CLraw.to_feather(mc_feather)

        CL = LUloader(
            LUs=Func_types, asProb=True, LUType='V',
            LUname=f'luMC_{rseed}', LUcol='lu',
            driver='feather', cmap=Func_cmap, path='CL'
        )

        # Build weight vectors
        w = pd.read_csv('CL/Ws.csv').set_index('PAC')
        wx = CL.raw.copy()

        if suit is not None:
            base = {i: j for i, j in zip([2, 4, 6, 8, 10], suitX_local[suit])}
            wx['food'] = wx['food'].map(base)
            wx['tree'] = wx['tree'].map(base)

        wx['grid_code'] = wx['grid_code'].astype(int)

        wx['food'] = wx['priority2'] * wx['food'] / 10 * wx['grid_code'].apply(lambda x: w['FOOD'][x])
        wx['tree'] = wx['priority2'] * wx['tree'] / 10 * wx['grid_code'].apply(lambda x: w['NEP'][x])
        wx['biod'] = wx['priority2'] * wx['tree'] / 10 * wx['grid_code'].apply(lambda x: w['BIO'][x])
        wx['foodvaluec'] = wx['priority2'] * wx['food'] / 10 * wx['grid_code'].apply(lambda x: w['FOODVALUEC'][x])
        wx['foodvalued'] = wx['priority2'] * wx['food'] / 10 * wx['grid_code'].apply(lambda x: w['FOODVALUED'][x])
        wx['treevaluec'] = wx['priority2'] * wx['tree'] / 10 * wx['grid_code'].apply(lambda x: w['TREEVALUEC'][x])
        wx['treevalued'] = wx['priority2'] * wx['tree'] / 10 * wx['grid_code'].apply(lambda x: w['TREEVALUED'][x])

        wx = wx[['pointid', 'food', 'tree', 'biod',
                 'foodvaluec', 'foodvalued', 'treevaluec', 'treevalued', 'priority2']]

        maskR = np.ones_like(CL.var.LU)
        CCM = np.loadtxt('CL/CCM.csv', delimiter=',')
        CCM = 1 - CCM / CCM.max()

        M = {'LU': CL}
        con = dict(
            food=wx['food'].to_numpy(), tree=wx['tree'].to_numpy(), biod=wx['biod'].to_numpy(),
            foodvaluec=wx['foodvaluec'].to_numpy(), foodvalued=wx['foodvalued'].to_numpy(),
            treevaluec=wx['treevaluec'].to_numpy(), treevalued=wx['treevalued'].to_numpy(),
            CCM=CCM, oLU=CL.var.LU.to_numpy().copy()
        )
        Model = GDAmodel(M, con, maskR, objs, thC=thC / 100., GPU=1, path='CL')

        # Warm-up: single-objective benchmarks
        Model.init_train(500, lr=0.1)

        # Main optimization with optional objective emphasis
        pre = [1] * len(objs.n_obj[0])
        if objX is not None:
            pre[objX] = 1.2
        Model.perference(pre)
        Model.train(n_iter=1000, thStop=1, nprint=None)

        # Extract and save results
        LUmax = pd.Series(
            Model.get_opt()['LU'].argmax(axis=1),
            index=Model.M['LU'].raw.pointid, name='LU'
        )
        res = Model.eval_objs()

        LUmax.to_frame().reset_index().to_csv(f'{path}/LU/{rseed}.csv', index=False)
        np.save(f'{path}/obj/{rseed}', np.array(res[0]))

        # Cleanup temp file
        if os.path.exists(mc_feather):
            os.remove(mc_feather)

        status_queue.put(('success', rseed))

    except Exception as e:
        status_queue.put(('error', rseed, str(e)))


# ============================================================
# Monte Carlo dispatcher
# ============================================================
def MC(rate=0.01, rep=100, random_state=42, suit=None, objX=None):
    """
    Run Monte Carlo experiments sequentially in subprocesses.
    Results are saved incrementally so interrupted runs can be resumed.
    """
    suffix = f'_{"suit" + suit}' if suit else ''
    path = f'sensitivity/MC_rate{rate}_rs{random_state}{suffix}'
    os.makedirs(f'{path}/LU', exist_ok=True)
    os.makedirs(f'{path}/obj', exist_ok=True)

    frac = 1.0 - rate
    status_queue = Queue()

    pbar = tqdm(range(rep), desc=path)
    for i in pbar:
        rseed = random_state + i

        # Skip already completed runs
        if os.path.exists(f'{path}/obj/{rseed}.npy'):
            pbar.set_postfix_str(f'seed={rseed} [skip]')
            continue

        p = Process(target=run_single, args=(path, rseed, frac, suit, objX, status_queue))
        p.start()
        p.join()  # Sequential execution; GPU memory released each run

        if not status_queue.empty():
            result = status_queue.get()
            if result[0] == 'success':
                pbar.set_postfix_str(f'seed={result[1]} [done]')
            else:
                pbar.set_postfix_str(f'seed={result[1]} [ERROR: {result[2]}]')
                print(f"\n  ERROR at seed={result[1]}: {result[2]}")

        if p.exitcode != 0:
            print(f"\n  WARNING: subprocess exited with code {p.exitcode} at seed={rseed}")


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    # Suitability-score sensitivity: binary, convex, concave
    for s in ['binary', 'convex', 'concave']:
        MC(rate=0, rep=100, random_state=42, suit=s)

    # Baseline: original unelevated scores
    MC(rate=0, rep=100, random_state=42, suit=None)

    # Mask contraction: increasing severity levels
    for r in [0.001, 0.002, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02, 0.05]:
        MC(rate=r, rep=100, random_state=42)
