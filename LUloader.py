# -*- coding: utf-8 -*-
"""
Land-Use Data Loader (LUloader)

Reference:
    Adapted from https://github.com/codeRimoe/pSO/tree/main/pMOLU
    (Luo & Huang, 2023 — A probabilistic framework with the gradient-based
     method for multi-objective land use optimization. Int. J. Geogr. Inf. Sci.)

This module provides utilities for loading, processing, and visualizing
land-use data in both vector and raster formats.
"""
import os
try:
    from osgeo import gdal
except Exception:
    gdal = None
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LUloader:
    '''
    path
    z_LU: 0 must be Restrict or just skip and begin from 1
    isProb
    VorR
    raw
    var
    '''
    def __init__(self, LUs, asProb, LUType, noR=False, path='.', cmap=None, **kwargs):
        self.path = path
        self.z_LU = len(LUs)
        if not noR:
           self.z_LU -= 1
        self.legend = LUs
        self.cmap = cmap
        self.isProb = asProb
        self.VorR = LUType
        self.noR = noR
        self.load(**kwargs)
        # try:
        #     self.load(**kwargs)
        # except:
        #     print('Load Error.')
        #     pass
    
    def load(self, LUname, driver, **kwargs):
        if self.VorR == 'R':
            self.load_raster(LUname, driver=driver, **kwargs)
        elif self.VorR == 'V':
            self.load_vector(LUname, driver=driver, **kwargs)


    def load_vector(self, LUname, LUcol, LID=None, driver='shapefile', encoding='utf-8', **kwargs):
        # kwargs: resample(int/2d-array), mask(int)
        #         LUcol(str), encoding(str)
        print(f'Loading vector landuse: {LUname}, LUCol: {LUcol}, Driver: {driver}/{encoding}')
        LUname = os.path.join(self.path, LUname)
        # read the raw data
        if driver == 'feather':
            vector = gpd.read_feather(f'{LUname}.feather')
        else:
            vector = gpd.read_file(LUname, driver=driver, encoding=encoding)
        if LID is not None:
            vector = vector.set_index(LID)
        # make the variable data
        _LU = np.array(vector[LUcol], dtype=np.int8)
        if self.noR:
            _LU += 1
        _R = _LU == 0
        if self.isProb:
            _LU = pd.DataFrame(self.prob_LU(_LU), index=vector.index, columns=pd.MultiIndex.from_product([["LU"], [i + 1 for i in range(self.z_LU)]]))
            _LU['R'] = _R
            gdf = gpd.GeoDataFrame(_LU, geometry=vector.geometry, crs=vector.crs)
        else:
            gdf = gpd.GeoDataFrame({'LU': _LU, 'R': _R}, geometry=vector.geometry, crs=vector.crs)
        self.raw = vector
        self.var = gdf
        print(f"Loaded vector: {LUname}[{driver}/{encoding}: {len(self.raw)}]")

    def load_raster(self, LUname, mask=None, resample=1, driver='TIF', **kwargs):
        # kwargs: resample(int/2d-array), mask(int)
        #         LUcol(str), encoding(str)
        print(f'Loading raster landuse: {LUname}, MaskDN: {mask}, Driver: {driver}/{resample}x')
        LUname = os.path.join(self.path, LUname)
        raster = read_raster(LUname, driver=driver, resample=resample, mask=mask)
        _LU = np.array(raster['img'], dtype=np.int8)
        _R = _LU == 0
        _M = _LU == mask
        if mask is not None:
            _LU[_M] = 0
        if self.isProb:
            _LU = self.prob_LU(_LU)
        self.raw = raster
        self.var = {'LU': _LU, 'R': _R, 'M': _M}
        print(f"Loaded raster: {LUname}[{driver}/{resample}x: {self.var['LU'].shape}]")

    def save_vector(self, name, optLU=None, driver='shapefile', encoding='utf-8'):
        print(f'Saving {name}({driver})')
        optLU = self.to_LU(optLU)
        name = os.path.join(self.path, name)
        self.var['opt'] = optLU
        if driver == 'feather':
            self.var.to_feather(f'{name}.feather')
        self.var.to_file(name, driver=driver, encoding=encoding)

    def save_raster(self, name, optLU=None, driver='TIF'):
        print(f'Saving {name}({driver})')
        optLU = self.to_LU(optLU, self.raw['mask'])
        name = os.path.join(self.path, name)
        write_raster(name, optLU, self.raw, driver=driver)

    def save(self, name, optLU=None, **kwargs):
        if self.VorR == 'V':
            self.save_vector(name, optLU, **kwargs)
        if self.VorR == 'R':
            self.save_raster(name, optLU, **kwargs)

    def prob_LU(self, LU=None):
        if LU is None:
            LU = self.var['LU']
            if self.isProb == True:
                return np.array(LU, dtype=np.float32)
        LU = np.array(LU, dtype=np.int8)
        LU = np.array([LU == (c + 1) for c in range(self.z_LU)], dtype=np.float32)
        return np.moveaxis(LU, 0, -1)
    
    def cate_LU(self, pLU=None):
        if pLU is None:
            pLU = self.var['LU']
            if self.isProb == False:
                return np.array(pLU, dtype=np.int8)
        LU = np.argmax(np.array(pLU, dtype=np.float32), axis=-1) + 1
        return np.array(LU, dtype=np.int8)

    def to_LU(self, LU=None, mask=None):
        if LU is None:
            LU = self.cate_LU(LU)
        _R = np.array(self.var['R'], dtype=bool)
        LU[_R] = 0
        if self.VorR == 'R':
            LU[self.var['M']] = self.z_LU + 1 if mask is None else mask
        return LU

    def plot_pLU(self, pLU=None, show=True, save_name=None, bg=None, bgarg={}, legendOrient='horizontal', path='plot'):
        LU = self.cate_LU(pLU)
        self.plot_LU(LU=LU, show=show, save_name=save_name, bg=bg, bgarg=bgarg, legendOrient=legendOrient, path=path)

    def plot_LU(self, LU=None, show=True, save_name=None, bg=None, bgarg={}, legendOrient='horizontal', path='plot'):
        LU = self.to_LU(LU)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.axis('off')
        if bg is not None:
            bg.plot(ax=ax, **bgarg)
        M = self.z_LU + 1
        try:
            cmap = ListedColormap(self.cmap)
        except:
            print('Use default colormap.')
            cmap = None
        # if legend is None:
        #     print('Use default legend.')
        #     legend = lgend
        if self.VorR == 'V':
            try:
                self.var.plot(LU, ax=ax, cmap=cmap, vmin=0, vmax=M, legend=False) # TODO: legend
            except:
                self.raw.plot(LU, ax=ax, cmap=cmap, vmin=0, vmax=M, legend=False)
        elif self.VorR == 'R':
            ax.imshow(LU, cmap=cmap, vmin=0, vmax=M, interpolation='none')
            # TODO:legend
            plt.colorbar(ax=ax, ticks=[0,0.5, 1], orientation=legendOrient).set_ticklabels(legend)
        fig.tight_layout()
        if save_name is not None:
            path = os.path.join(self.path, path)
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, save_name))
        if show:
            fig.show()

# %%
