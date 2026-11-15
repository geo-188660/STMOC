# -*- coding: utf-8 -*-
"""
Objective functions for the pMOLU model (STMOC application).

Reference:
    Adapted from https://github.com/codeRimoe/pSO/tree/main/pMOLU
    (Luo & Huang, 2023 — A probabilistic framework with the gradient-based
     method for multi-objective land use optimization. Int. J. Geogr. Inf. Sci.)

Defines five sustainability objectives and one conversion-cost constraint
used in the STMOC spatiotemporal optimization framework.
"""
import numpy as np
import tensorflow as tf
n_obj = [["food", "nep", "bio", "ecoc", "ecod"], ["Cov"]]
pre = [1, 1, 1, 1, 1]

# Conversion cost: NumPy implementation (for post-optimization evaluation)
def coversion_(A, B, CCM, weights=None):
    if weights is not None:
        A += np.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = np.transpose(np.reshape(A, (-1, z1)))
    B = np.reshape(B, (-1, z2))
    ABm = np.matmul(A, B) * CCM
    ABm = np.sum(ABm, axis=-1)
    return np.sum(ABm)

# Conversion cost: TensorFlow implementation (for gradient-based optimization graph)
def coversion(A, B, CCM, weights=None):
    if weights is not None:
        A += tf.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = tf.transpose(tf.reshape(A, (-1, z1)))
    B = tf.reshape(B, (-1, z2))
    ABm = tf.matmul(A, B) * CCM
    ABm = tf.reduce_sum(ABm, axis=-1)
    return tf.reduce_sum(ABm)


# pMOLU objective construction (called by GDAmodel.build_objs)
def Objs(self):
    foodland, treeland = self.V['LU'][:, 1], self.V['LU'][:, 2]
    food = tf.reduce_sum(self.W['food'] * foodland)
    nep = tf.reduce_sum(self.W['tree'] * treeland)
    bio = tf.reduce_sum(self.W['biod'] * treeland)
    ecoc = tf.reduce_sum(self.W['foodvaluec'] * foodland) + tf.reduce_sum(self.W['treevaluec'] * treeland)
    ecod = tf.reduce_sum(self.W['foodvalued'] * foodland) + tf.reduce_sum(self.W['treevalued'] * treeland)

    cov = coversion(self.V['LU'], self.W['oLU'], self.W['CCM'])
    objs = [[-food, -nep, -bio, -ecoc, -ecod], [cov]]
    return n_obj, objs

def Eval(self, var):
    var = var['LU']
    foodland, treeland = var[:, 1], var[:, 2]
    food = np.sum(self.W_['food'] * foodland)
    nep = np.sum(self.W_['tree'] * treeland)
    bio = np.sum(self.W_['biod'] * treeland)
    ecoc = np.sum(self.W_['foodvaluec'] * foodland) + np.sum(self.W_['treevaluec'] * treeland)
    ecod = np.sum(self.W_['foodvalued'] * foodland) + np.sum(self.W_['treevalued'] * treeland)

    cov = coversion_(var, self.W_['oLU'], self.W_['CCM'])
    objs = [[-food, -nep, -bio, -ecoc, -ecod], [cov]]

    return objs
