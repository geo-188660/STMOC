# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
This code include implemented objective functions for cultivated land optimization study in the artile:


"""
import numpy as np
import tensorflow as tf
n_obj = [["food", "nep", "bio", "ecoc", "ecod"], ["Cov"]]
pre = [1, 1, 1, 1, 1]

# Objective: conversion (constraint)
def coversion_(A, B, CCM, weights=None):
    if weights is not None:
        A += np.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = np.transpose(np.reshape(A, (-1, z1)))
    B = np.reshape(B, (-1, z2))
    ABm = np.matmul(A, B) * CCM
    ABm = np.sum(ABm, axis=-1)
    return np.sum(ABm)

# Objective: conversion (constraint) Tensorflow implement
def coversion(A, B, CCM, weights=None):
    if weights is not None:
        A += tf.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = tf.transpose(tf.reshape(A, (-1, z1)))
    B = tf.reshape(B, (-1, z2))
    ABm = tf.matmul(A, B) * CCM
    ABm = tf.reduce_sum(ABm, axis=-1)
    return tf.reduce_sum(ABm)


# for gradient desent
def Objs(self):
    foodland, treeland = self.V['LU'][:, 1], self.V['LU'][:, 2]
    food = tf.reduce_sum(self.W['food'] * foodland)
    nep = tf.reduce_sum(self.W['tree'] * treeland)
    bio = tf.reduce_sum(self.W['biod'] * treeland)
    ecoc = tf.reduce_sum(self.W['foodvaluec'] * foodland) + tf.reduce_sum(self.W['treevaluec'] * treeland) 
    ecod = tf.reduce_sum(self.W['foodvalued'] * foodland) + tf.reduce_sum(self.W['treevalued'] * treeland) 
    
    #gdp = tf.reshape(foodland, (1, -1)) @ self.W['p'] * 180000 + tf.reshape(treeland, (1, -1)) @ self.W['p'] * 30000 + self.W['gdp']
    #gdp /= tf.reduce_mean(gdp)
    #TI = tf.reduce_mean(gdp * tf.math.log(gdp))
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

    #gdp = foodland @ self.W_['p'] * 180000 + treeland @ self.W_['p'] * 30000 + self.W_['gdp']
    #gdp /= gdp.mean()
    #TI = np.mean(gdp * np.log(gdp))
    cov = coversion_(var, self.W_['oLU'], self.W_['CCM'])
    objs = [[-food, -nep, -bio, -ecoc, -ecod], [cov]]

    return objs