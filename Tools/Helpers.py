#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue October 19 2021

@author: cyrilvallez
__________
This files contains helpers functions
"""

import numpy as np
import tensorflow as tf

#------------------------------------------ LOADING DATA ----------------------------------------------------------

def Load1D(filename):
    """ Load 1D data files """
    data = np.loadtxt(filename)
    features = data[:,0]
    labels = data[:,1]
    
    return features, labels

def split_data(features, labels, seed=6, ratio=0.9):
    """ Split a dataset into training and testing according to the ratio given """
    np.random.seed(seed)
    N = len(features)
    stop = round(ratio*N)
    perm = np.random.permutation(N)
    features_training = features[perm[0:stop]]
    features_testing = features[perm[stop:N]]
    labels_training = labels[perm[0:stop]]
    labels_testing = labels[perm[stop:N]]
    
    return features_training, labels_training, features_testing, labels_testing

def Load_and_split_1D(filename, seed=6, ratio=0.9):
    """ Convenient shortcut for loading and spliting data in 1D """
    np.random.seed(seed)
    features, labels = Load1D(filename)
    
    return split_data(features, labels, seed, ratio)
    
    
#------------------------------------------ FEM functions ----------------------------------------------------------

def weights_FEM_first_layer_1D(N=100):
    """ Creates the weights for an exact FEM initialization of the first dense layer
    in 1D using a grid of N points """
    core1 = np.ones(N)
    core1 = np.expand_dims(core1, axis=0)
    x = np.linspace(0, 1, N)
    bias1 = -x
    
    return [core1, bias1]

#----------------------------------------- LOSSES FUNCTIONS ----------------------------------------------------------

def MSE(model, features, labels):
    """ Computes the MSE of a model on the given features/labels """
    pred = model.predict(features).ravel()
    assert(pred.shape == labels.shape)
    mse = tf.keras.losses.MeanSquaredError()
    
    return mse(labels, pred)

def MAE(model, features, labels):
    """ Computes the MAE of a model on the given features/labels """
    pred = model.predict(features).ravel()
    assert(pred.shape == labels.shape)
    mae = tf.keras.losses.MeanAbsoluteError()
    
    return mae(labels, pred)

#----------------------------------------- BASIS FUNCTIONS ----------------------------------------------------------

def basis_functions_first_layer(x, layer1):
    """ Compute the basis functions created by layer 1 on the space interval denoted by x """
    x2 = np.expand_dims(x, axis=-1)
    basis_func = layer1(x2)

    return basis_func.numpy().T

def basis_functions_second_layer(x, layer1, layer2, conv=False):
    """ Compute the basis functions created by layer 1 and 2 on the space interval denoted by x """
    x2 = np.expand_dims(x, axis=-1)
    x3 = layer1(x2)
    if (conv):
        x4 = tf.expand_dims(x3, axis=-1)
        x5 = layer2(x4)
        x6 = tf.squeeze(x5, axis=-1)
        basis_func = x6.numpy().T
    else:
        x4 = layer2(x3)
        basis_func = x4.numpy().T
    
    return basis_func
