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

#----------------------------------------- METRICS FUNCTIONS ----------------------------------------------------------

def MSE(model, features, labels):
    """ Computes the MSE of a model on the given features/labels """
    pred = model.predict(features)
    labels = np.expand_dims(labels, axis=-1)
    assert(pred.shape == labels.shape)
    mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
    
    return mse(labels, pred).numpy()

def MAE(model, features, labels):
    """ Computes the MAE of a model on the given features/labels """
    pred = model.predict(features)
    labels = np.expand_dims(labels, axis=-1)
    assert(pred.shape == labels.shape)
    mae = tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size')
    
    return mae(labels, pred).numpy()

def MaAE(model, features, labels):
    """ Computes the MaAE of a model on the given features/labels """
    pred = model.predict(features)
    labels = np.expand_dims(labels, axis=-1)
    assert(pred.shape == labels.shape)
    mae = tf.keras.losses.MeanAbsoluteError(reduction='none')
    error = mae(labels, pred).numpy()
    
    return np.max(error)

#----------------------------------------- BASIS FUNCTIONS ----------------------------------------------------------

def basis_functions_layer(x, layers):
    """ Compute the basis functions created by an arbitrary layer on the space interval denoted by x
    
    layers : list of all the layers until the layer of interest
    
    For example if we want to visualize what happens in layer 3
    --> layers = [layer1, layer2, layer3] 
    
    """
    x2 = np.expand_dims(x, axis=-1)
    for i in range(len(layers)):
        x2 = layers[i](x2)
    basis_func = x2.numpy().T
    
    return basis_func

def basis_functions_conv_layer(x, layer1, layer2):
    """ Compute the basis functions created by a dense followed by conv layer on the space interval denoted by x """
    x2 = np.expand_dims(x, axis=-1)
    x3 = layer1(x2)
    x4 = tf.expand_dims(x3, axis=-1)
    x5 = layer2(x4)
    x6 = tf.squeeze(x5, axis=-1)
    basis_func = x6.numpy().T
    
    return basis_func
