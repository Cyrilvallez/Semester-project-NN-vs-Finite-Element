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

def Load(filename, D):
    """ Load data files in D dimensions"""
    data = np.loadtxt(filename)
    features = data[:,0:D]
    labels = data[:,D]
    
    return features, labels

def split_data_val(features, labels, seed=6, ratio_test=0.1, ratio_val=0.1):
    """ Split a dataset into training, validation and testing according to the given ratio """
    np.random.seed(seed)
    N = len(features)
    stop_test = round(ratio_test*N)
    stop_val = round(ratio_val*N)
    perm = np.random.permutation(N)
    features_test = features[perm[0:stop_test]]
    labels_test = labels[perm[0:stop_test]]
    features_val = features[perm[stop_test:stop_test+stop_val]]
    labels_val = labels[perm[stop_test:stop_test+stop_val]]
    features_train = features[perm[stop_test+stop_val:]]
    labels_train = labels[perm[stop_test+stop_val:]]
    
    return features_train, labels_train, features_val, labels_val, features_test, labels_test


def Load_and_split_val(filename, D, seed=6, ratio_test=0.1, ratio_val=0.1):
    np.random.seed(seed)
    features, labels = Load(filename, D)
    
    return split_data_val(features, labels, seed, ratio_test, ratio_val)
    
    
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


def basis_functions_layer_2D(coordinates, layers):
    """ Compute the basis functions created by an arbitrary layer on the coordinates in 2D
    
    layers : list of all the layers until the layer of interest
    
    For example if we want to visualize what happens in layer 3
    --> layers = [layer1, layer2, layer3] 
    
    """
    x = coordinates
    for i in range(len(layers)):
        x = layers[i](x)
    basis_func = x.numpy().T
    
    N = int(np.sqrt(len(coordinates)))
    basis_func_grid = []
    
    for i in range(len(basis_func)):
        basis_func_grid.append(basis_func[i, :].reshape(N, N))
    
    return np.array(basis_func_grid)


def normalize_basis_2D(basis_functions, tol=1e-8):
    """ Normalize the basis functions as returned by basis_functions_layer_2D so that the functions are in
        the range [0; 1]. Moreover, we only keep functions which are really activated (non-zero) up to tol, so that
        the plots are more efficient.
    """
    basis_func = basis_functions.copy()   # So that we don't modify the original variable
    useful_indices = []
    for i in range(len(basis_func)):
        if (np.linalg.norm(basis_func[i], 'fro') > tol):
            basis_func[i] = (basis_func[i] - np.min(basis_func[i]))/(np.max(basis_func[i]) - np.min(basis_func[i]))
            useful_indices.append(i)
            
    return basis_func[useful_indices]


def basis_functions_radius(r, layers):
    """ Compute the basis functions created by an arbitrary layer on the space interval denoted by the radius r, on the line
        x1 = x2 = x3 = ... = xN  where N is the dimension of the input function
    
    layers : list of all the layers until the layer of interest
    
    For example if we want to visualize what happens in layer 3
    --> layers = [layer1, layer2, layer3] 
    
    """
    r2 = r.copy()
    for i in range(len(layers)):
        r2 = layers[i](r2)
    basis_func = r2.numpy().T
    
    return basis_func