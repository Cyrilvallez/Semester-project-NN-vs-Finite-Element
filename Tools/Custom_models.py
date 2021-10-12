#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue October 5 2021

@author: cyrilvallez
__________
This files contains different model architecture
"""

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

#-----------------------------------------------------------------------

class Model_1D(tf.keras.Model):
    
    def __init__(self, N_nodes, N_output=1, reg_kernel=False, reg_biases=False, reg_value=0.01):
        super(Model_1D, self).__init__()
        self.dense1 = Dense(N_nodes, activation='relu', name='dense1')
        if (reg_kernel & reg_biases):
            self.dense2 = Dense(N_nodes - 2, activation='relu',
            kernel_regularizer=regularizers.l1(l1=reg_value),
            bias_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        elif (reg_kernel & (not reg_biases)):
            self.dense2 = Dense(N_nodes - 2, activation='relu',
            kernel_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        else:
            self.dense2 = Dense(N_nodes - 2, activation='relu', name='dense2')
        self.dense3 = Dense(N_output, activation=None, name='dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = self.dense2(x)
        x3 = self.dense3(x2)
        return x3
                      
#-----------------------------------------------------------------------     
       
class Model_1D_non_active(tf.keras.Model):
    
    def __init__(self, N_nodes, N_output=1, reg_kernel=False, reg_biases=False, reg_value=0.01):
        super(Model_1D_non_active, self).__init__()
        self.dense1 = Dense(N_nodes, activation='relu', name='dense1')
        if (reg_kernel & reg_biases):
            self.dense2 = Dense(N_nodes - 2, activation=None,
            kernel_regularizer=regularizers.l1(l1=reg_value),
            bias_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        elif (reg_kernel & (not reg_biases)):
            self.dense2 = Dense(N_nodes - 2, activation=None,
            kernel_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        else:
            self.dense2 = Dense(N_nodes - 2, activation=None, name='dense2')
        self.dense3 = Dense(N_output, activation=None, name='dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = self.dense2(x)
        x3 = self.dense3(x2)
        return x3
                      
#-----------------------------------------------------------------------  

class Model_1D_full_active(tf.keras.Model):
    
    def __init__(self, N_nodes, N_output=1, reg_kernel=False, reg_biases=False, reg_value=0.01):
        super(Model_1D_full_active, self).__init__()
        self.dense1 = Dense(N_nodes, activation='relu', name='dense1')
        if (reg_kernel & reg_biases):
            self.dense2 = Dense(N_nodes - 2, activation='relu',
            kernel_regularizer=regularizers.l1(l1=reg_value),
            bias_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        elif (reg_kernel & (not reg_biases)):
            self.dense2 = Dense(N_nodes - 2, activation='relu',
            kernel_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        else:
            self.dense2 = Dense(N_nodes - 2, activation='relu', name='dense2')
        self.dense3 = Dense(N_output, activation='relu', name='dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = self.dense2(x)
        x3 = self.dense3(x2)
        return x3
                      
#-----------------------------------------------------------------------   