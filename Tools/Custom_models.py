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
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras import regularizers

#---------------------------------------------- CUSTOM SCHEDULERS ------------------------------------------------------------

def step_scheduler(step, rate):
    """ Step learning rate scheduler. Every step epochs, it decreases the learning rate by rate.
        It returns a function, to be used in a Callback during training """
    def scheduler(epoch, lr):
        if (epoch % step == 0 and epoch>0):
            return lr*rate
        else:
            return lr
        
    return scheduler


def exponential_scheduler(start, rate):
    """ exponential learning rate scheduler. after start epochs, it decreases the learning rate by rate every epoch.
        It returns a function, to be used in a Callback during training """
    def scheduler(epoch, lr):
        if (epoch < start):
            return lr
        else:
            return lr*rate
        
    return scheduler

#------------------------------------------------ CUSTOM INITIALIZATION -------------------------------------------------------

def my_bias_initializer_1D(shape, dtype=None):
    """ Custom initialization for the non-trainable first layer in 1D"""
    N = shape[-1]
    bias = -tf.linspace(-1, 1, N)
    return tf.cast(bias, dtype=dtype)
    
#------------------------------------------------------MODELS -------------------------------------------------------------------    

class Model_depth(tf.keras.Model):
    
    def __init__(self, K1, K_output=1, depth=1, name=None):
        super(Model_depth, self).__init__(name=name)
        self.sequence = tf.keras.Sequential(name='sequential')
        
        for i in range(depth):
            self.sequence.add(Dense(K1, activation='relu', name=f'dense{i+1}'))
        self.sequence.add(Dense(K_output, activation=None, name=f'dense{depth+1}'))

    def call(self, inputs):
        x = self.sequence(inputs)
        return x

#----------------------------------------------------------------------- 

class Model_depth_reg(tf.keras.Model):
    
    def __init__(self, K1, K_output=1, depth=1, reg=1e-2, name=None):
        super(Model_depth_reg, self).__init__(name=name)
        self.sequence = tf.keras.Sequential(name='sequential')
        
        for i in range(depth):
            self.sequence.add(Dense(K1, activation='relu', name=f'dense{i+1}', 
                                    kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.sequence.add(Dense(K_output, activation=None, name=f'dense{depth+1}',
                               kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))

    def call(self, inputs):
        x = self.sequence(inputs)
        return x

#----------------------------------------------------------------------- 
       
class Model_1_layer_trainable(tf.keras.Model):
    
    def __init__(self, K1, K_output=1, train_first_layer=True, name=None):
        super(Model_1D_1_layer, self).__init__(name=name)
        if (train_first_layer):
            self.dense1 = Dense(K1, activation='relu', name='dense1')
        else:
            self.dense1 = Dense(K1, activation='relu', name='dense1', trainable=False, kernel_initializer='ones', 
                                bias_initializer=my_bias_initializer_1D)
        self.dense2 = Dense(K_output, activation=None, name='dense2')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = self.dense2(x)
        return x2
                      
#----------------------------------------------------------------------- 

class Model_conv(tf.keras.Model):
    
    def __init__(self, K1, kernel=3, K_output=1, train_first_layer=True, name=None):
        super(Model_1D_conv, self).__init__(name=name)
        if (train_first_layer):
            self.dense1 = Dense(K1, activation='relu', name='dense1')
        else:
            self.dense1 = Dense(K1, activation='relu', name='dense1', trainable=False, kernel_initializer='ones', 
                                bias_initializer=my_bias_initializer_1D)
        self.conv = Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation=None, name='conv')
        self.dense2 = Dense(K_output, activation=None, name='dense2')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = tf.expand_dims(x, axis=-1)
        x3 = self.conv(x2)
        x4 = tf.squeeze(x3, axis=-1)
        x5 = self.dense2(x4)
        return x5
                      
#----------------------------------------------------------------------- 

class Model_2_layers_reg(tf.keras.Model):
    
    def __init__(self, K1, K_output=1, reg_kernel=False, reg_biases=False, reg_value=0.01, name=None):
        super(Model_1D_2_layers, self).__init__(name=name)
        self.dense1 = Dense(K1, activation='relu', name='dense1')
        if (reg_kernel & reg_biases):
            self.dense2 = Dense(K1, activation='relu',
            kernel_regularizer=regularizers.l1(l1=reg_value),
            bias_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        elif (reg_kernel & (not reg_biases)):
            self.dense2 = Dense(K1, activation='relu',
            kernel_regularizer=regularizers.l1(l1=reg_value), name='dense2')
        else:
            self.dense2 = Dense(K1, activation='relu', name='dense2')
        self.dense3 = Dense(K_output, activation=None, name='dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = self.dense2(x)
        x3 = self.dense3(x2)
        return x3

#----------------------------------------------------------------------- 

class Model_2_layers_trainable(tf.keras.Model):
    
    def __init__(self, K1, K_output=1, train_first_layer=True, name=None):
        super(Model_1D_2_layers_trainable, self).__init__(name=name)
        if (train_first_layer):
            self.dense1 = Dense(K1, activation='relu', name='dense1')
        else:
            self.dense1 = Dense(K1, activation='relu', name='dense1', trainable=False, kernel_initializer='ones', 
                                bias_initializer=my_bias_initializer_1D)
        self.dense2 = Dense(K1 - 2, activation='relu', name='dense2')
        self.dense3 = Dense(K_output, activation=None, name='dense3')

    def call(self, inputs):
        x = self.dense1(inputs)
        x2 = self.dense2(x)
        x3 = self.dense3(x2)
        return x3
    
#-----------------------------------------------------------------------
                  
