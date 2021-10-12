#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:39:26 2021

@author: cyrilvallez
"""

import matplotlib.pyplot as plt

#plt.rc('font', family=['serif'])
#plt.rc('font', serif=['Computer Modern Roman'])
plt.rc('savefig', dpi=1000)
plt.rc('figure', titlesize=20)
plt.rc('figure', dpi=100)
plt.rc('legend', fontsize=15)
plt.rc('lines', linewidth=2.5)
plt.rc('lines', markersize=7)
plt.rc('axes', labelsize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

#### print(plt.rcParams) FOR A FULL LIST OF PARAMETERS