# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:56:03 2023

@author: petersdorf
"""


import mlreflect
from mlreflect.utils import check_gpu
import matplotlib.pyplot as plt
import numpy as np
from mlreflect.data_generation import Layer, Substrate, AmbientLayer, MultilayerStructure
from mlreflect.training import Trainer
from mlreflect.data_generation import ReflectivityGenerator
from mlreflect.curve_fitter import CurveFitter
import pandas as pd
from mlreflect.models import TrainedModel

class prediction_sample():
    
    def __init__(self, qz, inties):

path_file = "trained_0_layer.h5"

model = TrainedModel()
model.from_file(path_file)
curve_fitter = CurveFitter(model)

df= pd.read_csv('Data/rbbr_Reflectivity_rbbr_3mol_s1_3_off.dat', sep = "\t")
q_exp = np.array(df["//qz"])[:-3]
intensity_exp = np.array(df["intensity_normalized"])[:-3]/7.58493186e-01

experimental_fit_output = curve_fitter.fit_curve(intensity_exp, q_exp, polish = True, optimize_q = True)
pred_experimental_reflectivity = experimental_fit_output["predicted_reflectivity"]
pred_experimental_test_labels = experimental_fit_output["predicted_parameters"]

fig = plt.figure(dpi = 300)
plt.semilogy(q_exp, intensity_exp, 'o', markerfacecolor = "white", markeredgecolor = "blue", label = "Experiment")
plt.semilogy(q_exp, pred_experimental_reflectivity[0], label = "Prediction", color="red")
plt.legend()
plt.xlabel("q [1/A]")
plt.ylabel("Reflectivity [norm]")
plt.show()

print(pred_experimental_test_labels)