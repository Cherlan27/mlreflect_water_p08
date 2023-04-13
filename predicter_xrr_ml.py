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
import pandas

class prediction_sample():
    def __init__(self, qz, inties, inties_e, scan_number, pathfile = "trained_0_layer.h5", save_directory = "./processed/"):
        self.qz = qz
        self.inties = inties
        self.inties_e = inties_e
        self.path_file = pathfile
        self.scan_number = scan_number
        self.save_directory = save_directory

    def __call__(self):
        model = TrainedModel()
        model.from_file(self.path_file)
        curve_fitter = CurveFitter(model)
    
        experimental_fit_output = curve_fitter.fit_curve(self.inties, self.qz, polish = True, optimize_q = True)
        pred_experimental_reflectivity = experimental_fit_output["predicted_reflectivity"]
        pred_experimental_test_labels = experimental_fit_output["predicted_parameters"]
        
        fig = plt.figure(dpi = 300)
        plt.semilogy(self.qz, self.inties, 'o', markerfacecolor = "white", markeredgecolor = "blue", label = "Experiment")
        plt.semilogy(self.qz, pred_experimental_reflectivity[0], label = "Prediction", color="red")
        plt.legend()
        plt.xlabel("q [1/A]")
        plt.ylabel("Reflectivity [norm]")
        plt.show()
        save_directory_comp = self.save_directory + str(self.scan_number) + ".png"
        plt.savefig(save_directory_comp)
        
        out_filename = self.save_directory + str(self.scan_number) + "xrr_data.dat"
        df = pandas.DataFrame()
        df["qz"] = self.qz
        df["intensity_normalized"] = self.inties
        df["e_intensity_normalized"] = self.inties_e
        df.to_csv(out_filename, sep="\t", index=False)
        
        print(pred_experimental_test_labels)