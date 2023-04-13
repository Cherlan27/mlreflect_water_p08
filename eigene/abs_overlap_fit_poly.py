7#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:54:02 2018

@author: warias
"""

import numpy
import symfit
            
            
class Absorber(object):
    
    def __init__(self, values={x: 10.0 for x in range(1, 10)}):
        super(Absorber, self).__init__()
        self._values = values
        self._data = []
        
    def __call__(self, n):
        if n == 0:
            return 1.0
        else:
            return self._values[n] * self.__call__(n-1)
            
    def add_dataset(self, abs_value, qz, intensity):
        """
        Add dataset for absorber factor determination from overlaps.
        """
        self._data.append((int(abs_value+0.05), qz, intensity))
        
    def calculate_from_overlaps(self):
        temp = {}
        zero_test = 0
        # compare each dataset with others with abs_value-1
        for n in range(len(self._data)):
            abs_value, qz, intensity = self._data[n]
            for m in range(len(self._data)):
                abs_value_a, qz_a, intensity_a = self._data[m]
                if abs_value_a == abs_value - 1 and zero_test == 0:
                    fit_mask = numpy.where(qz >= qz_a[0])[0]
                    if fit_mask.size == 1:
                        fit_mask = numpy.append(numpy.array([fit_mask[0]-1]), fit_mask)
                    fit_mask_a = numpy.where(qz_a <= qz[len(qz)-1])[0]
                    if fit_mask_a.size == 1:
                        fit_mask_a = numpy.append(fit_mask_a, numpy.array([1]))
                    
                    x_1, x_2, y_1, y_2 = symfit.variables('x_1, x_2, y_1, y_2')
                    a_1, b_1, c_1, u = symfit.parameters('a_1, b_1, c_1, u')
                    globalmodel = symfit.Model({
                        y_1: a_1+b_1*x_1+c_1*x_1**2,
                        y_2: (a_1+b_1*x_2+c_1*x_2**2)*u
                    })
                    
                    globalfit = symfit.Fit(globalmodel, x_1 = qz[fit_mask], x_2 = qz_a[fit_mask_a], y_1 = intensity[fit_mask], y_2 = intensity_a[fit_mask_a])
                    globalfit_result = globalfit.execute()
                    
                    if abs_value not in temp:
                        temp[abs_value] = []
                    temp[abs_value].append(globalfit_result.value(u))
                    
                    if abs_value_a == 0:
                        zero_test =+ 1
        result = {n: temp[n][0] for n in temp.keys()}
        print(result)
        self._values.update(result)