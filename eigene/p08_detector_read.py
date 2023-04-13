#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:22:02 2019

@author: warias
"""

import h5py
import numpy


class p08_detector_read(object):
    
    def __init__(self, path, experiment, scannumber, detector):
        self._path = path
        self._experiment = experiment
        self._scannumber = scannumber
        self._detector = detector
        
    def __call__(self):
        func_name = self._detector + "_"
        func = getattr(self,func_name)
        return func()
   
    def lambda_(self):
        detector_file = f"{self._path}/{self._experiment}_{self._scannumber:05}/{self._detector}/{self._experiment}_{self._scannumber:05}_{0:05}.nxs"
        det_file = h5py.File(detector_file, "r")
        img = numpy.array(det_file["/entry/instrument/detector/data"])
        det_file.close()
        return img
    
    def eiger_(self):
        img = "not yet implemented"
        print(img)
        return img
    
    def pilc_(self):
        detector_file_adc = f"{self._path}/{self._experiment}_{self._scannumber:05}/{self._detector}/{self._experiment}_{self._scannumber:05}_adc.nxs"
        detector_file_cnt = f"{self._path}/{self._experiment}_{self._scannumber:05}/{self._detector}/{self._experiment}_{self._scannumber:05}_cnt.nxs"
        det_file_adc = h5py.File(detector_file_adc, "r")
        det_file_cnt = h5py.File(detector_file_cnt, "r")
        data = dict(ion1 = numpy.array(det_file_adc["/entry/data/value3"]), 
                 ion2 = numpy.array(det_file_adc["/entry/data/value4"]),
                 apd2 = numpy.array(det_file_cnt["/entry/data/value3"])
                 )
        det_file_adc.close()
        det_file_cnt.close()
        return data
