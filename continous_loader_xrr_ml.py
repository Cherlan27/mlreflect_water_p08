# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:06:55 2023

@author: petersdorf
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:56:56 2023

@author: petersdorf
"""


# File for checking after finished scan

from eigene.fio_reader import read
from matplotlib.ticker import MultipleLocator
import numpy
import h5py
import os
from eigene.abs_overlap_fit_poly import Absorber
import sys
from PIL import Image
from eigene.p08_detector_read import p08_detector_read
from predicter_xrr_ml import prediction_sample
from matplotlib import pyplot as plt
import time

class load_xrr():
    def __init__(self):
        self.data_directory = "K:/SYNCHROTRON/Murphy/2022-04_P08_11013326_giri_hoevelmann_petersdorf/raw_data/raw",
        self.flatfield = "./Module_2017-004_GaAs_MoFluor_Flatfielddata.tif",
        self.pixel_mask = "./Module_2017-004_GaAs_mask.tif",
        self.use_flatfield = True,
        self.use_mask = None,
        self.experiment = "timing",
        self.detector = "lambda",
        #self.scan_numbers = list(range(1803,1811+1)),
        self.detector_orientation = "vertical",
        self.footprint_correct = True,
        self.beam_width = 20e-6, #2e-5, 0.2e-4, 0.02e-3,
        self.sample_length = 81.4e-3,
        self.roi = (14, 91, 60, 22), # (1480,390,70,30),# (1506, 185, 18, 15)
        self.roi_offset = 30,
        self.monitor = "ion2",
        self.calculate_abs = None,           # If "True" the absorber factors wíll be calculated and the table will be ignored
        self.absorber_factors = {1: 12.617,
                            2: 11.0553,
                            3: 11.063,
                            4: 11.048,
                            5: 11.7,
                            6: 12.813},
        self.primary_intensity = "auto",
        self.auto_cutoff = [0.015, 0.003],
        self.auto_cutoff_nom = [0.0184, 0.0013],
        self.qc = 0.0217,
        self.save_directory = "./processed/"
           
    def __call__(self, scan_numbers):
        while True:
            self.scan_numbers = scan_numbers
            scan_numbers_check = self.xrr_recognizer(scan_numbers)
            print(scan_numbers_check)
            if scan_numbers_check != []:
                self.scan_numbers = scan_numbers_check
                print(self.scan_numbers)
                qz, intensities, e_intensities = self.xrr_calculate()
                print("test hier")
                print(intensities)
                
                fig2 = plt.figure()
                fig2.patch.set_color("white")
                ax2 = fig2.gca()
                ax2.set_yscale('log', nonposy='clip')
                ax2.errorbar(qz, intensities, ls='none',
                            marker='o', mec='#cc0000', mfc='white', color='#ee0000',
                            mew=1.2)
                ax2.errorbar(qz, self.fresnel(0.0217, qz, 2.4), linestyle='--', color='#424242')
                ax2.set_xlabel('qz')
                ax2.set_ylabel('R')
                plt.show()
                

                prediction = prediction_sample(qz, intensities, e_intensities, self.scan_numbers[0], save_directory = self.save_directory)
                prediction()
                scan_numbers = scan_numbers_check[-1] + 1
            time.sleep(0.001)
    
    def xrr_recognizer(self, scan_number):
        scan_xrr = False
        scan_xrr_finished = False
        xrr_scan_numbers = []
        while scan_xrr_finished == False:
            self.experiment = self.file_finder(scan_number)
            if self.experiment != None:
                experimental_part = self.experiment.split("_")[0]
                fio_filename = self.data_directory[0] + "/" + experimental_part + "_" + str(scan_number).rjust(5, "0") + ".fio"
                header, column_names, data, scan_cmd = read(fio_filename)
                if scan_cmd != None:
                    scan_type = scan_cmd.split()[0]
                elif scan_cmd == None:
                    scan_type = "point"
                elif scan_xrr_finished == True:
                    scan_xrr_finished = True
                    
                if scan_xrr == True and scan_type != "qscan":
                    scan_xrr_finished = True
                if scan_type == "qscan" and scan_xrr_finished == False:
                    scan_xrr = True
                    xrr_scan_numbers.append(scan_number)
                scan_number = scan_number + 1
            else:
                break
        if scan_xrr_finished == False:
            scan_number = scan_number + 1
        return xrr_scan_numbers
            
        
    def file_finder(self, scan_number):
        file_search = self.data_directory[0]
        matching = [s for s in os.listdir(file_search) if str(scan_number).rjust(5, "0") in s]
        print(matching)
        if matching != []:
            file_searched = [s for s in matching if not "." in s][0]
            return file_searched
        else:
            return None
        
        
    # == helper functions
    def abs_fac(self, abs_val):
        abs_val = int(abs_val + 0.2)
        if abs_val == 0:
            return 1.0
        else:
            return self.absorber_factors[0][abs_val] * self.abs_fac(abs_val - 1)
        

    def fresnel(self, qc, qz, roughness=2.5):
        """
        Calculate the Fresnel curve for critical q value qc on qz assuming
        roughness.
        """
        return (numpy.exp(-qz**2 * roughness**2) *
                abs((qz - numpy.sqrt((qz**2 - qc**2) + 0j)) /
                (qz + numpy.sqrt((qz**2 - qc**2)+0j)))**2)

    # wav = wavelength
    # L = sample width
    # b = width of the beam
    def footprint_correction(self, q, intensity, b, L, wl = 68.88e-12):
        """ 
        calculates the footprint corrected data
        """
        #print(b)
        #print(L)
        q_b = (4*numpy.pi/wl*b/L)*10**(-10)
        #print(q_b)
        intensity2 = intensity
        #print(q[len(q)-1] > q_b)
        i = 0
        for i in range(0,len(q),1):
            if q[i] < q_b:
                #print(i)
                #print(q[i])
                intensity2[i] = intensity2[i]/(q[i]/q_b)
                i += 1
        else:
            None
        return intensity2, q_b

    def xrr_calculate(self):
        if self.use_flatfield == True:
            flatfield_2 = numpy.ones((516,1556))
            flatfield = numpy.array(Image.open(self.flatfield))
        
        # if self.use_mask:
        #     file_mask = h5py.File(self.mask, "r")
        #     img_mask = numpy.array(file_mask["/entry/instrument/detector/data"])[0]
        #     file_mask.close()
        #     mask = numpy.zeros_like(img_mask)
        #     mask[(img_mask > 1)] = 1
        #     mask = (mask == 1)
        #     mask_value = 0
        

        # == prepare data structures
        intensity = numpy.array([])
        e_intensity = numpy.array([])
        qz = numpy.array([])
        
        # == make data
        absorbers = Absorber()
        temp_intens = {}
        temp_e_intens = {}
        print("fdsafas")
        print(self.scan_numbers)
        
        for scan_number in self.scan_numbers:
        # == load .fio file
            self.experiment = self.file_finder(scan_number)
            experimental_part = self.experiment.split("_")[0]
            fio_filename = self.data_directory[0] + "/" + experimental_part + "_" + str(scan_number).rjust(5, "0") + ".fio"
            header, column_names, data, scan_cmd = read(fio_filename)
            # load monitor
            s_moni = data[self.monitor[0]]
            # make qz
            wl = 12.38/18 * 1e-10
            s_alpha = data["alpha_pos"]
            s_beta = data["beta_pos"]
            s_qz = ((4 * numpy.pi / wl) *
                    numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)
            # prepare data structures
            s_intens = []
            s_e_intens = []
            # load detector data  
            roi = self.roi[0]
            roi_offset = self.roi_offset[0]
        
            detector_images = p08_detector_read(self.data_directory[0], experimental_part, scan_number, self.detector[0])()
            n_images = detector_images.shape[0]
            n_points = min(n_images, len(s_alpha))
            
            for n in range(n_points):
                img = detector_images[n]
                
                # flatfield correction
                if self.use_flatfield == True:
                    img = img / flatfield
                
                # if self.use_mask == True:
                #     img[mask] = mask_value
                
                p_specular = img[roi[1]:(roi[1]+roi[3]),roi[0]:(roi[0]+roi[2])].sum()            
        
                if self.detector_orientation[0] == "horizontal":            
                    p_bg0 = img[roi[1]:(roi[1]+roi[3]),
                                (roi[0]+roi[2]+roi_offset):(roi[0]+2*roi[2]+roi_offset)].sum()
                    p_bg1 = img[roi[1]:(roi[1]+roi[3]),
                                (roi[0]-roi[2]-roi_offset):(roi[0]-roi_offset)].sum()            
                elif self.detector_orientation[0] == "vertical":            
                    p_bg0 = img[(roi[1]+roi[3]+roi_offset):(roi[1]+2*roi[3]+roi_offset),
                                (roi[0]):(roi[0]+roi[2])].sum()
                    p_bg1 = img[(roi[1]-roi[3]-roi_offset):(roi[1]-roi_offset),
                                (roi[0]):(roi[0]+roi[2])].sum()       
                    
                
                p_intens = ((p_specular - (p_bg0 + p_bg1) / 2.0) / s_moni[n])
                
                
                if self.monitor[0] == "Seconds":
                    p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / s_moni[n])
                else:    
                    p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / s_moni[n] 
                                 + abs (0.1 * (p_specular - (p_bg0 + p_bg1) / 2.0) / s_moni[n]))
                s_intens.append(p_intens)
                s_e_intens.append(p_e_intens)
            s_intens = numpy.array(s_intens)
            s_e_intens = numpy.array(s_e_intens)
            #print(s_intens)
            
            qz = numpy.concatenate((qz, s_qz))
            if self.footprint_correct[0] == True:
                    temp_intensities = s_intens
                    proof = s_intens
                    #print(s_qz)
                    temp_intensities, q_b = self.footprint_correction(s_qz, temp_intensities, self.beam_width[0], self.sample_length[0])
                    s_intens = temp_intensities
            
            if self.calculate_abs[0] == True:
                absorbers.add_dataset(header["abs"], s_qz, s_intens)
                temp_intens[scan_number] = int(header["abs"]+0.1), s_intens
                temp_e_intens[scan_number] = int(header["abs"]+0.1), s_e_intens
            elif self.calculate_abs[0] == None:
                s_intens = s_intens * self.abs_fac(header["abs"])
                s_e_intens = s_e_intens * self.abs_fac(header["abs"])
                intensity = numpy.concatenate((intensity, s_intens))
                e_intensity = numpy.concatenate((e_intensity, s_e_intens))
            else:
                sys.exit("Bitte Absorberfaktoren spezifizieren")
            
        if self.calculate_abs[0] == True:
            absorbers.calculate_from_overlaps()
            intensity = numpy.concatenate([absorbers(x[0])*x[1] for x in list(temp_intens.values())])
            e_intensity = numpy.concatenate([absorbers(x[0])*x[1] for x in list(temp_e_intens.values())])
            
        # == sort data points
        m = numpy.argsort(qz)
        qz = qz[m]
        intensity = intensity[m]
        e_intensity = e_intensity[m]
            
        # == normalize
        if self.primary_intensity[0] == "auto":
            primary = intensity[(qz > self.auto_cutoff[0][0]) & (qz<(self.qc[0] - self.auto_cutoff[0][1]))].mean()
        else:
            primary = self.primary_intensity[0]
        
        #print(intensity)
        #print(primary)
        return qz, intensity/primary, e_intensity/primary
    
    
xrr_water = load_xrr()
xrr_water(1439)