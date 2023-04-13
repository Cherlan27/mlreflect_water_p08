# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:18:21 2018

@author: florian.bertram@desy.de
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


import matplotlib.pyplot as plt

import cProfile


def read(filename, header_only = False):
    
    motor_positions = {}
    
    data_block = False
    param_block = False
    comment_block = False
    
    data_columns = False
    
    column_names = []
    
    scan_cmd = None
    
    data = {}
    
    file = open(filename, 'r') 
    
    
    for line in file:
        #print( line )
        
        if line.find('%c') > -1:
            data_block = False
            param_block = False
            comment_block = True            
            #print('entering comment block')
            continue
        elif line.find('%p') > -1:
            data_block = False
            param_block = True
            comment_block = False            
            #print('entering parameter block')
            continue
        elif line.find('%d') > -1:
            data_block = True
            param_block = False
            comment_block = False
            #print('entering data block')
            continue
        elif line.find('!') > -1:
            continue
        
        if param_block:
            if line.find('=', 1) > -1:
                try:
                    spl = line.strip().split('=')
                    #print("%s = %f" % (spl[0], float(spl[1])))
                    motor_positions[spl[0].strip()] = float(spl[1])
                except:
                    pass
        elif comment_block:
            if line.find('scan') > -1 or line.find('mesh') > -1 or line.find('_burst') > -1:
                scan_cmd = line.strip()
        elif data_block and not data_columns:
            if line.find('Col ') > -1:
                spl = line.split()
                
                column_names.append(spl[2])
                
            elif len(column_names) > 0:
            
                if header_only:
                    break

                spl = line.split()
                
                for idx in range(len(column_names)):
                    try:
                        data[column_names[idx]] = np.array([ float(spl[idx]) ])
                    except:
                        data[column_names[idx]] = np.array([ float('nan') ])
                
                data_columns = True
        
        elif data_columns:
        
            spl = line.split()        
                
            for idx in range(len(column_names)):
                try:
                    data[column_names[idx]] = np.append(data[column_names[idx]], np.array([float(spl[idx])]) )
                except:
                    data[column_names[idx]] = np.append(data[column_names[idx]], np.array([float("nan")]) )
            
        #print ( line.find('=') )

    file.close()
    
    return motor_positions, column_names, data, scan_cmd


if __name__ == '__main__':
    
    cProfile.run("read('./data/test_00065.fio')",sort="tottime")
    
    header, column_names, data = read('./data/test_00065.fio')
    
    
    #for key in sorted(header):
    #    print ("%s = %f" % (key, header[key]))
    
    for col in column_names:
        print (col)
    
    if 'om' in column_names:
        print (data['om'])
        print (len(data['om']))
        
