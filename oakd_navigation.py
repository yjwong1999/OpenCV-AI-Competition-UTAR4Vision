# -*- coding: utf-8 -*-
"""
Created on Tue May 11 23:38:25 2021

@author: Wong Yi Jie
"""

from depthai_utils import NavPipeline
                  
#-------------------------------------------------------------------------------
# get model path for object detection
#-------------------------------------------------------------------------------
model_path = r'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
        
        
#-------------------------------------------------------------------------------
# MobilenetSSD label nnLabels
#-------------------------------------------------------------------------------
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


#-------------------------------------------------------------------------------
# define the resolution 
#-------------------------------------------------------------------------------
try:
    navPipeline = NavPipeline(nnPath=model_path, labelMap=labelMap)
    navPipeline.run()
except:
    raise
