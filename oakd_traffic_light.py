# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 01:31:31 2021

@author: Wong Yi Jie
"""
from depthai_utils import TrafficPipeline

modelPath = r'models/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
label = ['red', 'green']
trafficPipeline = TrafficPipeline(modelPath, label)

try:
    trafficPipeline.run()
except:
    raise
