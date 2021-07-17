# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 01:26:43 2021

@author: e-default
"""
from depthai_utils import SegmentPipeline

segmentationModelPath = r'models\path_segmentation.blob'
segmentPipeline = SegmentPipeline(segmentationModelPath)
segmentPipeline.run()