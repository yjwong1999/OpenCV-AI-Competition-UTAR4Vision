#!/usr/bin/env python3

from depthai_utils import PpesPipeline

# model to detect protective equipment (ppe)
ppesModelPath = r'models/peps_model.blob'
ppesLabelMap = ["sanitizer", "facemask", "thermometer"]
syncNN = True
ppesPipeline = PpesPipeline(modelPath=ppesModelPath,
                            labelMap=ppesLabelMap,
                            syncNN=syncNN)


# run the pipeline
try:
    ppesPipeline.run_without_audio()
except: 
    raise


