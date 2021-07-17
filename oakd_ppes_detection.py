# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 19:28:28 2021

@author: e-default
"""
from task_utils import SpeechRecognizer
from depthai_utils import PpesPipeline

# model to detect protective equipment (ppe)
ppesModelPath = r'models/peps_model.blob'
ppesLabelMap = ["sanitizer", "facemask", "thermometer"]
syncNN = True
ppesPipeline = PpesPipeline(modelPath=ppesModelPath,
                            labelMap=ppesLabelMap,
                            syncNN=syncNN)

# activate the speech recognizer
speech = SpeechRecognizer()
speech.start()
speech.task = 'search'

# run the pipeline
try:
    ppesPipeline.run(speech)
except:
    speech.listen = False
    speech.end = True     
    speech.join()    