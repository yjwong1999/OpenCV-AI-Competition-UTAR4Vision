from depthai_utils import NavPipeline, SegmentPipeline, TrafficPipeline, PpesPipeline
from task_utils import SpeechRecognizer
import cv2

#-------------------------------------------------------------------------------
# define the Pipelines
#-------------------------------------------------------------------------------
# navigation (obstacle avoidance + social distancing)
commonObjectPath = r'models/mobilenet-ssd_openvino_2021.2_6shave.blob'
commonObjectLabelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
navPipeline = NavPipeline(nnPath=commonObjectPath,
                          labelMap=commonObjectLabelMap)

# walkable path segmentation
segModelPath = r'models\path_segmentation.blob'
segmentPipeline = SegmentPipeline(segModelPath=segModelPath)

# model to detect pedestrian traffic light
trafficModelPath = r'models/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
trafficLabelMap = ['red', 'green']
trafficPipeline = TrafficPipeline(modelPath = trafficModelPath, 
                                  labelMap = trafficLabelMap)

# model to detect protective equipment (ppe)
ppesModelPath = r'models/peps_model.blob'
ppesLabelMap = ["sanitizer", "facemask", "thermometer"]
syncNN = True
ppesPipeline = PpesPipeline(modelPath=ppesModelPath,
                            labelMap=ppesLabelMap,
                            syncNN=syncNN)

# run the speech recognizer
speech = SpeechRecognizer()
speech.start()

try:
    navPipeline.run(speech)
    while True:
        if speech.task == 'navigation':
            navPipeline.run(speech)
        if speech.task == 'segmentation':
            segmentPipeline.run(speech)
        if speech.task == 'traffic light':
            trafficPipeline.run(speech)
        if speech.task == 'search':
            ppesPipeline.run(speech)
            
except:
    cv2.destroyAllWindows()
    speech.end = True     
    speech.join()       
    raise
    
    
