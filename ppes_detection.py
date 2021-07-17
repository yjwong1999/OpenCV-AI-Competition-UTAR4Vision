#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import sched, time

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

# Tiny yolo v3/4 label texts
labelMap = ["sanitizer", "facemask", "thermometer"]

syncNN = True

# Get argument first
peps_model_path = 'models/peps_model.blob'
if len(sys.argv) > 1:
    peps_model_path = sys.argv[1]

if not Path(peps_model_path).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(416, 416)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# mono cam nodes
monoLeft = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.createMonoCamera()
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# stereo node
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(255)

# create yolo spatial network node
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
spatialDetectionNetwork.setBlobPath(peps_model_path)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)
# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(3)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# link the nodes
camRgb.preview.link(spatialDetectionNetwork.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# output node
xoutRgb_2 = pipeline.createXLinkOut()
xoutRgb_2.setStreamName("rgb")
xoutNN = pipeline.createXLinkOut()
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")

# connect to output node
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
else:
    camRgb.preview.link(xoutRgb_2.input)
    
spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect and start the pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    item = None

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    #(new)
    s = sched.scheduler(time.time, time.sleep)
    start_time = time.time()

    #(new)
    def direction():
        # initialize variable
        x_command, y_command, z_command = None, None, None
        
        # X-direction
        if detection.spatialCoordinates.x >= 400:
            x_command = "Right"
        if 400 > detection.spatialCoordinates.x >= 100:
            x_command = "Slight Right"

        if detection.spatialCoordinates.x <= -400:
            x_command = "Left"
        if -400 < detection.spatialCoordinates.x <= -100:
            x_command = "Slight Left"

        # Y-direction
        if detection.spatialCoordinates.y >= 400:
            y_command = "Up"
        if 400 > detection.spatialCoordinates.y >= 100:
            y_command = "Slight Up"

        if detection.spatialCoordinates.y <= -400:
            y_command = "Down"
        if -400 < detection.spatialCoordinates.y <= -100:
            y_command = "Slight Down"

        # Z-direction
        if detection.spatialCoordinates.z >= 1000:
            z_command = "Move forward 1 step"
        if 1000 > detection.spatialCoordinates.z >= 500:
            z_command = "Move slightly forward"
        if 500 > detection.spatialCoordinates.z >= 0:
            z_command = "The item is 50cm infront of you"
            
        return x_command, y_command, z_command

    while True:
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        # (new)
        current_time = time.time()
        elapsed_time = current_time - start_time
        notify_time = int(elapsed_time) % 5

        # (new)
        if item is None:
            item = input("Enter object to find: ")
            print("Turn around slowly to search for " + item)

        # (new)
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections
        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            # (new)
            if label == item:
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)/10} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)/10} cm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)/10} cm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                x_command, y_command, z_command = direction()
                cv2.putText(frame, x_command, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, y_command, (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, z_command, (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)


        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        # (new)
        key = cv2.waitKey(1)
        if key == ord('r'):
            item = None

        if key == ord('q'):
            cv2.destroyAllWindows() 
            break
