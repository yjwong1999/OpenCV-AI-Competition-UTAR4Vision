import depthai as dai
import numpy as np
import time, sched, timeit
import cv2

from task_utils import BirdEye

#-------------------------------------------------------------------------------
# Navigation Pipeline
#-------------------------------------------------------------------------------
class NavPipeline(BirdEye):
    def __init__(self, nnPath, labelMap, 
                 syncNN=True, flipRectified = True,
                 erosionKernelSize=5, obstacleDistance=1000):   
        # initialize BirdEye
        BirdEye.__init__(self)
        
        # declare variables
        self.__nnPath = nnPath
        self.__labelMap = labelMap
        self.__syncNN = syncNN
        self.__flipRectified = flipRectified
        
        # variables for obstacle avoidance
        assert erosionKernelSize in [2, 3, 4, 5, 6, 7], 'arg "erosionKernelSize" must be in [2, 3, 4, 5, 6, 7]'
        self.__erosionKernel = np.ones((erosionKernelSize,erosionKernelSize),np.uint8)
        self.__obstacleDistance = obstacleDistance
        
        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        
        # Define sources and outputs
        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        stereo = self.pipeline.createStereoDepth()
        spatialDetectionNetwork = self.pipeline.createMobileNetSpatialDetectionNetwork()
        imageManip = self.pipeline.createImageManip()
        
        xoutManip = self.pipeline.createXLinkOut()
        nnOut = self.pipeline.createXLinkOut()
        depthRoiMap = self.pipeline.createXLinkOut()
        xoutDepth = self.pipeline.createXLinkOut()
        
        xoutManip.setStreamName("right")
        nnOut.setStreamName("detections")
        depthRoiMap.setStreamName("boundingBoxDepthMapping")
        xoutDepth.setStreamName("depth")
        
        # Properties
        imageManip.initialConfig.setResize(300, 300)
        # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
        imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # StereoDepth
        stereo.setConfidenceThreshold(255)
        
        # Define a neural network that will make predictions based on the source frames
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.setBlobPath(self.__nnPath)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        
        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        
        imageManip.out.link(spatialDetectionNetwork.input)
        if self.__syncNN:
            spatialDetectionNetwork.passthrough.link(xoutManip.input)
        else:
            imageManip.out.link(xoutManip.input)
        
        spatialDetectionNetwork.out.link(nnOut.input)
        spatialDetectionNetwork.boundingBoxMapping.link(depthRoiMap.input)
        
        stereo.rectifiedRight.link(imageManip.inputImage)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

  
    def __obstacleAvoidance(self, crop=None, depthThreshold=600, sumThreshold=8000):
        '''
        purpose:
        - to guide user to avoid close obstalce on left or right side
        
        args
        1) crop
        - the cropped depth frame (ROI that we want)
        2) depthThreshold
        - the max distance to be considered CLOSE OBSTACLE (heurestically obtained)
        3) sumThreshold
        - the threshold for the sum of the left ROI and right ROI (heurestically obtained)
        '''
        if crop is not None:
            # create a frame
            obsFrame = np.full((30, 93, 3), 70, np.uint8)
            
            # mask the depth map
            masked = crop > depthThreshold
            masked = np.array(masked, dtype=np.float32)
            
            # erode the masked depth map
            #erosion = cv2.erode(masked, self.__erosionKernel, iterations = 1)
            erosion = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.__erosionKernel)
            
            # divide the eroded depth map to left and right
            left = erosion[:, :erosion.shape[1] // 2]
            right = erosion[:, erosion.shape[1] // 2 :]
            
            # get the sum of the pixels in left and right
            leftSum = np.sum(1 - left)
            rightSum = np.sum(1 - right)
            
            # make decision
            if leftSum < sumThreshold:
                if rightSum < sumThreshold:
                    directionMessage =  'Clear'
                else:
                    directionMessage = 'Turn left to avoid Obstacle'
                    obsFrame = cv2.rectangle(obsFrame, (47, 0), (93, 30), (0, 0, 255), -1) 
            elif rightSum < sumThreshold:
                directionMessage = 'Turn Right to avoid Obstacle'
                obsFrame = cv2.rectangle(obsFrame, (0, 0), (46, 30), (0, 0, 255), -1) 
            else:
                directionMessage = 'Obstacle in Front!'
                obsFrame = cv2.rectangle(obsFrame, (0, 0), (93, 30), (0, 0, 255), -1) 
    
            # show the images
            erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
            
            cv2.putText(erosion, directionMessage, (10, 14), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
            #cv2.imshow('Masked', masked)
            cv2.imshow('Crop', erosion) 
            
            
            return obsFrame, erosion
      
    # new  
    def run(self, speech=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthRoiMapQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
            rectifiedRight = None
            detections = []
        
            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            labelFrame = np.ones((330, 93, 3), np.uint8)
            labelFrame = np.full((330, 93, 3), 70, np.uint8)
            #labelFrame[:,0:2,:] = 255
            cv2.putText(labelFrame, 'far', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (100, 255, 0))
            cv2.putText(labelFrame, 'obstacle', (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (100, 255, 0))
            cv2.putText(labelFrame, 'close', (10, 305), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(labelFrame, 'obstacle', (10, 320), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
        
            while True:
                inRectified = previewQueue.get()
                inDet = detectionNNQueue.get()
                inDepth = depthQueue.get()
        
                counter += 1
                currentTime = time.monotonic()
                if (currentTime - startTime) > 1:
                    fps = counter / (currentTime - startTime)
                    counter = 0
                    startTime = currentTime
        
                rectifiedRight = inRectified.getCvFrame()
                if self.__flipRectified:
                    rectifiedRight = cv2.flip(rectifiedRight, 1)
        
                depthFrame = inDepth.getFrame()
                if depthFrame is not None:
                    # not all region is useful for obstacle avoidance
                    # crop the ROI
                    depthCropped = depthFrame.copy()[200:,120:520]
                    obsFrame, cropFrame = self.__obstacleAvoidance(depthCropped)
                    
                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        
                detections = inDet.detections
                if len(detections) != 0:
                    boundingBoxMapping = depthRoiMapQueue.get()
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
        
                # If the rectifiedRight is available, draw bounding boxes on it and show the rectifiedRight
                height = rectifiedRight.shape[0]
                width = rectifiedRight.shape[1]
                X, Z = [], []
                for detection in detections:
                    if self.__flipRectified:
                        swap = detection.xmin
                        detection.xmin = 1 - detection.xmax
                        detection.xmax = 1 - swap
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    
                    X.append(detection.spatialCoordinates.x)
                    Z.append(detection.spatialCoordinates.z)
                    try:
                        label = self.__labelMap[detection.label]
                    except:
                        label = detection.label
                    
                    color = (100, 255, 0)
                    if detection.spatialCoordinates.z < self.__obstacleDistance:
                        color = (0, 0, 255)
                        if label == 'person':
                            msg = "Social Distance!"
                        else:
                            msg = 'Obstacle alert!'
                        cv2.putText(rectifiedRight, f"{msg}", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        
                    cv2.putText(rectifiedRight, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, "{:.2f}%".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #cv2.putText(rectifiedRight, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #cv2.putText(rectifiedRight, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    #cv2.putText(rectifiedRight, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, "Distance:", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, f"{int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        
                    cv2.rectangle(rectifiedRight, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                
                # get bird eye frame
                birdEyeFrame = self.plotBirdEye(X, Z)
                
                # combine the four frame
                frame = np.vstack((birdEyeFrame, obsFrame))  
                frame = np.hstack((frame, labelFrame))
                '''
                cropFrame = cv2.resize(cropFrame, (300, 150), interpolation = cv2.INTER_AREA)
                rectifiedRight[150:,:] =rectifiedRight[150:,:] * cropFrame
                '''
                rectifiedRight = cv2.resize(rectifiedRight, (330, 330), interpolation = cv2.INTER_AREA)
                rectifiedRight = np.hstack((rectifiedRight, frame))
                
                # add inference rate
                cv2.putText(rectifiedRight, "NN fps: {:.2f}".format(fps), (2, rectifiedRight.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
                cv2.imshow("rectified right", rectifiedRight)
                
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    if speech is not None:
                        speech.task = None
                        speech.listen = True
                    break


#-------------------------------------------------------------------------------
# Road Segmentation Pipeline
#-------------------------------------------------------------------------------
class SegmentPipeline:
    def __init__(self, segModelPath):
        self.segModelPath = segModelPath
        
        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        
        # Create color cam node
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(256, 256)
        camRgb.setInterleaved(False)
        camRgb.setFps(40)
        
        # create image manip node
        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(256, 256)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        
        # create nn node
        nn = self.pipeline.createNeuralNetwork()
        nn.setBlobPath(self.segModelPath)
        nn.setNumInferenceThreads(2)
        nn.setNumPoolFrames(4)
        nn.input.setBlocking(False)
        
        # link the nodes
        camRgb.preview.link(manip.inputImage)
        manip.out.link(nn.input)
        
        # output nodes
        xoutRgb_1 = self.pipeline.createXLinkOut()
        xoutRgb_1.setStreamName("rgb_1")
        xoutRgb_1.input.setBlocking(False)
        nn.passthrough.link(xoutRgb_1.input)
        
        nnOut = self.pipeline.createXLinkOut()
        nnOut.setStreamName("segmentation")
        nnOut.input.setBlocking(False)
        nn.out.link(nnOut.input)


    def run(self, speech=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
            qRgb = device.getOutputQueue(name="rgb_1", maxSize=4, blocking=False)
            qNet = device.getOutputQueue(name="segmentation", maxSize=4, blocking=False)
            
            def customReshape(x, target_shape):
              x = np.reshape(x, target_shape, order='F')
              for i in range(3):
                  x[:,:,i] = np.transpose(x[:,:,i])
            
              return x
            
            def show_deeplabv3p(output_colors, mask):
                mask = ((mask + 1) / 2 * 255).astype(np.uint8)
                return cv2.addWeighted(mask,0.8, output_colors,0.5,0)
            
            startTime = 0
            counter = 0
            fps = 0
            
            # start looping
            while True:
                # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
                inRGB = qRgb.tryGet()
                inNet = qNet.tryGet()
        
                if inRGB is not None:
                    rgb = inRGB.getCvFrame()
                    cv2.imshow('rgb_1', rgb)
                
                counter += 1
                current_time = timeit.default_timer()
                if inRGB is not None and inNet is not None:                                        
                    mask = inNet.getFirstLayerFp16()
                    mask = np.array(mask)
                    mask = customReshape(mask, (256, 256, 3))
                    if (current_time - startTime) > 1 :  
                        fps = counter / (current_time - startTime)
                        counter = 0                      
                        startTime = current_time
                        
                    mask = show_deeplabv3p(rgb, mask)
                    cv2.putText(mask, "NN fps: {:.2f}".format(fps), (2, mask.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
                    cv2.imshow('mask', mask)
        
                # quit if user pressed 'q'
                if cv2.waitKey(1) == ord('q'):
                    if speech is not None:
                        speech.task = None
                        speech.listen = True
                    cv2.destroyAllWindows() 
                    break


#-------------------------------------------------------------------------------
# Pedestrian Traffic Light Detection Pipeline
#-------------------------------------------------------------------------------
class TrafficPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath= modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN
        
        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        
        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        
        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
        spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
        spatialDetectionNetwork.setIouThreshold(0.5)
        
        # link the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        
        # output node
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
        
        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)
            
        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)
        
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


    def run(self, speech=None):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
            frame = None
            detections = []
            
            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)
        
            #(new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()
        
        
            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()
        
                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5
        
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
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label
                        
                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)/10} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)/10} cm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)/10} cm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                
                msg = None
                if len(detections) == 1:
                    if label == 'red':
                        msg = 'Red light! Dont Cross Yet'
                        color = red
                    else:
                        msg = 'Green light! Cross Now'
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)
        
                # (new)
                key = cv2.waitKey(1)      
                if key == ord('q'):
                    cv2.destroyAllWindows() 
                    if speech is not None:
                        speech.task = None
                        speech.listen = True                    
                    break
                  

#-------------------------------------------------------------------------------
# PPE (Protective Equipment) Detection Pipeline
#-------------------------------------------------------------------------------
class PpesPipeline:
    def __init__(self, modelPath, labelMap, syncNN=True):
        # define some variable
        self.modelPath= modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN
        
        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        
        # Define a source - color camera
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # mono cam nodes
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        
        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # stereo node
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        
        # create yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
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
        xoutRgb_2 = self.pipeline.createXLinkOut()
        xoutRgb_2.setStreamName("rgb")
        xoutNN = self.pipeline.createXLinkOut()
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
        
        # connect to output node
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb_2.input)
        else:
            camRgb.preview.link(xoutRgb_2.input)
            
        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)
        
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


    def run(self, speech):
        # if speech exist
        if speech is not None:
            speech.listen = True
            
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
            frame = None
            detections = []
            
            # default is set to sanitizer
            speech.item = 'sanitizer'
            
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
                '''
                # (new)
                while speech.item is None:
                    pass
                '''
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
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label
        
                    # (new)
                    if label == speech.item:
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
                    speech.item = None
        
                if key == ord('q'):
                    cv2.destroyAllWindows() 
                    if speech is not None:
                        speech.task = None
                        speech.listen = True
                    break
                  
    def run_without_audio(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
            frame = None
            detections = []
            
            # default set to sanitizer
            item = 'sanitizer'
        
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
            
            print('Instruction:')
            print('(i) To switch item to be found, press "r"')
            print('(ii) To quit, press "q"\n')
            print('If the button not working:')
            print('- click the any of the image frames first')
            print('- then, try clicking the button again\n')
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
                    cv2.destroyAllWindows() 
                    while True:
                        item = input("Items: sanitizer, facemask, thermometer\nEnter object to find: ")
                        if item in self.labelMap:
                            break
                        print('Invalid Item!')
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
                        label = self.labelMap[detection.label]
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