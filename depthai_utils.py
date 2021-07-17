import depthai as dai
import numpy as np
import time, sched
import cv2
from task_utils import Navigator

#-------------------------------------------------------------------------------
# Navigation Pipeline
#-------------------------------------------------------------------------------
class NavPipeline:
    def __init__(self, nnPath, labelMap, extended_disparity = False,
                 subpixel = False, lr_check = False, median=True, 
                 monoCamResolution=400):
        '''
        args
        1) nnPath
        - the path for MobileNetSSD object detection model
        
        2) labelMap
        - label map for the prediction output labels
        
        3) extended_disparity
        - Closer-in minimum depth, disparity range is doubled, 
        - from 0-95 (default, not changeable) to 0-190
        
        4) subpixel
        - Better accuracy for longer distance, fractional disparity 32-levels
        
        5) lr_check
        - Better handling for occlusions
        
        6) monoCamResolution
        - can be 400, 720 or 800 only
        '''
      
        self.__nnPath = nnPath
        self.__labelMap = labelMap
        self.__extended_disparity = extended_disparity
        self.__subpixel = subpixel
        self.__lr_check = lr_check
        self.__median = median
        
        assert monoCamResolution in [400, 720, 800], "arg monoCamResolution must be 400, 720 or 800!"
        self.__monoCamResolution = monoCamResolution
        
        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        
        #-------------------------------------------------------------------------------
        # Create 2 nodes for left and right mono (grayscale) cameras
        #-------------------------------------------------------------------------------
        left = self.pipeline.createMonoCamera()
        right = self.pipeline.createMonoCamera()
        
        if self.__monoCamResolution == 400:
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            
        if self.__monoCamResolution == 720:
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)  
            
        if self.__monoCamResolution == 800:
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        
        #-------------------------------------------------------------------------------
        # Create a stereo node
        #-------------------------------------------------------------------------------
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
      
        # Normal disparity values range from 0..95, will be used for normalization
        max_disparity = 95
        
        # extend disparity
        if self.__extended_disparity: max_disparity *= 2 # Double the range
        stereo.setExtendedDisparity(self.__extended_disparity)
        
        # subpixel
        if self.__subpixel: max_disparity *= 32 # 5 fractional bits, x32
        stereo.setSubpixel(self.__subpixel)
        
        # set left right check
        stereo.setLeftRightCheck(self.__lr_check)
        
        # Median filter
        if self.__median:
            # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
            median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
            stereo.setMedianFilter(median_filter)
        
        # When we get disparity to the host, we will multiply all values with the multiplier
        # for better visualization
        self.disparity_multiplier = 255 / max_disparity
        
        
        #-------------------------------------------------------------------------------
        # Create a node to convert the grayscale frame into the nn-acceptable form
        #-------------------------------------------------------------------------------
        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(300, 300)
        # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        
        
        #-------------------------------------------------------------------------------
        # Define a neural network that will make predictions based on the source frames
        #-------------------------------------------------------------------------------
        nn = self.pipeline.createMobileNetDetectionNetwork()
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(self.__nnPath)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
      
      
        #-------------------------------------------------------------------------------
        # Link all the nodes (excluding outputs)
        #-------------------------------------------------------------------------------
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.rectifiedRight.link(manip.inputImage)
        manip.out.link(nn.input)
      
      
        #-------------------------------------------------------------------------------
        # Create outputs
        #-------------------------------------------------------------------------------
        depthOut = self.pipeline.createXLinkOut()
        depthOut.setStreamName("depth")
        stereo.depth.link(depthOut.input)
        
        xoutRight = self.pipeline.createXLinkOut()
        xoutRight.setStreamName("rectifiedRight")
        manip.out.link(xoutRight.input)
        
        nnOut = self.pipeline.createXLinkOut()
        nnOut.setStreamName("nn")
        nn.out.link(nnOut.input)
          
    def run(self, speech=None):
        # launched the navigator
        nav = Navigator(self.__labelMap, self.__monoCamResolution, 
                        depthLowerThresh=100,depthUpperThresh=7000,
                        erosionKernelSize=5, obstacleDistance=1000)   
        
        # launched the pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
            qRight = device.getOutputQueue("rectifiedRight", maxSize=4, blocking=False)
            qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)
            qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)
            
            # initialize variables to store video frame
            rightFrame = None
            depthFrame = None
            
            # flip the image or not
            # The rectified streams are horizontally mirrored by default
            flipRectified = not self.__lr_check #True
            
            # start looping
            while True:
                # initialize variables needed to plot the frame
                names = []
                frames = []
                detections = []
                
                # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
                inRight = qRight.tryGet()
                inDet = qDet.tryGet()
                inDepth = qDepth.tryGet()
        
                if inRight is not None:
                    rightFrame = inRight.getCvFrame()
                    if flipRectified:
                        rightFrame = cv2.flip(rightFrame, 1)
        
                if inDet is not None:
                    detections = inDet.detections
                    if flipRectified:
                        for detection in detections:
                            swap = detection.xmin
                            detection.xmin = 1 - detection.xmax
                            detection.xmax = 1 - swap
        
                if inDepth is not None:
                    depthFrame = inDepth.getFrame()
                    names.append("depth")
                    frames.append(depthFrame)
        
                if rightFrame is not None:
                    names.append("rectified right")  
                    frames.append(rightFrame)
                
                # make decision 
                nav.parseInputs(names, frames, detections)
        
                # quit if user pressed 'q'
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
        camRgb.setFps(10)
        
        # create image manip node
        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(256, 256)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        
        # create nn node
        nn = self.pipeline.createNeuralNetwork()
        nn.setBlobPath(self.segModelPath)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        
        # link the nodes
        camRgb.preview.link(manip.inputImage)
        manip.out.link(nn.input)
        
        # output nodes
        xoutRgb_1 = self.pipeline.createXLinkOut()
        xoutRgb_1.setStreamName("rgb_1")
        camRgb.preview.link(xoutRgb_1.input)
        
        nnOut = self.pipeline.createXLinkOut()
        nnOut.setStreamName("segmentation")
        nn.out.link(nnOut.input)

    def run(self, speech=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
            qRgb = device.getOutputQueue(name="rgb_1", maxSize=4, blocking=False)
            qNet = device.getOutputQueue(name="segmentation", maxSize=4, blocking=False)
            
            def customReshapeV1(x, target_shape):
              row, col, ch = target_shape
              arr3d = []
              arr2d = None
            
              for i in range(len(x)//col):
                  if i % col == 0 and i != 0:
                      arr3d.append(arr2d)
                      arr2d = None
              
                  idx1 = i * col
                  idx2 = idx1 + col
                  arr1d = np.reshape(x[idx1:idx2], (1, col, 1))
              
                  if arr2d is None:
                      arr2d = arr1d.copy()
                  else:
                      arr2d = np.concatenate((arr2d, arr1d), axis = 0)
                
              arr3d.append(arr2d)
              arr3d = np.concatenate(arr3d, axis=-1)
            
              return arr3d
            
            def customReshape(x, target_shape):
              x = np.reshape(x, target_shape, order='F')
              for i in range(3):
                  x[:,:,i] = np.transpose(x[:,:,i])
            
              return x
            
            def show_deeplabv3p(output_colors, mask):
                mask = ((mask + 1) / 2 * 255).astype(np.uint8)
                return cv2.addWeighted(mask,0.8, output_colors,0.5,0)
            
            t1 = 0
            t2 = 0
            # start looping
            while True:
                # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
                inRGB = qRgb.tryGet()
                inNet = qNet.tryGet()
        
                if inRGB is not None:
                    rgb = inRGB.getCvFrame()
                    cv2.imshow('rgb_1', rgb)
                    
                if inNet is not None:
                    '''
                    t2 = timeit.default_timer()
                    print(t2-t1, t1, t2)
                    t1 = t2
                    '''
                    mask = inNet.getFirstLayerFp16()
                    mask = np.array(mask)
                    mask = customReshape(mask, (256, 256, 3))
                    mask = show_deeplabv3p(rgb, mask)
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

    def run(self, speech):
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
                        cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                    else:
                        msg = 'Green light! Cross Now'
                        cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    
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