import threading
from threading import Lock
import speech_recognition as sr
import time
import numpy as np
import cv2
# peopleCount, and what count, and TEST, and direction

class SpeechRecognizer(threading.Thread):
    def __init__(self):
        '''
        Define a new subclass of the Thread class
        that performs speech-to-text and keyword detection
        for voice command
        '''
        threading.Thread.__init__(self)
        self.dataLock = Lock()
        
        # this var is for ppe detection using voice command
        self.item = None
        
        # these var are for the functioning of the loop
        self.task = None
        self.end = False
        self.listen = False
      
    def __recognizeSpeech(self, recognizer, microphone):
        
        """
        Transcribe speech from recorded from `microphone`.
    
        Returns a dictionary with three keys:
        "success"      : a boolean indicating whether or not the API request was
                         successful
        "error"        : `None` if no error occured, otherwise a string containing
                         an error message if the API could not be reached or
                         speech was unrecognizable
        "transcription": `None` if speech could not be transcribed,
                         otherwise a string containing the transcribed text
        """
        
        # check that recognizer and microphone arguments are appropriate type
        if not isinstance(recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")
    
        if not isinstance(microphone, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")
    
        # adjust the recognizer sensitivity to ambient noise and record audio
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio = recognizer.listen(source)
    
        # initialize the response object
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
    
        # try recognizing the speech in the recording
        # if a RequestError or UnknownValueError exception is caught,
        # update the response object accordingly
        try:
            # store the transcription
            response["transcription"] = recognizer.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"
    
        return response
            
    def run(self):
        '''
        Prompt for voice command from user
        Process the voice command for keyword detection
        '''
        # NEW MARK LINE
        # stop looping or not
        self.end = False
        
        # listen or not
        self.listen = False
        
        # Number of prompts for error trials before it turn off
        PROMPT_LIMIT = 3
    
        # create recognizer and mic instances
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
    
        while not self.end:
            # if a transcription is returned, break out of the loop and
            #     continue
            # if no transcription returned and API request failed, break
            #     loop and continue
            # if API request succeeded but no transcription was returned,
            #     re-prompt the user to say their guess again. Do this up
            #     to PROMPT_LIMIT times
            
            while self.listen:                
                # loop until limit is reached
                for i in range(PROMPT_LIMIT):
                    if self.end:
                        break
                      
                    # show instructions and wait 1 second(s) before starting
                    if self.task != 'search' or self.task is None:
                        print("Hi, how can I help you?")
                        time.sleep(1)
                    else:
                        print('What item do you want to search for?')
                        time.sleep(1)                      
                    print("...listening...")
                    
                    command = self.__recognizeSpeech(recognizer, microphone)
                    
                    # if transcription is successfully stored
                    if command["transcription"]:
                        break
                        
                    # if API was unreachable or unresponsive
                    if not command["success"]:
                        break
                        
                    # if speech was unintelligible, prompt again
                    print("I didn't catch that. What did you say?\n")
        
                # if there was an error, stop the recognition and display the error message
                if command["error"]:
                    with self.dataLock:
                        self.task = 'navigation'
                    print("ERROR: {}".format(command["error"]))
                    break
        
                # if no error, show the user the transcription
                print("You said: {}".format(command["transcription"]))
        
                # TODO:----------------------------------------------------------------------------------------------------
                
                speech = command["transcription"].lower()
                
                #-------------------------------------------------------------
                # This part is dedicated for search ppes
                #-------------------------------------------------------------
                if self.task == 'search':
                    # update item to search
                    if 'sanitizer' in speech or 'sanitiser' in speech:
                        with self.dataLock:
                            self.item = 'sanitizer'
                        print("Find sanitizer\n")
                        
                    elif 'facemask' in speech or 'face mask' in speech:
                        with self.dataLock:
                            self.item = 'facemask'
                        print("Find facemask\n")
    
                    elif 'thermometer' in speech:
                        with self.dataLock:
                            self.item = 'thermometer'
                        print("Find thermometer\n")
                    else:
                        print('Invalid items! Try again\n')
                    continue
                
                #-------------------------------------------------------------
                # This part is to get command
                #-------------------------------------------------------------
                if 'navigate' in speech or 'navigation' in speech:
                    # update task to navigation
                    print('Proceed to navigation\n')
                    with self.dataLock:
                        self.task = "navigation"
                        self.listen = False
                    break
                    
                elif 'road' in speech or 'route' in speech or 'segmentation' in speech:
                    # update task to segmentation
                    print('Proceed to segmentation\n')
                    with self.dataLock:
                        self.task = 'segmentation'
                        self.listen = False
                    break
                        
                elif 'traffic' in speech or 'light' in speech:
                    # update task to traffic light setection
                    print('Proceed to traffic light detection\n')
                    with self.dataLock:
                        self.task = 'traffic light'
                        self.listen = False
                    break
                
                # new markline
                elif 'search' in speech or 'find' in speech:
                    # update task to search
                    print('Proceed to search\n')
                    with self.dataLock:
                        self.task = 'search'
                        self.listen = False
                    break
                               
                else:
                    print("Sorry, the function is not available\n")
                

class SpatialCalculator:
    '''
    purpose:
    - an class to handle the computation involving spatial calculation for 
      detected objects
    '''
    
    # predefined the coordinate of the user
    userCoord = (0, 3)
    # focal length of OAK-D
    focalLength = 857.06
    
    
    #-------------------------------------------------------------------------------
    # constructor
    #-------------------------------------------------------------------------------
    def __init__(self, depthLowerThresh=100, depthUpperThresh=7000):
        '''
        purpose:
        - the depth map is very noisy even after median filtering
        - to get the estimation of depth in a ROI of depth map, 
          it is important to remove outlier "depth" values from the ROI
        - hence, the argument below are the threshold for the 
          preliminary filtering of the outliers
          
        args:
        1) depthLowerThresh
        - the minimum depth to be considered as non-outliers
        
        2) depthUpperThresh
        - the maximum depth to be considered as non-outliers
        '''
        self.depthLowerThresh = depthLowerThresh
        self.depthUpperThresh = depthUpperThresh


    #-------------------------------------------------------------------------------
    # Get the spatial data of detected object based on ROI in the depth map
    #-------------------------------------------------------------------------------
    def getSpatialData(self, depthFrame, bboxCoord):
        '''
        purpose:
        - we do not use the predefined spatialLocationCalculator node provided
          in depthai module due to the inflexibility
        - logic of our method:
            * remove some obvious outliers based on predefined threshold
            * remove the remaining outliers based on statistical approach
            * we cannot directly take the min/max or mean of the depth in 
              the ROI of depth map as the depth representation of that region
            * it is better to take the median (or any relevant percentile) of 
              the depth values in that ROI
          
        args:
        1) depthFrame
        - the depth map
        
        2) bboxCoord
        - the bbox coordinate for the ROI in the depth map (xmin, ymin, xmax, ymax)
        
        return:
        1) (x, z)
        - spatial coordinate (in xz form)
        - can be modify into xyz form
        - perhaps future update will add a choice to select xz or xyz 
        '''
        
        # get the x center of the bbox
        xmin, ymin, xmax, ymax = bboxCoord
        xCentre = (xmax + xmin)//2
        xCentre -= (depthFrame.shape[0]) / 2 # adjust the coord from image cartesian plane to camera (actual) cartesian plane
        
        # flatten the ROI
        cropFrame = depthFrame[ymin:ymax, xmin:xmax]
        cropFrame = np.array(cropFrame)
        cropFrame = cropFrame.flatten()
        
        # remove outlier depth value
        cropFrame = cropFrame[cropFrame<self.depthUpperThresh] 
        cropFrame = cropFrame[cropFrame>self.depthLowerThresh]
        
        # remove the outlier in the remaining values
        # https://www.kite.com/python/answers/how-to-remove-outliers-from-a-numpy-array-in-python
        mean = np.mean(cropFrame)
        standard_deviation = np.std(cropFrame)
        distance_from_mean = abs(cropFrame - mean)
        max_deviations = 200 # so far 200 the best, but no visible difference between 200 and 100
        not_outlier = distance_from_mean < max_deviations * standard_deviation
        cropFrame = cropFrame[not_outlier]
        
        # get the median or any percentile value as the depth representation of ROI
        if cropFrame.shape[0] != 0:
            z = np.percentile(cropFrame, 30) #30 not bad for 400 resolution
        else:
            z = np.nan
        
        # get actual x in real life
        x = xCentre / self.focalLength * z #focal length for all current/updated depthai device is 857.06mm

        return (x,z)
      
    
    #-------------------------------------------------------------------------------
    # get the distance between the coordinates
    #-------------------------------------------------------------------------------
    def getDistance(self, coord):
        '''
        purpose:
        - to get the distance of the object measured from user
        
        args:
        1) coord
        - coord of the object (support both xz and xyz form)
        
        return:
        1) dist
        - euclidean distance between user and the detected object
        '''
        
        dist = 0
        for i in range(len(coord)):
            dist += (coord[i] - self.userCoord[i]) ** 2
        dist = np.sqrt(dist)
        
        if str(dist) == "nan": dist = np.nan
        
        return dist
      

class Navigator(SpatialCalculator):
    '''
    purpose:
    - an class to handle the decision making for navigation
    '''
    
    #-------------------------------------------------------------------------------
    # constructor 
    #-------------------------------------------------------------------------------   
    def __init__(self, labelMap, resolution=400,
                 depthLowerThresh=100, depthUpperThresh=7000,
                 erosionKernelSize=5, obstacleDistance=1000):
        '''
        purpose:
        - to get the min/max threshold for outliers 
          (refer the documentation of constructor for class "SpatialCalculator")
          
        args:
        1) labelMap
        - the neural network labels (in sequence)
        
        2) depthLowerThresh
        - the minimum depth to be considered as non-outliers
        
        3) depthUpperThresh
        - the maximum depth to be considered as non-outliers
        
        4) erosionKernelSize
        - the kernel size for erosion
        
        5) obstacleDistance
        - the maximum distance to be considered obstacle
        '''
        
        # initialize the label map
        self.__labelMap = labelMap
        
        # based on the resolution, get the index needed to crop the depthframe
        assert resolution in [400, 720, 800], 'arg "resolution" must be in [400, 720, 800]'
        
        if resolution == 400:
            aspectRatio = 640 / 400
        if resolution == 720:
            aspectRatio = 1280 / 720
        if resolution == 800:
            aspectRatio = 1280 / 800
            
        self.__idx1 = int((aspectRatio * resolution - resolution) // 2)
        self.__idx2 = self.__idx1 + resolution
    
        # initialize the setting for spatial calculator
        SpatialCalculator.__init__(self, depthLowerThresh, depthUpperThresh)
        
        # initialize the erosion kernel
        assert erosionKernelSize in [2, 3, 4, 5, 6, 7], 'arg "kernelSize" must be in [2, 3, 4, 5, 6, 7]'
        self.__erosionKernel = np.ones((erosionKernelSize,erosionKernelSize),np.uint8)
        
        # initialize the maximum distance to be considered obstacle
        self.__obstacleDistance = obstacleDistance


    #-------------------------------------------------------------------------------
    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    #-------------------------------------------------------------------------------
    def __frameNorm(self, frame, bbox):
        '''
        purpose:
        - the received bbox is in [0, 1] range
        - we must normalized it to the frame width/height
        
        args
        1) frame
        - any frame to be handled 
        
        2) bbox
        - the original bbox in [0, 1] range
          
        return
        - a normalized bbox
        '''
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    #-------------------------------------------------------------------------------
    # make sure there is no obstacles in front of user
    #-------------------------------------------------------------------------------
    def __obstacleAvoidance(self, depthFrame):
        '''
        purpose:
        - to detect if there is obstacle in a certain region of the depth map
        - currently, we take lower half of the cropped depthFrame to estimate
          if there is obstacle or not
        
        args:
        1) depthFrame
        - cropped depth frame
        
        return:
        - direction to go (to avoid obstacles)
        '''
        
        # mask the depth map
        masked = depthFrame > 600
        masked = np.array(masked, dtype=np.float32)
        
        # erode the masked depth map
        #erosion = cv2.erode(masked, self.__erosionKernel, iterations = 1)
        erosion = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.__erosionKernel)
        
        # crop the roi out
        crop = erosion[erosion.shape[0] // 2:, :]
        
        # divide the eroded depth map to left and right
        left = crop[:, :erosion.shape[1] // 2]
        right = crop[:, erosion.shape[1] // 2 :]
        
        # get the sum of the pixels in left and right
        leftSum = np.sum(1 - left)
        rightSum = np.sum(1 - right)
        
        # make decision
        if leftSum < self.__obstacleDistance:
            if rightSum < self.__obstacleDistance:
                directionMessage =  'Clear'
            else:
                directionMessage = 'Turn left to avoid Obstacle'
        elif rightSum < self.__obstacleDistance:
            directionMessage = 'Turn Right to avoid Obstacle'
        else:
            directionMessage = 'Obstacle in Front!'

        # show the images
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cv2.putText(crop, directionMessage, (10, 14), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
        #cv2.imshow('Masked', masked)
        cv2.imshow('Crop', crop)
            
        return directionMessage


    #-------------------------------------------------------------------------------
    # detect and draw bounding box on object
    #-------------------------------------------------------------------------------
    def __objectDetection(self, names, frames, detections):
        '''
        purpose:
        - detect and draw bounding box on object
        
        args:
        1) names
        - a list of name, which correspond to the name of frame in argument frames
        
        2) frames
        - a list of image frame that is acquired in this timestamp
        
        3) detections
        - a list of detection (object that is detected by the neural network)
        - each detection contains xmin, ymin, xmax, ymax of the bbox of the particular detected object
        
        return:
        1) peopleCount
        - total count of people detected
        
        2) objectCount
        - total count of object detected
        '''
                
        # relevant information in this loop of detections
        peopleCount, objectCount = 0, 0
        dists = []
        
        # loop thru all detected objects and get relevant information
        if "depth" in names:
            # get index of depth frame in variable frames
            depthFrameIdx = names.index("depth")
            
            for detection in detections:
                dist = None
                bbox = None
                
                # get alert message
                if self.__labelMap[detection.label] == "person":
                    alertMessage = "PLEASE MAINTAIN SOCIAL DISTANCING!"
                    peopleCount += 1
                else:
                    alertMessage = "OBSTACLE ALERT!"
                    objectCount += 1
    
                # get distance of the object away from user
                frame = frames[depthFrameIdx][:,self.__idx1:self.__idx2]
                bbox = self.__frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                coord = self.getSpatialData(frame, bbox)
                dist = self.getDistance(coord)
                dists.append(dist)
                
            # convert the depth map into coloured image
            frame = frames[depthFrameIdx]
            frame = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            frame = cv2.equalizeHist(frame)
            frames[depthFrameIdx] = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        
        # add text and bounding boxes to the frames
        for dist, detection in zip(dists, detections):
            bbox = None
                    
            # get distance of the object away from user
            for name, frame in zip(names, frames):
                # make sure the depth frame is labeled correctly 
                # because right frame took the center of depth frame only
                if name == "depth":
                    frame = frame[:,self.__idx1:self.__idx2]
                bbox = self.__frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                # add text and boxes into the frame
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, self.__labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                if (dist is not None) and (dist is not np.nan):
                    cv2.putText(frame, "{:.2f} m".format(dist / 1000), (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                    if dist <= 1000: # 1000 mm (1m)
                        cv2.putText(frame, f"{self.__labelMap[detection.label]}", (bbox[0] + 10, bbox[1] + 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                        cv2.putText(frame, alertMessage, (bbox[0] + 10, bbox[1] + 120), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                        
        return peopleCount, objectCount
  
        
    #-------------------------------------------------------------------------------
    # get the necessary inputs for futher decision making
    #-------------------------------------------------------------------------------
    def parseInputs(self, names, frames, detections):
        '''
        purpose:
        - to feed all the necessary inputs for further decision making, such as direction giving
        - plot the required frames
        
        args:
        1) names
        - a list of name, which correspond to the name of frame in argument frames
        
        2) frames
        - a list of image frame that is acquired in this timestamp
        
        3) detections
        - a list of detection (object that is detected by the neural network)
        - each detection contains xmin, ymin, xmax, ymax of the bbox of the particular detected object
        '''
        # if depth frame exist, call obstacleAvoidance for direction
        direction = None
        if "depth" in names:
            depthFrameIdx = names.index("depth")
            depthCropped = frames[depthFrameIdx].copy()[:,self.__idx1:self.__idx2]
            direction = self.__obstacleAvoidance(depthCropped)
        
        # call objectDetection to locate objects
        peopleCount, objectCount = self.__objectDetection(names, frames, detections)
        
        # Show the frame
        for name, frame in zip(names, frames):
            '''
            if name == "depth":
                frame = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                frame = cv2.equalizeHist(frame)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            '''
            cv2.imshow(name, frame)