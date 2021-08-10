import threading
from threading import Lock
import speech_recognition as sr
import time

import numpy as np
import math
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


class BirdEye:
    max_z = 4
    min_z = 1
    max_x = 0.9
    min_x = -0.7
    
    def __init__(self):
        self.fov = 68.7938
        self.min_distance = 1
        self.shape = (320, 100, 3)
        pass
      
    def __make_bird_frame(self):
        frame = np.zeros(self.shape, np.uint8)
        min_y = int((1 - (self.min_distance - self.min_z) / (self.max_z - self.min_z)) * frame.shape[0])
        cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

        alpha = (180 - self.fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, frame.shape[0]),
            (frame.shape[1], frame.shape[0]),
            (frame.shape[1], max_p),
            (center, frame.shape[0]),
            (0, max_p),
            (0, frame.shape[0]),
        ])
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def __calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.shape[1]
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, self.shape[1])
        return int(bottom_x), int(top_x)

    def __calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.shape[0]
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, self.shape[0])
        return int(bottom_z), int(top_z)
      
    def plotBirdEye(self, X, Z):
        frame = self.__make_bird_frame()
        for x, z in zip(X, Z):
            if z < 1000:
                continue
            x, z = x/1000, z/1000
            left, right = self.__calc_x(x)
            top, bottom = self.__calc_z(z)
            frame = cv2.rectangle(frame, (left, top), (right, bottom), (100, 255, 0), 2)  
        width = int(frame.shape[1] * 300 / frame.shape[0])
        height = int(frame.shape[0] * 300 / frame.shape[0])
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        return frame