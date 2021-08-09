# Demonstration

## 1. Navigation
Far Object (beyond 1m)
- An object detector is used to detect potential obstacles that may block the user.
- The detected objects will be plotted into a bird-eye view map.

Close Object (within 1m)
- Not all objects can be detected by object detector.
- An object detector may also missed some object (if the object is partially occluded).
- An obect avoidance system is designed to detect if there is/are close obstalce(s) on the left/right side of the user.
- If there is obstacles on the left side of user, a "RED FLAG" will be raised under the bird-eye view map (and vice cersa).

### Avoiding Obstacle on the Left Side
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/nav%20demo%202.png" 
     alt="Navigation 2" width=600><br>
### Avoiding Obstacle on the Right Side
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/nav%20demo%201.png" 
     alt="Navigation 1" width=600><br>
### Avoiding Obstacle that Could Not be Detected by Object Detector
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/nav%20demo%203.png" 
     alt="Navigation 3" width=600><br>

## 2. Segmentation
- Most road surface semantic segmentation convolutional neural network (CNN) models avalailable online are trained from a self-driving car's perspective. 
- A lightweight CNN for walkable path road segmentation, which can deployed in an OpenCV AI Kit with Depth (OAK-D) device.
- [Refer this link to see how the model is trained](https://github.com/yjwong1999/Walkable_Path_Segmentation)

### Outdoor Path
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/Walkable%20Path%20Segmentation%20-%20Outdoor.png" 
     alt="Outdoor Path" width=400><br>
### Busy Road Path
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/Walkable%20Path%20Segmentation%20-%20Busy%20Road.png" 
     alt="Busy Road Path" width=400><br>
### Indoor Path
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/Walkable%20Path%20Segmentation%20-%20Indoor.png" 
     alt="Indoor Path" width=400><br>

## 3. Pedestrian Traffic Light Detection
- A Yolo model is desiged to detect for pedestrian traffic light, and classify if it is red or green light now.
- The model will not detected normal traffic light for car.

### Red Light
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/traffic%20red.png" 
     alt="Red Light" width=400><br>
### Green Light
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/traffic%20green.png" 
     alt="Green Light" width=400><br>
### Non-Pedestrian Traffic Light
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/traffic%20-%20non%20pedestrian.png" 
     alt="Non-Pedestrian Traffic Light" width=400><br>
     
## 4. Personal Protection Equipment (PPE) Detector
A Yolo model is used to detect PPE for the user, including:
- Facemask
- Sanitizer
- Thermometer

### Face Mask
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/ppe-facemask.png" 
     alt="Face Mask" width=400><br>
### Sanitizer
<img src="https://raw.githubusercontent.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/main/others/ppe-sanitizer.png" 
     alt="Sanitizer" width=400><br>
