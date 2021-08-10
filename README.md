# OpenCV-AI-Competition-UTAR4Vision
Please refer the links before for the demonstrations of the project.<br>
[Demonstration Sample Images](https://github.com/yjwong1999/OpenCV-AI-Competition-UTAR4Vision/blob/main/demo/Demo.md)<br>
[Demonstration Video](https://drive.google.com/file/d/17MhLnoijZUo6YUl7zlxEkXu-qxgSOTll/view?usp=sharing)

# Terminology
Some important terminologies for a better understanding on the readme
### 1. OAK-D
- an OpenCV AI Kit (with depth) device 
- has a stereo camera setup, and have Intel VPU chip in it
- [OAK-D specs](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html#bw1098oak)
### 2. Pipeline
- a pipeline is the workflow on the OAK-D side, 
- it consists of nodes (such as monocam node, neural network node, and etc.), together with the connections between them
![depthai](https://user-images.githubusercontent.com/55955482/126036064-95e69b4f-7579-44f1-bb06-b003ed24fb72.png)

# Utilites file

## 1. depthai_utils
Contains the code for each of the Pipeline required in this project
The Pipelines involved:
### Navigation
- obstacle detection
- obstacle avoidance
- social distancing
### Walkable road path segmentation
- road path segmentation for pedestrian's perspective
### Pedestrian traffic light detection
- to detect and classify the traffic light for pedestrian
### Protective Equipment (PPE) detection
- to detect hand sanitizer/face mask/thermometer

## 2. task_utils
### Speech Recognition
- an OOP object for speech recognition and voice command
- designed for multithreading
### Navigator
- contains the functions for obstacle avoidance, social distancing

# Sample Codes
### 1. main.py
- the consolidated code
- flowchart will be added soon
### 2. oakd_navigation.py
- demo code for navigation
### 3. oakd_segmentation.py
- demo code for road segmentation
### 4. oakd_traffic_light.py
- demo code for pedestrian traffic ligth detection
### 5. oakd_ppes_detection.py
- demo code for ppe detection
