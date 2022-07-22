# ros_mobilenet

ROS Package for MobileNet SSD

### Node
The topic names can be changed in the param.yaml 
#### Subscribed Topics
1. /webcam/image_raw (sensor_msgs/Image)<br>
The input RGB image stream
<br>

#### Published Topics
1. /mobilenet/image_raw (sensor_msgs/Image) <br>
The resultant image after being passed through the detector. <br>

2. /mobilenet/prediction (ros_mobilenet/Prediction) <br>
Custom message that contains the class, prediction score, & the cordinates of the bounding box

#### Services
1. is_detect (std_srv/SetBool) <br>
Switches on or off the detector. 

### Requirements
Run `bash requirements.sh` <br>


### Usage
`roslaunch ros_mobilenet model.launch` <br>
`rosservice call /is_detect "data: false/true`  to set the detection off/on