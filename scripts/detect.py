#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse
import cv2
import numpy as np
import os
from tensorflow.lite.python.interpreter import Interpreter

class MobileNet:
    def __init__(self) -> None:
        '''
        Initializes subscribers, publishers, service client,
        & MobileNet SSD model
        '''
        self.sub_topic = rospy.get_param("input_image", "/image_raw")
        self.pub_topic = rospy.get_param("output_image", "/detect/image_raw")
        rospy.Subscriber(self.sub_topic, Image, self.detect)
        self.pubimage = rospy.Publisher(self.pub_topic, Image, queue_size= 1)
        rospy.Service("is_detect", SetBool, self.service_handler)
        self.isDetect = True
        
        self.bridge = CvBridge()

        CWD_PATH = os.path.join( os.path.dirname( __file__ ) )
        MODEL_NAME = "model"
        GRAPH_NAME = "detect.tflite"
        LABELMAP_NAME = "labelmap.txt"

        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        if self.labels[0] == '???':
            del(self.labels[0])

        self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = self.output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else: # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

    def service_handler(self, request):
        '''
        Handles the flag for whether to detect or not
        '''
        if request.data:
            self.isDetect = request.data
            resp = SetBoolResponse(True, "Detection is set to True")
            return resp
        else:
            self.isDetect = request.data
            resp = SetBoolResponse(False, "Detection is set to False")
            return resp

    def detect(self, msg):
        '''
        Image is passed through here and passed to the model for inference.
        The detection is only done if the isDetect flag is set to True
        '''
        imW = msg.width
        imH = msg.height
        cv_image =  self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        if self.isDetect:
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()
            # Retrieve detection results
            boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
            scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > .6) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(cv_image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                    # Draw label
                    object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(cv_image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(cv_image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        output_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
        self.pubimage.publish(output_msg)

if __name__ == "__main__":
    rospy.init_node("detector_node")
    MobileNet()
    rospy.spin()
