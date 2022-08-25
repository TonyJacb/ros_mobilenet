#!/usr/bin/env python
import rospy
from ros_mobilenet.msg import Prediction
from geometry_msgs.msg import Twist
import numpy as np


class Follow:
    def __init__(self) -> None:
        rospy.Subscriber("/mobilenet/prediction", Prediction, callback= self.gen_cmd_vel)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist_msg = Twist()

        self.LINEAR_SPEED = .5
        self.MINI_LINEAR_SPEED = .05
        self.ANGULAR_SPEED = np.pi/12
        self.MINI_ANGULAR_SPEED = np.pi / 24

        self.FORWARD = self.LINEAR_SPEED
        self.BACK = -self.LINEAR_SPEED
        self.LEFT = self.ANGULAR_SPEED
        self.RIGHT = -self.ANGULAR_SPEED
        self.MINI_FORWARD = self.MINI_LINEAR_SPEED
        self.MINI_BACK = -self.MINI_LINEAR_SPEED
        self.MINI_LEFT = self.MINI_ANGULAR_SPEED
        self.MINI_RIGHT = -self.MINI_ANGULAR_SPEED

    def gen_cmd_vel(self, prediction):
        if prediction.label == "person":
            vertical_ratio = self.get_v_ratio(prediction.ymax, prediction.ymin)
            linear_speed = self.get_linear_speed(vertical_ratio)

            lateral_percentage = self.get_l_percentage(prediction.xmax, prediction.xmin)
            angular_speed = self.get_angular_speed(lateral_percentage)

            self.twist_msg.linear.x = linear_speed
            self.twist_msg.angular.z = angular_speed * 4
            self.pub.publish(self.twist_msg)
    
    def get_v_ratio(self, ymax, ymin):
        height = ymax - ymin
        ratio = float(height) / 480
        return ratio
    
    def get_linear_speed(self,ratio):
        low = .7
        high = .9
        low_mid = low
        mid_high = high
        if ratio < low:
            return self.FORWARD
        elif low <= ratio < low_mid:
            return self.MINI_FORWARD
        elif low_mid <= ratio < mid_high:
            return 0.
        elif mid_high <= ratio < high:
            return self.MINI_BACK
        elif ratio >= high:
            return self.BACK

    def get_l_percentage(self,max, min):
        centre = (max + min)/2
        percentage = float(centre) / 640
        return percentage

    def get_angular_speed(self,l_percentage):
        wide_margin = .15
        small_margin = .05
        mid = .5

        a = mid - wide_margin
        b = mid - small_margin
        c = mid + small_margin
        d = mid + wide_margin

        if l_percentage < a:
            return self.LEFT
        elif a <= l_percentage < b:
            return self.MINI_LEFT
        elif b <= l_percentage < c:
            return 0.
        elif c <= l_percentage < d:
            return self.MINI_RIGHT
        elif l_percentage >= d:
            return self.RIGHT

if __name__ == "__main__":
    rospy.init_node("follower")
    Follow()
    rospy.spin()