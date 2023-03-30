#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import cv2
import numpy
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImagePublisher(Node):
    def __init__(self, img_folder):
        super().__init__('hps_image_publisher')
        self.publisher_ = self.create_publisher(Image, '/hps/robot/cam_front', 10)
        timer_period = 0.0333  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.img_folder = img_folder
        self.img_files = []
        # img_file_example = "1679992067688454000.jpg"
        for img_file in os.listdir(img_folder):
            tmp = img_file.split(".")
            if len(tmp) == 2 and tmp[1] == "jpg":
                self.img_files.append(img_file)
        self.img_files.sort()

    def timer_callback(self):

        time_before_publish = time.time()

        # reads image data
        img_path = os.path.join(self.img_folder, self.img_files[self.i])
        # cvimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        cvimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        ts = int(self.img_files[self.i].split(".")[0])



        # processes image data and converts to ros 2 message
        msg = Image()
        msg.header.stamp.sec = ts // 1000000000
        msg.header.stamp.nanosec = ts % 1000000000
        msg.header.frame_id = 'base_link'
        msg.height = numpy.shape(cvimg)[0]
        msg.width = numpy.shape(cvimg)[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = numpy.shape(cvimg)[2] * numpy.shape(cvimg)[1]

        ## assignment to msg.data is very slow for unknown reasons.
        ## here we assign to the private member ._data instead to circumvent it.
        ## see: https://stackoverflow.com/questions/71939189/numpy-tobytes-method-is-very-slow

        # msg.data = numpy.array(cvimg).tobytes()  
        msg._data = numpy.array(cvimg).tobytes()



        # publishes message
        self.publisher_.publish(msg)

        time_after_publish = time.time()
        cost = time_after_publish - time_before_publish

        # image counter increment
        self.i += 1

        # self.get_logger().info('%d Images Published, cost %f seconds' % (self.i , cost))     
        if  self.i  >=  len(self.img_files):
            
        return None


def main(argv=None):
    rclpy.init(args=argv)

    img_publisher = ImagePublisher(argv[1])

    rclpy.spin(img_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    img_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)

