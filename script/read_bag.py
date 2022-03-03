#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import os

data_dir = "/home/cadit/Data/snu_lib_rect/"
depth_topic = "/camera/depth/image_rect_raw"
rgb_topic = "/camera/color/image_raw"

f = open(data_dir + "associations.txt", 'w')

try:
  os.mkdir(data_dir+"rgb")
  os.mkdir(data_dir+"depth")
except:
  pass

class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.depth_sub = message_filters.Subscriber(depth_topic, Image)
    self.rgb_sub = message_filters.Subscriber(rgb_topic, Image)
   

  def callback(self,depth, rgb):
    try:
      # Read rgb and depth image from ros message
      depth_img = self.bridge.imgmsg_to_cv2(depth, depth.encoding)
      rgb_img = self.bridge.imgmsg_to_cv2(rgb, rgb.encoding)
      rgb_img = cv2.resize(rgb_img, (848, 480))
      
      # Read time stamp
      t_d_str = str(depth.header.stamp)
      t_rgb_str = str(rgb.header.stamp)
      time_depth = t_d_str[:len(t_d_str)-9] + '.' + t_d_str[len(t_d_str)-9: len(t_d_str)-3]
      time_rgb = t_rgb_str[:len(t_rgb_str)-9] + '.' + t_rgb_str[len(t_rgb_str)-9: len(t_rgb_str)-3]
      
      # Save rgb ans depth image
      cv2.imwrite(data_dir+"rgb/"+time_rgb+".png", cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB))
      cv2.imwrite(data_dir+"depth/"+time_depth+".png", depth_img)

      # Write associations.txt
      f.write(time_depth+" "+"depth/"+time_depth+".png"+" "+time_rgb+" "+"rgb/"+time_rgb+".png\n")
      
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  sync = message_filters.ApproximateTimeSynchronizer([ic.depth_sub, ic.rgb_sub], 1, 1)
  sync.registerCallback(ic.callback)
  print("Ready!")
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    f.close()
