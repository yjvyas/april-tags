#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from dt_apriltags import Detector
import rospkg 


"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")


        calibration_fname = '/data/config/calibrations/camera_extrinsic/default.yaml'
        if 'CALIBRATION_FILE' in os.environ:
            calibration_fname = os.environ['CALIBRATION_FILE']
        
        self.homography = self.readYamlFile(calibration_fname)
        H = np.reshape(np.array(self.homography['homography']), [3, 3])

        intrinsic_fname = '/data/config/calibrations/camera_intrinsic/default.yaml'
        if 'INTRINSICS_FILE' in os.environ:
            intrinsic_fname = os.environ['CALIBRATION_FILE']
        self.intrinsics = self.readYamlFile(intrinsic_fname)

        self.K = np.reshape(np.array(self.intrinsics['camera_matrix']['data']), [3,3])

        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=4,
                                    quad_decimate=2.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)


        rospack = rospkg.RosPack()

        # subscriber and publisher
        self.sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.callback)
        self.pub = rospy.Publisher(f'/{self.veh_name}/augmented_reality/image/compressed', CompressedImage, queue_size=10)

        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')
        self.bridge = CvBridge()

    def callback(self, msg):
        """Detects april-tags and renders duckie on them."""
        img = self.readImage(msg)

        K = self.K
        tags = self.at_detector.detect(img[:,:,0], 
                                       estimate_tag_pose=True, 
                                       camera_params=[K[0], K[4], K[2], K[5]],
                                       tag_size=0.0675)

        for t in tags:
            P = self.projection_matrix(t.homography, self.K)
            img = self.renderer.render(img, P)
        
        self.pub.publish(self.cvbr.cv2_to_compressed_imgmsg(img))

    
    def projection_matrix(self, H, K):
        
        # Compute rotation along the x and y axis as well as the translation
        rt = np.dot(np.linalg.inv(K), H)
        col_1 = rt[:, 0]
        col_2 = rt[:, 1]
        col_3 = rt[:, 2]

        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)

        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(K, projection)

    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()