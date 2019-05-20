from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from PIL import Image
import rospy
import os
import cv2
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        if is_site:
            self.SSD_GRAPH_FILE_PATH = r'../models/ssd_udacity/frozen_inference_graph.pb'
            #self.SSD_GRAPH_FILE_PATH = rospy.get_param('~/site_model_path', "not found")
        else:
            #self.SSD_GRAPH_FILE_PATH = r'../models/ssd_sim/frozen_inference_graph_2.pb'
            self.SSD_GRAPH_FILE_PATH = rospy.get_param('~/sim_model_path', "not found")
        rospy.logdebug('#### is_site %s? SSD_GRAPH_FILE_PATH read = %s',is_site, self.SSD_GRAPH_FILE_PATH)

        self.detection_graph = self.load_graph(self.SSD_GRAPH_FILE_PATH)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.detection_graph)
        # yellow_image_path = rospy.get_param('~/test_image_yellow', "not found")
        # print('classfiy yellow')
        # img_binary = Image.open(yellow_image_path)
        # self.get_classification(img_binary)


    def filter_boxes(self,min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def load_graph(self,graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # graph_file_1='models/ssd_sim/frozen_inference_graph.pb'
        # with open(graph_file_1, 'rb') as f:
        #     serialized = f.read()
        #     detector_graph_def = tf.GraphDef()
        #     detector_graph_def.ParseFromString(serialized)
        #     tf.import_graph_def(detector_graph_def, name='detector')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        # image_np = np.expand_dims(image, axis=0)
        # start = datetime.datetime.now()
        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        if len(classes) == 0:
            return TrafficLight.UNKNOWN
        switcher = {
            1: TrafficLight.GREEN,
            2: TrafficLight.RED,
            3: TrafficLight.YELLOW,
            }

        class_result = switcher.get(classes[0], TrafficLight.UNKNOWN)
        rospy.logdebug('get_classification end class = %d', class_result)
        # end = datetime.datetime.now()
        # c = end - start
        return class_result
