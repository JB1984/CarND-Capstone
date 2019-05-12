from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from PIL import Image


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.SSD_GRAPH_FILE = rospy.get_param("/sim_model")
        self.SSD_TESTAREA_FILE = 'models/ssd_udacity/frozen_inference_graph.pb'
        image_path = rospy.get_param("/test_image")
        image = Image.open(image_path)

        self.detection_graph = self.load_graph(self.SSD_GRAPH_FILE)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')



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
        # with open(graph_file, 'rb') as f:
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
        image = Image.open('./alex-lechner-udacity-traffic-light-dataset/udacity_testarea_rgb/left1122.jpg')

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.8
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            #
            # # The current box coordinates are normalized to a range between 0 and 1.
            # # This converts the coordinates actual location on the image.
            # width, height = image.size
            # box_coords = to_image_coords(boxes, height, width)
            #
            # # Each class with be represented by a differently colored box
            # draw_boxes(image, box_coords, classes)
            #
            # plt.figure(figsize=(12, 8))
            # plt.imshow(image)
            if len(classes) == 0:
                return TrafficLight.UNKNOWN
            switcher = {
                1: TrafficLight.GREEN,
                2: TrafficLight.RED,
                3: TrafficLight.YELLOW,
                }

            return switcher.get(classes[0], TrafficLight.UNKNOWN)
