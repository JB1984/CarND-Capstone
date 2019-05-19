import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import datetime

#python tl_classifier_test.py
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.SSD_SIM_FILE_PATH = '../models/ssd_sim/frozen_inference_graph.pb'
        self.SSD_TESTAREA_FILE = '../models/ssd_udacity/frozen_inference_graph.pb'
        test_color = "Unknown"
        dir_path = '../alex-lechner-udacity-traffic-light-dataset/simulator_dataset_rgb/'+test_color

        # sim_model_path = rospy.get_param('~/sim_model_path', "not found")
        self.detection_graph = self.load_graph(self.SSD_SIM_FILE_PATH)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.detection_graph)
        # red_image_path = '../models/camera_images/red_0.png'
        # green_image_path = '../models/camera_images/green_2.png'
        # yellow_image_path = '../models/camera_images/yellow_1.png'
        # print('classfiy red')
        # img_binary = Image.open(red_image_path)
        #
        # # img_binary = cv2.imread(image_path)
        # # image = cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB)
        # self.get_classification(img_binary)
        # print('classfiy yellow')
        # img_binary = Image.open(yellow_image_path)
        # self.get_classification(img_binary)
        #
        # print('classfiy green')
        # img_binary = Image.open(green_image_path)
        # self.get_classification(img_binary)


        cnt = 0
        correct_cnt = 0
        dirs = os.listdir(dir_path)
        for file_name in dirs:
            if not file_name.endswith('.png'):
                continue
            cnt += 1
            file_path = '../alex-lechner-udacity-traffic-light-dataset/simulator_dataset_rgb/'+test_color+'/'+file_name
            img_binary = Image.open(file_path)
            class_result = self.get_classification(img_binary)
            if class_result == "TrafficLight.UNKNOWN":
                correct_cnt +=1
            else:
                print('!!!wrong',class_result,file_path)

        print(test_color,"=======correct/cnt", correct_cnt, cnt)







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
        # image_path = rospy.get_param("/test_image")
        # image = Image.open(image_path)
        # start = datetime.datetime.now()

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        # image_np = np.expand_dims(image, axis=0)
        # with tf.Session(graph=self.detection_graph) as sess:
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
            return "TrafficLight.UNKNOWN"
        switcher = {
            1: "TrafficLight.GREEN",
            2: "TrafficLight.RED",
            3: "TrafficLight.YELLOW",
            }

        # if classes[0] != 3:
        #     print("!!! incorrect: original classes/scores: ",classes,scores)
        class_result = switcher.get(classes[0], "TrafficLight.UNKNOWN")

        # print('####classification class = %d', class_result)
        # end = datetime.datetime.now()
        # print('-- total time', (end-start).total_seconds())
        return class_result



if __name__ == '__main__':
    TLClassifier()
