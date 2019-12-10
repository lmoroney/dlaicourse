# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tflite_runtime.interpreter import Interpreter

from visualization_utils import *
from utils import *


class ObjectDetectorLite:
    def __init__(self, model_path, label_path):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load label_map
        self._load_label(label_path)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input size
        input_shape = self.input_details[0]['shape']
        self.size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

    def get_input_size(self):
        return self.size

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """
        # Add a batch dimension
        frame = np.expand_dims(image, axis=0)
        
        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num = self.interpreter.get_tensor(self.output_details[3]['index'])

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                                        np.squeeze(boxes[0]),
                                        np.squeeze(classes[0]+1).astype(np.int32),
                                        np.squeeze(scores[0]),
                                        min_score_thresh=threshold)

    def close(self):
        pass
      
    def _boxes_coordinates(self,
                        image,
                        boxes,
                        classes,
                        scores,
                        max_boxes_to_draw=20,
                        min_score_thresh=.5):
        """
        This function groups boxes that correspond to the same location
        and creates a display string for each detection 

        Args:
          image: uint8 numpy array with shape (img_height, img_width, 3)
          boxes: a numpy array of shape [N, 4]
          classes: a numpy array of shape [N]
          scores: a numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
          max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
            all boxes.
          min_score_thresh: minimum score threshold for a box to be visualized
        """

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        number_boxes = min(max_boxes_to_draw, boxes.shape[0])
        detected_boxes = []
        probabilities = []
        categories = []

        for i in range(number_boxes):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                detected_boxes.append(box)
                probabilities.append(scores[i])
                categories.append(self.category_index[classes[i]])
        return np.array(detected_boxes), probabilities, categories

    def _load_label(self, path):
        """
            Loads labels
        """
        categories = load_labelmap(path)
        self.category_index = create_category_index(categories)
