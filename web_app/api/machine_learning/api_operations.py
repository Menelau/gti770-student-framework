#!/usr/bin/env python
# encoding: utf-8

"""
  Lists the available API operations on Machine Learning models.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

# Import generic Python package.
import datetime
import time
from email import utils

import cv2
import numpy as np
import tensorflow as tf

from web_app import config_file


class ApiOperations(object):
    """ Lists the available Machine Learning operations for API.

      This class uses the Machine Learning models and does the necessary back end processing.
    """

    def __init__(self):
        """Initialization function.

        This function reloads a previously exported TensorFlow model.
        """
        self.session, tf_graph = self._create_session()

        # Gets the placeholders from the graph by name.
        self.input_x = tf_graph.get_operation_by_name("input/X").outputs[0]
        self.dropout_keep_prob = tf_graph.get_operation_by_name("input/dropout_keep_probability").outputs[0]

        # The Tensors we want to evaluate.
        self.predictions = tf_graph.get_operation_by_name("accuracy/correct_prediction/predictions").outputs[0]
        self.top_scores = tf_graph.get_operation_by_name("accuracy/correct_prediction/top_scores").outputs[0]

        print("MLP model restored successfully.")

    def _create_session(self):
        tf_graph = tf.Graph()

        with tf_graph.as_default():
            session = tf.Session()

            with session.as_default():
                # Load the saved meta graph and restore variables
                print("Reading DNN model parameters from %s" % config_file.DNN_CHECKPOINT_PATH)
                tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.TRAINING],
                                           config_file.DNN_CHECKPOINT_PATH)

                return session, tf_graph

    @staticmethod
    def _get_time_stamp():
        """ Get the time stamp of an API request.

        The API has a time_stamp field. This method create the value of this time stamp based on the time of the
        API request.

        Returns:
          str: The request's time stamp in a string object.
        """

        time_now = datetime.datetime.now()
        time_tuple = time_now.timetuple()
        time_stamp = time.mktime(time_tuple)
        formatted_time_stamp = utils.formatdate(time_stamp)

        return formatted_time_stamp

    def _preprocess_image(self, image):
        """ Preprocess an image submitted via the web API.

        Args:
            image: an image file saved on disk.

        Returns:
            image: an OpenCV standard image format.

        """
        # Use OpenCV to read the image.
        image = cv2.imread(config_file.UPLOAD_FOLDER + image)

        # Transform image as a 32-bit numpy array.
        image = image.astype(np.float32)

        # Normalize the image.
        image = np.multiply(image, 1.0 / 255.0)

        return image

    def calssify(self, image_filename):
        """ Classify a galaxy image.

        Makes a classification of the submitted galaxy image file using an in-RAM, previously loaded TensorFlow model.

        Args:
          image_filename: The file name on disk of the submitted image.

        Returns:
          dict: The JSON formatted response.
        """

        image_list = list()

        # Preprocess the image as the learned model requires it.
        processed_image = self._preprocess_image(image_filename)

        image_list.append(processed_image)

        # Inference process.
        score, prediction = self.session.run([self.top_scores, self.predictions],
                                             feed_dict={self.input_x: image_list, self.dropout_keep_prob: 1.0})

        # Gets a time stamp of the request.
        time_stamp = self._get_time_stamp()

        # Prepares the HTTP response in JSON format.
        data = \
            {
                'time_stamp': time_stamp,
                'galaxy_class':
                    {
                        'class': str(prediction[0]),
                        'score': score[0][0],
                    },
            }

        return data
