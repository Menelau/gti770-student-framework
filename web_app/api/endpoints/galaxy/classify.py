#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" API endpoint class for sentence analysis.

  This is the endpoint for galaxy image classification.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

import os

from flask import request
from flask_restplus import Resource, Namespace
from werkzeug.utils import secure_filename

from web_app.api.helpers.parsers.parsers import galaxy_image_arguments
from web_app.api.helpers.serializers.serializers import classification
from web_app.api.machine_learning import api_operations

operations = api_operations.ApiOperations()

# Creates a namespace
api = Namespace('classify', description='Complete analysis of a sentence')


@api.route('/galaxy')
@api.response(500, 'Galaxy image file cannot be processed.')
class MusicGenreClassification(Resource):
    """ RESTFul operations for a complete analysis operation.

    This class lists the available REST methods for the complete analysis operation.

    Analyse operation gets the intent and entities out of a sentence using TensorFlow pre-trained neural networks.
    """

    @api.doc("Returns the class of a galaxy image.")
    @api.expect(galaxy_image_arguments)
    @api.marshal_with(classification)
    @api.response(200, 'Galaxy image successfully processed.')
    def post(self):
        """ Returns the genre of a music file..

        Use this method to extract features out of an audio file and make them pass into a classifier.

        * Takes an HTTP parameter the audio file. For example :

         ```
         http://[server_name:server_port]/api/v1.0/analyze/genre_classification
         ```

         Returns:
           The HTTP response containing the music genre analysis.
        """
        args = galaxy_image_arguments.parse_args(request)
        filename = secure_filename(args["galaxy_image"].filename)
        args["galaxy_image"].save(os.environ["VIRTUAL_ENV"] + "/web_app/uploads/" + filename)

        return operations.calssify(filename), 200
