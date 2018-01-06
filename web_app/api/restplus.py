#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  This file represents the instance of the API itself.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

import logging

from flask import Blueprint
from flask_restplus import Api

log = logging.getLogger(__name__)

""" The API declaration. """

blueprint = Blueprint('api', __name__, url_prefix='/api/v1')

api = Api(blueprint,
          version='1.0',
          ui=True,
          title="GTI770 Team # X Machine Learning API",
          description='A RESTful API for music analysis.')


@api.errorhandler
def default_error_handler(e):
    """ Default error handler of the API.

      Defines the default error handler for the API in case of an error. Display a message on screen in case the algorithm
      cannot process the request.

      Args:
        e (error): The error.

      Returns:
          tuple: {message (str), 500 (int)}: The first value in the returned tuple is a string containing the error message
          The second value in the tuple is the HTTP 500 code which means 'Internal Server Error'
    """
    message = 'An unhandled exception occurred.'
    log.exception(message)

    return {'message': message}, 500
