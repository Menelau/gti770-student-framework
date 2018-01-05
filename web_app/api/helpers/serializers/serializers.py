#!/usr/bin/env python
# encoding: utf-8

"""
  Describes the JSON models used in the API endpoint.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

from flask_restplus import fields

from web_app.api.restplus import api

galaxy_class_analysis = api.model('Galaxy', {
    'class': fields.String(required=True,
                           description='The actual galaxy class found.'),
    'score': fields.Float(required=True,
                          description='The global confidence score of the galaxy classification algorithm.'),
})

classification = api.model('Classification of a galaxy image.', {
    'time_stamp': fields.DateTime(dt_format='rfc822', readOnly=True,
                                  description='The unique time stamp of an extracted galaxy image sent back from the server.'),
    'galaxy_class': fields.Nested(galaxy_class_analysis)
})
