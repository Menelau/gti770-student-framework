#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  This file lists the available parser arguments.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

from flask_restplus import reqparse
from werkzeug.datastructures import FileStorage

galaxy_image_arguments = reqparse.RequestParser()
galaxy_image_arguments.add_argument("galaxy_image",
                                    type=FileStorage,
                                    required=True,
                                    location="files",
                                    help="Galaxy image to classify.")
