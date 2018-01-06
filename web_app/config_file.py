#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  This file contains all the necessary settings, constants and paths the Flask web application uses.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

import os

# General Flask server settings.
FLASK_DEBUG = False  # Do not use debug mode in production


# General Flask_RestPlus API module setting.
RESTPLUS_SWAGGER_UI_DOC_EXPANSION = 'list'
RESTPLUS_VALIDATE = True
RESTPLUS_MASK_SWAGGER = False
RESTPLUS_ERROR_404_HELP = False


# Deep neural network evaluation parameters.
DNN_CHECKPOINT_PATH = os.environ["VIRTUAL_ENV"] + "/data/models/exports/MLP/my_mlp/export"
DROPOUT_KEEP_PROB = 1.0

UPLOAD_FOLDER = os.environ["VIRTUAL_ENV"] + "/web_app/uploads/"

ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = True

