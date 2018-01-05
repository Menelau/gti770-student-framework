#!/usr/bin/env python
# encoding: utf-8

"""
  Flask application script. This is the entry point of the web application.

  :Date: 2017-12-10
  :Author: Pierre-Luc Delisle
  :Version: 1.0
  :Credit: None
  :Maintainer: Pierre-Luc Delisle
  :Email: pierre-luc.delisle.1@ens.etsmtl.ca
"""

# Import generic Python package.
from flask import Flask

from web_app import config_file
from web_app.api.endpoints.classify import api as analyze_galaxy
from web_app.api.restplus import api, blueprint

app = Flask(__name__)


def configure_app(flask_app):
    """ Configure the Python Flask application.

      Configure the current Python Flask application containing the API and serving machine learning models.

      Args:
        flask_app (Flask): A Flask application instance.
    """

    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = config_file.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = config_file.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = config_file.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = config_file.RESTPLUS_ERROR_404_HELP
    flask_app.config['UPLOAD_FOLDER'] = config_file.UPLOAD_FOLDER


def initialize_app(flask_app):
    """ Initialize the web application.

      Initialize the Flask application and API.

      Args:
        flask_app (Flask): A Flask application instance.
    """

    configure_app(flask_app)
    api.add_namespace(analyze_galaxy)
    flask_app.register_blueprint(blueprint)


def main():
    """ Main application function.

     Initializes and starts the Flask application.
    """

    initialize_app(app)
    app.run(debug=config_file.FLASK_DEBUG, host='0.0.0.0')


if __name__ == '__main__':
    main()
