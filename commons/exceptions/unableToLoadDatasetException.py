"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # X - Lab's name

Students :
    Names — Permanent Code

Group :
    GTI770-H18-0X
"""

class UnableToLoadDatasetException(Exception):
    def __init__(self, message):
        self.message = message
