#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from mdlp.discretization import MDLP

class SupervisedDiscretizer(object):

    def __init__(self):
        self.transformer = MDLP()

    def discretize(self, X, y):
        X_discretized = self.transformer.fit_transform(X=X, y=y)
        return X_discretized

