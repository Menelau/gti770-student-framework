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

import numpy as np

class UnsupervisedDiscretizer(object):

    def __init__(self):
        pass

    def find_range(self, X):
        min = np.amin(X, axis=0)
        max = np.amax(X, axis=0)

        return min, max

    def discretize(self, X, nb_bins=10):

        digitized = np.zeros(shape=X.shape)

        for i in range(0, X[0].size):

            min, max = self.find_range(X[:, i])

            bins = np.linspace(min, max, nb_bins)

            digitized[:, i] = (np.digitize(X[:, i], bins))

        return digitized