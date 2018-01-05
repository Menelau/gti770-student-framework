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

from sklearn.neighbors import KernelDensity

class GaussianKernelDensity(object):

    def __init__(self):
        self.kde = KernelDensity(bandwidth=1.0, algorithm="auto", kernel="gaussian")

    def train(self, X):
        self.kde.fit(X=X)

    def score_samples(self, X):
        return self.kde.score_samples(X=X)
