"""
    Auteur : 
        Modification apporté par Robin Quillivic Data Scientsit chez StarClay, rquilivic@starclay.fr
        Basé sur le code de cet article : https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
        
    Description : 
    Le but de ce script est de construire une classe permettant de transformer un classifier scikitLearn multiclasse en classifier multiclasse ordinale. 
    L'approche a été développée dans le papier de Eibe Frank and Mark Hal, (lECML 2001. 12th European Conference ): 
    https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
"""


from sklearn.base import clone
import numpy as np
import logging


class OrdinalClassifier:

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(
                    clfs_predict[y-1][:, 1] - clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
