import numpy as np
from sklearn import neighbors, datasets
import pickle

# Config
weight = 'distance'
n_neighbors = 1

# Class wihch stores the configure classifier with the information needed to
# normalize the feature vector
class Classifier:
    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
        self.norm_min = None
        self.norm_range = None

    def trainClassifier(self, featureVectors, classVectors):
        featureVectors = ( np.array(featureVectors) - self.norm_min ) / self.norm_range
        self.clf.fit(featureVectors, classVectors)

    def saveTrainedClassifier(self, path):
        with open(path, 'wb') as file_handle:
            pickle.dump(self.clf, file_handle)
            pickle.dump(self.norm_min, file_handle)
            pickle.dump(self.norm_range, file_handle)

    def loadTrainedClassifier(self, path):
        with open(path, 'rb') as file_handle:
            self.clf = pickle.load(file_handle)
            self.norm_min = pickle.load(file_handle)
            self.norm_range = pickle.load(file_handle)

    def predictData(self, featureVectors):
        featureVectors = ( np.array(featureVectors) - self.norm_min ) / self.norm_range
        return self.clf.predict(featureVectors)

    def setNormalization(self, norm_min, norm_range):
        self.norm_min = norm_min
        self.norm_range = norm_range
