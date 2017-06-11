import numpy as np
from sklearn import neighbors, datasets
import pickle

# Config
weight = 'distance'
n_neighbors = 1

class Classifier:
    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)

    def trainClassifier(self, featureVectors, classVectors):
        self.clf.fit(featureVectors, classVectors)

    def saveTrainedClassifier(self, path):
        with open(path, 'wb') as file_handle:
            pickle.dump(self.clf, file_handle) #, protocol=pickle.HIGHEST_PROTOCOL)

    def loadTrainedClassifier(self, path):
        with open(path, 'rb') as file_handle:
            self.clf = pickle.load(file_handle)

    def predictData(self, featureVectors):
        return self.clf.predict(featureVectors)

    def testClassifier(self, featureVectors, classVectors):
        predictedData = self.clf.predict(featureVectors)

        correct = 0
        false = 0

        for i in range( 0, len(predictedData) ):
            if predictedData[0] == classVectors[0]:
                correct += 1
            else:
                false += 1

        print("Accuracy ", (correct/(correct+false)))


'''
a = np.array([[1, 2], [1, 3], [1, 1], [2, 1], [2, 2], [2, 3], [1, 4], [2, 4]])
b = np.array([1, 1, 1, 2, 2, 2, 1, 2])

#clf = Classifier()
#clf.trainClassifier(a, b)

a = np.array([[1, 10], [1, 31], [1, 11], [2, 12], [2, 22], [2, 43], [1, 41], [2, 41]])

#clf.predictData(a)
#clf.testClassifier(a, b)

#clf.saveTrainedClassifier('./classie.pickle')

cla = Classifier()
cla.loadTrainedClassifier('./classie.pickle')

cla.predictData(a)
cla.testClassifier(a, b)
'''
