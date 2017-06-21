from NeuronalNet.Oli.libs.NetManager import NetManager
from NeuronalNet.Oli.libs.Preprocessor import *

#__________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = '../../images/Dataset_1'
# Name for model when saved
MODEL_LOAD_PATH = "SavedModels/CNN_RAND_80"

# Pipeline managing working with keras model
netManager: NetManager = NetManager()

# Loads all images and extract features and labels
preprocessor: IPreprocessor = ConvPreprocessor()
x, y = netManager.load_features_and_preprocess(IMG_PATH, img_preprocessor=preprocessor)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

# Load the model from disk
netManager.load_model(MODEL_LOAD_PATH)

# Make predictions with loaded model
y_pred = netManager.predict(x)

# Evaluate model on test images and show summary
netManager.evaluate(y, y_pred)



