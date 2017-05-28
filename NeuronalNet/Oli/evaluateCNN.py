from NeuronalNet.Oli.libs.Pipeline import Pipeline
from NeuronalNet.Oli.libs.Preprocessor import *

#__________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = '../../DataGenerator/images/text_it_mgmt'
# Name for model when saved
MODEL_LOAD_PATH = "../SavedModels/ConvLong"

# Pipeline managing working with keras model
pipeline: Pipeline = Pipeline()

# Loads all images and extract features and labels
preprocessor: IPreprocessor = ConvPreprocessor()
x, y = pipeline.load_features(IMG_PATH, img_preprocessor=preprocessor)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

# Load the model from disk
pipeline.load_model(MODEL_LOAD_PATH)

# Make predictions with loaded model
y_pred = pipeline.predict(x)

# Evaluate model on test images and show summary
pipeline.evaluate(y, y_pred)



