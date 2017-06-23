from NeuralNet.Oli.libs.ProcessingPipeline import ProcessingPipeline
from NeuralNet.Oli.libs.Preprocessor import *

#__________Configuration__________#
# Path to folder which contains subfolders with the images
IMG_PATH = '../../images/Dataset_2'
# Name for model when saved
MODEL_LOAD_PATH = "SavedModels/CNN_RAND_80"

# Pipeline managing working with keras model
pipeline: ProcessingPipeline = ProcessingPipeline()

# Loads all images and extract features and labels
preprocessor: IPreprocessor = ConvPreprocessor()
x, y = pipeline.load_features_and_preprocess(IMG_PATH, img_preprocessor=preprocessor)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

# Load the model from disk
pipeline.load_model(MODEL_LOAD_PATH)

# Make predictions with loaded model
y_pred = pipeline.predict(x)

# Evaluate model on test images and show summary
pipeline.evaluate(y, y_pred)



