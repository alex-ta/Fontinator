from NeuronalNet.Oli.libs.ProcessingPipeline import ProcessingPipeline
from NeuronalNet.Oli.libs.Preprocessor import SimplePreprocessor, IPreprocessor

#__________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = '../../images/Dataset_3'
# Name for model when saved
MODEL_LOAD_PATH = "SavedModels/LT2"

# Pipeline managing working with keras model
pipeline: ProcessingPipeline = ProcessingPipeline()

# Loads all images and extract features and labels
preprocessor: IPreprocessor = SimplePreprocessor()
x, y = pipeline.load_features_and_preprocess(IMG_PATH, img_preprocessor=preprocessor)

# Load the model from disk
pipeline.load_model(MODEL_LOAD_PATH)

# Make predictions with loaded model
y_pred = pipeline.predict(x)

# Evaluate model on test images and show summary
pipeline.evaluate(y, y_pred)



