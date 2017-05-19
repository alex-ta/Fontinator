from NeuronalNet.Oli.libs.Pipeline import Pipeline
from NeuronalNet.Oli.libs.Preprocessor import SimplePreprocessor, IPreprocessor

#__________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = 'X:\WichtigeDaten\GitProjects\Fontinator\DataGenerator\images\\raw_char_groups'
# Name for model when saved
MODEL_LOAD_PATH = "SavedModels/ALL2500_AC0.87"

# Pipeline managing working with keras model
pipeline: Pipeline = Pipeline(MODEL_LOAD_PATH, IMG_PATH)

# Loads all images and extract features and labels
preprocessor: IPreprocessor = SimplePreprocessor()
x, y = pipeline.load_features(img_preprocessor=preprocessor)

# Load the model from disk
pipeline.load_model()

# Make predictions with loaded model
y_pred = pipeline.predict(x)

# Evaluate model on test images and show summary
pipeline.evaluate(y, y_pred)



