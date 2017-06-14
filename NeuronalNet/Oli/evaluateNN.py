from NeuronalNet.Oli.libs.NetManager import NetManager
from NeuronalNet.Oli.libs.Preprocessor import SimplePreprocessor, IPreprocessor

#__________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = '../../DataGenerator/images/text_it_mgmt'
# Name for model when saved
MODEL_LOAD_PATH = "../SavedModels/LT2"

# Pipeline managing working with keras model
netManager: NetManager = NetManager()

# Loads all images and extract features and labels
preprocessor: IPreprocessor = SimplePreprocessor()
x, y = netManager.load_features(IMG_PATH, img_preprocessor=preprocessor)

# Load the model from disk
netManager.load_model(MODEL_LOAD_PATH)

# Make predictions with loaded model
y_pred = netManager.predict(x)

# Evaluate model on test images and show summary
netManager.evaluate(y, y_pred)



