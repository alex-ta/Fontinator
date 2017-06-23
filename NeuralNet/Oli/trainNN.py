from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import plot_model

from NeuralNet.Oli.libs.ProcessingPipeline import ProcessingPipeline
from NeuralNet.Oli.libs.Preprocessor import *

# __________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = '../../DataGenerator/images/text_zfs'
# Count of epoches when learning the NN model
TRAIN_EPOCHS = 1000
# Name for model when saved
MODEL_OUTPUT_PATH = "SavedModels/Demo"
# The ratio of data to use for training (0.0 < x < 1.0)
TRAIN_RATIO = 0.8

# Pipeline managing working with keras model
pipeline: ProcessingPipeline = ProcessingPipeline()

# Loads all images and extrakt features and labels
preprocessor: IPreprocessor = SimplePreprocessor()
x, y = pipeline.load_features_and_preprocess(IMG_PATH, img_preprocessor=preprocessor)

# Defining the Network structure
model = Sequential()
model.add(Dense(2400, input_shape=(x.shape[1],), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(120, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(60, activation='relu'))
model.add(Dense(int(y.max() + 1), activation='softmax'))

# Train the NN model and save to disk
pipeline.train_model(model, x, y, epos=TRAIN_EPOCHS, train_ratio=TRAIN_RATIO, batch_size=0.25)

# Saves the model structure, weights and additional metadata about the training
pipeline.save_model(MODEL_OUTPUT_PATH)
plot_model(model, to_file=MODEL_OUTPUT_PATH + '/model_structure.svg', show_layer_names=True, show_shapes=True)
plot_model(model, to_file=MODEL_OUTPUT_PATH + '/model_structure.png', show_layer_names=True, show_shapes=True)
