from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import plot_model

from NeuronalNet.Oli.libs.NetManager import NetManager
from NeuronalNet.Oli.libs.Preprocessor import *

# __________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = 'X:\WichtigeDaten\GitProjects\\tmp\\100_Images'
# Count of epoches when learning the NN model
TRAIN_EPOCHS = 2
# Name for model when saved
MODEL_OUTPUT_PATH = "SavedModels/Demo"
# The ratio of data to use for training (0.0 < x < 1.0)
TRAIN_RATIO = 0.8

# Pipeline managing working with keras model
netManager: NetManager = NetManager()

# Loads all images and extrakt features and labels
preprocessor: IPreprocessor = SimplePreprocessor()
x, y = netManager.load_features(IMG_PATH, img_preprocessor=preprocessor)

# Defining the Network structure
model = Sequential()
model.add(Dense(2400, input_shape=(x.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dense(int(y.max() + 1)))
model.add(Activation('softmax'))

# Train the NN model and save to disk
netManager.train_model(model, x, y, epos=TRAIN_EPOCHS, train_ratio=TRAIN_RATIO, batch_size=0.25)

# Saves the model structure, weights and additional metadata about the training
netManager.save_model(MODEL_OUTPUT_PATH)
plot_model(model, to_file=MODEL_OUTPUT_PATH + '/model_structure.svg', show_layer_names=True, show_shapes=True)
plot_model(model, to_file=MODEL_OUTPUT_PATH + '/model_structure.png', show_layer_names=True, show_shapes=True)
