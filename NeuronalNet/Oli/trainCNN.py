from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import plot_model

from NeuronalNet.Oli.libs.Pipeline import Pipeline
from NeuronalNet.Oli.libs.Preprocessor import *

# __________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = '/home/oli/Downloads/Fontinator/DataGenerator/images/text_vietnam'
# Count of epoches when learning the NN model
TRAIN_EPOCHS = 1
# Name for model when saved
MODEL_OUTPUT_PATH = "SavedModels/CNN"
# The ratio of data to use for training (0.0 < x < 1.0)
TRAIN_RATIO = 0.8

# Pipeline managing working with keras model
pipeline: Pipeline = Pipeline()

# Loads all images and extrakt features and labels
preprocessor: IPreprocessor = ConvPreprocessor()
x, y = pipeline.load_features(IMG_PATH, img_preprocessor=preprocessor)

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

# Defining the Network structure
model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(20, kernel_size=(3, 3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(int(y.max() + 1), activation='softmax'))

# Train the NN model and save to disk
pipeline.train_model(model, x, y, epos=TRAIN_EPOCHS, train_ratio=TRAIN_RATIO, batch_size=150)

# Saves the model structure, weights and additional metadata about the training
pipeline.save_model(MODEL_OUTPUT_PATH)
plot_model(model, to_file=MODEL_OUTPUT_PATH + '/model_structure.svg', show_layer_names=True, show_shapes=True)
plot_model(model, to_file=MODEL_OUTPUT_PATH + '/model_structure.png', show_layer_names=True, show_shapes=True)
