import os

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from NeuronalNet.DataLoader import *
from NeuronalNet.ModelSerializer import ModelSerializer
from NeuronalNet.Preprocessor import *

# Load the NN model from disk
print("Loading model from disk")
model_serializer = ModelSerializer("LongTrained")
model = model_serializer.load_model_from_files()

print("Compiling NN model ...")
nn_optimizer = RMSprop(lr=0.0001)
model.compile(optimizer=nn_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Loads the images from the defined path
data_loader: DataLoader = DataLoader('X:\WichtigeDaten\GitProjects\\tmp\\1000_Images')

image_count = data_loader.get_image_count()
font_count = data_loader.get_font_count()
print("Found {0} images with {1} different fonts".format(image_count, font_count))

preprocessor: Preprocessor = Preprocessor()

font_names: list = data_loader.get_font_names()

label_encoder = LabelEncoder()
label_encoder.fit(font_names)

print("Start preprocessing images ...")
features = []
labels = []
# Iterate over all fonts
for f_name in font_names:
    print(" -> {0}".format(f_name))
    label_id = label_encoder.transform([f_name])
    font_labels = np.full(data_loader.get_img_count_for_font(f_name), label_id)
    labels.extend(font_labels)

    # Iterate over all images for one font
    for img_path in data_loader.iterate_images_for_fontname(f_name):
        nd_img: ndarray = preprocessor.prepare_image(img_path)
        features.append(nd_img)

x: ndarray = np.array(features)
y: ndarray = np.array(labels)

# Convert labels to categorical one-hot encoding; e.g. [1, 2, 3] -> [[1,0,0], [0,1,0], [0,0,1]]
y_onehotenc = np_utils.to_categorical(y)

# Make predctions
y_pred_onehotenc = model.predict(x)

# Retransform one hot encoding to indexes
y_pred = y_pred_onehotenc.argmax(axis=1)

# Calculate correct and wrong prediction count
correct_pred_items = np.equal(y, y_pred)

cor_pred_count = np.sum(correct_pred_items)
wrong_pred_count = y_pred.size - cor_pred_count
cor_pred_ratio = cor_pred_count / y_pred.size

print("Summary:")
print("Correct predictions: {0} | Wrong predictions: {1}"
      .format(cor_pred_count, wrong_pred_count))
print("{0}".format(cor_pred_ratio))



