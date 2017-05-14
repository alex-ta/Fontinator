from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import ndarray

from NeuronalNet.Oli.libs.DataLoader import DataLoader
from NeuronalNet.Oli.libs.ModelSerializer import ModelSerializer
from NeuronalNet.Oli.libs.Preprocessor import Preprocessor

#__________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = 'X:\WichtigeDaten\GitProjects\Fontinator\DataGenerator\images'
# Name for model when saved
MODEL_NAME = "SavedModels/ALL2500_AC0.87"

# Load the NN model from disk
print("Loading model from disk")
model_serializer = ModelSerializer(MODEL_NAME)
model = model_serializer.load_model_from_files()

print("Compiling NN model ...")
nn_optimizer = RMSprop(lr=0.0001)
model.compile(optimizer=nn_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Loads the images from the defined path
data_loader: DataLoader = DataLoader(IMG_PATH)
font_names: list = data_loader.get_font_names()

image_count = data_loader.get_image_count()
font_count = data_loader.get_font_count()
print("Found {0} images with {1} different fonts".format(image_count, font_count))

# Map labels(str) to class_ids(int)
label_encoder = LabelEncoder()
label_encoder.fit(font_names)
label_ids = label_encoder.transform(label_encoder.classes_)
print("Mapping labels:\n{0} \n -> {1}".format(label_encoder.classes_, label_ids))

print("Start preprocessing images ...")
preprocessor: Preprocessor = Preprocessor()
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



