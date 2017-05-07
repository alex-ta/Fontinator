import os

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from NeuronalNet.DataLoader import *
from NeuronalNet.Preprocessor import *


data_loader: DataLoader = DataLoader('..\DataGenerator\images')

img_size = 48000
image_count = data_loader.get_image_count()
font_count = data_loader.get_font_size()

x: ndarray = np.empty(shape=(image_count, img_size))
y: ndarray = np.empty(shape=(image_count))

preprocessor: Preprocessor = Preprocessor()

font_names: list = data_loader.get_fonts()

label_encoder = LabelEncoder()
label_encoder.fit(font_names)

i = 0
# Iterate over all fonts
for font_name in font_names:
    label_id = label_encoder.transform([font_name])
    # Iterate over all images for one font
    for img_path in data_loader.iterate_images_for_fontname(font_name):
        nd_img: ndarray = preprocessor.prepare_image(img_path)
        x[i] = nd_img
        y[i] = label_id
        i += 1

# Convert labels to categorical one-hot encoding; e.g. [1, 2, 3] -> [[1,0,0], [0,1,0], [0,0,1]]
y_onehotenc = np_utils.to_categorical(y)

train_X, test_X, train_y, test_y = train_test_split(x, y_onehotenc, train_size=0.75, random_state=0)

model = Sequential()
model.add(Dense(2400, input_shape=(img_size,)))
model.add(Activation('sigmoid'))
model.add(Dense(120))
model.add(Activation('sigmoid'))
model.add(Dense(6))
model.add(Activation('sigmoid'))
model.add(Dense(font_count))
model.add(Activation('softmax'))

print("Compiling NN model ...")
nn_optimizer = RMSprop(lr=0.0001)
model.compile(optimizer=nn_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
print("Training the NN model")
model.fit(train_X, train_y, epochs=1000, batch_size=int(0.8*x.size))

loss_and_metrics = model.evaluate(test_X, test_y, batch_size=img_size)
print(loss_and_metrics)

