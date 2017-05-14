from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from NeuronalNet.Oli.libs.Preprocessor import *
from NeuronalNet.Oli.libs.DataLoader import DataLoader
from NeuronalNet.Oli.libs.ModelSerializer import ModelSerializer

# __________Configuration__________#
# Path to folder which contains subfolders which with the images
IMG_PATH = 'X:\WichtigeDaten\GitProjects\\tmp\\100_Images'
# Count of epoches when learning the NN model
TRAIN_EPOCHS = 2
# Name for model when saved
MODEL_NAME = "Demo"
# The ratio of data to use for training (0.0 < x < 1.0)
TRAIN_SIZE = 0.8

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

# save the mapping to disk
model_serializer = ModelSerializer(MODEL_NAME)
model_serializer.save_label_mapping(label_encoder.classes_, label_ids)

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

# Splitting to train- /test data
train_X, test_X, train_y, test_y = train_test_split(x, y_onehotenc, train_size=TRAIN_SIZE)

# Defining the Network structure
model = Sequential()
model.add(Dense(2400, input_shape=(x.shape[1],)))
model.add(Activation('sigmoid'))
model.add(Dropout(rate=0.2))
model.add(Dense(120))
model.add(Activation('sigmoid'))
model.add(Dropout(rate=0.2))
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
model.fit(train_X, train_y, epochs=TRAIN_EPOCHS, batch_size=int(0.8 * x.size),
                    validation_data=(test_X, test_y))

loss_and_metrics = model.evaluate(test_X, test_y, batch_size=int(0.8 * x.size))
print(loss_and_metrics)

# Save the NN model to disk
print("Saving NN model and the label index mapping")
model_serializer.serialize_to_disk(model)
