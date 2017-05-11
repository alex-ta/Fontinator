from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from dataloader import loader
from dataloader import serialize
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
import cv2 as cv
import numpy as np

# replaces text with int ["x","y","x"] => [0,1,0]
def replace_groups(data):
    a,b,c, = np.unique(data, True, True)
    _, ret = np.unique(b[c], False, True)
    return ret

# load dataset
dx,dyd = loader.read_to_xy_array("C:/Users/Alex/Downloads/images", 0)
# cast to numpy array instead of list
dx = np.array(dx, dtype='int32')
dyd = np.array(dyd)
#dy = np.array(dy, dtype='a16')
classes = np.unique(dyd)
# convert to one_hot labels
dy = np_utils.to_categorical(replace_groups(dyd),len(classes))
# split in test and train
train_X, test_X, train_y, test_y = train_test_split(dx, dy, train_size=0.75, random_state=0)

#img_dimen = 40 * 1200 * 3
#cv.imshow("new",dx[0])
#cv.waitKey(0)
# create model

#model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape=(40, 1200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes)))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Fit the model
model.fit(train_X, train_y, epochs=10, batch_size=10)
# save model
serialize.save_model(model)
# check accracy
loss_and_metrics = model.evaluate(test_X, test_y, batch_size=10)
print(loss_and_metrics)
