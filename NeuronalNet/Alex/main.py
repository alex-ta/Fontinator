from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from dataloader import loader
from dataloader import serialize
from dataloader import plot
from keras.optimizers import RMSprop
import cv2 as cv
import numpy as np

# load dataset
real_x,real_y = loader.read_to_xy_array("C:/Users/Alex/Downloads/images", 0)
train_x, test_x, train_y, test_y, label_encoder, classes = serialize.get_train_testxy_set(real_x,real_y,train_size=0.1)

#img_dimen = 40 * 1200 * 3
#cv.imshow("new",dx[0])
#cv.waitKey(0)
# create model

#model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape=(40, 1200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
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
result = model.fit(train_x, train_y, epochs=2, batch_size=10)
plot.write_csv(result.history)

# save model
serialize.save_model(model)
# check accracy
loss_and_metrics = model.evaluate(test_x, test_y, batch_size=10)



print(loss_and_metrics)
