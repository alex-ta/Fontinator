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

def replace_groups(data):
    a,b,c, = np.unique(data, True, True)
    _, ret = np.unique(b[c], False, True)
    return ret

# load dataset
real_x,real_y = loader.read_to_xy_array("E:\images\pirates", flatten=0, print_out=0)
train_x, test_x, train_y, test_y, label_encoder, classes = serialize.get_train_testxy_set(real_x,real_y,train_size=0.1)

model = serialize.load_model()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

loss_and_metrics = model.evaluate(test_x, test_y, batch_size=10)
# calculate predictions
#predictions = model.predict(dx[0])
# round predictions
print(loss_and_metrics)
#print(model.history)
#print (model.predict(np.array([test_X[0],test_X[1]])))
#print (test_y[0],test_y[1])
