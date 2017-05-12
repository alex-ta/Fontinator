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
dx,dyd = loader.read_to_xy_array("C:/Users/Alex/Downloads/images", 0)
dx = np.array(dx, dtype='int32')
dyd = np.array(dyd)
#dy = np.array(dy, dtype='a16')
classes = np.unique(dyd)
dy = np_utils.to_categorical(replace_groups(dyd),len(classes))

train_X, test_X, train_y, test_y = train_test_split(dx, dy, train_size=0.75, random_state=0)

model = serialize.load_model()

loss_and_metrics = model.evaluate(test_X, test_y, batch_size=10)
# calculate predictions
#predictions = model.predict(dx[0])
# round predictions
print(loss_and_metrics)

#print (model.predict(np.array([test_X[0],test_X[1]])))
#print (test_y[0],test_y[1])
