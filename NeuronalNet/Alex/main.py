import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from dataloader import loader
from dataloader import serialize
from dataloader import plot
#from dataloader import TrainingLogger
from keras.optimizers import SGD
import numpy as np

# load dataset
real_x,real_y = loader.read_to_xy_array("E:\images\images", flatten=0, print_out=0)
train_x, test_x, train_y, test_y, label_encoder, classes = serialize.get_train_testxy_set(real_x,real_y,train_size=0.6)

#print(len(train_x[0]))
#print(len(train_x[0][0]))
#print(len(train_x[0][0][0]))

#img_dimen = 40 * 1200 * 3
#cv.imshow("new",dx[0])
#cv.waitKey(0)
# create model

#model
model = Sequential()

model.add(Conv2D(8, kernel_size=(7, 7),
				 activation='relu',
				 input_shape=(40,1200,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3),
				 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, kernel_size=(3, 3),
				 activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(classes), activation='softmax'))

sgd = SGD(lr=0.01, clipvalue=0.5)
model.compile(loss=keras.losses.mean_squared_error,
			  optimizer=sgd,
			  metrics=['accuracy'])


# Fit the model
#logger = TrainingLogger("train_logger", frequent_write=False)
result = model.fit(train_x, train_y, epochs=5000, batch_size=100)
plot.write_csv(result.history)

# save model
serialize.save_model(model)
# check accracy
#loss_and_metrics = model.evaluate(test_x, test_y, batch_size=10)



print(loss_and_metrics)
