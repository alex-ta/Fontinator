# the model gets created
# in this file specific layers can be defined and changed
# the default data contains 40 x 1200 x 3 data as defined by the input dataformat
# if the data for test and validation is change the first layer format can change
# model contains a sequential keras model that can be applied with different layers

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
