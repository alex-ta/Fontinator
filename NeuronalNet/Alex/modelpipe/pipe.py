# installed modules 
import keras
import os
import shutil
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import numpy as np
# local modules
from dataloader import loader
from dataloader import serialize
from dataloader import plot
from .Callback import Logger

class Pipe:
	def __init__(self, data_path = "C:/Users/Alex/Downloads/images", train_size=0.6):
		self.data_path = data_path
		self.train_size = train_size
		
	def load_data(self, flatten=0, print_out=1):
		real_x,real_y = loader.read_to_xy_array(self.data_path, flatten, print_out)
		train_x, test_x, train_y, test_y, label_encoder, classes = serialize.get_train_testxy_set(real_x,real_y,train_size=self.train_size)
		self.train_x = train_x
		self.test_x = test_x
		self.train_y = train_y
		self.test_y = test_y
		self.label_encoder = label_encoder
		self.classes = classes
		return test_x, test_y

	def run(self, model_name = "models/model_01", epochsize = 1, batchsize = 100):
		self.model_name = model_name
		self.epochsize = epochsize
		self.batchsize = batchsize
		logger = Logger();
		
		#img_dimen = 40 * 1200 * 3
		#prepare model and global data
		model = Sequential()
		classes = self.classes
		#define the modelfile to load the model
		command_file_name = self.model_name+".py"
		#excute and load model as pythoncommands
		exec(compile(open(command_file_name, "rb").read(), command_file_name, 'exec'))
		# fit the rasult
		result = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y), epochs=self.epochsize, batch_size=self.batchsize, callbacks=[logger])
		#create model folder and delete existing before
		if os.path.exists(self.model_name):
			shutil.rmtree(self.model_name)
		os.mkdir(self.model_name)
		# save plot
		plot.write_csv(result.history, path=self.model_name+"/plot.csv")
		# save model
		serialize.save_model(model,path=self.model_name)

		loss_and_metrics = model.evaluate(self.test_x, self.test_y, batch_size=self.batchsize)
		return loss_and_metrics
	
	def eval(self):
		#load the model
		model = serialize.load_model(path=self.model_name)
		model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
		#create the metrics
		loss_and_metrics = model.evaluate(self.test_x, self.test_y, batch_size=self.batchsize)
		return loss_and_metrics;
		
	def predict(self, imgs, one_hot = 1):
		#load model
		model = serialize.load_model(path=self.model_name)
		model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
		# calculate predictions
		prediction = model.predict(imgs)
		if one_hot:
			prediction = serialize.get_label_from_one_hot(one_hot=prediction, label_encoder=self.label_encoder)
		return prediction