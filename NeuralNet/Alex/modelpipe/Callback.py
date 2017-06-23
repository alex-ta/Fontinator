import keras
import datetime
from dataloader import plot
from sklearn.metrics import roc_auc_score
 
class Logger(keras.callbacks.Callback):
	def __init__(self):
		super().__init__()
	
	def on_train_begin(self, logs={}):
		self.begin = 1
		self.aucs = []
		self.losses = []

	def on_epoch_begin(self, epoch, logs={}):
		self.begin_epoch = datetime.datetime.now()
		return

	def on_epoch_end(self, epoch, logs={}):
		self.end_epoch = datetime.datetime.now()
		logs["time"] = (self.end_epoch - self.begin_epoch).total_seconds()
		if self.begin:
			plot.write_line_csv(list(logs.keys()))
			self.begin = 0
		plot.write_line_csv(list(logs.values()))
		print(logs)
		return