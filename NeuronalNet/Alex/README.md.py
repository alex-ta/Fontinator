# Convolutional Neuronal Network font-style recongnition

### Basics

####Pipeline
This Code is designed for easy use. All scripts are tightened togehter in a pipeline. The following Code shows any option provided by this pipe:

```python
# include data pipe
from modelpipe import pipe
import numpy as np

#create pipe
#data_path => path to the training and testdata 
#	data must be provided in a folderstructure
#   data
# 	  - label 1
#		- image1
#		- image2
#     - label 2
#train_size => the size of test (1-x) and trainings data (x)
#
#this call creates an pipe object and sets up variables
datapipe = pipe.Pipe(data_path = "E:\images\pirates", train_size=0.6)



#load data
#flatten => provides a flattend datastructure, where all images are provided in a single array instead of a [h][w][c] #array (width, height, channel)
#print => enabels or disables logging
#test_x and test_y contain the testdata (x) and the testlabels (y)
#all other data gets saved in the pipe for later use
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)

#train model
#model_name => path to a keras model descriped in [] without .py (models/model_01.py gets loaded)
#the created model gets saved in a folder with the model_name path. (models/model_01 folder contains the created data)
#epochsize => amount of epochs run by the pipe
#batchsize => amount of images in one batch with same weigth
#result contains the loss and metricies from the model run with test data.
result = datapipe.run(model_name = "models/model_01", epochsize = 3000, batchsize = 100)
print(result)


#evaluate model (to use with existing models)
#takes optional parameters 
#model_name => model_name defaults to the data in the run methode
#test_x => test_x data defaults to the data loaded in the load_data methode 
#test_y => test_y data defaults to the data loaded in the load_data methode
#batch_size => defaults to batchsize defined in the run methode or undefined
result = datapipe.eval()
print(result)

#predict
#model_name => model_name defaults to the data in the run methode 
#imgs => images that should get predicted as np.array
#one_hot => set labels to one_hot encode, default is percentages
result = datapipe.predict(np.array([test_x[0]]), one_hot=1)
print(result)

```

#### Visualization

```python
# include plot options
from dataloader import plot

#read data from csv and plot it. The csv gets created in the model_name folder and contains the process per epoch data.

#plots loss, percentage and time per epoch charts
plot.plot_csv_multipath("models/model_01/plot.csv")
#plot all data in one image
plot.plot_csv("models/model_01/plot.csv")
```

### Structure




