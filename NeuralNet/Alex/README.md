# Convolutional Neural Network font-style recongnition

### CNN

The structure of a CNN differs by the possibility of processing multi-dimensional data. For this purpose, filters are used which show excerpts of the data. These sections are again compared and used as feature recognition features. The following figure shows an overview of a CNN.

![image](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-4-59-29-pm.png?w=748?raw=true "Cnn")

As input, the network expects an image, which runs through three filters in the first step. Each filter produces an equal-sized output image in which each pixel is replaced by the sum of the filter. With a poolinglayer the individual pixel features are then reduced to image excerpts, which reduces the feature matrices (visible in the picture by a reduction in width and height). Subsequently, the steps of the network are repeated twice until the data is reduced to a one-dimensional vector. The network shown is supplemented by two Fully Connected Layers, which link each compressed feature to an output euron. By linking all input and output eurons, probabilities are created for different classifiers, which are assigned to the four classifiers dog, cat, boat and bird.

### Basics

The project NN for fonts recognition is based on CNN. It uses image data, which is read in the same way as for other NNs. A folder structure maps different images with fonts to a label. An example structure is as follows:

![image](Images.png?raw=true "Folder Structure")

The individual images consist of texts in the respective font. The text is generated variably via a script. The exact structure is explained in chapter [Data Generator].

Python was selected with Keras and Tensorflow for the font recognition application. All necessary frameworks as well as the installation instructions can be read in [How to Setup].
The structure of the software consists of several modules. All essential functionalities are to be explained using the folder structure.

#### Folders
- dataloader: contains all important scripts for data from folder structures for loading and dividing in test and training data. It also offers possibilities for one-hot encoding and import as a flat vector.
- modelpipe: this folder contains the simplification of the background processes. Thus, the important functionalities: load data, train network, check network, assign data as well as simplify loading the model. Thus, several networks can be loaded and trained in a batch process.
- model: in this folder are the defined models and their results.

#### Scripts:
- main.py: this script starts the training process by loading the training data and training several models.
- state.py: this script generates a statistics from existing CSV data of a model. In addition, a suitable image is created for each model that shows the input and output layers.
- evaluate.py: this script performs an evaluation of the models against the given data sets.

#### Pipeline
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

#### Minimal Code Samples

1) Training
```
from modelpipe import pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0.6)
datapipe.load_data(flatten=0, print_out=1)
result = datapipe.run(model_name = "models/model_01", epochsize = 300, batch_size = 100)
```

The example creates a pipe object with the data path and the percentage of training data. The data is then loaded as a multi-dimensional array using the load_data command. Then the model "model / model_01.py" is trained with 300 epochs and a batch size of 300. The trained model is stored in the "model / model_01 /" folder with meta information.

2) Evaluation
```
from modelpipe import pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0.6)
datapipe.load_data(flatten=0, print_out=1)
result = datapipe.eval(model_name = "models/model_0"+str(i), batch_size = 100)
```

Similarly, in the first example, a pipe object is created and data is loaded. The data are then evaluated via the eval function. This requires the path to the model created in the first example. The model.h5 and model.json are loaded automatically in the respective folder. The resulting object is an array containing the error value in the first place and the accuracy in percent in the second place.

3) Creating images to analyze the model
```
from dataloader import plot
plot.plot_csv_multipath("models/model_01/plot.csv", figure="Model_1").savefig("model1.png")
```
With this snippet code, the plot.csv is loaded from the model folder and plotted. The plot is then saved as model1.png.

```
from modelpipe import pipe
from keras.utils import plot_model
model = pipe.Pipe().get_model(model_name="models/model_01")
plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
```
The second code snippet shows the creation of a model overview, the layers of the model are plotted. Additional software is required.

4) Predictions
```
from modelpipe import pipe
imgArray = "Your Images go here"
predictions = pipe.Pipe().predict(model_name="models/model_01", imgs=imgArray)
```
This example shows how multiple images are assigned using the predict method. To do this, the path to the model created in the first example must be specified. The images are given as imgArray. In the example, no images were loaded.

5) Creating a model
```
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(40,1200,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(24, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss=keras.losses.mean_squared_error, optimizer="rmsprop", metrics=['accuracy'])
```
In this last example, a model is displayed in "models / model_01.py". The model automatically has a model object, a class object, as well as various keras importe. The script is loaded at runtime by the run method and can use the objects and references instanced before. Thus, an optimal separation of dependencies is possible which makes it easy to exchange models.

### Structure

![image](Structure.png?raw=true "Project Structure")
