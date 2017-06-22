# include data pipe
from modelpipe import pipe
import numpy as np

#create pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0.6)
#load data
datapipe.load_data(flatten=0, print_out=1)
np.random.seed(10)
result = datapipe.run(model_name = "models/model_08", epochsize = 300, batch_size = 100)

#create model
#for i in range(9,10):
#	np.random.seed(10)
#	result = datapipe.run(model_name = "models/model_0"+str(i), epochsize = 300, batch_size = 100)
#	print(result)


	
#create pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate model

buffer = "";
buffer += "Dataset_1\n"
buffer += "Model_Name;err;acc"

for i in range(10):
	result = datapipe.eval(model_name = "models/model_0"+str(i), batch_size = 100)
	buffer += "\nmodel_"+str(i)
	for r in result:
		buffer += ";" + str(r)

#create pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_2", train_size=0)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate model

buffer += "Dataset_2\n"
buffer += "Model_Name;err;acc"

for i in range(10):
	result = datapipe.eval(model_name = "models/model_0"+str(i), batch_size = 100)
	buffer += "\nmodel_"+str(i)
	for r in result:
		buffer += ";" + str(r)

#create pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_3", train_size=0)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate model

buffer += "Dataset_3\n"
buffer += "Model_Name;err;acc"

for i in range(10):
	result = datapipe.eval(model_name = "models/model_0"+str(i), batch_size = 100)
	buffer += "\nmodel_"+str(i)
	for r in result:
		buffer += ";" + str(r)


file = open("evaluate.csv","w")
file.write(buffer)
file.close()