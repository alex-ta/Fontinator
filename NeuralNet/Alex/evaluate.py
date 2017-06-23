# include data pipe
from modelpipe import pipe

#create pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate models on dataset 1

buffer = "";
buffer += "Dataset_1\n"
buffer += "Model_Name;err;acc"

for i in range(10):
	result = datapipe.eval(model_name = "models/model_0"+str(i), batch_size = 100)
	buffer += "\nmodel_"+str(i)
	for r in result:
		buffer += ";" + str(r)

file = open("evaluate.csv","w")
file.write(buffer)
file.close()