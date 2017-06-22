# include data pipe
from modelpipe import pipe
import numpy as np

#create pipe
datapipe = pipe.Pipe(data_path = "../../images/Dataset_1", train_size=0.6)
#load data
datapipe.load_data(flatten=0, print_out=1)

#train model
for i in range(10):
	np.random.seed(10)
	result = datapipe.run(model_name = "models/model_0"+str(i), epochsize = 300, batch_size = 100)
	print(result)	
