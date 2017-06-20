# include data pipe
from modelpipe import pipe
import numpy as np

#create pipe
datapipe = pipe.Pipe(data_path = "E:/Model/Fontinator/images/untrained", train_size=0.6)

#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)

#create model
#result = datapipe.run(model_name = "models/model_01", epochsize = 300, batch_size = 100)
#print(result)
#result = datapipe.run(model_name = "models/model_02", epochsize = 300, batch_size = 100)
#print(result)
#result = datapipe.run(model_name = "models/model_03", epochsize = 300, batch_size = 100)
#print(result)
#result = datapipe.run(model_name = "models/model_04", epochsize = 300, batch_size = 100)
#print(result)
result = datapipe.run(model_name = "models/model_05", epochsize = 300, batch_size = 100)
print(result)
result = datapipe.run(model_name = "models/model_06", epochsize = 300, batch_size = 100)
print(result)
result = datapipe.run(model_name = "models/model_07", epochsize = 300, batch_size = 100)
print(result)
result = datapipe.run(model_name = "models/model_08", epochsize = 300, batch_size = 100)
print(result)
result = datapipe.run(model_name = "models/model_09", epochsize = 300, batch_size = 100)
print(result)
result = datapipe.run(model_name = "models/model_10", epochsize = 300, batch_size = 100)
print(result)


#evaluate model
#result = datapipe.eval()
#print(result)

#predict
#result = datapipe.predict(np.array([test_x[0]]), one_hot=1)
#print(result)