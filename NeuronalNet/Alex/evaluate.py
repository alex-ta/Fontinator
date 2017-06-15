# include data pipe
from modelpipe import pipe
import numpy as np
#create pipe
datapipe = pipe.Pipe(data_path = "C:/Users/Alex/Documents/Fontinator/images/Dataset_1/images", train_size=0.6)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate model
result = datapipe.eval(model_name = "models/model_01", batch_size = 100)
print("model_1")
print(result)
result = datapipe.eval(model_name = "models/model_02", batch_size = 100)
print("model_2")
print(result)
result = datapipe.eval(model_name = "models/model_03", batch_size = 100)
print("model_3")
print(result)
result = datapipe.eval(model_name = "models/model_04", batch_size = 100)
print("model_4")
print(result)
result = datapipe.eval(model_name = "models/model_05", batch_size = 100)
print("model_5")
print(result)
result = datapipe.eval(model_name = "models/model_06", batch_size = 100)
print("model_6")
print(result)


#create pipe
datapipe = pipe.Pipe(data_path = "C:/Users/Alex/Documents/Fontinator/images/Dataset_2/untrained", train_size=0.6)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate model
result = datapipe.eval(model_name = "models/model_01", batch_size = 100)
print("model_1")
print(result)
result = datapipe.eval(model_name = "models/model_02", batch_size = 100)
print("model_2")
print(result)
result = datapipe.eval(model_name = "models/model_03", batch_size = 100)
print("model_3")
print(result)
result = datapipe.eval(model_name = "models/model_04", batch_size = 100)
print("model_4")
print(result)
result = datapipe.eval(model_name = "models/model_05", batch_size = 100)
print("model_5")
print(result)
result = datapipe.eval(model_name = "models/model_06", batch_size = 100)
print("model_6")
print(result)

#create pipe
datapipe = pipe.Pipe(data_path = "C:/Users/Alex/Documents/Fontinator/images/Dataset_3/text_zfs", train_size=0.6)
#load data
test_x, test_y = datapipe.load_data(flatten=0, print_out=1)
#evaluate model
result = datapipe.eval(model_name = "models/model_01", batch_size = 100)
print("model_1")
print(result)
result = datapipe.eval(model_name = "models/model_02", batch_size = 100)
print("model_2")
print(result)
result = datapipe.eval(model_name = "models/model_03", batch_size = 100)
print("model_3")
print(result)
result = datapipe.eval(model_name = "models/model_04", batch_size = 100)
print("model_4")
print(result)
result = datapipe.eval(model_name = "models/model_05", batch_size = 100)
print("model_5")
print(result)
result = datapipe.eval(model_name = "models/model_06", batch_size = 100)
print("model_6")
print(result)