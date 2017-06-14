#testclass for dataloader
import loader
import serialize
import cv2 as cv
import numpy as np

#data
data = "C:/Users/Alex/Documents/Fontinator/NeuronalNet/Alex/dataloader/test_img"

# load dataset
dataset = loader.read_to_dataset(data, 1);
print(dataset[0].get_name())
print(dataset[0].get_data()[0][0].get_name())
print(dataset[0].get_data()[0][0].get_data())
cv.imshow("dataset",dataset[0].get_data()[0][0].get_data()[0]);
cv.waitKey();

nndataset = loader.parse_to_NN_dataset(dataset,1);
print(nndataset[0].get_name())
print(nndataset[0].get_label())
#print(nndataset[0].get_data())
cv.imshow("nndataset",nndataset[0].get_data())
cv.waitKey()
x,y = nndataset[0].to_xy_data();
#print(x)
print(y)
cv.imshow("nndatasetxy",x)
cv.waitKey()

x,y = loader.parse_to_xy_array(dataset, 1)
#print(x[0])
print(y[0])
cv.imshow("xydataset",x[0]);
cv.waitKey()

real_x,real_y = loader.read_to_xy_array(data, 0)
train_x, test_x, train_y, test_y, label_encoder, classes = serialize.get_train_testxy_set(real_x,real_y,train_size=0.1)
print(real_y[0])
cv.imshow("real_x",real_x[0]);
cv.waitKey()

train_x, test_x, train_y, test_y, label_encoder, classes = serialize.get_train_testxy_set(real_x,real_y,train_size=0.5)
print(serialize.get_label_from_one_hot(train_y[0],label_encoder))
cv.imshow("train_x",train_x[0]);
cv.waitKey()
print(serialize.get_label_from_one_hot(train_y[1],label_encoder))
cv.imshow("train_x",train_x[1]);
cv.waitKey()
