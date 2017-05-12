# load dataset from file
from os import listdir
from os.path import isfile, join, basename
import numpy as np
import cv2


# data class
class Data:
    def __init__(self):
        self.data = []
    def set_name(self, val):
        self.name = val
    def set_label(self, val):
        self.label = val
    def set_data(self, val):
        self.data.append(val)
    def set_file(self, val):
        self.file = val
    def get_name(self):
        return self.name
    def get_label(self):
        return self.label
    def get_data(self):
        return self.data
    def get_file(self):
        return self.file

# read data into a class
def read_to_dataset(dir, print_out = 0):
    dirs = listdir(dir)
    data = []
    for file in dirs:
        if print_out:
            print("read file " + file)
        file_data = Data()
        file_data.set_label(basename(dir))
        file_data.set_name(file)
        fPath = dir+"/"+file
        if isfile(fPath):
            file_data.set_file(1)
            file_data.set_data(cv2.imread(fPath))
        else:
            file_data.set_file(0)
            file_data.set_data(read_to_dataset(fPath, print_out))
        data.append(file_data)
    return data

# reads file to NN data
def read_to_NN_dataset(file, print_out=1):
    return parse_to_NN_dataset(read_to_dataset(file, print_out), print_out)

# reads files to arrays x and y
def read_to_xy_array(file, print_out=1):
    return parse_to_xy_array(read_to_dataset(file, print_out), print_out)


# parses normal data to NN data
def parse_to_NN_dataset(dataset, print_out=1):
    data_array = []
    for data in dataset:
        if data.get_file():
            data_array.append(NNData(data))
        else:
            for d_array in data.get_data():
                data_array.extend(parse_to_NN_dataset(d_array,0))
    if print_out:
        print(str(len(data_array))+" datasets read")
    return data_array

#parse to x and y dataset
def parse_to_xy_array(dataset, print_out=1):
    data_x = []
    data_y = []
    for data in dataset:
        if data.get_file():
            x,y = NNData(data).to_xy_data()
            data_x.append(x)
            data_y.append(y)
        else:
            for d_array in data.get_data():
                x,y = parse_to_xy_array(d_array,0)
                data_x.extend(x)
                data_y.extend(y)
    if print_out:
        print("x: "+str(len(data_x))+" y:"+str(len(data_y))+" datasets read")
    return np.array(data_x, dtype='int32'),np.array(data_y)



# MM Data class
class NNData:
    def __init__(self, data):
        if data.get_file():
            self.data = data.get_data()
            self.label = data.get_label()
            self.name = data.get_name()
    def get_label(self):
        return self.label
    def get_data(self):
        return self.data
    def get_name(self):
        return self.name
    def to_xy_data(self):
        # data is x (get the img array), label is y
        return self.data[0],self.label
