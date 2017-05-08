# load dataset from file
from os import listdir
from os.path import isfile, join, basename
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
def create_dataset(dir):
    dirs = listdir(dir)
    data = []
    for file in dirs:
        file_data = Data()
        file_data.set_label(basename(dir))
        file_data.set_name(file)
        fPath = dir+"/"+file
        if isfile(fPath):
            file_data.set_file(1)
            file_data.set_data(cv2.imread(fPath))
        else:
            file_data.set_file(0)
            file_data.set_data(create_map(fPath))
        data.append(file_data)
    return data

# reades folder into a map
def create_map(dir):
    data = []
    dirs = listdir(dir)
    for file in dirs:
        file_data = {}
        file_data['label'] = basename(dir)
        file_data['name'] = file
        fPath = dir+"/"+file
        if isfile(fPath):
            file_data['isFile'] = 1
            file_data['data'] = cv2.imread(fPath)
        else:
            file_data['isFile'] = 0
            file_data['data'] = create_map(fPath)
        data.append(file_data)
    return data


x = create_dataset("C:/Users/Alex/Downloads/images")
print(x[0].get_name())
y = create_map("C:/Users/Alex/Downloads/images")
print(y[0]["name"])
