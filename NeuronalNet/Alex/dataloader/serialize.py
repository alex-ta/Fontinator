from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

    # returns train/test data (x = data, y = label) and encoder and the classes
def get_train_testxy_set(x,y,train_size=0.75, one_hot = 1, random_state = 0):
    label_encoder, classes, one_hot_y = get_encoder(y)
    # transform to one hot
    if one_hot:
        one_hot_y = np_utils.to_categorical(one_hot_y,len(classes))
    # split in test and train
    train_x, test_x, train_y, test_y = train_test_split(x, one_hot_y, train_size = train_size, random_state = random_state)
    return train_x, test_x, train_y, test_y, label_encoder, classes

    # returns the label from one_hot array
def get_label_from_one_hot(one_hot, label_encoder):
    y_index = one_hot.argmax()#axis = 1
    return label_encoder.inverse_transform(y_index)

    # returns the label encoder for labels, the unique classes and the transformed labels
def get_encoder(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder, label_encoder.classes_, label_encoder.transform(labels)

def save_model(model, model_name = "model.json", weights_name = "model.h5"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_name)
    print("Saved model as " + model_name +" & "+ weights_name)

def load_model(model_name = "model.json", weights_name = "model.h5"):
    # load json and create model
    json_file = open(model_name, 'r')
    loaded_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json)
    # load weights into new model
    model.load_weights(weights_name)
    print("Loaded model as " + model_name +" & "+ weights_name)
    return model
