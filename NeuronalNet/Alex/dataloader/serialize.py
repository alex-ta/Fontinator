from keras.models import model_from_json

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
