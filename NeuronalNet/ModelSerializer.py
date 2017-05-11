import os
import uuid
from pathlib import Path
import json

from keras.models import model_from_json
from numpy.core.multiarray import ndarray


class ModelSerializer:

    _base_dir: Path = None
    _model_id_dir: Path = None
    _model_structure_file: Path = None
    _model_weights_file: Path = None
    _model_label_mapping_file = None

    # filenames
    MODEL_STRUCTURE_FN = 'model.json'
    MODEL_WEIGHTS_FN = "model.h5"
    MODEL_LABEL_MAP_FN = "label_mapping.json"

    def __init__(self, unique_name = None, base_dir: str = "SavedModels"):
        self._base_dir = Path(base_dir)

        # If unique_name is not set -> Replace with uuid
        if unique_name is None:
            unique_name = str(uuid.uuid4())

        # Create paths to dirs and files
        self._model_id_dir = self._base_dir.joinpath(unique_name)
        self._model_structure_file = self._model_id_dir.joinpath(self.MODEL_STRUCTURE_FN)
        self._model_weights_file = self._model_id_dir.joinpath(self.MODEL_WEIGHTS_FN)
        self._model_label_mapping_file = self._model_id_dir.joinpath(self.MODEL_LABEL_MAP_FN)

    # Save the NN model to disk
    def serialize_to_disk(self, model):

        # Create dirs if they don't exist
        if not self._base_dir.exists():
            self._base_dir.mkdir()
        if not self._model_id_dir.exists():
            self._model_id_dir.mkdir()

        # serialize model structure to JSON file
        model_json = model.to_json()
        with open(self._model_structure_file, "w") as json_file:
            json_file.write(model_json)

        # serialize model weights to HDF5 file
        model.save_weights(str(self._model_weights_file))
        print("Saved model to disk ")

    def load_model_from_files(self):
        # Throws exception if model can't be loaded from files
        if not self.canLoadModel():
            raise Exception("Can't load model from files. No Files found")

        # load json and create model
        json_file = open(self._model_structure_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights(str(self._model_weights_file))
        return model

    # Checks if model can be loaded from saved files
    def canLoadModel(self) -> bool:
        if not self._model_structure_file.is_file() and not self._model_weights_file.is_file():
            return False
        return True

    # Saves the labels and the mapping fo the output layer of the NN
    def save_label_mapping(self, labels: list, indexes: list):
        if type(labels) is ndarray:
            labels = labels.tolist()
        if type(indexes) is ndarray:
            indexes = indexes.tolist()

        data = {'labels': labels, 'indexes': indexes}

        with open(self._model_label_mapping_file, 'w') as outfile:
            json.dump(data, outfile)
