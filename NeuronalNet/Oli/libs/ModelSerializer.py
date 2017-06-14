import json
import uuid
from pathlib import Path

from keras.models import model_from_json
from numpy.core.multiarray import ndarray


class ModelSerializer:
    """
    Manages the saving and loading of a keras NN model from and to disk.
    Also allows saving of additional metadata about a trained NN model 
    """

    _base_dir: Path = None
    _model_id_dir: Path = None
    _model_structure_file: Path = None
    _model_weights_file: Path = None
    _model_label_mapping_file = None

    # filenames
    MODEL_STRUCTURE_FN = 'model.json'
    MODEL_WEIGHTS_FN = "model.h5"
    MODEL_LABEL_MAP_FN = "label_mapping.json"

    def __init__(self, base_dir: str = "SavedModels"):
        self._base_dir = Path(base_dir)

        # Create paths to dirs and files
        self._model_structure_file = self._base_dir.joinpath(self.MODEL_STRUCTURE_FN)
        self._model_weights_file = self._base_dir.joinpath(self.MODEL_WEIGHTS_FN)
        self._model_label_mapping_file = self._base_dir.joinpath(self.MODEL_LABEL_MAP_FN)

    def serialize_to_disk(self, model):
        '''
        Save the NN model to disk
        :param model: The keras NN model
        :return: None
        '''
        # Create dirs if they don't exist
        self.__create_dirs_if_necessary()

        # serialize model structure to JSON file
        model_json = model.to_json()
        with open(self._model_structure_file, "w") as json_file:
            json_file.write(model_json)

        # serialize model weights to HDF5 file
        model.save_weights(str(self._model_weights_file))
        print("Saved model to disk ")

    def load_from_path(self):
        '''
        Loads the keras NN model from disk 
        :return: None
        '''
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

    def canLoadModel(self) -> bool:
        '''
        Checks if model exists and can be loaded from saved files
        :return: bool indication if model can be loaded
        '''
        if not self._model_structure_file.is_file() and not self._model_weights_file.is_file():
            return False
        return True

    def save_label_mapping(self, labels: list, indexes: list):
        '''
        Saves the labels and the mapping fo the output layer of the NN
        :param labels: list or ndarray with the labels
        :param indexes: list or ndarray with index
        :return: None
        '''
        # Create dirs if they don't exist
        self.__create_dirs_if_necessary()

        if type(labels) is ndarray:
            labels = labels.tolist()
        if type(indexes) is ndarray:
            indexes = indexes.tolist()

        data = {'labels': labels, 'indexes': indexes}
        with open(self._model_label_mapping_file, 'w') as outfile:
            json.dump(data, outfile)

    def __create_dirs_if_necessary(self):
        '''
        Create the directorie if they don't exist
        :return: None
        '''
        # Create dirs if they don't exist
        if not self._base_dir.exists():
            self._base_dir.mkdir(parents=True)
