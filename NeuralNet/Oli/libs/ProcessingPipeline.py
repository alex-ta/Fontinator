from keras.optimizers import RMSprop
from keras.utils import np_utils
from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

from NeuralNet.Oli.libs.TrainingLogger import TrainingLogger
from NeuralNet.Oli.libs.ImageLoader import ImageLoader
from NeuralNet.Oli.libs.ModelSerializer import ModelSerializer
from NeuralNet.Oli.libs.Preprocessor import IPreprocessor


class ProcessingPipeline:
    '''
    Manages the whole pipeline from data loading, preprocessing to model training and evaluation
    '''
    def __init__(self):
        self.font_names: list = None
        self.model_path: str = None
        self.model = None
        self.preprocessor: IPreprocessor = None
        self.label_encoder = None
        pass

    def load_features_and_preprocess(self, img_path: str, img_preprocessor: IPreprocessor) -> (ndarray, ndarray):
        self.preprocessor = img_preprocessor

        # Loads the images from the defined path
        data_loader: ImageLoader = ImageLoader(img_path)
        self.font_names = data_loader.get_font_names()
        image_count = data_loader.get_image_count()
        font_count = data_loader.get_font_count()
        print("Found {0} images with {1} different fonts".format(image_count, font_count))

        # Map labels(str) to class_ids(int)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.font_names)
        label_ids = self.label_encoder.transform(self.label_encoder.classes_)
        print("Mapping labels:\n{0} \n -> {1}".format(self.label_encoder.classes_, label_ids))

        print("Start preprocessing images ...")
        features = []
        labels = []
        # Iterate over all fonts
        for f_name in self.font_names:
            print(" -> {0}".format(f_name))
            label_id = self.label_encoder.transform([f_name])
            font_labels = np.full(data_loader.get_img_count_for_font(f_name), label_id, dtype=np.float32)
            labels.extend(font_labels)

            # Iterate over all images for one font
            for img_path in data_loader.iterate_images_for_fontname(f_name):
                nd_img: ndarray = self.preprocessor.prepare_image(img_path)
                features.append(nd_img)

        x: ndarray = np.array(features)
        y: ndarray = np.array(labels)
        return x, y

    def __compile_model(self):
        print("Compiling NN model ...")
        nn_optimizer = RMSprop(lr=0.0001)
        self.model.compile(optimizer=nn_optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load_model(self, model_path):
        # Load the NN model from disk
        print("Loading model from disk")
        model_serializer = ModelSerializer(model_path)
        self.model = model_serializer.load_from_path()
        self.__compile_model()

    def train_model(self, keras_model, x: ndarray, y: ndarray, epos=1000, train_ratio=0.8, batch_size=100):

        self.model = keras_model

        # Convert labels to categorical one-hot encoding; e.g. [1, 2, 3] -> [[1,0,0], [0,1,0], [0,0,1]]
        y_onehotenc = np_utils.to_categorical(y)

        # Splitting to train- /test data
        train_X, test_X, train_y, test_y = train_test_split(x, y_onehotenc, train_size=train_ratio)

        # Saves stats while training the NN
        self.train_logger: TrainingLogger = TrainingLogger(self.model_path, frequent_write=False)

        self.__compile_model()

        print("Training the NN model")
        if type(batch_size) == float and 0.0 < batch_size <= 1.0:
            batch_size = int(batch_size * train_X.shape[0])
        self.model.fit(train_X, train_y, epochs=epos, batch_size=batch_size,
                       validation_data=(test_X, test_y), callbacks=[self.train_logger])

        # Calculate the metrics for the trained model
        loss_and_metrics = keras_model.evaluate(test_X, test_y, batch_size=batch_size)
        print(loss_and_metrics)

    def save_model(self, model_save_path: str, include_stats=True):

        # save the mapping of the features to disk
        model_serializer = ModelSerializer(model_save_path)
        label_ids = self.label_encoder.transform(self.label_encoder.classes_)
        model_serializer.save_label_mapping(self.label_encoder.classes_, label_ids)

        # Write csv file and plot image for training stats
        if include_stats is True:
            self.train_logger.set_basepath(model_save_path)
            self.train_logger.write_csv()
            self.train_logger.make_plots()

        # Save the NN model to disk
        print("Saving NN model and the label index mapping")
        model_serializer.serialize_to_disk(self.model)

    def predict(self, x):
        # Make predictions
        y_pred_onehotenc = self.model.predict(x)

        # Retransform one hot encoding to indexes
        y_pred = y_pred_onehotenc.argmax(axis=1)
        return y_pred

    def evaluate(self, y, y_pred):

        # Calculate correct and wrong prediction count
        correct_pred_items = np.equal(y, y_pred)

        cor_pred_count = np.sum(correct_pred_items)
        wrong_pred_count = y_pred.size - cor_pred_count
        cor_pred_ratio = cor_pred_count / y_pred.size

        print("Summary:")
        print("Correct predictions: {0} | Wrong predictions: {1}"
              .format(cor_pred_count, wrong_pred_count))
        print("{0}".format(cor_pred_ratio))
