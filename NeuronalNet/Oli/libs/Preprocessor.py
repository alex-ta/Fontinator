import numpy as np
from PIL import Image
from numpy.core.multiarray import ndarray

"""
Manages all necessary transformations go prepare the image data for the neuronal network
"""
class Preprocessor:

    def __init__(self):
        pass

    def prepare_image(self, img_name: str) -> ndarray:
        """
        Runs the preprocessing chain for the image
        :param img_name: The path to the image
        :return: A numpy ndarray with dtype=np.ubyte
        """
        img = Image.open(img_name)

        # Convert image to black and white
        img = img.convert('1')

        # Convert Pillow image to numpy ndarray
        np_array: ndarray = np.asarray(img, dtype=np.ubyte)

        # Flattens the 2Dim ndarray (e.g. shape (1200,40) -> (48000)
        flat_img: ndarray = np_array.flatten()

        return flat_img
