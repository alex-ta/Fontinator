import os.path
import os
from PIL import Image, ImageFont, ImageDraw
import string
import numpy as np
import cv2
import json

import extractFeatures
import classifier

local_path = os.path.dirname(os.path.realpath(__file__))
fonts_path = os.path.join(local_path, '..', 'DataGenerator', 'fonts')
fonts = os.listdir(fonts_path)

FONT_SIZE = 24
PADDING_TOP_BOTTOM = 8
IMG_WIDTH = 100

training_data = []
train_classifier_features = []
train_classifier_labels = []

train_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

# Iterate for all available fonts
i = 0
for font in fonts:
    if not font.upper().endswith('.TTF'):
        continue

    print(font)

    font_path = os.path.join(fonts_path, font)
    img_size = FONT_SIZE + 2 * PADDING_TOP_BOTTOM
    font_file = ImageFont.truetype(font_path, FONT_SIZE)

    font_char_features = {}

    for j, char in enumerate(train_chars):
        image = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        text = char
        draw.text((10, 3), text, (0, 0, 0), font=font_file)

        image = np.array(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Etract features
        char_features = extractFeatures.get_feature_vector(image)
        #char = {text: char_features}

        train_classifier_features.append( char_features )
        train_classifier_labels.append( j + i * len(train_chars) )
        print( j + i * len(train_chars) )
        font_char_features[char] = char_features

    font_features = {'font': font.split('.')[0], 'chars': font_char_features}

    training_data.append(font_features)

    i += 1

# Train Classifier
cla = classifier.Classifier()
cla.trainClassifier(train_classifier_features, train_classifier_labels)
cla.saveTrainedClassifier('./classie.pickle')

with open('result.json', 'w') as fp:
    json.dump(training_data, fp)
