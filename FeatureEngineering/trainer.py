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

train_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ÄÖÜäöü!"()[]?ß.,+-'

typo_lines = [[29,24,11,7],[24,19,8,4],[23,18,7,3],[35,28,16,10],[27,22,11,6],[27,22,11,7],[37,31,19,9],[30,25,13,9],[28,23,12,7],[29,24,13,4],[31,25,9,5],[26,26,4,3]]
spacing = [13, 11, 10, 13, 14, 12, 14, 12, 12, 14, 16, 16]
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
        #cv2.imshow('a',image)
        #cv2.waitKey(0)
        #print(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        for r in range(0, 5):
            train_image = image

            if r != 0:
                noise = np.zeros(image.shape, np.uint8)
                cv2.randu(noise, 0, 25)
                noise2 = np.zeros(image.shape, np.uint8)
                cv2.randu(noise2, 0, 25)
                train_image = train_image - noise #- noise2

                #print(noise)
                #print(noise2)
            #cv2.imshow('A', train_image)
            #cv2.waitKey(0)
            # Etract features
            char_features = extractFeatures.get_feature_vector(image)

            #char_features.append( (typo_lines[i][1]-typo_lines[i][2]) / (typo_lines[i][0]-typo_lines[i][3]) )
            #char_features.append( (typo_lines[i][0]-typo_lines[i][1]) / (typo_lines[i][0]-typo_lines[i][3]) )
            #char_features.append( (typo_lines[i][2]-typo_lines[i][3]) / (typo_lines[i][0]-typo_lines[i][3]) )

            #char_features.append( spacing[i] )
            #char = {text: char_features}

            train_classifier_features.append( char_features )
            train_classifier_labels.append( j + i * len(train_chars) )
            print( j + i * len(train_chars) )
            font_char_features[char] = char_features

    font_features = {'font': font.split('.')[0], 'chars': font_char_features}

    training_data.append(font_features)

    i += 1

# Normalization
norm_min = np.min( np.array(train_classifier_features), axis=0 )
norm_range = np.max( np.array(train_classifier_features), axis=0 ) - norm_min

#norm_data = (np.arry(train_classifier_features) - norm_min) / norm_range


# Train Classifier
cla = classifier.Classifier()
cla.setNormalization(norm_min, norm_range)
cla.trainClassifier(train_classifier_features, train_classifier_labels)
#cla.trainClassifier(norm_data, train_classifier_labels)
cla.saveTrainedClassifier('./classie.pickle')

with open('result.json', 'w') as fp:
    json.dump(training_data, fp)
