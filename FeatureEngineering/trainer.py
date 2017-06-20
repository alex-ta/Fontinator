import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import pickle

import extractFeatures
import classifier

# Static Config
FONT_SIZE = 24
PADDING_TOP_BOTTOM = 8
TRAIN_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ÄÖÜäöü!"()[]?ß.,+-'

# Get list of all fonts for training
local_path = os.path.dirname(os.path.realpath(__file__))
fonts_path = os.path.join(local_path, '..', 'DataGenerator', 'fonts')
fonts = os.listdir(fonts_path)
fonts.sort(key=lambda s: s.lower())
cleaned_fonts = []
for font in fonts:
    # Skip everything which is not a Font File (TTF)
    if font.upper().endswith('.TTF'):
        cleaned_fonts.append(font)
fonts = cleaned_fonts

# Save the font, char, label assignment
with open('labels.pickle', 'wb') as file_handle:
    pickle.dump(TRAIN_CHARS, file_handle)
    pickle.dump(fonts, file_handle)

train_classifier_features = []
train_classifier_labels = []

# Iterate for all available fonts
font_label = 0
for font in fonts:
    print(font)

    # Load Fontfile
    font_path = os.path.join(fonts_path, font)
    img_size = FONT_SIZE + 2 * PADDING_TOP_BOTTOM
    font_file = ImageFont.truetype(font_path, FONT_SIZE)

    font_char_features = {}

    # Iterate over all Characters
    for char_label, char in enumerate(TRAIN_CHARS):
        # Draw the Character in the specified Font
        image = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        text = char
        draw.text((10, 3), text, (0, 0, 0), font=font_file)

        # Convert to Grayscale image
        image = np.array(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Extract features
        char_features = extractFeatures.get_feature_vector(image)

        # Append the FeatureVector and the Label to the training data
        train_classifier_features.append( char_features )
        train_classifier_labels.append( char_label + font_label * len(TRAIN_CHARS) )
        font_char_features[char] = char_features

    font_label += 1

# Feature normalization
norm_min = np.min( np.array(train_classifier_features), axis=0 )
norm_range = np.max( np.array(train_classifier_features), axis=0 ) - norm_min

# Train Classifier
cla = classifier.Classifier()
cla.setNormalization(norm_min, norm_range)
cla.trainClassifier(train_classifier_features, train_classifier_labels)
cla.saveTrainedClassifier('./classie.pickle')
