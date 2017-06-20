import cv2
import os
import numpy as np
import math
import pickle

import extractGlyphes
import extractFeatures
import classifier

# Load saved clasifier and label data
cla = classifier.Classifier()
cla.loadTrainedClassifier('./classie.pickle')
with open('labels.pickle', 'rb') as file_handle:
    TRAIN_CHARS = pickle.load(file_handle)
    FONT_LIST = pickle.load(file_handle)

local_path = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(local_path, '..', 'TestSets','images', 'Dataset_1')
#images_path = os.path.join(local_path, '..', 'TestSets','images', 'Dataset_2')
#images_path = os.path.join(local_path, '..', 'TestSets','images', 'Dataset_3')

font_folders = os.listdir(images_path)
font_folders.sort(key=lambda s: s.lower())

image_path_font_list = []
font_label = 0
all_prediction = [];
all_label = [];

# Get paths to the images and determine the correct results
# (from the folder structure) in order to test the accuracy later on
for folder in font_folders:
    if not folder.startswith('.'):
        font_images_path = os.path.join(images_path, folder)
        images = os.listdir(font_images_path)
        images.sort(key=lambda s: s.lower())

        for image in images:
            if not image.startswith('.'):
                image_path = os.path.join(font_images_path, image)
                image_path_font_list.append([image_path, folder])
                all_label.append(font_label)

        font_label += 1

i = 0
correct = 0
false = 0
all_glyph_features = []

# Iterate over all images and predict the font
for image_path_font in image_path_font_list:
    print(image_path_font[0])
    image = cv2.imread(image_path_font[0])

    glyphs = extractGlyphes.extract_glyphs(image)
    glyph_features = []
    font_predictions = []

    if len(glyphs) == 0:
        print('ERROR: No glyphs extracted')
        cv2.imshow('a', image)
        cv2.waitKey(0)

    # Iterate over all found Glyphs
    for glyph in glyphs:
        glyph_features = extractFeatures.get_feature_vector(glyph)

        all_glyph_features.append(glyph_features)
        prediction = cla.predictData([glyph_features])
        pred_char = prediction % len(TRAIN_CHARS)
        pred_font = math.floor(prediction / len(TRAIN_CHARS))
        font_predictions.append(pred_font)

        # Determine the Accuracy fpr each glyph individualy
        if pred_font == all_label[i]:
            correct += 1
        else:
            false += 1

    print( np.bincount(font_predictions, minlength=12) )
    print( FONT_LIST[np.argmax(np.bincount(font_predictions))] )

    all_prediction.append( np.argmax(np.bincount(font_predictions)) )
    i += 1

print('Glyph Font Accuracy: ' + str(correct/(correct+false)))

correct = 0
false = 0
for i in range(0, len(all_prediction)):
    if all_prediction[i] == all_label[i]:
        correct += 1
    else:
        false += 1

print('Text Accuracy: ' + str(correct/(correct+false)))
