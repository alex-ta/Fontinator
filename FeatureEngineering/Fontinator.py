import cv2
import os
import numpy as np
import sys
import math
import pickle

import classifier
import extractGlyphes
import extractFeatures

# Load the classifier and labels
cla = classifier.Classifier()
cla.loadTrainedClassifier('./classie.pickle')
with open('labels.pickle', 'rb') as file_handle:
    TRAIN_CHARS = pickle.load(file_handle)
    FONT_LIST = pickle.load(file_handle)

# Check if image path was provided
if len(sys.argv) < 2:
    print('Usage: python3.6 Fontinator.py Path_To_Image')
    sys.exit(0)

# Check if file exists
image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print('Invalid image path')
    sys.exit()

# Read image and extract glyphs
image = cv2.imread(image_path)
glyphs = extractGlyphes.extract_glyphs(image)

if len(glyphs) == 0:
    print('ERROR: No glyphs extracted')
    sys.exit()

font_predictions = []
# Iterate over all found Glyphs
for glyph in glyphs:
    glyph_features = extractFeatures.get_feature_vector(glyph)

    prediction = cla.predictData([glyph_features])
    pred_char = prediction % len(TRAIN_CHARS)
    pred_font = math.floor( prediction / len(TRAIN_CHARS) )
    font_predictions.append(pred_font)

# Rank the identified fonts by their propability
font_prediction = np.bincount(font_predictions, minlength=12)
font_prediction = font_prediction / np.sum(font_prediction)
font_ranking = np.argsort(font_prediction)
font_ranking = np.flip(font_ranking, axis=0)

# Output the result to the console
print('Fontinator\n')
print('Font prediction for: ' + os.path.split(image_path)[1] + '\n' )

for font_idx in font_ranking:
    if font_prediction[font_idx] > 0:
        percent = font_prediction[font_idx] * 100
        percent = "{:8.2f}".format(percent)
        font_name = FONT_LIST[font_idx]
        print(percent + '% : ' + font_name)
print('')
