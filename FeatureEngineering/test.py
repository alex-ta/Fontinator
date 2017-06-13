import cv2
import os.path
import os
import numpy as np
import extractGlyphes
import extractFeatures
from matplotlib import pyplot as plt
import math

import classifier

from sklearn.svm import SVC
from sklearn.feature_selection import RFE

n_test_pictures = 1

cla = classifier.Classifier()
cla.loadTrainedClassifier('./classie.pickle')

local_path = os.path.dirname(os.path.realpath(__file__))
images_path = os.path.join(local_path, '..', 'images')
font_folders = os.listdir(images_path)
image_path_font_list = []
for folder in font_folders:
    if not folder.startswith('.'):
        font_images_path = os.path.join(images_path, folder)
        images = os.listdir(font_images_path)

        i = 0
        for image in images:
            if not image.startswith('.'):
                image_path = os.path.join(font_images_path, image)
                image_path_font_list.append([image_path, folder])

                i += 1
                if i == n_test_pictures:
                    break

#image_path_font_list = [['/Users/Sebastian/Desktop/Master/IDA_Projekt/Repo/Fontinator/images/arial/arial_101.png', 'a']]
#image_path_font_list = [['/Users/Sebastian/Desktop/Master/IDA_Projekt/Repo/Fontinator/images/jokerman/jokerman_0.png', 'a']]
#image_path_font_list = [['/Users/Sebastian/Desktop/Master/IDA_Projekt/Repo/Fontinator/images/forte/forte_43.png', 'a']]
#image_path_font_list = [['/Users/Sebastian/Desktop/Master/IDA_Projekt/Repo/Fontinator/images/times_new_romance/times_new_romance_43.png', 'a']]
all_prediction = [];
all_label = [];

for i in range(0, 12):
    for j in range(0, n_test_pictures):
        all_label.append(i)

all_glyph_features = []
for image_path_font in image_path_font_list:
    print(image_path_font[0])
    image = cv2.imread(image_path_font[0])
    '''DEBUG SIFT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    #kp = sift.detect(gray, None)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp, image)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis 8
    plt.show()
    '''
    '''DEBUG SURF
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(10000)
    #kp = sift.detect(gray, None)
    kp, des = surf.detectAndCompute(gray, None)
    print(len(kp))
    print(des.shape)
    img = cv2.drawKeypoints(gray, kp, image)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis 8
    plt.show()
    '''
    '''DEBUG BRISK
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brisk = cv2.BRISK_create()
    #kp = sift.detect(gray, None)
    kp, des = brisk.detectAndCompute(gray, None)
    print(len(kp))
    print(des)
    img = cv2.drawKeypoints(gray, kp, image)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis 8
    plt.show()
    '''
    '''DEBUG Harris Corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    print(corners)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis 8
    plt.show()
    #cv2.imshow('dst', image)
    #if cv2.waitKey(0) & 0xff == 27:
        #cv2.destroyAllWindows()'''

    glyphs, p_line, ground_line, middle_line, t_line, spacing = extractGlyphes.extract_glyphs(image)
    glyph_features = []
    font_predictions = []
    '''DEBUG Glyphs'''
    for glyph in glyphs:
        #print(glyph)
        #weight = extractFeatures.get_mean_vertical_position(glyph)
        #print(weight)

        glyph_features = extractFeatures.get_feature_vector(glyph)

        #glyph_features.append( (ground_line-middle_line) / (p_line-t_line) )
        #glyph_features.append( (p_line-ground_line) / (p_line-t_line) )
        #glyph_features.append( (middle_line-t_line) / (p_line-t_line) )
        #glyph_features.append( spacing )

        all_glyph_features.append(glyph_features)
        prediction = cla.predictData([glyph_features])
        pred_char = prediction % 80
        pred_font = math.floor(prediction / 80)
        font_predictions.append(pred_font)
        #print("Prediction " + str(prediction) + " Font " + str(pred_font) + " Char " + str(pred_char))
        #cv2.imshow('dst', glyph)
        #cv2.waitKey(0)
        train_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ÄÖÜäöü!"()[]?ß.,+-'
        print( train_chars[pred_char[0]] )
        print( pred_font )

        cv2.imshow('dst', glyph)
        cv2.waitKey(0)

    print( np.bincount(font_predictions, minlength=12) )
    print( np.argmax(np.bincount(font_predictions)) )

    all_prediction.append( np.argmax(np.bincount(font_predictions)) )

correct = 0
false = 0
for i in range(0, len(all_prediction)):
    if all_prediction[i] == all_label[i]:
        correct += 1
    else:
        false += 1

print('Accuracy: ' + str(correct/(correct+false)))
