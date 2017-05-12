import cv2
import os.path
import os

import numpy as np

def extract_glyphs(image):
    originalImage = np.copy(image)
    # Binarisize
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Contour dtection works better with white letters on black background -> inverted
    ret,binary = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)

    # Find contours
    im2, contours, hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    filt_contours = []
    # Select only contours withour inner contour
    for i, item in enumerate(hierarchy[0]):
        if item[3] == -1:
            filt_contours.append(contours[i])

    boundings = []
    for contur in filt_contours:
        boundRect = cv2.boundingRect( contur )
        boundings.append( boundRect )

    combined_boundings = []
    for i in range(0, len(boundings)):
        for j in range(i, len(boundings)):
            # is inner element x-wise in outer element ?
            if ( boundings[j][0] >= boundings[i][0] and boundings[j][0]+boundings[j][2] <= boundings[i][0]+boundings[i][2] ):
                combined_contour = np.concatenate([filt_contours[i], filt_contours[j]])
                combined_boundings.append( cv2.boundingRect( combined_contour ) )


    x_segments = [(bound[0], bound[0] + bound[2]) for bound in boundings]
    filt_x_segments = []
    x_segments.sort()
    cur = 0
    nex = 1
    while nex < len(x_segments):
        cur_s, cur_e = x_segments[cur]
        next_s, next_e = x_segments[nex]

        if cur_e < next_s:
            filt_x_segments.append(x_segments[cur])
        else:
            #while (x_segments[cur][0] <= x_segments[nex][0]) and (x_segments[cur][1] > x_segments[nex][1]):
            while segement_in_segment(x_segments[nex][0],x_segments[nex][1],x_segments[cur][0],x_segments[cur][1]):
                nex += 1

            filt_x_segments.append(x_segments[cur])
            cur = nex

        cur = nex
        nex += 1

    # get rid of to small things
    characters = []
    image = np.copy(originalImage)
    for seg in filt_x_segments:
        characters.append( image[ 0:40, seg[0]:seg[1] ] )

    return characters

def segement_in_segment(inner_s, inner_e, outer_s, outer_e):
    inner_middle = round((inner_e - inner_s)/2) + inner_s
    return inner_middle >= outer_s and inner_middle <= outer_e
