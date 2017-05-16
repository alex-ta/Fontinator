import cv2
#import os.path
#import os

import numpy as np

import math

def extract_glyphs(image):
    # Save copy of original image
    originalImage = np.copy(image)
    # To graryscale and binarisize
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Contour dtection works better with white letters on black background -> inverted
    #ret,binary = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
    binary = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)

    # Determine typographical ground and middle line
    p_line, ground_line, middle_line, t_line = get_typographical_lines(binary)
    ''' DEBUG Typo lines
    img = np.copy(originalImage)
    cv2.line(img, (0,ground_line), (1000, ground_line), (255,0,0))
    cv2.line(img, (0,middle_line), (1000, middle_line), (0,255,0))
    cv2.line(img, (0,p_line), (1000, p_line), (0,0,255))
    cv2.line(img, (0,t_line), (1000, t_line), (255,0,255))
    cv2.imshow('test', img)
    cv2.waitKey(0)
    '''
    # Find contours
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter out inner contours e.g. the inner circle of a o
    filt_contours = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
            filt_contours.append(contours[idx])

    # combine the found contours to glyph contours if possible
    contours_glyphs = combine_contours_to_glyph_contours(filt_contours, ground_line, middle_line, t_line)

    ''' DEBUG
    for cont_list in contours_glyphs:
        img = np.copy(originalImage)
        for cont in cont_list:
            rect = cv2.boundingRect(cont)
            cv2.rectangle(img, (rect[0],rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,0,0))
        cv2.imshow('test', img)
        cv2.waitKey(0)
    '''

    ''' DEBUG
    debug_image = np.copy(originalImage)
    for cont_list in contours_glyphs:
        cont = np.vstack(cont_list)
        unified = cv2.convexHull(cont)
        rect = cv2.boundingRect(unified)
        cv2.drawContours(debug_image, cont_list, -1,(0,255,0), -1)
        cv2.rectangle(debug_image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,255))
    cv2.imshow('Debug', debug_image)
    cv2.waitKey(0)
    return
    '''

    # Padding around the cropping area which is the filtered by a mask
    crop_area_padding = 6

    glyphs = []
    gray = np.copy(originalImage)
    for cont_list in contours_glyphs:
        # Determine region of interest (ROI)
        cont = np.vstack(cont_list)
        unified = cv2.convexHull(cont)
        center = np.array(get_center_of_contour(unified))
        rect = cv2.boundingRect(unified)

        # Detemine size of crop of ROI
        crop_size = [rect[2]+crop_area_padding, rect[3]+crop_area_padding]

        # Determine start and endpoint for cropping
        # Check if the cropping area is outside of the image
        while True:
            offset = np.array([round(crop_size[0]/2), round(crop_size[1]/2)])
            crop_area_start = center - offset
            crop_area_end = crop_area_start + crop_size

            # if the cropping are is outside of the image
            # -> shrink the cropping area till it fits
            if crop_area_start[0] < 0 or crop_area_start[1] < 0 or crop_area_end[1] >= gray.shape[0] or crop_area_end[0] >= gray.shape[1]:
                crop_size = [crop_size[0]-2, crop_size[1]-2]
            else:
                break

        # Create mask form the contour
        mask = np.zeros((crop_size[1], crop_size[0], 1), np.uint8)
        mask[:,:] = 0
        contour_list = [(cont - center + offset) for cont in cont_list]
        cv2.drawContours(mask, contour_list, -1,255 , -1)
        kernel = np.ones((3,3),np.uint8)
        # Make the mask bigger, so the whole glyph is later cropped
        mask = cv2.dilate(mask,kernel, iterations = 1)
        mask = np.array(mask, np.bool)

        # Create blank image for the glyph
        glyph = np.zeros((crop_size[1], crop_size[0]), np.uint8)
        glyph[:,:] = (255)

        # Crop out the glyph from the grayscale image, apply the mask and copy it to the new create image
        crop = imgray[ crop_area_start[1]:crop_area_end[1], crop_area_start[0]:crop_area_end[0] ]
        np.copyto(glyph, crop, where=mask)

        ''' DEBUG
        debug_image = np.copy(originalImage)
        cv2.rectangle(debug_image, (rect[0],rect[1]),(rect[0]+rect[2], rect[1]+rect[3]), (255,0,0))
        cv2.imshow('Debug',debug_image)
        cv2.imshow('glyph',glyph)
        cv2.waitKey(0)
        '''
        glyphs.append(glyph)

    return glyphs

def get_center_of_contour(contour):
    rect = cv2.boundingRect(contour)
    cx = round(rect[0] + rect[2]/2)
    cy = round(rect[1] + rect[3]/2)
    return cx, cy

# returns typographical lines
# it's about this
# https://de.wikipedia.org/wiki/Typografie#/media/File:Typografische_Begriffe.svg
def get_typographical_lines(binary_image):
    line_pixels = []
    for line in binary_image:
        line_pixels.append(sum(line))

    diffs = []
    for i in range(0, len(line_pixels)-1):
        diffs.append( line_pixels[i+1] - line_pixels[i] )
    # append 0 for last val
    diffs.append(0)

    max_val = max(diffs)
    # find smallest val bigger 0
    min_val = min(i for i in diffs if i > 0)
    threshold = (max_val - min_val) / 2 + min_val
    # TODO
    threshold *= 1.5

    t_line, middle_line, ground_line, p_line = None, None, None, None
    for idx, val in enumerate(line_pixels):
        diff = diffs[idx]

        if t_line == None:
            if val > 0:
                t_line = idx

        elif middle_line == None:
            if diff > threshold:
                middle_line = idx

        elif ground_line == None:
            if diff < -threshold:
                ground_line = idx

        elif val == 0:
            p_line = idx-1
            break

    return p_line, ground_line, middle_line, t_line

def combine_contours_to_glyph_contours(filt_contours, ground_line, middle_line, t_line):
    # try to compute the area threshold from the typographical lines
    contour_area_threshold = (ground_line - middle_line) / 6
    contour_area_threshold *= contour_area_threshold

    # try to calculate a threshold for the max distance between two contours whuch shall be combined
    # from the typo lines
    distance_threshold = ground_line - middle_line + middle_line - t_line
    #distance_threshold *= 2
    # seperate contours which are above or below the middle line
    contours_above = []
    contours_below = []
    for contour in filt_contours:
        center = ( get_center_of_contour(contour) )
        area = cv2.contourArea(contour)

        # if ABOVE middle_line
        if center[1] < middle_line:
            contours_above.append(contour)
        else:
            # if there a small contours below the middle line we skip them, they are
            # propabliy not that important
            if area < contour_area_threshold:
                continue
            # copy contours which are below the middle line into a list
            # so we can append the corresponding elemnts from above the middle_line
            # to these
            contours_below.append([contour])

    # we try to assign each contour above the middle line to a contour below the
    # middle line by comparing the distance between the center of both (and a threshold)
    for cont in contours_above:
        min_distance = 999.9
        best_index = -1

        center = ( get_center_of_contour(cont) )

        for i in range(0, len(contours_below)):
            center_below = ( get_center_of_contour(contours_below[i][0]) )
            # it works better when the x distance is more important than the y distnace
            cur_distance_x = (center_below[0]-center[0])**2
            cur_distance_x *= 2
            cur_distance_y = (center_below[0]-center[0])**2
            cur_distance = math.sqrt(cur_distance_x+cur_distance_y)

            if cur_distance < distance_threshold and cur_distance < min_distance:
                min_distance = cur_distance
                best_index = i

        # assign the above contour to the below contour wiht the smallest distance
        if best_index != -1:
            contours_below[best_index].append(cont)

    return contours_below









'''
def get_formated_contour(in_contour):
    contour = []
    # contours are in a strange format ???
    for contour_point in in_contour:
        contour.append(contour_point[0])

    return np.array([contour])
'''

'''
def draw_glyph_from_contour_list(contour_list, glyph_width, glyph_height):
    glyph = np.zeros((glyph_height, glyph_width, 3), np.uint8)
    glyph[:,:] = (255, 255, 255)

    cont = np.vstack(contour_list)
    unified = cv2.convexHull(cont)
    unified = get_formated_contour(unified)

    center = np.array(get_center_of_contour(unified))
    offset = np.array([round(glyph_width/2), round(glyph_height/2)])
    contour_list = [(cont - center + offset) for cont in contour_list]

    cv2.drawContours(glyph, contour_list, -1,(0,0,0), 1)

    #cv2.imshow('abc', glyph)
    #cv2.waitKey(0)

    return glyph
'''
'''
def draw_mask_from_contour_list(contour_list, glyph_width, glyph_height):
    mask = np.zeros((glyph_height, glyph_width, 1), np.uint8)
    mask[:,:] = 0

    cont = np.vstack(contour_list)
    unified = cv2.convexHull(cont)
    unified = get_formated_contour(unified)

    center = np.array(get_center_of_contour(unified))
    offset = np.array([round(glyph_width/2), round(glyph_height/2)])
    contour_list = [(cont - center + offset) for cont in contour_list]
    #contour_list = [(cont - center + offset) for cont in unified]

    cv2.drawContours(mask, contour_list, -1,255 , -1)

    return mask
'''
