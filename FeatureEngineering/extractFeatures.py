import cv2
#import os.path
#import os

import numpy as np

import math

def get_num_of_pixels(binary_image):
    sum_white_pixels = len(np.where(binary_image == 255)[0])
    sum_black_pixels = len(np.where(binary_image < 255)[0])
    sum_zero_pixels = len(np.where(binary_image == 0)[0])
    sum_total_pixels = binary_image.shape[0] * binary_image.shape[1]
    sum_gray_pixels = sum_total_pixels - sum_white_pixels - sum_zero_pixels

    return sum_total_pixels, sum_white_pixels, sum_black_pixels, sum_gray_pixels

def get_horizontal_position(binary_image):
    width = get_width(binary_image)
    left_edge = get_left_edge_pos(binary_image)
    center = width / 2

    return left_edge + center

def get_vertical_position(binary_image):
    height = get_height(binary_image)
    top_edge = get_top_edge_pos(binary_image)
    center = height / 2

    return top_edge + center

def get_width(binary_image):
    return get_height( np.transpose(binary_image) )

def get_height(binary_image):
    height = 0
    height_begin = None
    height_end = None
    for i, line in enumerate(binary_image):
        if height_begin == None:
            if np.any(line < 255):
                height_begin = i
                height_end = i
        if height_begin != None:
            if np.any(line < 255):
                height_end = i
    height_end += 1

    return height_end - height_begin

def get_left_edge_pos(binary_image):
    return get_top_edge_pos( np.transpose(binary_image) )

def get_top_edge_pos(binary_image):
    start_pos = None
    for i, line in enumerate(binary_image):
        if np.any(line < 255):
                start_pos = i
                break
    return start_pos

def get_mean_horizontal_position(binary_image):
    center = get_horizontal_position(binary_image)
    width = get_width(binary_image)

    dist_on_pix_center = np.where(binary_image < 255)[1] - center
    dist_on_pix_center = dist_on_pix_center / width

    mean_horizontal_pos = np.mean(dist_on_pix_center)

    return mean_horizontal_pos

def get_mean_vertical_position(binary_image):
    center = get_vertical_position(binary_image)
    height = get_height(binary_image)

    dist_on_pix_center = np.where(binary_image < 255)[0] - center
    dist_on_pix_center = dist_on_pix_center / height

    mean_vertical_pos = np.mean(dist_on_pix_center)

    return mean_vertical_pos

def get_mean_squared_horizontal_position(binary_image):
    center = get_horizontal_position(binary_image)
    width = get_width(binary_image)

    dist_on_pix_center = np.where(binary_image < 255)[1] - center
    dist_on_pix_center = dist_on_pix_center / width

    dist_on_pix_center = dist_on_pix_center * dist_on_pix_center

    mean_squared_horizontal_pos = sum(dist_on_pix_center) / len(dist_on_pix_center)

    return mean_squared_horizontal_pos

def get_mean_squared_vertical_position(binary_image):
    center = get_vertical_position(binary_image)
    height = get_height(binary_image)

    dist_on_pix_center = np.where(binary_image < 255)[0] - center
    dist_on_pix_center = dist_on_pix_center / height

    dist_on_pix_center = dist_on_pix_center * dist_on_pix_center

    mean_squared_vertical_pos = sum(dist_on_pix_center) / len(dist_on_pix_center)

    return mean_squared_vertical_pos

def get_mean_diagonal_position(binary_image):
    center_v = get_vertical_position(binary_image)
    height = get_height(binary_image)
    center_h = get_horizontal_position(binary_image)
    width = get_width(binary_image)

    dist_on_pix_center_v = np.where(binary_image < 255)[0] - center_v
    dist_on_pix_center_h = np.where(binary_image < 255)[1] - center_h

    dist_on_pix_center = dist_on_pix_center_v * dist_on_pix_center_h
    dist_on_pix_center = dist_on_pix_center / (height * width)

    mean_diagonal_pos = sum(dist_on_pix_center) / len(dist_on_pix_center)

    return mean_diagonal_pos

def get_correlation_of_horizontal_variance(binary_image):
    center_v = get_vertical_position(binary_image)
    height = get_height(binary_image)
    center_h = get_horizontal_position(binary_image)
    width = get_width(binary_image)

    dist_on_pix_center_v = (np.where(binary_image < 255)[0] - center_v) / width
    dist_on_pix_center_h = (np.where(binary_image < 255)[1] - center_h) / height

    dist_on_pix_center = dist_on_pix_center_v * dist_on_pix_center_h * dist_on_pix_center_h

    corr_hor_variance = sum(dist_on_pix_center) / len(dist_on_pix_center)

    return corr_hor_variance

def get_correlation_of_vertical_variance(binary_image):
    center_v = get_vertical_position(binary_image)
    height = get_height(binary_image)
    center_h = get_horizontal_position(binary_image)
    width = get_width(binary_image)

    dist_on_pix_center_v = (np.where(binary_image < 255)[0] - center_v) / width
    dist_on_pix_center_h = (np.where(binary_image < 255)[1] - center_h) / height

    dist_on_pix_center = dist_on_pix_center_v * dist_on_pix_center_v * dist_on_pix_center_h

    corr_vert_variance = sum(dist_on_pix_center) / len(dist_on_pix_center)

    return corr_vert_variance

def get_horizontal_variance(binary_image):
    dist =  np.where(binary_image < 255)[1]
    return np.var(dist)

def get_vertical_variance(binary_image):
    dist =  np.where(binary_image < 255)[0]
    return np.var(dist)

def get_sum_of_vertical_edges(binary_image):
    pos_on_pix_v = np.where(binary_image < 255)[0]
    pos_on_pix_h = np.where(binary_image < 255)[1]

    max_width = binary_image.shape[1] - 1
    edge_counter = 0

    for i in range(0, len(pos_on_pix_v)):
        if pos_on_pix_h[i] < max_width:
            if binary_image[ pos_on_pix_v[i] ][ pos_on_pix_h[i]+1 ] == 255:
                edge_counter += 1

    return edge_counter

def get_sum_of_horizontal_edges(binary_image):
    return get_sum_of_vertical_edges( np.transpose(binary_image) )

def get_sum_of_diagonal_edges(binary_image):
    pos_on_pix_v = np.where(binary_image < 255)[0]
    pos_on_pix_h = np.where(binary_image < 255)[1]

    max_width = binary_image.shape[1] - 1
    max_height = binary_image.shape[0] - 1
    edge_counter = 0

    for i in range(0, len(pos_on_pix_v)):
        if pos_on_pix_h[i] < max_width and pos_on_pix_v[i] < max_height:
            if binary_image[ pos_on_pix_v[i]+1 ][ pos_on_pix_h[i]+1 ] == 255:
                edge_counter += 1

    return edge_counter

def get_mean_number_of_vertical_edges(binary_image):
    pos_on_pix_v = np.where(binary_image < 255)[0]
    pos_on_pix_h = np.where(binary_image < 255)[1]

    center = get_horizontal_position(binary_image)
    width = get_width(binary_image)
    max_width = binary_image.shape[1] - 1
    edges = []

    for i in range(0, len(pos_on_pix_v)):
        if pos_on_pix_h[i] < max_width:
            if binary_image[ pos_on_pix_v[i] ][ pos_on_pix_h[i]+1 ] == 255:
                dist_to_center = (pos_on_pix_v[i] - center) / width
                edges.append(dist_to_center*dist_to_center)

    if len(edges) > 0:
        return sum(edges) / len(edges)
    else:
        return 0

def get_mean_number_of_horizontal_edges(binary_image):
    pos_on_pix_v = np.where(binary_image < 255)[0]
    pos_on_pix_h = np.where(binary_image < 255)[1]

    center = get_vertical_position(binary_image)
    height = get_height(binary_image)
    max_height = binary_image.shape[1] - 1
    edges = []

    for i in range(0, len(pos_on_pix_v)):
        if pos_on_pix_v[i] < max_height:
            if binary_image[ pos_on_pix_v[i]+1 ][ pos_on_pix_h[i] ] == 255:
                dist_to_center = (pos_on_pix_h[i] - center) / height
                edges.append(dist_to_center*dist_to_center)

    if len(edges) > 0:
        return sum(edges) / len(edges)
    else:
        return 0

def get_sum_of_horizontal_singles(binary_image):
    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];
    connections = 0

    for line in range(0, line_count-2):
        for col in range(0, col_count):
            if np.any(binary_image[line+1][col] < 255):
                if np.any(binary_image[line][col] == 255) and np.any(binary_image[line+2][col] == 255):
                    connections += 1
    return connections

def get_sum_of_horizontal_doubles(binary_image):
    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];
    connections = 0

    for line in range(0, line_count-2):
        for col in range(0, col_count):
            if np.any(binary_image[line][col] < 255):
                if np.any(binary_image[line][col] < 255):
                    connections += 1
    return connections

def get_sum_of_vertical_doubles(binary_image):
    return get_sum_of_horizontal_doubles( np.transpose(binary_image) )

def get_sum_of_vertical_singles(binary_image):
    return get_sum_of_horizontal_singles( np.transpose(binary_image) )

def get_mean_number_of_horizontal_edges(binary_image):
    return get_mean_number_of_vertical_edges( np.transpose(binary_image) )

def get_sum_perimeter(binary_image):
    perimeter = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter += cv2.arcLength(contour,True)

    return perimeter

def get_sum_compactness(binary_image):
    compactness = []
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if (cv2.contourArea(contour) != 0):
            comp = cv2.contourArea(contour) / (cv2.arcLength(contour,True)**2)
            compactness.append(comp)
        else:
            compactness.append(1)

    if len(compactness) > 0:
        return sum(compactness) / len(compactness)
    else:
        return 0

def get_num_hohles(binary_image):
    regions = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            regions += 1

    return regions

def get_num_comps(binary_image):
    regions = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
            regions += 1

    return regions

def get_horizontal_sharpness(binary_image):
    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];
    sharp_edges = 0

    for line in range(0, line_count-1):
        for col in range(0, col_count):
            if np.any(binary_image[line][col] == 255):
                if np.any(binary_image[line+1][col] == 0):
                    sharp_edges += 1
    return sharp_edges

def get_wider_horizontals(binary_image):
    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];
    widths = []
    cur_width = 0
    col = 0

    for line in range(0, line_count):
        while col < (col_count-1):
            if np.any(binary_image[line, col] < 255):
                col += 1

                while (col + 1 < col_count) and np.any(binary_image[line, col+1] < 255):
                    cur_width += 1
                    col  += 1

                widths.append(cur_width)
            col += 1
        col = 0

    median_width = np.median( widths )

    wider = 0
    for width in widths:
        if width > median_width:
            wider += 1

    return wider / len(widths)

def get_vertical_sharpness(binary_image):
    return get_horizontal_sharpness( np.transpose(binary_image) )

def get_num_of_connected_componens(binary_image):
    conn_comp = cv2.connectedComponents(binary_image)

    # Minus background
    return len(conn_comp) - 1

def get_num_of_connected_hohles(binary_image):
    binary_image = 255 - binary_image
    conn_comp = cv2.connectedComponents(binary_image)

    # Minus background
    return len(conn_comp) - 1

def get_skeletation(binary_image):
    size = np.size(binary_image)
    skel = np.zeros(binary_image.shape,np.uint8)

    ret,img = cv2.threshold(binary_image,127,255,cv2.THRESH_BINARY_INV)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    done = False

    #cv2.imshow('a', img)
    #cv2.waitKey(0)

    i = 0
    zeros = cv2.countNonZero(img)
    while( zeros != size):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        #if zeros==size:
        #    done = True

        #i += 1
        #if i == 100:
        #    cv2.imshow('a', binary_image)
        #    cv2.waitKey(0)

    return cv2.countNonZero(skel)

def get_mean_brightness(binary_image):
    top_edge = get_top_edge_pos(binary_image)
    left_edge = get_left_edge_pos(binary_image)
    width = get_width(binary_image)
    height = get_height(binary_image)

    crop_img = binary_image[top_edge:top_edge+height, left_edge:left_edge+width]

    return ( np.sum(crop_img) / np.size(crop_img) )


def get_feature_vector(binary_image):
    feature_vector = []

    height = get_height(binary_image)
    width = get_width(binary_image)
    norm, _, black, gray = get_num_of_pixels(binary_image)

    feature_vector.append( get_sum_perimeter(binary_image) / (width + height) )

    skel = get_skeletation(binary_image)
    if skel != 0:
        feature_vector.append( get_sum_perimeter(binary_image) / get_skeletation(binary_image) )
    else:
       feature_vector.append( 0 )

    feature_vector.append( get_mean_horizontal_position(binary_image) )
    feature_vector.append( get_mean_vertical_position(binary_image) )
    feature_vector.append( get_sum_of_horizontal_edges(binary_image) / black )
    feature_vector.append( get_num_hohles(binary_image) )
    feature_vector.append( get_num_comps(binary_image) )
    feature_vector.append( get_vertical_variance(binary_image) / height )
    feature_vector.append( get_horizontal_variance(binary_image) / width )

    #feature_vector.append( height / black )
    #feature_vector.append( width )
    #feature_vector.append( height )


    #feature_vector.append( gray / (width * height) )
    #feature_vector.append( black / (width * height) )
    #feature_vector.append( black )

    #feature_vector.append( black / height )
    #feature_vector.append( black / width )

    #feature_vector.append( get_mean_brightness(binary_image) )

    # BAD
    #feature_vector.append( get_mean_horizontal_position(binary_image) )

    # GOOD
    #feature_vector.append( get_mean_vertical_position(binary_image) )

    # NEUTRAL
    #feature_vector.append( get_mean_squared_horizontal_position(binary_image) )

    # GOOD
    #feature_vector.append( get_mean_squared_vertical_position(binary_image) )

    # GOOD
    #feature_vector.append( get_mean_diagonal_position(binary_image) )

    # GOOD
    #feature_vector.append( get_sum_of_vertical_edges(binary_image) / (black) )
    #feature_vector.append( get_sum_of_vertical_edges(binary_image) / (width*height) )

    # GOOD
    #feature_vector.append( get_sum_of_horizontal_edges(binary_image) / (black) )
    #feature_vector.append( get_sum_of_horizontal_edges(binary_image) / (width*height) )

    #feature_vector.append( get_num_hohles(binary_image) )

    #feature_vector.append( get_num_comps(binary_image) )

    #feature_vector.append( get_sum_perimeter(binary_image) / (width + height) )

    # GOOD
    #feature_vector.append( get_sum_of_diagonal_edges(binary_image) / (black) )

    # SCHLECHTER
    #feature_vector.append( get_sum_of_diagonal_edges(binary_image) / (width*height) )

    # NEUTRAL
    #feature_vector.append( get_correlation_of_vertical_variance(binary_image) )

    # BAD
    #feature_vector.append( get_skeletation(binary_image) / black )
    #feature_vector.append( get_skeletation(binary_image) / (width*height) )
    #print ( get_skeletation(binary_image) / black )
    #cv2.waitKey(0)

    # BUG feature_vector.append( get_num_of_connected_componens(binary_image) )
    #feature_vector.append( get_num_of_connected_hohles(binary_image) )

    # BAD
    #feature_vector.append( get_mean_number_of_vertical_edges(binary_image) )

    # NEUTRAL
    #feature_vector.append( get_mean_number_of_horizontal_edges(binary_image) )

    # NOT TESTED
    #if horizontal_edges != 0:
    #    feature_vector.append( get_horizontal_sharpness(binary_image) )
    #else:
    #    feature_vector.append( 0 )

    # NOT TESTED
    #if vertical_edges != 0:
    #    feature_vector.append( get_vertical_sharpness(binary_image) )
    #else:
    #    feature_vector.append( 0 )

    # BAD
    #feature_vector.append( get_wider_horizontals(binary_image) )
    ''' '''

    # NEUTRAL
    #feature_vector.append( get_sum_compactness(binary_image) )

    # GOOD
    #skel = get_skeletation(binary_image)
    #if skel != 0:
    #    feature_vector.append( get_sum_perimeter(binary_image) / get_skeletation(binary_image) )
    #else:
    #   feature_vector.append( 0 )

    # NEUTRAL
    #feature_vector.append( get_sum_of_vertical_singles(binary_image) / black )
    #feature_vector.append( get_sum_of_vertical_singles(binary_image) / (width*black) )

    # BAD
    #feature_vector.append( get_sum_of_horizontal_singles(binary_image) / (black) )
    #feature_vector.append( get_sum_of_horizontal_singles(binary_image) / (width*height) )

    # GOOD
    #feature_vector.append( get_correlation_of_horizontal_variance(binary_image) )

    # BAD
    #feature_vector.append( get_mean_number_of_vertical_edges(binary_image) )

    # NEUTRAL
    #feature_vector.append( get_mean_number_of_horizontal_edges(binary_image) )

    return feature_vector
