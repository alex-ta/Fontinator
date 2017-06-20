import cv2
import numpy as np
import math

# Get the number of pixels with specific characteristica
def get_num_of_pixels(iamge):
    sum_white_pixels = len(np.where(iamge == 255)[0])
    sum_black_pixels = len(np.where(iamge < 255)[0])
    sum_zero_pixels = len(np.where(iamge == 0)[0])
    sum_total_pixels = iamge.shape[0] * iamge.shape[1]
    sum_gray_pixels = sum_total_pixels - sum_white_pixels - sum_zero_pixels

    return sum_total_pixels, sum_white_pixels, sum_black_pixels, sum_gray_pixels

# Get the horizontal center of the glpyh
def get_horizontal_center(iamge):
    width = get_width(iamge)
    left_edge = get_left_edge_pos(iamge)
    center = width / 2

    return left_edge + center

# Get the vertical center of the glpyh
def get_vertical_center(iamge):
    height = get_height(iamge)
    top_edge = get_top_edge_pos(iamge)
    center = height / 2

    return top_edge + center

# Get the width of the glyph
def get_width(iamge):
    return get_height( np.transpose(iamge) )

# Get the height of the glyph
def get_height(iamge):
    height = 0
    height_begin = None
    height_end = None
    for i, line in enumerate(iamge):
        if height_begin == None:
            if np.any(line < 255):
                height_begin = i
                height_end = i
        if height_begin != None:
            if np.any(line < 255):
                height_end = i
    height_end += 1

    return height_end - height_begin

# Get the Horizontal Pos from the left edge to the begin of the glpyh
def get_left_edge_pos(iamge):
    return get_top_edge_pos( np.transpose(iamge) )

# Get the Vertical Pos from the top edge to the begin of the glyph
def get_top_edge_pos(iamge):
    start_pos = None
    for i, line in enumerate(iamge):
        if np.any(line < 255):
                start_pos = i
                break
    return start_pos

# Get the mean horizontal position of all 'on' pixels
def get_mean_horizontal_position(iamge):
    center = get_horizontal_center(iamge)
    width = get_width(iamge)

    dist_on_pix_center = np.where(iamge < 255)[1] - center
    dist_on_pix_center = dist_on_pix_center / width

    mean_horizontal_pos = np.mean(dist_on_pix_center)

    return mean_horizontal_pos

# Get the mean vertical position of all 'on' pixels
def get_mean_vertical_position(iamge):
    center = get_vertical_center(iamge)
    height = get_height(iamge)

    dist_on_pix_center = np.where(iamge < 255)[0] - center
    dist_on_pix_center = dist_on_pix_center / height

    mean_vertical_pos = np.mean(dist_on_pix_center)

    return mean_vertical_pos

# Get the horizontal variance of all 'on' pixels
def get_horizontal_variance(iamge):
    dist =  np.where(iamge < 255)[1]
    return np.var(dist)

# Get the vertical variance of all 'on' pixels
def get_vertical_variance(iamge):
    dist =  np.where(iamge < 255)[0]
    return np.var(dist)

# Get the sum of all vertical edges in the picture
# a vertical edge is a 'on' pixel with a 'off' pixel directly to right
def get_sum_of_vertical_edges(iamge):
    pos_on_pix_v = np.where(iamge < 255)[0]
    pos_on_pix_h = np.where(iamge < 255)[1]

    max_width = iamge.shape[1] - 1
    edge_counter = 0

    for i in range(0, len(pos_on_pix_v)):
        if pos_on_pix_h[i] < max_width:
            if iamge[ pos_on_pix_v[i] ][ pos_on_pix_h[i]+1 ] == 255:
                edge_counter += 1

    return edge_counter

# Get the sum of all vertical edges in the picture
# a vertical edge is a 'on' pixel with a 'off' pixel directly below
def get_sum_of_horizontal_edges(iamge):
    return get_sum_of_vertical_edges( np.transpose(iamge) )

# Get the perimter of all contours in the image
def get_sum_perimeter(iamge):
    perimeter = 0
    binary = cv2.adaptiveThreshold(iamge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter += cv2.arcLength(contour,True)

    return perimeter

# Get the number of holes in the image
# a hohle is e.g. the inner of an O
def get_num_holes(iamge):
    regions = 0
    binary = cv2.adaptiveThreshold(iamge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            regions += 1

    return regions

# Get the number of all connected components with on pixels in the image
def get_num_comps(iamge):
    regions = 0
    binary = cv2.adaptiveThreshold(iamge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
            regions += 1

    return regions

# Reduce the 'on' areas in the pixel in order to get the skelet of each reagion
def get_skeletation(iamge):
    size = np.size(iamge)
    skel = np.zeros(iamge.shape,np.uint8)

    ret,img = cv2.threshold(iamge,127,255,cv2.THRESH_BINARY_INV)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    done = False

    i = 0
    zeros = cv2.countNonZero(img)
    while( zeros != size):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)

    return cv2.countNonZero(skel)

# Creates the feautre vector
def get_feature_vector(iamge):
    feature_vector = []

    height = get_height(iamge)
    width = get_width(iamge)
    norm, _, black, gray = get_num_of_pixels(iamge)

    feature_vector.append( get_sum_perimeter(iamge) / (width + height) )

    skel = get_skeletation(iamge)
    if skel != 0:
        feature_vector.append( get_sum_perimeter(iamge) / get_skeletation(iamge) )
    else:
       feature_vector.append( 0 )

    feature_vector.append( get_mean_horizontal_position(iamge) )
    feature_vector.append( get_mean_vertical_position(iamge) )
    feature_vector.append( get_sum_of_horizontal_edges(iamge) / black )
    feature_vector.append( get_num_holes(iamge) )
    feature_vector.append( get_num_comps(iamge) )
    feature_vector.append( get_vertical_variance(iamge) / height )
    feature_vector.append( get_horizontal_variance(iamge) / width )

    return feature_vector
