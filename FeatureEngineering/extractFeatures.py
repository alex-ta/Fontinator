import cv2
import numpy as np
import math

# Get the number of pixels with specific characteristica
def get_num_of_pixels(image):
    sum_white_pixels = len(np.where(image == 255)[0])
    sum_black_pixels = len(np.where(image < 255)[0])
    sum_zero_pixels = len(np.where(image == 0)[0])
    sum_total_pixels = image.shape[0] * image.shape[1]
    sum_gray_pixels = sum_total_pixels - sum_white_pixels - sum_zero_pixels

    return sum_total_pixels, sum_white_pixels, sum_black_pixels, sum_gray_pixels

# Get the horizontal center of the glpyh
def get_horizontal_center(image):
    width = get_width(image)
    left_edge = get_left_edge_pos(image)
    center = width / 2

    return left_edge + center

# Get the vertical center of the glpyh
def get_vertical_center(image):
    height = get_height(image)
    top_edge = get_top_edge_pos(image)
    center = height / 2

    return top_edge + center

# Get the width of the glyph
def get_width(image):
    return get_height( np.transpose(image) )

# Get the height of the glyph
def get_height(image):
    height = 0
    height_begin = None
    height_end = None
    for i, line in enumerate(image):
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
def get_left_edge_pos(image):
    return get_top_edge_pos( np.transpose(image) )

# Get the Vertical Pos from the top edge to the begin of the glyph
def get_top_edge_pos(image):
    start_pos = None
    for i, line in enumerate(image):
        if np.any(line < 255):
                start_pos = i
                break
    return start_pos

# Get the mean horizontal position of all 'on' pixels
def get_mean_horizontal_position(image):
    center = get_horizontal_center(image)
    width = get_width(image)

    dist_on_pix_center = np.where(image < 255)[1] - center
    dist_on_pix_center = dist_on_pix_center / width

    mean_horizontal_pos = np.mean(dist_on_pix_center)

    return mean_horizontal_pos

# Get the mean vertical position of all 'on' pixels
def get_mean_vertical_position(image):
    center = get_vertical_center(image)
    height = get_height(image)

    dist_on_pix_center = np.where(image < 255)[0] - center
    dist_on_pix_center = dist_on_pix_center / height

    mean_vertical_pos = np.mean(dist_on_pix_center)

    return mean_vertical_pos

# Get the horizontal variance of all 'on' pixels
def get_horizontal_variance(image):
    dist =  np.where(image < 255)[1]
    return np.var(dist)

# Get the vertical variance of all 'on' pixels
def get_vertical_variance(image):
    dist =  np.where(image < 255)[0]
    return np.var(dist)

# Get the sum of all vertical edges in the picture
# a vertical edge is a 'on' pixel with a 'off' pixel directly to right
def get_sum_of_vertical_edges(image):
    pos_on_pix_v = np.where(image < 255)[0]
    pos_on_pix_h = np.where(image < 255)[1]

    max_width = image.shape[1] - 1
    edge_counter = 0

    for i in range(0, len(pos_on_pix_v)):
        if pos_on_pix_h[i] < max_width:
            if image[ pos_on_pix_v[i] ][ pos_on_pix_h[i]+1 ] == 255:
                edge_counter += 1

    return edge_counter

# Get the sum of all vertical edges in the picture
# a vertical edge is a 'on' pixel with a 'off' pixel directly below
def get_sum_of_horizontal_edges(image):
    return get_sum_of_vertical_edges( np.transpose(image) )

# Get the perimter of all contours in the image
def get_sum_perimeter(image):
    perimeter = 0
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter += cv2.arcLength(contour,True)

    return perimeter

# Get the number of holes in the image
# a hohle is e.g. the inner of an O
def get_num_holes(image):
    regions = 0
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            regions += 1

    return regions

# Get the number of all connected components with on pixels in the image
def get_num_comps(image):
    regions = 0
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
            regions += 1

    return regions

# Reduce the 'on' areas in the pixel in order to get the skelet of each reagion
def get_skeletation(image):
    size = np.size(image)
    skel = np.zeros(image.shape,np.uint8)

    ret,img = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
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
def get_feature_vector(image):
    feature_vector = []

    height = get_height(image)
    width = get_width(image)
    norm, _, black, gray = get_num_of_pixels(image)

    feature_vector.append( get_sum_perimeter(image) / (width + height) )

    skel = get_skeletation(image)
    if skel != 0:
        feature_vector.append( get_sum_perimeter(image) / get_skeletation(image) )
    else:
       feature_vector.append( 0 )

    feature_vector.append( get_mean_horizontal_position(image) )
    feature_vector.append( get_mean_vertical_position(image) )
    feature_vector.append( get_sum_of_horizontal_edges(image) / black )
    feature_vector.append( get_num_holes(image) )
    feature_vector.append( get_num_comps(image) )
    feature_vector.append( get_vertical_variance(image) / height )
    feature_vector.append( get_horizontal_variance(image) / width )

    return feature_vector
