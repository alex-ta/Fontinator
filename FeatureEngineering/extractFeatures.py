import cv2
import numpy as np
import math

# Get the number of pixels with specific characteristica
def get_num_of_pixels(binary_image):
    sum_white_pixels = len(np.where(binary_image == 255)[0])
    sum_black_pixels = len(np.where(binary_image < 255)[0])
    sum_zero_pixels = len(np.where(binary_image == 0)[0])
    sum_total_pixels = binary_image.shape[0] * binary_image.shape[1]
    sum_gray_pixels = sum_total_pixels - sum_white_pixels - sum_zero_pixels

    return sum_total_pixels, sum_white_pixels, sum_black_pixels, sum_gray_pixels

# Get the horizontal center of the glpyh
def get_horizontal_center(binary_image):
    width = get_width(binary_image)
    left_edge = get_left_edge_pos(binary_image)
    center = width / 2

    return left_edge + center

# Get the vertical center of the glpyh
def get_vertical_center(binary_image):
    height = get_height(binary_image)
    top_edge = get_top_edge_pos(binary_image)
    center = height / 2

    return top_edge + center

# Get the width of the glyph
def get_width(binary_image):
    return get_height( np.transpose(binary_image) )

# Get the height of the glyph
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

# Get the Horizontal Pos from the left edge to the begin of the glpyh
def get_left_edge_pos(binary_image):
    return get_top_edge_pos( np.transpose(binary_image) )

# Get the Vertical Pos from the top edge to the begin of the glyph
def get_top_edge_pos(binary_image):
    start_pos = None
    for i, line in enumerate(binary_image):
        if np.any(line < 255):
                start_pos = i
                break
    return start_pos

# Get the mean horizontal position of all 'on' pixels
def get_mean_horizontal_position(binary_image):
    center = get_horizontal_center(binary_image)
    width = get_width(binary_image)

    dist_on_pix_center = np.where(binary_image < 255)[1] - center
    dist_on_pix_center = dist_on_pix_center / width

    mean_horizontal_pos = np.mean(dist_on_pix_center)

    return mean_horizontal_pos

# Get the mean vertical position of all 'on' pixels
def get_mean_vertical_position(binary_image):
    center = get_vertical_center(binary_image)
    height = get_height(binary_image)

    dist_on_pix_center = np.where(binary_image < 255)[0] - center
    dist_on_pix_center = dist_on_pix_center / height

    mean_vertical_pos = np.mean(dist_on_pix_center)

    return mean_vertical_pos

# Get the horizontal variance of all 'on' pixels
def get_horizontal_variance(binary_image):
    dist =  np.where(binary_image < 255)[1]
    return np.var(dist)

# Get the vertical variance of all 'on' pixels
def get_vertical_variance(binary_image):
    dist =  np.where(binary_image < 255)[0]
    return np.var(dist)

# Get the sum of all vertical edges in the picture
# a vertical edge is a 'on' pixel with a 'off' pixel directly to right
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

# Get the sum of all vertical edges in the picture
# a vertical edge is a 'on' pixel with a 'off' pixel directly below
def get_sum_of_horizontal_edges(binary_image):
    return get_sum_of_vertical_edges( np.transpose(binary_image) )

# Get the perimter of all contours in the image
def get_sum_perimeter(binary_image):
    perimeter = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter += cv2.arcLength(contour,True)

    return perimeter

# Get the number of holes in the image
# a hohle is e.g. the inner of an O
def get_num_holes(binary_image):
    regions = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            regions += 1

    return regions

# Get the number of all connected components with on pixels in the image
def get_num_comps(binary_image):
    regions = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
            regions += 1

    return regions

# Reduce the 'on' areas in the pixel in order to get the skelet of each reagion
def get_skeletation(binary_image):
    size = np.size(binary_image)
    skel = np.zeros(binary_image.shape,np.uint8)

    ret,img = cv2.threshold(binary_image,127,255,cv2.THRESH_BINARY_INV)
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
    feature_vector.append( get_num_holes(binary_image) )
    feature_vector.append( get_num_comps(binary_image) )
    feature_vector.append( get_vertical_variance(binary_image) / height )
    feature_vector.append( get_horizontal_variance(binary_image) / width )

    return feature_vector
