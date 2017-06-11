import cv2
#import os.path
#import os

import numpy as np

import math

def get_num_of_pixels(binary_image):
    sum_white_pixels = 0
    sum_black_pixels = 0
    sum_total_pixels = 0
    for line in binary_image:
        for pixel in line:
            sum_total_pixels = sum_total_pixels + 1
            if np.any(pixel == 255):
                sum_white_pixels = sum_white_pixels + 1
            else:
                sum_black_pixels = sum_black_pixels + 1
    return sum_total_pixels, sum_white_pixels, sum_black_pixels

def get_horizontal_position(binary_image):
    width = get_width(binary_image)
    left_edge = get_left_edge_pos(binary_image)
    center = width / 2
    mod = center % 2
    if mod > 0:
        center = center - 0.5
    else:
        center = center + 0.5
    return left_edge + center

def get_vertical_position(binary_image):
    height = get_height(binary_image)
    top_edge = get_top_edge_pos(binary_image)
    center = height / 2
    mod = center % 2
    if mod > 0:
        center = center - 0.5
    else:
        center = center + 0.5
    return top_edge + center

def get_width(binary_image):
    width = 0
    ret_width = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                width = width + 1
        if width > ret_width:
            ret_width = width
        width = 0
    return ret_width

def get_height(binary_image):
    height = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                height = height + 1
                break
    return height

def get_left_edge_pos(binary_image):
    pos = 1000
    col_count = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255) and (col_count < pos):
                pos = col_count
            col_count = col_count + 1
        col_count = 0
    return pos

def get_top_edge_pos(binary_image):
    line_count = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                return line_count
        line_count = line_count + 1
    return line_count

def get_mean_horizontal_position(binary_image):
    center = get_horizontal_position(binary_image)
    width = get_width(binary_image)
    weight = 0
    weight_array =  []
    for line in binary_image:
        col_count = 0
        for pixel in line:
            if np.any(pixel < 255):
                weight = (col_count - center) / width
                weight_array.append(weight)
            col_count = col_count + 1
        col_count = 0
    horizontal_weight = sum(weight_array) / len(weight_array)
    return horizontal_weight

def get_mean_vertical_position(binary_image):
    center = get_vertical_position(binary_image)
    height = get_height(binary_image)
    weight = 0
    weight_array = []
    line_count = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                weight = (line_count - center) / height
                weight_array.append(weight)
        line_count = line_count + 1
    vertical_weight = sum(weight_array) / len(weight_array)
    return vertical_weight

def get_mean_squared_horizontal_position(binary_image):
    center = get_horizontal_position(binary_image)
    width = get_width(binary_image)
    weight = 0
    weight_array =  []
    for line in binary_image:
        col_count = 0
        for pixel in line:
            if np.any(pixel < 255):
                weight = (col_count - center) / width
                weight_array.append(weight*weight)
            col_count = col_count + 1
        col_count = 0
    horizontal_weight = sum(weight_array) / len(weight_array)
    return horizontal_weight

def get_mean_squared_vertical_position(binary_image):
    center = get_vertical_position(binary_image)
    height = get_height(binary_image)
    weight = 0
    weight_array = []
    line_count = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                weight = (line_count - center) / height
                weight_array.append(weight*weight)
        line_count = line_count + 1
    vertical_weight = sum(weight_array) / len(weight_array)
    return vertical_weight

# TODO Wie normalisieren?
def get_correlation_of_horizontal_variance(binary_image):
    line_count = 0
    col_count = 0
    correlation_array = []
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                correlation_array.append(col_count * col_count * line_count)
            col_count = col_count + 1
        line_count = line_count + 1
        col_count = 0
    mean = sum(correlation_array) / len(correlation_array)
    return mean

# TODO Wie normalisieren?
def get_correlation_of_vertical_variance(binary_image):
    line_count = 0
    col_count = 0
    correlation_array = []
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                correlation_array.append(col_count * line_count * line_count)
            col_count = col_count + 1
        line_count = line_count + 1
        col_count = 0
    mean = sum(correlation_array) / len(correlation_array)
    return mean

def get_mean_number_of_vertical_edges(binary_image):
    line_count = 0
    col_count = 0
    edges = 0
    edge_array = []
    edge_pos_array = []

    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];

    for line in range(0, line_count):
        for col in range(0, col_count-1):
            if np.any(binary_image[line][col] < 255):
                if np.any(binary_image[line][col] < 255):
                    edges = edges + 1
                    edge_pos_array.append([line, col])

        if np.any(binary_image[line][col_count-1]) < 255:
            edges = edges + 1
            edge_pos_array.append([line, col])

        edge_array.append(edges)
        edges = 0

    mean_edges = sum(edge_array) / len(edge_array)
    return mean_edges, edge_pos_array

def get_sum_of_vertical_edges(binary_image):
    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];
    edges = 0

    for line in range(0, line_count-1):
        for col in range(0, col_count):
            if np.any(binary_image[line][col] == 255):
                if np.any(binary_image[line+1][col] < 255):
                    edges += 1
    return edges

    mean_edges, edge_pos_array = get_mean_number_of_vertical_edges(binary_image)
    line_array = []
    for line in edge_pos_array:
        for first_item in line:
            line_array.append(first_item)
            break
    sum_pos = sum(line_array)
    return sum_pos

def get_sum_of_diagonal_edges(binary_image):
    col_count = binary_image.shape[1];
    line_count = binary_image.shape[0];
    edges = 0

    for line in range(0, line_count-1):
        for col in range(0, col_count-1):
            if np.any(binary_image[line][col] == 255):
                if np.any(binary_image[line][col+1] < 255) and np.any(binary_image[line+1][col] < 255) and np.any(binary_image[line+1][col+1] < 255):
                    edges += 1
    return edges

    mean_edges, edge_pos_array = get_mean_number_of_vertical_edges(binary_image)
    line_array = []
    for line in edge_pos_array:
        for first_item in line:
            line_array.append(first_item)
            break
    sum_pos = sum(line_array)
    return sum_pos

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

    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                if binary_image[line_count + 1][col_count ] == 255:
                    edges = edges + 1
                    edge_pos_array.append([line_count, col_count])
            col_count = col_count + 1
            edge_array.append(edges)
            edges = 0
        line_count = line_count + 1
        col_count = 0
    mean_edges = sum(edge_array) / len(edge_array)
    return mean_edges, edge_pos_array

def get_sum_of_horizontal_edges(binary_image):
    return get_sum_of_vertical_edges( np.transpose(binary_image) )

    mean_edges, edge_pos_array = get_mean_number_of_horizontal_edges(binary_image)
    line_array = []
    for line in edge_pos_array:
        for first_item in line:
            line_array.append(first_item)
            break
    sum_pos = sum(line_array)
    return sum_pos

def get_sum_perimeter(binary_image):
    perimeter = 0
    binary = cv2.adaptiveThreshold(binary_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        perimeter += cv2.arcLength(contour,True)

    return perimeter

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

    for line in range(0, line_count):
        while col < (col_count-1):
            if np.any(binary_image[line][col] < 255):
                line += 1

                while (col + 1 < col_count) and np.any(binary_image[line][col+1] < 255):
                    cur_width += 1
                    col  += 1

                widths.append(cur_width)

    median_width = np.median( widths )

    wider = 0
    for width in widths:
        if width > median_width:
            wider += 1

    return wider / len(widths)

def get_vertical_sharpness(binary_image):
    return get_horizontal_sharpness( np.transpose(binary_image) )

def get_feature_vector(binary_image):
    feature_vector = []

    height = get_height(binary_image)
    width = get_width(binary_image)
    feature_vector.append( width/height )

    norm, _, black = get_num_of_pixels(binary_image)
    feature_vector.append( black/(width*height) )

    feature_vector.append( width / black )

    feature_vector.append(  get_mean_horizontal_position(binary_image) )

    feature_vector.append(  get_mean_vertical_position(binary_image) )

    feature_vector.append(  get_mean_squared_horizontal_position(binary_image) )

    feature_vector.append(  get_mean_squared_vertical_position(binary_image) )

    feature_vector.append(  get_sum_perimeter(binary_image) / black )

    vertical_edges = get_sum_of_vertical_edges(binary_image)
    feature_vector.append( vertical_edges / black )

    horizontal_edges = get_sum_of_horizontal_edges(binary_image)
    feature_vector.append( horizontal_edges / black )

    feature_vector.append( get_sum_of_diagonal_edges(binary_image) / black )

    feature_vector.append( get_sum_of_horizontal_doubles(binary_image) / black )

    feature_vector.append( get_sum_of_vertical_doubles(binary_image) / black )

    if horizontal_edges != 0:
        feature_vector.append( get_horizontal_sharpness(binary_image) / horizontal_edges )
    else:
        feature_vector.append( 0 )

    if vertical_edges != 0:
        feature_vector.append( get_vertical_sharpness(binary_image) / vertical_edges )
    else:
        feature_vector.append( 0 )

    ''' '''

    #feature_vector.append( get_wider_horizontals(binary_image) )

    #feature_vector.append( get_sum_of_vertical_singles(binary_image) / black )

    #feature_vector.append( get_sum_of_horizontal_singles(binary_image) / black )


    #feature_vector.append( get_sum_of_vertical_edges(binary_image) / get_sum_of_horizontal_edges(binary_image) )

    #feature_vector.append(  get_correlation_of_horizontal_variance(binary_image) / get_correlation_of_vertical_variance(binary_image) )

    #feature_vector.append(  get_correlation_of_horizontal_variance(binary_image) )

    #feature_vector.append(  get_correlation_of_vertical_variance(binary_image) )

    #feature_vector.append(  get_mean_number_of_vertical_edges(binary_image)[0] / black )

    #feature_vector.append(  get_sum_of_vertical_edges(binary_image)/(height*width) )

    #feature_vector.append(  get_mean_number_of_horizontal_edges(binary_image)[0] / black )

    #feature_vector.append(  get_sum_of_horizontal_edges(binary_image)/(height*width) )

    return feature_vector
