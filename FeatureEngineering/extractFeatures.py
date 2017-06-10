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
    horizontal_weight = sum(weight_array)
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
        line_count = line_count + 1
    vertical_weight = sum(weight_array)
    return vertical_weight

def get_mean_squared_horizontal_position(binary_image):
    res = get_mean_horizontal_position(binary_image)
    res = res * res
    return res

def get_mean_squared_vertical_position(binary_image):
    res = get_mean_vertical_position(binary_image)
    res = res * res
    return res

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
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                if binary_image[line_count][col_count + 1] == 255:
                    edges = edges + 1
                    edge_pos_array.append([line_count, col_count])
            col_count = col_count + 1
            edge_array.append(edges)
            edges = 0
        line_count = line_count + 1
        col_count = 0
    mean_edges = sum(edge_array) / len(edge_array)
    return mean_edges, edge_pos_array

def get_sum_of_vertical_edges(binary_image):
    mean_edges, edge_pos_array = get_mean_number_of_vertical_edges(binary_image)
    line_array = []
    for line in edge_pos_array:
        for first_item in line:
            line_array.append(first_item)
            break
    sum_pos = sum(line_array)
    return sum_pos

def get_mean_number_of_horizontal_edges(binary_image):
    line_count = 0
    col_count = 0
    edges = 0
    edge_array = []
    edge_pos_array = []
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
    mean_edges, edge_pos_array = get_mean_number_of_horizontal_edges(binary_image)
    line_array = []
    for line in edge_pos_array:
        for first_item in line:
            line_array.append(first_item)
            break
    sum_pos = sum(line_array)
    return sum_pos