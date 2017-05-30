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
    line_count = 0
    col_count = 0
    pos_array = []
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                #falls es eine weitere zeile gibt, in welcher der col_count kleiner ist als der bisher gefundene, müsste man den col_count nehmen!
                pos_array = [line_count, col_count]
                return pos_array
            col_count = col_count + 1
        line_count = line_count + 1
        col_count = 0
    return [0,0]

def get_vertical_position(binary_image):
    line_count = 0
    col_count = 0
    pos_array = []
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                pos_array = [line_count, col_count]
                break
            col_count = col_count + 1
        line_count = line_count + 1
        col_count = 0
    return pos_array

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
    #total, white, black = get_num_of_pixels(binary_image)
    width = get_width(binary_image)
    left_edge_pos = get_left_edge_pos(binary_image)
    center = (width / 2)
    mod = center % 2
    center_left = 0
    center_right = 0
    if mod > 0:
        center_left = center - mod - 1 + left_edge_pos
        center_right = center + mod + left_edge_pos
    else:
        center_left = center + left_edge_pos
        center_right = center + 1 + left_edge_pos
    weight = 0
    weight_array =  []
    for line in binary_image:
        col_count = 0
        for pixel in line:
            if np.any(pixel < 255):
                if col_count <= center_left:
                    weight = (col_count - center_left) / width
                    weight_array.append(weight)
                elif col_count >= center_right:
                    weight = (col_count - center_right) / width
                    weight_array.append(weight)
                else:
                    weight = 0
                    weight_array.append(weight)
            col_count = col_count + 1
        col_count = 0
    horizontal_weight = sum(weight_array)
    return horizontal_weight

def get_mean_vertical_position(binary_image):
    # total, white, black = get_num_of_pixels(binary_image)
    height = get_height(binary_image)
    top_edge_pos = get_top_edge_pos(binary_image)
    center = (height / 2)
    mod = center % 2
    center_top = 0
    center_bottom = 0
    if mod > 0:
        center_top = center - mod - 1 + top_edge_pos
        center_bottom = center + mod + top_edge_pos
    else:
        center_top = center + top_edge_pos
        center_bottom = center + 1 + top_edge_pos
    weight = 0
    weight_array = []
    line_count = 0
    for line in binary_image:
        for pixel in line:
            if np.any(pixel < 255):
                if line_count <= center_top:
                    weight = (line_count - center_top) / height
                    weight_array.append(weight)
                elif line_count >= center_bottom:
                    weight = (line_count - center_bottom) / height
                    weight_array.append(weight)
                else:
                    weight = 0.0
                    weight_array.append(weight)
        line_count = line_count + 1
    vertical_weight = sum(weight_array)
    return vertical_weight

