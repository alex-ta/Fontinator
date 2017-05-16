import cv2
import os.path
import os
import extractGlyphes

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
                if i == 1:
                    break

#image_path_font_list = [['/Users/Sebastian/Desktop/Master/IDA_Projekt/Repo/Fontinator/images/canterbury/canterbury_101.png', 'a']]
#image_path_font_list = [['/Users/Sebastian/Desktop/Master/IDA_Projekt/Repo/Fontinator/images/jokerman/jokerman_0.png', 'a']]

for image_path_font in image_path_font_list:
    print(image_path_font[0])
    image = cv2.imread(image_path_font[0])

    glyphs = extractGlyphes.extract_glyphs(image)

    for glyph in glyphs:
        cv2.imshow('a', glyph)
        cv2.waitKey(0)
