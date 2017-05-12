import cv2
import os.path
import os
import extractGlyphes

# Read Image
localPath = os.path.dirname(os.path.realpath(__file__))
imgPath = os.path.join('..', 'images', 'arial', 'arial_0.png')
imgPath = os.path.join('..', 'images', 'waltograph_UI', 'waltograph_UI_0.png')
imgPath = os.path.join('..', 'images', 'times_new_romance', 'times_new_romance_0.png')
#imgPath = os.path.join('..', 'images', 'jokerman', 'jokerman_0.png')
#imgPath = os.path.join('..', 'images', 'forte', 'forte_0.png')
image = cv2.imread(imgPath)

os.listdir(localPath)
chars = []
chars = extractGlyphes.extract_glyphs(image)

for char in chars:
    cv2.imshow('a', char)
    cv2.waitKey(0)
