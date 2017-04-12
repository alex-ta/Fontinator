import os
import sys
import os.path
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageOps
import textwrap

# Textfile containing sentence
input_file="inText.txt"
# Font to generate
font_type = "files/Aaargh.ttf"
# Image default name
img_name = "sample"
# height of the images
height = 200
# width of the images
width = 800
# border to top left (drawing start)
border = 10
# counter to label sample count
count = 0

# read commandline arguments
arg_len = len(sys.argv)
print(arg_len)
if arg_len >= 2:
    input_file = sys.argv[1]
elif arg_len >= 3:
    font_type = sys.argv[2]
elif arg_len >= 4:
    img_name = sys.argv[3]
elif arg_len >= 5:
    width = sys.argv[4]
elif arg_len >= 6:
    height = sys.argv[5]
else:
    print("Expected arguments: 1) file with sentence 2) font file path 3) img name 4) img width 5) img height")

# converts string to image
def to_img(txt, name, style):
    global count
    global width
    global height
    global border
    draw = Image.new("RGBA", (width, height), (255,255,255))
    text_img = ImageDraw.Draw(draw)
    font = ImageFont.truetype(style, 20)
    # converts text to array of width matching strings
    lines = textwrap.wrap(txt, width=width/11)
    _height = border
    for line in lines:
        w, h = font.getsize(line)
        text_img.text((border, _height), line+".", (0,0,0), font=font)
        _height += h
    count += 1
    draw.save(name+str(count)+".jpg")

# read file and add sentences to array
array = open(input_file,"r").read().replace("\n","").replace(". ",".").split(".")
file_name = font_type.rsplit('.',1)[0]
os.makedirs(file_name)
for sentence in array:
    #create image from array
    to_img(sentence, str(os.path.join(file_name, img_name)), font_type)
