########################################################################
###     Skript liest Fonts ein und erzeugt Trainings- und Testbilder ###
###     Configuration erfolgt in "config.py"                         ###
########################################################################
import random
from pathlib import Path

import shutil
from PIL import Image, ImageFont, ImageDraw
from DataGenerator.libs.WordDict import *
import DataGenerator.config as cfg

fonts_dir: Path = Path(cfg.FONT_INPUT_PATH)
img_dir: Path = Path(cfg.IMG_OUTPUT_PATH)

# Validation of config data
if not fonts_dir.exists():
    raise Exception("Path '{}' does not exists".format(fonts_dir))

if not Path(cfg.TEXT_INPUT_FILE_PATH).exists():
    raise Exception("Path '{}' does not exists".format(cfg.TEXT_INPUT_FILE_PATH))

if cfg.IMAGE_COUNT < 1:
    raise Exception("IMAGE_COUNT must be 1 oder greater")

# Remove old created images
if img_dir.exists():
    for font_img_dir in img_dir.iterdir():
        shutil.rmtree(str(font_img_dir))
else:
    # Create img dir
    img_dir.mkdir(parents=True)

# Loading data from file into dict for sentence generation
word_dict = WordDict.load_from_textfile(cfg.TEXT_INPUT_FILE_PATH)
print("Loaded {0} words ...".format(word_dict.get_word_count()))

# Iterate for all available fonts
for font_file in fonts_dir.iterdir():
    print("Creating images for font '{}'".format(font_file.stem))

    # Create path to directory for images with this font
    img_font_dir: Path = img_dir.joinpath(Path(font_file.stem))

    # Create dir for font images.
    if not img_font_dir.exists():
        img_font_dir.mkdir()

    # Create multiple images for each font
    for i in range(cfg.IMAGE_COUNT):

        # Create font
        font_size = random.randint(cfg.FONT_SIZE_MIN, cfg.FONT_SIZE_MAX)
        font = ImageFont.truetype(str(font_file), font_size)

        # Create raw image
        image = Image.new("RGBA", (cfg.IMG_WIDTH, cfg.IMG_HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Draw random sentence to image
        word_count = random.randint(cfg.WORD_COUNT_MIN, cfg.WORD_COUNT_MAX)
        text = word_dict.get_sentence(word_count)

        padding_left = random.randint(cfg.PADDING_LEFT_MIN, cfg.PADDING_LEFT_MAX)
        padding_top = random.randint(cfg.PADDING_TOP_MIN, cfg.PADDING_TOP_MAX)
        draw.text((padding_left, padding_top), text, (0, 0, 0), font=font)

        # Save file in folder for used font
        file_name = "{}_{}.png".format(font_file.stem, i + cfg.START_NUM)
        img_path = img_font_dir.joinpath(file_name)
        image.save(img_path)
