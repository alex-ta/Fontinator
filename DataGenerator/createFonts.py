########################################################################
###     Skript liest Fonts ein und erzeugt Trainings- und Testbilder ###
###     Eingabefonts        -> Order 'fonts'
###     Erzeute Bilderdaten -> Ordner 'images'
########################################################################

from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from libs.WordDict import *

# Pfad zur Eingabedatei, aus der Zufallssätze erzeugt werden
input_file_path = "inputText.txt"
# Pfad zum Order der alle fonts enthält
fonts_dir: Path = Path("fonts")
# Zu erzeugende images pro font
image_count = 3
# Pfad zum Order der alle erzeugten images enthält
img_dir: Path = Path("images")

word_dict = WordDict.load_from_textfile(input_file_path)

# Iterate for all available fonts
for font_file in fonts_dir.iterdir():

    # Create path to directory for images with this font
    img_font_dir: Path = img_dir.joinpath(Path(font_file.stem))

    # Create dir for font images.
    if not img_font_dir.exists():
        img_font_dir.mkdir()
    # If folder exists remove all img_files
    else:
        for img_file in img_font_dir.iterdir():
            img_file.unlink()

    # Create multiple images for each font
    for i in range(image_count):
        # Create raw image
        image = Image.new("RGBA", (1200, 30), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Draw random sentence to image
        font = ImageFont.truetype(str(font_file), 25)
        text = word_dict.get_sentence(10)
        draw.text((10, 0), text, (0, 0, 0), font=font)

        # Save file in folder for used font
        file_name = "{}_{}.png".format(font_file.stem, i)
        img_path = img_font_dir.joinpath(file_name)
        image.save(img_path)
