from collections import Iterable
from pathlib import Path


class ImageLoader:
    """
    Manages loading of all data
    """

    _img_base_dir: Path = None
    _font_names = []
    _img_count_per_font = []
    _img_count_global = 0

    def __init__(self, img_dir_path: str):
        self._img_base_dir = Path(img_dir_path)

        # Iterate over all font directories
        for (i, font_dir) in enumerate(self._img_base_dir.iterdir()):
            self._font_names.append(font_dir.stem)

            # Iterate over all image files and increase image counter
            self._img_count_per_font.append(0)
            for img in font_dir.iterdir():
                if img.is_file():
                    self._img_count_per_font[i] += 1

            # Append image count per font to global counter
            self._img_count_global += self._img_count_per_font[i]

    def iterate_images_for_fontname(self, font_name: str) -> Iterable:
        """
        Returns an iterator which iterates over all image Paths for the given font
        :param font_name: The name of the font
        :return: An iterator which iterates over all image Paths for the given font
        """
        font_dir: Path = self._img_base_dir.joinpath(font_name)
        return font_dir.iterdir()

    def get_img_count_for_font(self, font_name: str) -> int:
        """
        Returns the amount of images available for the given fontname
        :param font_name: The name of the font
        :return: A Int count of how much images are available for the font
        """
        for (i, fname) in enumerate(self._font_names):
            if font_name is fname:
                return self._img_count_per_font[i]

    def get_font_names(self) -> list:
        return self._font_names

    def get_font_count(self) -> int:
        return len(self._font_names)

    def get_image_count(self) -> int:
        return self._img_count_global
