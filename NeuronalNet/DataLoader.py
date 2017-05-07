from collections import Iterable
from pathlib import Path

''' Manages loading of all data '''
class DataLoader:

    _img_dir: Path = None
    _font_names = []
    _img_count = 0

    def __init__(self, img_dir_path: str):
        self._img_dir = Path(img_dir_path)

        # Iterate over all font directories
        for font_dir in self._img_dir.iterdir():
            self._font_names.append(font_dir.stem)

            # Iterate over all image file and increase image counter
            for img in font_dir.iterdir():
                if img.is_file():
                    self._img_count += 1

    def iterate_images_for_fontname(self, font_name: str) -> Iterable:
        font_dir: Path = self._img_dir.joinpath(font_name)
        return font_dir.iterdir()

    def get_fonts(self) -> list:
        return self._font_names

    def get_font_size(self) -> int:
        return len(self._font_names)

    def get_image_count(self) -> int:
        return self._img_count



