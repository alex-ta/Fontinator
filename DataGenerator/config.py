
# Zu erzeugende images pro Font
IMAGE_COUNT = 500

# Erster Index im Name des erzeugten Bildes
START_NUM = 0

# Name der Datei, die Text für Zufallssätze enthält
INPUT_TXT_FILE = "text_zfs"

# Pfad zur Eingabedatei, aus der Zufallssätze erzeugt werden
TEXT_INPUT_FILE_PATH = "text_res/" + INPUT_TXT_FILE + ".txt"

# Pfad zum Order der alle Fonts enthält (Fonts sollten im ttf-Format sein)
FONT_INPUT_PATH = "fonts"

# Pfad zum Order für die erzeugten images
IMG_OUTPUT_PATH = "images/" + INPUT_TXT_FILE

# Breite und Höhe der erzeugten Bilder
IMG_WIDTH = 1200
IMG_HEIGHT = 40

# Minimale und maximale Anzahl der Wörter pro Satz
WORD_COUNT_MAX = 8
WORD_COUNT_MIN = 1

# Minimale und maximale Schriftgröße für jeden erzeugten Satz
FONT_SIZE_MIN = 14
FONT_SIZE_MAX = 26

# Minimaler und maximaler Rand nach oben und unten
PADDING_TOP_MIN = 0
PADDING_TOP_MAX = 10
PADDING_LEFT_MIN = 10
PADDING_LEFT_MAX = 300