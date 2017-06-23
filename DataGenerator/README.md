# Data Generator
Der folgendes Text bezieht sich auf das Modul unter dem Pfad "Fontinator/DataGenerator".
Das Programm-Modul ermöglicht die Erzeugung von Trainings- und Testbildern für den Fontinator.
Die Bilder bestehen aus einem einzeiligen Text mit unterschiedlichen Schriftstilen (Fonts).
Der erzeugte Text wird zufällig erzeugt.
Für eine nähere Beschreibung siehe Präsentationfolien (S. 5-8)

## Abhängigkeiten
Das Modul besitzt die folgenden externen Abhängigkeiten:
 * PIL

## Konfiguration:
Die Konfiguration erfolgt in der Datei 'config.py'.
Es kann beispielsweise die Zahl der zu erzeugenden Bilder oder die Fontgröße festgelegt werden.
Die zu verwendenden Fonts werden standardmäßig aus dem Ordner 'fonts' gelesen.
Die Fonts müssen im .ttf Format vorliegen.
Die erzeuten Bilderdaten werden standardmäßig im Ordner 'images' abgelegt.
Im Ordner "text_res" befinden sich einige Textdateien, welche die Wörter für die Zufallssätze enthalten.

## Ausführung
Zum Starten muss das Skript "createImages.py" ausgeführt werden.
Es werden alle Fonts eingelesen und die Trainings- und Testbilder erzeugt.

## Hilfsklassen
In dem Ordner "libs" befinden sich alle vom Data Generator verwendeten Hilfsklassen.

### WordDict
Die Klasse "WordDict" ermöglicht die Erzeugung von Zufallssätzen.
Die Zufallswörter können aus einer Textdatei eingelesen werden.