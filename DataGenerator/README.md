# Data Generator
Das Programm ermöglicht die Erzeugung von Trainings- und Testbildern für den Fontinator.
Die Bilder bestehen aus einem einzeiligen Text mit unterschiedlichen Schriftstilen (Fonts). 

## Konfiguration:
Die Konfiguration erfolgt in der Datei 'config.py'.
Es kann beispielsweise die Zahl der zu erzeugenden Bilder festgelegt werden.
Die zu verwendenden Fonts werden standardmäßig aus dem Ordner 'fonts' gelesen.
Die Fonts müssen im .ttf Format vorliegen.
Die erzeuten Bilderdaten werden standardmäßig im Ordner 'images' abgelegt.

## Starten
Ausführen des Skripts "createImages.py".
werden alle Fonts eingelesen und Trainings- und Testbilder erzeugt.