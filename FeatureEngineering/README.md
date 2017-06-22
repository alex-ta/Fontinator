# Fontinator - Feature Engineering

Das Unterprojekt Fontinator Feature Engineering identifiziert, wie das Gesamtprojekt Fontinator, die in einem Bild verwendete Schriftart. Dabei setzt das Unterprojekt auf klassische Ansätze, wie die Extraktion von bestimmten Features, welche für die anschließende Klassifikation der Schriftart benötigt wird.

Nachfolgend werden die Abhängigkeiten sowie die Verwendung des Systems aufgezeigt. Anschließend erfolgt eine kurze Beschreibung des Systemkonzepts samt Funktionsweise.

## Abhängigkeiten

Im Folgenden werden die externen Abhängigkeiten anhand der Entwicklungsumgebung kurz aufgezeigt:

* Python 3.6
* OpenCV 3.2
* Numpy 1.12.1
* PIL 4.1.1
* scikit-learn 0.18.1

## Verwendung

Die folgenden Skripte bilden die Hauptkomponenten des Programms:

* *trainer.py* - Erstellen der Trainings- und Labeldaten samt Training des Klassifikators
* *test.py* - Test des Programms mit vorgegebenen Bilddaten
* *Fontinator.py* - Schriftartenklassifikation eines Bildes

### *trainer.py*

Zunächst erfolgt das Training des Klassifikators. Dazu wird das Skript *trainer.py* ausgeführt. Hierzu werden alle Schriftarten, welche als *.ttf*- Dateien, in dem Ordner `Fontinator/DataGenerator/fonts` vorliegen, herangezogen. Das verwendete Glyphenset kann im Skript *trainer.py* mithilfe der Variable `TRAIN_CHARS` modifiziert werden. Die ezeugten Parameter des Klassifiaktors werden in der Datei `classie.pickle` gepseichert. Die Datei `labels.pickle` enthält die zugehörigen Labels und erlaubt somit die spätere Zuordnung eines Klassifikationsergebnisses zu dem entsprechenden Namen der Schriftart.

### *test.py*

Nach dem Training kann das System mithilfe des Skriptes *test.py* überprüft werden. Hierbei werden in der aktuellen Konfiguration die Bilder des Ordners `Fontinator/TestSets/images/Dataset_1` verwendet. Der Pfad zum Datenset kann in dem Skript *test.py* mithilfe der Variable `images_path` angepasst werden. Dabei muss die im folgenden Dargestellte Ordnerstruktur eingehalten werden.

```
images_path
├── FORTE
│   ├── forte_0.png
│   ├── forte_1.png
│   ├── ...
├── arial
│   ├── arial_0.png
│   ├── arial_1.png
│   ├── ...
├── ...
```

Wichtig ist dabei, dass die Ordner, welche die Bilder enthalten entsprechend der zugehörigen Schriftart benannt sind.

Dabei wird für jedes Bild die Schriftart, welche erkannt wurde, ausgegeben. Zusätzlich wird jede erkannte Glyphe des Bildes individuell einer Schriftart zugeordnet. Wurden alle Bilder untersucht, so wird die Genauigkeit der Ergebnisse für die einzelnen Glyphen selbst, sowie für das Gesamtergebniss der Bilder ausgegeben.

### *Fontinator.py*

Ist das Training abgeschlossen kann das Skript *Fontinator.py* für die Klassifikation der Schriftart eines Bildes verwendet werden. Dazu wird das Skript mit Python aufgerufen und der Pfad zum Bild übergeben.

`python Fontinator.py /Pfad/zum/Bild`

Das Skript liefert eine absteigende Liste mit den Wahrscheinlichkeiten der erkannten Schriftarten.

## Systemkonzept

Das Verarbeitungskonzept kann in mehrere Schritte unterteilt werden, welche für jede Verarbeitung eines Bildes durchgeführt werden:

* Extraktion einzelner Glyphen aus dem Text
* Extraktion von Features aus den einzelnen Glyphen
* Klassifikation anhand der extrahierten Features

Da die Bilddaten erzeugt werden und damit einen idealen Datensatz darstellen, können auf größere Vorverarbeitungsschritte zur Bildverbesserung verzichtet werden.

Zusätzlich ist das Training des Klassifikators ein weiterer Schritt, welcher einmalig für die Verwendung des Programms durchgeführt werden muss.

### Glyphenextraktion

Zunächst wird der im Bild vorhandene Text in einzelne Glyphen zerlegt. Eine Glyphe stellt dabei einen Buchstaben, eine Ziffer, ein Satzzeichen oder ähnliche Textartefakte dar. Hierzu werden zunächst die typografischen Eigenschaften nach dem Vierliniensystem berechnet. Die Linien beschreiben die Ober- und Unterkante von Kleinbuchstaben ohne Oberlänge (z.B. m, o, a). Eine  weitere Linie beschreibt die Oberkante von Großbuchstaben, sowie von Kleinbuchstaben mit Oberlänge (l, ö, i). Die vierte Linen beschreibt die Unterkannte von Buchstaben mit Unterlänge (z.B. g, p, q). [[Wikipedia Liniensystem](https://de.wikipedia.org/wiki/Liniensystem_(Typografie)] 

![Liniensystem](https://upload.wikimedia.org/wikipedia/commons/3/39/Typography_Line_Terms.svg)

*Bildquelle: [Wikipedia](https://en.wikipedia.org/wiki/Baseline_(typography)*

Anschließend werden die gefundenen Linien für die eigentliche Glyphenextraktion verwendet. Hierbei werden zunächst zusammenhängende Artefakte mithilfe der OpenCv Methode [findContours](http://docs.opencv.org/3.2.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a) identifiziert. Die gefundenen Artefakte stellen dabei entweder eine gesamte Glyphe oder einen Teil einer Glyphe dar. Die gefundenen typografischen Linien helfen dabei, einzelne Artefakte wie z.B. i-Punkte dem zugehörigen Artefakt zuzuordnen und somit die Glyphe zu vervollständigen. Die vervollständigten Glyphen werden anschließend an den nächsten Verarbeitungsschritt, die Featureextraktion, weitergereicht.

### Featureextraktion

Bei der Featureextraktion werden markante Merkmale (Features) aus den einzelnen Glyphen ermittelt. Entscheidend ist es hierbei Features zu wählen, welche die individuellen Charakteristiken der Schriftarten beschreiben und somit eine Differenziation zwischen diesen erlauben. Als Startpunkt wurden hierbei einige Features nach [Andrew Bray](https://github.com/andrewpbray/lab-7/blob/master/lab-7.Rmd) implementiert und durch weitere eigens definierte Features ergänzt. Hierbei werden die im folgenden vorgestellten Features verwendet:

* Anzahl dunkler Pixel
* Länge des Umfangs aller Komponenten einer Glyphe
* Anzahl der Pixel der skelletierten Glyphe
* Mittlere horizontale Position alle dunklen Pixel
* Mittlere vertikale Position aller dunklen Pixel
* Anzahl der horizontalen Kanten im Bild
* Anzahl zusammenhängender Komponenten
* Anzahl von Löcher (z.B. innerer Kreis im Buchstaben O)
* Horizontale Varianz der Position aller dunklen Pixel
* Vertikale Varianz der Position aller dunklen Pixel

Die ermittelten Features werden anschließend mithilfe von Merkmalen, welche die Größe der Glyphen beschreiben, normiert und sind somit weitestgehend unabhängig von der Schriftgröße. Diese Features bilden anschließend den Featurevektor, welcher für die Klassifikation verwendet wird.

### Klassifikation

Mithilfe des gefundenen Featurevektors erfolgt anschließend die Klassifikation der Schriftart. Um den Einfluss aller Features gleichzuhalten, wird der Featurevektor auf den Bereich Null bis Eins normiert.

Für die Klassifikation kommt ein [One-Nearest-Neighbor](http://scikit-learn.org/stable/modules/neighbors.html) Klassifikator von scikit-learn zum Einsatz.

Hierbei wird jede Glyphe einzeln klassifiziert und anschließend mitihlfe aller Glyphen eines Textes eine Mehrheitsabstimmung für die Schriftart durchgeführt.

### Training

Das Training des Klassifikators erfolgt mithilfe von generierten Bilder der einzelnen Buchstaben einer jeden Schriftart. Hierbei wird somit für ein einfacheres Labeling der Trainingsdaten auf die Glyphenextraktion verzichtet. Anschließend erfolgt die Featureextraktion, wie bereits oben beschrieben. Mithilfe der erzeugten Bild- und Labeldaten erfolgt das Training des Klassifikators, sowie die Berechnung der für die Normalisierung notwendigen Parameter. Anschließend werden alle Parameter für die spätere Verwendung abgespeichert.



### Auswertung

Abschließend kann das Programm auf seine Funktionsweise getestet werden. Dieses geschieht entweder über das Skript `Fontinator.py` oder über das Test-Skript `test.py` (wie bereits oben beschrieben). Aus dem jeweiligen Bild werden folglich die Glyphen extrahiert, sowie daraus die Features und der Featurevektor ermittelt. Der Klassifikator vergleicht die gefundenen Features mit den bereits antrainierten Features und weist jedem Bild eine Schriftart zu.
