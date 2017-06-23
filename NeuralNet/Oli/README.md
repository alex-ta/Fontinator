# Neural Networks for font-style recongnition (Autor O.Feucht)
Der folgendes Text bezieht sich auf das Modul unter dem Pfad "Fontinator/NeuralNet/Oli".
Dieses ermöglicht die klassifizierung der Bilder mithilfe von Neuronalen Netzen.
Dafür werden zwei Ansätze untersucht:
 * Fully Connected Neural Networks (Siehe Präsentationsfolien S. 21-24)
 * Convolutional Neural Networks

## Abhängigkeiten
Das Modul besitzt die folgenden externen Abhängigkeiten:
 * numpy
 * pandas
 * pillow
 * matplotlib
 * scipy
 * scikit-learn
 * theano
 * keras
 * h5py
 * graphviz
 * pydot-ng
 * keras
 * opencv
 
 In der Datei "Fontinator/Documents/HOW_TO_SETUP.md" wird die Installation der benötigten Python Umgebung erklärt\
 Hierfür wird [Anaconda](https://www.continuum.io/downloads) verwendet.

## Ausführbare Skripte
Dieser Abschnitt beschreibt alle ausführbaren Skripte im Modul.\
Alle ausfürbaren Skripte enthalten am Anfang einen kurzen Bereich zur Konfiguration.\
Alle Skripte informieren den Benutzer in der Console über die aktuellen Arbeitsschritte.\
Die Skripte können direkt in der Konsole aufgerufen werden => "python \<skriptname>".\

### trainNN
Das Skript "trainNN.py" startet das Trainieren eines Fully Connected Neural Networks.

### evaluateNN
Das Skript "evaluateNN.py" erlaubt die Evaluation eines mit "trainNN.py" erzeugten Models.

### trainCNN
Das Skript "trainCNN.py" startet das Trainieren eines Convolutional Neural Networks.

### evaluateNN
Das Skript "evaluateCNN.py" erlaubt die Evaluation eines mit "trainCNN.py" erzeugten Models.

## Abgespeicherte Modelle

### Download
Da die vortrainierten Keras-Modelle eine Größe von 200-500mb besitzen, sind diese nicht im Repository enthalten.
Die vortrainierten Modelle können unter folgendem
[Link](https://www.oliver-feucht.de/nextcloud/s/OrPDp7G2uxLo5av) gefunden werden.
Die Keras-Modelle sollten in dem Ordner "Fontinator/NeuralNet/SavedModels" abgespeichert werden.

### Aufbau
Für jedes Model gibt es einen eigenen Unterordner (z.B. LT2, CNN_RAND_80).
Für jedes Model wurden die Struktur und die Gewichtungen des Netzes abgespeichert.\
Zusätzlich sind noch weitere Metadaten verfügbar:
 * Ein Diagram, das anzeigt wie sich die Accuracy beim Trainieren verhalten hat.
 * SVG- und PNG-Grafiken, die den Aufbau des Models zeigen.
 * JSON-Datei, die das Label Mapping gespeichert hat
 * CSV-Datei, die jegliche Informationen über das Training des Models enthält (epoch,val_loss,val_acc,loss,acc,tdiff)

## Hilfsklassen
Im Ordner "libs" befinden sich alle von diesem Modul verwendeten Hilfsklassen.

### ModelSerializer
Die Klasse übernimmt das Abspeichern und Laden eines Keras-Models von der Festplatte.

### ImageLoader
Die Klasse unterstützt beim Laden aller Bilddaten.

### TrainingLogger
Die Klasse protokolliert alle wichtigen Informationen, die beim Training eines Keras-Models anfallen.
Die protokollierten Informatioen werden in einem Pandas-Dataframe gespeichert.
Dieses kann auch in einer CSV-Datei abgespeichert werden.
Außerdem können auch direkt einige aussagekräftige Diagramme erzeugt werden.

### ProcessingPipeline
Die Klasse managt den kompletten Lifecycle eines Keras-Models.

Dazu gehören:
 * Laden der Bilder und Labels
 * Preprocessing
 * Laden eines Models aus einer Datei
 * Trainieren eines Models
 * Abspeichern von Struktur und der Gewichtungen eines Models
 * Prediction
 * Evaluierung von Bildern
 
### Preprocessor
Enthält mehrere Preprocessor-Klassen, welche die Vorverarbeitung der Bilder für das Neuronale Netz übernehmen.

Der "SimplePreprocessor" übernimmt die Vorverarbeitung eines Bildes für ein Fully Connected NN.\
Dazu gehören die Schritte:
 * Binarisierung
 * Flattening
 
Der "ConvPreprocessor" übernimmt die Vorverarbeitung eines Bildes für ein Convolutional NN.
Dabei wird das Bild nur binarisiert.
