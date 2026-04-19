# hand-gesture-drawing-opencv
Real-time hand gesture drawing application using OpenCV and computer vision for finger tracking and gesture recognition.
📌 Projektbeschreibung

Dieses Projekt dient zur Erkennung von Handzeichen mithilfe von Machine Learning bzw. Computer Vision.
Ziel ist es, Handgesten über eine Kamera zu erfassen und automatisch einem bestimmten Zeichen oder einer Aktion zuzuordnen.
Das System kann z. B. verwendet werden für:
Gestensteuerung
Lernprojekte im Bereich KI
Mensch-Maschine-Interaktion
Echtzeit-Objekterkennung

🛠 Technologien
Python 3.x
OpenCV (Bildverarbeitung)
TensorFlow / PyTorch (falls ML verwendet wird)
NumPy
MediaPipe

🕹️ **Steuerung & Gesten**

- **Zeichnen:** Zeigefinger ausstrecken.
- **Radieren:** Alle Finger ausstrecken (offene Handfläche). Der Radierer orientiert sich an der Mitte der Handfläche und deren Größe.
- **Farbe wechseln:** Nur den kleinen Finger ausstrecken.
- **Strichdicke ändern:** Mit dem hochgestreckten Zeigefinger über den virtuellen Slider am _linken_ Bildschirmrand fahren und auf/ab wischen.
- **Leinwand löschen:** Taste `c` drücken.
- **Beenden:** Taste `q` drücken.
