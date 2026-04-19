"""
Modul 1 – Frame Acquisition
Öffnet die Webcam, liest Frames und spiegelt sie horizontal.
"""

import cv2


class Camera:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        """Initialisiert die Kamera mit gewünschter Auflösung."""
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("Kamera konnte nicht geöffnet werden!")

        # Tatsächliche Auflösung auslesen
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self):
        """Liest einen Frame und spiegelt ihn horizontal (Spiegel-Effekt).

        Returns:
            frame (ndarray | None): Das gespiegelte Bild oder None bei Fehler.
        """
        success, frame = self.cap.read()
        if not success:
            return None
        # Horizontal spiegeln für natürliche Interaktion
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        """Gibt die Kamera-Ressourcen frei."""
        self.cap.release()
