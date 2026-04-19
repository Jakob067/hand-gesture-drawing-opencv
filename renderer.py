"""
Modul 6 – Rendering Engine
Kombiniert Kamerabild mit Canvas und zeichnet ein HUD-Overlay.
"""

import cv2
import numpy as np


class Renderer:
    def __init__(self):
        """Initialisiert den Renderer."""
        self.window_name = "Gesture Drawing"

    def render(self, frame, canvas, color: tuple, thickness: int,
               mode: str, fps: float, landmarks=None):
        """Kombiniert Frame und Canvas und zeigt das Ergebnis an.

        Args:
            frame: Kamerabild (BGR).
            canvas: Zeichenfläche (BGR, schwarzer Hintergrund).
            color: Aktuelle Zeichenfarbe (BGR).
            thickness: Aktuelle Strichdicke.
            mode: Aktueller Modus-Name.
            fps: Aktuelle Bildrate.
            landmarks: Optionale Landmark-Daten für Debugging.

        Returns:
            combined: Das kombinierte Bild.
        """
        # Canvas über Frame legen (Additiv-Blending)
        # Nur dort wo Canvas nicht schwarz ist
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Frame-Anteil dort wo Canvas Farbe hat abdunkeln
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame_dimmed = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0, 0)
        frame_dimmed_masked = cv2.bitwise_and(frame_dimmed, frame_dimmed, mask=mask)

        # Canvas-Anteil
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

        # Zusammensetzen
        combined = cv2.add(frame_bg, frame_dimmed_masked)
        combined = cv2.add(combined, canvas_fg)

        # HUD zeichnen
        self._draw_hud(combined, color, thickness, mode, fps)

        return combined

    def _draw_hud(self, image, color: tuple, thickness: int, mode: str, fps: float):
        """Zeichnet das Head-Up Display.

        Args:
            image: Bild auf dem das HUD gezeichnet wird.
            color: Aktuelle Farbe für den Farbindikator.
            thickness: Aktuelle Strichdicke.
            mode: Aktueller Modus.
            fps: FPS-Wert.
        """
        h, w = image.shape[:2]

        # Halbtransparenter Hintergrund für HUD (oben links)
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (280, 140), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Rahmen
        cv2.rectangle(image, (10, 10), (280, 140), (80, 80, 80), 1)

        # Modus-Anzeige
        mode_colors = {
            "IDLE": (150, 150, 150),
            "DRAW": (50, 255, 50),
            "ERASE": (50, 50, 255),
            "THICKNESS_ADJUST": (0, 255, 255),
            "COLOR_SWITCH": (255, 0, 255),
        }
        mode_color = mode_colors.get(mode, (255, 255, 255))
        cv2.putText(image, f"Modus: {mode}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Farb-Indikator
        cv2.putText(image, "Farbe:", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(image, (90, 50), (130, 70), color, -1)
        cv2.rectangle(image, (90, 50), (130, 70), (200, 200, 200), 1)

        # Dicke-Anzeige
        cv2.putText(image, f"Dicke: {thickness}px", (20, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Visuelle Dicke-Vorschau
        bar_start = 140
        bar_end = bar_start + int((thickness / 30) * 120)
        cv2.rectangle(image, (bar_start, 80), (260, 95), (60, 60, 60), -1)
        cv2.rectangle(image, (bar_start, 80), (bar_end, 95), color, -1)

        # FPS
        cv2.putText(image, f"FPS: {fps:.0f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Steuerungshinweise (unten)
        help_y = h - 20
        cv2.putText(image, "Zeigefinger=Zeichnen | Alle Finger=Radieren | Kleiner Finger=Farbe | q=Beenden | c=Loeschen",
                    (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # --- Virtueller Slider ---
        slider_x = 20
        slider_y = 160
        slider_w = 40
        slider_h = 300

        # Hintergrund
        cv2.rectangle(image, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (40, 40, 40), -1)
        cv2.rectangle(image, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (200, 200, 200), 1)

        # Thumb berechnen (Dicke von 30 bis 2)
        ratio = (30 - thickness) / 28.0
        thumb_y = slider_y + int(ratio * slider_h)
        thumb_radius = max(5, thickness // 2) + 5

        # Thumb zeichnen
        cv2.circle(image, (slider_x + slider_w // 2, thumb_y), thumb_radius, color, -1)
        cv2.circle(image, (slider_x + slider_w // 2, thumb_y), thumb_radius, (255, 255, 255), 2)

        # Label
        cv2.putText(image, "Dicke", (slider_x - 3, slider_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def show(self, image):
        """Zeigt das Bild im Fenster an.

        Args:
            image: Das anzuzeigende Bild.
        """
        cv2.imshow(self.window_name, image)

    def destroy(self):
        """Schließt alle OpenCV-Fenster."""
        cv2.destroyAllWindows()
