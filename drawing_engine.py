"""
Modul 5 – Action Execution
Verwaltet den Zeichenzustand und führt Aktionen aus.
Nutzt die Handfläche als Radiergummi mit dynamischer Größe.
"""

import cv2
import numpy as np


# Farbpalette (BGR-Format für OpenCV)
COLOR_PALETTE = [
    (255, 50, 50),    # Blau
    (50, 50, 255),    # Rot
    (50, 255, 50),    # Grün
    (0, 255, 255),    # Gelb
    (255, 0, 255),    # Magenta
    (255, 165, 0),    # Orange
    (255, 255, 255),  # Weiß
]

# Dicke-Grenzen
MIN_THICKNESS = 2
MAX_THICKNESS = 30
DEFAULT_THICKNESS = 5


class DrawingEngine:
    def __init__(self, width: int, height: int):
        """Initialisiert die Zeichen-Engine.

        Args:
            width: Breite des Canvas.
            height: Höhe des Canvas.
        """
        self.width = width
        self.height = height

        # Schwarzes Canvas (transparenter Hintergrund bei Overlay)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Zeichenzustand
        self.color_index = 0
        self.color = COLOR_PALETTE[self.color_index]
        self.thickness = DEFAULT_THICKNESS
        self.prev_position = None
        self.mode = "IDLE"

    def execute(self, gesture: str, fingertip: tuple[int, int] | None = None,
                landmarks=None):
        """Führt die dem Gesture entsprechende Aktion aus.

        Args:
            gesture: Aktueller Systemzustand (DRAW, ERASE, etc.).
            fingertip: (x, y) Position der Zeigefingerspitze.
            landmarks: Alle 21 Landmarks für Palm-Berechnung.
        """
        self.mode = gesture

        if gesture == "DRAW":
            if self._is_on_slider(fingertip):
                self._adjust_thickness_from_slider(fingertip)
            else:
                self._draw(fingertip)
        elif gesture == "ERASE":
            self._erase_with_palm(landmarks)
        elif gesture == "COLOR_SWITCH":
            self._switch_color()
        else:
            # IDLE – Position zurücksetzen
            self.prev_position = None

    def _draw(self, fingertip: tuple[int, int] | None):
        """Zeichnet eine Linie zum Fingertip.

        Args:
            fingertip: (x, y) aktuelle Fingerposition.
        """
        if fingertip is None:
            self.prev_position = None
            return

        if self.prev_position is not None:
            cv2.line(self.canvas, self.prev_position, fingertip,
                     self.color, self.thickness, lineType=cv2.LINE_AA)

        self.prev_position = fingertip

    def _erase_with_palm(self, landmarks):
        """Radiert mit der Handfläche — nutzt die tatsächliche Handflächenform.

        Die Handfläche wird als Polygon aus den MCP-Gelenken und dem
        Handgelenk gebildet. Alles innerhalb dieses Polygons wird radiert.

        Args:
            landmarks: Alle 21 Landmarks oder None.
        """
        self.prev_position = None

        if landmarks is None or len(landmarks) < 21:
            return

        # Handflächenpolygon: Handgelenk (0) + MCP-Gelenke (5, 9, 13, 17)
        palm_indices = [0, 5, 9, 13, 17]
        palm_points = np.array(
            [(landmarks[i][0], landmarks[i][1]) for i in palm_indices],
            dtype=np.int32
        )

        # Handflächen-Zentrum berechnen
        cx = int(np.mean(palm_points[:, 0]))
        cy = int(np.mean(palm_points[:, 1]))

        # Dynamischen Radius berechnen (basierend auf Handgröße)
        # Abstand von Handgelenk zu Mittelfinger-MCP als Referenz
        wrist = np.array([landmarks[0][0], landmarks[0][1]])
        middle_mcp = np.array([landmarks[9][0], landmarks[9][1]])
        palm_size = np.linalg.norm(middle_mcp - wrist)
        erase_radius = int(palm_size * 0.6)
        erase_radius = max(30, min(120, erase_radius))  # Klemmen

        # Großen schwarzen Kreis zeichnen (radiert auf schwarzem Canvas)
        cv2.circle(self.canvas, (cx, cy), erase_radius, (0, 0, 0), -1)

    def _switch_color(self):
        """Wechselt zur nächsten Farbe in der Palette."""
        self.prev_position = None
        self.color_index = (self.color_index + 1) % len(COLOR_PALETTE)
        self.color = COLOR_PALETTE[self.color_index]

    def _is_on_slider(self, fingertip: tuple[int, int] | None) -> bool:
        """Prüft, ob die Fingerspitze über dem virtuellen Slider liegt."""
        if not fingertip:
            return False
        x, y = fingertip
        slider_x = 20
        slider_y = 160  # Platziert unterhalb des HUDs
        slider_w = 40
        slider_h = 300
        
        # Etwas Toleranz-Bereich (Padding) für leichtere Bedienung
        if slider_x - 20 <= x <= slider_x + slider_w + 30:
            if slider_y - 30 <= y <= slider_y + slider_h + 30:
                return True
        return False

    def _adjust_thickness_from_slider(self, fingertip: tuple[int, int]):
        """Passt die Strichdicke basierend auf der Y-Position am Slider an."""
        _, y = fingertip
        slider_y = 160
        slider_h = 300
        
        clamped_y = max(slider_y, min(slider_y + slider_h, y))
        ratio = (clamped_y - slider_y) / slider_h
        
        # Oben = MAX_THICKNESS, Unten = MIN_THICKNESS
        thickness_range = MAX_THICKNESS - MIN_THICKNESS
        new_thickness = MAX_THICKNESS - int(ratio * thickness_range)
        
        self.thickness = max(MIN_THICKNESS, min(MAX_THICKNESS, new_thickness))
        self.prev_position = None  # Linie unterbrechen, während man den Slider bedient

    def clear_canvas(self):
        """Löscht das gesamte Canvas."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
