"""
Modul 4 – Gesture Classification (Regelbasiert)
Klassifiziert Fingerzustände in Systemzustände mittels State Machine.
Verwendet temporale Glättung für stabilere Gestenerkennung.
"""

import time
from finger_state import get_finger_count


# Systemzustände
IDLE = "IDLE"
DRAW = "DRAW"
ERASE = "ERASE"
THICKNESS_ADJUST = "THICKNESS_ADJUST"
COLOR_SWITCH = "COLOR_SWITCH"

# Anzahl aufeinanderfolgender Frames für stabile Gestenerkennung
STABILITY_FRAMES = 3


class GestureClassifier:
    def __init__(self, color_switch_cooldown: float = 0.8):
        """Initialisiert den Gestenklassifizierer.

        Args:
            color_switch_cooldown: Sekunden zwischen Farbwechseln.
        """
        self.color_switch_cooldown = color_switch_cooldown

        self.last_color_switch_time = 0.0
        self.color_switched = False

        # Temporale Glättung
        self._gesture_history = []
        self._current_stable_gesture = IDLE

    def classify(self, fingers: list[int], index_y: int | None = None) -> tuple[str, int]:
        """Klassifiziert den aktuellen Fingerzustand in einen Systemzustand.

        Args:
            fingers: [thumb, index, middle, ring, pinky] als [0|1].
            index_y: Y-Position der Zeigefingerspitze (Pixel).

        Returns:
            Tupel (gesture_name, delta_y):
                gesture_name: Einer der Systemzustände.
                delta_y: Vertikale Bewegung (nur relevant bei THICKNESS_ADJUST).
        """
        delta_y = 0
        finger_count = get_finger_count(fingers)
        raw_gesture = IDLE

        # Mindestens 4 Finger oben → ERASE (toleranter als exakt 5)
        if finger_count >= 4 and fingers[1] == 1:
            raw_gesture = ERASE

        # Nur kleiner Finger oben (oder kleiner + Daumen)
        elif fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
            raw_gesture = COLOR_SWITCH

        # Nur Zeigefinger oben (Daumen darf auch oben sein)
        elif fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            raw_gesture = DRAW

        # Temporale Glättung: Geste muss STABILITY_FRAMES konsekutiv sein
        stable_gesture = self._apply_temporal_smoothing(raw_gesture)

        # Aktionen basierend auf stabilem Gesture
        if stable_gesture == ERASE:
            self._reset_tracking()
            self.color_switched = False
            return ERASE, 0

        if stable_gesture == COLOR_SWITCH:
            self._reset_tracking()
            now = time.time()
            if not self.color_switched and (now - self.last_color_switch_time) > self.color_switch_cooldown:
                self.last_color_switch_time = now
                self.color_switched = True
                return COLOR_SWITCH, 0
            return IDLE, 0

        # Kleiner Finger nicht mehr oben → Debounce zurücksetzen
        if stable_gesture != COLOR_SWITCH:
            self.color_switched = False

        if stable_gesture == DRAW:
            return DRAW, 0

        # IDLE
        self._reset_tracking()
        return IDLE, 0

    def _apply_temporal_smoothing(self, raw_gesture: str) -> str:
        """Wendet temporale Glättung an — Geste muss N Frames stabil sein.

        Args:
            raw_gesture: Die aktuelle rohe Geste.

        Returns:
            Die stabile Geste (ändert sich nur nach N konsekutiven gleichen Frames).
        """
        self._gesture_history.append(raw_gesture)

        # Nur die letzten N Frames behalten
        if len(self._gesture_history) > STABILITY_FRAMES:
            self._gesture_history = self._gesture_history[-STABILITY_FRAMES:]

        # Prüfe ob alle letzten N Frames die gleiche Geste haben
        if len(self._gesture_history) >= STABILITY_FRAMES:
            if all(g == raw_gesture for g in self._gesture_history[-STABILITY_FRAMES:]):
                self._current_stable_gesture = raw_gesture

        return self._current_stable_gesture

    def _reset_tracking(self):
        """Setzt Tracking-Variablen zurück."""
        pass
