"""
Modul 2 – Hand Landmark Detection
Erkennt Handlandmarks mittels MediaPipe Tasks API (ab v0.10.x).
Lädt das Modell automatisch herunter, falls es nicht vorhanden ist.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks API Imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Modell-URL und lokaler Pfad
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def _ensure_model():
    """Lädt das Hand-Landmarker-Modell herunter, falls es nicht existiert."""
    if not os.path.exists(MODEL_PATH):
        print(f"Lade Hand-Landmarker-Modell herunter...")
        print(f"  URL: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Gespeichert: {MODEL_PATH}")


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7):
        """Initialisiert den MediaPipe Hand-Tracker mit der Tasks API.

        Args:
            max_hands: Maximale Anzahl erkannter Hände.
            detection_confidence: Mindestkonfidenz für die Erkennung.
            tracking_confidence: Mindestkonfidenz für das Tracking.
        """
        _ensure_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def detect(self, frame):
        """Erkennt Handlandmarks im Frame.

        Args:
            frame: BGR-Bild von OpenCV.

        Returns:
            landmarks: Liste von 21 (x, y, z) Tupeln in Pixel-Koordinaten
                       oder None wenn keine Hand erkannt.
            result: MediaPipe HandLandmarkerResult für optionales Zeichnen.
        """
        # MediaPipe Tasks API erwartet RGB als mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp inkrementieren (muss monoton steigend sein)
        self._frame_timestamp_ms += 33  # ~30 FPS angenommen
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]  # Nur erste Hand
            h, w, _ = frame.shape

            landmarks = []
            for lm in hand:
                px = int(lm.x * w)
                py = int(lm.y * h)
                landmarks.append((px, py, lm.z))

            return landmarks, result

        return None, result

    def draw_landmarks(self, frame, landmarks):
        """Zeichnet die erkannten Landmarks manuell auf den Frame.

        Args:
            frame: BGR-Bild auf dem gezeichnet wird.
            landmarks: Liste von 21 (x, y, z) Tupeln oder None.
        """
        if landmarks is None:
            return

        # Verbindungen zwischen Landmarks (MediaPipe Hand Connections)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Daumen
            (0, 5), (5, 6), (6, 7), (7, 8),       # Zeigefinger
            (0, 9), (9, 10), (10, 11), (11, 12),   # Mittelfinger
            (0, 13), (13, 14), (14, 15), (15, 16), # Ringfinger
            (0, 17), (17, 18), (18, 19), (19, 20), # Kleiner Finger
            (5, 9), (9, 13), (13, 17),             # Handfläche
        ]

        # Verbindungslinien zeichnen
        for start, end in connections:
            pt1 = (landmarks[start][0], landmarks[start][1])
            pt2 = (landmarks[end][0], landmarks[end][1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Landmark-Punkte zeichnen
        for i, lm in enumerate(landmarks):
            cv2.circle(frame, (lm[0], lm[1]), 4, (255, 0, 0), -1)

    def release(self):
        """Gibt MediaPipe-Ressourcen frei."""
        self.landmarker.close()
