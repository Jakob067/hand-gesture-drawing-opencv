"""
Gesture Drawing – Hauptprogramm
Gestengesteuerte Zeichenanwendung mit MediaPipe-Handtracking.

Steuerung:
  - Zeigefinger hoch: Zeichnen
  - Alle Finger hoch (offene Handfläche): Radieren
  - Kleiner Finger hoch: Farbe wechseln
  - Zeigefinger + vertikale Bewegung: Strichdicke ändern
  - 'q': Beenden
  - 'c': Canvas löschen
"""

import time
import cv2

from camera import Camera
from hand_tracker import HandTracker
from finger_state import estimate_finger_states
from gesture_classifier import GestureClassifier
from drawing_engine import DrawingEngine
from renderer import Renderer


def get_fingertip(landmarks) -> tuple[int, int] | None:
    """Gibt die Position der Zeigefingerspitze zurück (Landmark 8)."""
    if landmarks and len(landmarks) > 8:
        return (landmarks[8][0], landmarks[8][1])
    return None


def main():
    """Hauptschleife der Anwendung."""
    # Module initialisieren
    camera = Camera()
    tracker = HandTracker()
    classifier = GestureClassifier()
    engine = DrawingEngine(camera.width, camera.height)
    renderer = Renderer()

    print("=== Gesture Drawing gestartet ===")
    print("Steuerung:")
    print("  Zeigefinger          → Zeichnen")
    print("  Offene Handfläche    → Radieren")
    print("  Kleiner Finger       → Farbe wechseln")
    print("  Vertikale Bewegung   → Dicke ändern")
    print("  'q'                  → Beenden")
    print("  'c'                  → Canvas löschen")
    print("================================")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # 1. Frame lesen
            frame = camera.read_frame()
            if frame is None:
                break

            # 2. FPS berechnen
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = current_time

            # 3. Handlandmarks erkennen
            landmarks, results = tracker.detect(frame)

            # 4. Optional: Landmarks auf Frame zeichnen
            tracker.draw_landmarks(frame, landmarks)

            # 5. Fingerzustand bestimmen
            fingers = estimate_finger_states(landmarks)

            # 6. Geste klassifizieren
            index_y = None
            fingertip = get_fingertip(landmarks)
            if fingertip:
                index_y = fingertip[1]

            gesture, _ = classifier.classify(fingers, index_y)

            # 7. Aktion ausführen — Landmarks für Palm-Eraser übergeben
            engine.execute(gesture, fingertip=fingertip,
                           landmarks=landmarks)

            # 8. Rendern
            combined = renderer.render(
                frame, engine.canvas, engine.color,
                engine.thickness, engine.mode, fps
            )
            renderer.show(combined)

            # 9. Tastatureingabe prüfen
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                engine.clear_canvas()
                print("Canvas gelöscht.")

    except KeyboardInterrupt:
        print("\nAbgebrochen.")
    finally:
        # Ressourcen freigeben
        camera.release()
        tracker.release()
        renderer.destroy()
        print("Ressourcen freigegeben. Auf Wiedersehen!")


if __name__ == "__main__":
    main()
