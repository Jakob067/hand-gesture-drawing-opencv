"""
Modul 3 – Finger State Estimation
Bestimmt den binären Zustand jedes Fingers (oben/unten).
Nutzt einen Schwellenwert (Margin) für stabilere Erkennung.
"""


# MediaPipe Landmark-Indizes
# Daumen: 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
# Zeigefinger: 5 (MCP), 6 (PIP), 7 (DIP), 8 (Tip)
# Mittelfinger: 9 (MCP), 10 (PIP), 11 (DIP), 12 (Tip)
# Ringfinger: 13 (MCP), 14 (PIP), 15 (DIP), 16 (Tip)
# Kleiner Finger: 17 (MCP), 18 (PIP), 19 (DIP), 20 (Tip)

FINGER_TIP_IDS = [4, 8, 12, 16, 20]
FINGER_PIP_IDS = [3, 6, 10, 14, 18]  # Für Daumen: IP-Gelenk (Landmark 3)
FINGER_MCP_IDS = [2, 5, 9, 13, 17]   # MCP-Gelenke für bessere Erkennung

# Schwellenwert: Finger muss mindestens diese Pixel-Differenz haben
# um als "oben" erkannt zu werden (verhindert Flackern)
FINGER_MARGIN = 15


def estimate_finger_states(landmarks) -> list[int]:
    """Bestimmt welche Finger oben (gestreckt) sind.

    Verwendet Tip vs. PIP UND Tip vs. MCP für robustere Erkennung.
    Ein Finger gilt nur als "oben", wenn der Tip deutlich über dem
    PIP-Gelenk liegt (mit Margin).

    Args:
        landmarks: Liste von 21 (x, y, z) Tupeln.

    Returns:
        Liste von 5 Integern [thumb, index, middle, ring, pinky],
        wobei 1 = Finger oben, 0 = Finger unten.
    """
    if landmarks is None or len(landmarks) < 21:
        return [0, 0, 0, 0, 0]

    fingers = []

    # Daumen: Vergleich x-Koordinate (Tip vs. IP-Gelenk)
    thumb_tip = landmarks[FINGER_TIP_IDS[0]]
    thumb_ip = landmarks[FINGER_PIP_IDS[0]]
    thumb_mcp = landmarks[FINGER_MCP_IDS[0]]

    # Prüfe Handseite anhand von Handgelenk vs. Pinky-MCP x-Position
    wrist = landmarks[0]
    pinky_mcp = landmarks[17]

    if wrist[0] < pinky_mcp[0]:
        # Rechte Hand (im gespiegelten Bild): Daumen oben wenn Tip.x < IP.x
        thumb_diff = thumb_ip[0] - thumb_tip[0]
    else:
        # Linke Hand: Daumen oben wenn Tip.x > IP.x
        thumb_diff = thumb_tip[0] - thumb_ip[0]

    fingers.append(1 if thumb_diff > FINGER_MARGIN else 0)

    # Zeigefinger bis kleiner Finger: y-Koordinate Tip vs. PIP
    # Finger oben wenn Tip.y deutlich kleiner als PIP.y (y-Achse geht nach unten)
    for i in range(1, 5):
        tip = landmarks[FINGER_TIP_IDS[i]]
        pip_joint = landmarks[FINGER_PIP_IDS[i]]

        # Tip muss mindestens FINGER_MARGIN Pixel über PIP sein
        diff = pip_joint[1] - tip[1]
        fingers.append(1 if diff > FINGER_MARGIN else 0)

    return fingers


def get_finger_count(fingers: list[int]) -> int:
    """Zählt die Anzahl gestreckter Finger.

    Args:
        fingers: Binäre Liste [thumb, index, middle, ring, pinky].

    Returns:
        Anzahl der gestreckten Finger (0-5).
    """
    return sum(fingers)
