import cv2
import mediapipe as mp
from openai_analyzer import analyze_expression
import time

# Setup face mesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

last_detected = set()
last_detect_time = {}
cooldown_seconds = 5

GESTURE_DEFINITIONS = [
    ("raised left eyebrow", lambda lm: (lm[159].y - lm[65].y) > 0.03),
    ("pressed lips", lambda lm: abs(lm[13].y - lm[14].y) < 0.01),
    ("squinting eyes", lambda lm: abs(lm[159].y - lm[145].y) < 0.01),
    ("raised right eyebrow", lambda lm: (lm[386].y - lm[295].y) > 0.03),
    ("mouth open", lambda lm: abs(lm[13].y - lm[14].y) > 0.04),
    ("eyes wide open", lambda lm: abs(lm[159].y - lm[145].y) > 0.035),
    ("head tilt left", lambda lm: lm[234].y - lm[454].y > 0.03),
    ("head tilt right", lambda lm: lm[454].y - lm[234].y > 0.03),
    ("cheek puff", lambda lm: abs(lm[50].x - lm[280].x) > 0.25),
    ("nostril flare", lambda lm: abs(lm[94].x - lm[331].x) > 0.05),
    ("smile", lambda lm: abs(lm[61].x - lm[291].x) > 0.045),
    ("frown", lambda lm: abs(lm[61].x - lm[291].x) < 0.035),
    ("jaw drop", lambda lm: abs(lm[152].y - lm[13].y) > 0.05),
    ("brow furrow", lambda lm: abs(lm[65].x - lm[295].x) < 0.03),
    ("brow lift", lambda lm: (lm[65].y + lm[295].y) / 2 < lm[10].y - 0.03),
    ("pursed lips", lambda lm: abs(lm[61].x - lm[291].x) < 0.025),
    ("eye roll up", lambda lm: lm[468].y < lm[474].y - 0.02),
    ("eye roll down", lambda lm: lm[468].y > lm[474].y + 0.02),
    ("chin thrust forward", lambda lm: lm[152].z < -0.1),
    ("chin tuck", lambda lm: lm[152].z > 0.1),
    ("smirk left", lambda lm: lm[61].y > lm[291].y + 0.015),
    ("smirk right", lambda lm: lm[291].y > lm[61].y + 0.015),
    ("eye blink left", lambda lm: abs(lm[159].y - lm[145].y) < 0.005),
    ("eye blink right", lambda lm: abs(lm[386].y - lm[374].y) < 0.005),
    ("nose wrinkle", lambda lm: abs(lm[6].y - lm[168].y) < 0.02),
    ("tongue out", lambda lm: lm[19].y < lm[14].y - 0.03),
    ("lip bite", lambda lm: abs(lm[13].y - lm[14].y) < 0.008 and abs(lm[61].x - lm[291].x) < 0.03),
    ("glare left", lambda lm: lm[33].x - lm[133].x > 0.02),
    ("glare right", lambda lm: lm[263].x - lm[362].x > 0.02),
    ("brows raised and mouth open", lambda lm: (lm[159].y - lm[65].y) > 0.03 and abs(lm[13].y - lm[14].y) > 0.04),
    ("brows lowered and lips pressed", lambda lm: (lm[159].y - lm[65].y) < 0.01 and abs(lm[13].y - lm[14].y) < 0.01),
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()
    new_gestures = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            landmarks = face_landmarks.landmark

            for name, condition in GESTURE_DEFINITIONS:
                try:
                    if condition(landmarks):
                        if name not in last_detected or (current_time - last_detect_time.get(name, 0) > cooldown_seconds):
                            print(f"🟢 New gesture: {name}")
                            new_gestures.append(name)
                            last_detected.add(name)
                            last_detect_time[name] = current_time
                except Exception as e:
                    print(f"⚠️ Error in gesture '{name}': {e}")

    for gesture in new_gestures:
        response = analyze_expression(gesture)
        print("💬 GPT says:", response)

    cv2.imshow('Emoticon - Face Tracker', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to stop
        break

cap.release()
cv2.destroyAllWindows()
