import cv2
import os
from datetime import datetime

save_folder = "captured_photos"
os.makedirs(save_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open the webcam. Check that it's connected and not in use by another app.")
    raise SystemExit(1)

print("Camera started! Press S to save a photo, Q to quit.")

known_faces = []   # each: {'center': (x, y), 'name': str, 'last_seen': frame_index}
MAX_DIST = 80       # max pixel distance to consider it the "same" face between frames
MAX_MISSING_FRAMES = 30  # drop a tracked face after this many frames without a match
next_person_number = 1
frame_index = 0


def get_center(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def find_matching_known_face(center):
    best_idx = None
    best_dist = MAX_DIST
    for i, kf in enumerate(known_faces):
        d = ((center[0] - kf['center'][0]) ** 2 + (center[1] - kf['center'][1]) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


cv2.namedWindow('Face Detection')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: failed to read frame from camera. Stopping.")
        break

    frame_index += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        center = get_center((x, y, w, h))
        idx = find_matching_known_face(center)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if idx is not None:
            known_faces[idx]['center'] = center
            known_faces[idx]['last_seen'] = frame_index
            label = known_faces[idx]['name']
        else:
            label = f"Person {next_person_number}"
            known_faces.append({'center': center, 'name': label, 'last_seen': frame_index})
            next_person_number += 1

        cv2.putText(frame, label, (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Drop tracked faces that haven't been seen in a while, so the list
    # doesn't grow forever and stale positions don't "steal" new faces.
    known_faces = [
        kf for kf in known_faces
        if frame_index - kf['last_seen'] <= MAX_MISSING_FRAMES
    ]

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        filename = os.path.join(save_folder, f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    if key == ord('q') or key == ord('Q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
