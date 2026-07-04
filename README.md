# Webcam Face Detection

A Python script using OpenCV that detects faces from a webcam feed in real time, assigns each detected face a simple persistent label (`Person 1`, `Person 2`, ...), and lets you save snapshots.

## Demo

The app opens a live window showing your webcam feed with a green box around each detected face, a label above it, and a live face count in the corner.

## Requirements

- Python 3.7+
- A webcam

## Installation

```bash
pip install opencv-python
```

## Usage

```bash
python face_detection.py
```

A window will open and start detecting faces immediately.

### Controls

| Key | Action |
|-----|--------|
| `S` | Save the current frame as a photo |
| `Q` or `Esc` | Quit the application |

Saved photos go into a `captured_photos/` folder, named with a timestamp (e.g. `photo_20260629_143052.png`).

## How it works

1. Each frame is converted to grayscale (Haar Cascade detection works on grayscale images).
2. `cv2.CascadeClassifier` with the bundled `haarcascade_frontalface_default.xml` model detects face bounding boxes.
3. For each detected face, its center point is compared against previously tracked faces. If it's within 80 pixels of a known face's last position, it keeps that face's label; otherwise it gets a new label (`Person N`).
4. Tracked faces that haven't been seen for more than 30 consecutive frames are dropped, so memory doesn't grow unbounded and stale positions don't get reused.
5. The total face count is drawn in the top-left corner of the frame.

## Limitations

This is **not** real face recognition — it's lightweight position-based tracking between consecutive frames, not identity recognition based on facial features. Because of that:

- If two people's faces cross paths, their labels may swap.
- If one person leaves the frame and another appears in roughly the same spot, the new person may incorrectly inherit the previous label.
- If the same person leaves and re-enters the frame after a short gap (more than 30 frames, roughly ~1 second), they'll get a brand new number instead of their original one.

For persistent identity recognition (recognizing the same person even after they leave and return), you'd need an actual face-recognition library with facial embeddings, such as [`face_recognition`](https://github.com/ageitgey/face_recognition) or a deep learning embedding model (e.g. FaceNet, dlib, or a model via `deepface`).

## Troubleshooting

**"Error: could not open the webcam"**
This means OpenCV couldn't access your camera. Check that:
- Your webcam is properly connected
- No other application (Zoom, browser tab, another script) is currently using it
- You've granted camera permissions to your terminal/Python if on macOS

## License

Feel free to use and modify this project as you like.
