from flask import Flask, render_template, Response
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

FRAME_SKIP = 15
frame_count = 0
last_faces = []

suggestions = {
    "happy": "You look happy 😊 Keep spreading positivity!",
    "sad": "Take a deep breath. Listening to calm music may help.",
    "angry": "Try relaxation or deep breathing exercises.",
    "neutral": "Stay focused and productive.",
    "fear": "Relax. Everything will be okay.",
    "surprise": "Take a moment to process your feelings.",
    "disgust": "Try shifting your focus to something positive."
}

def generate_frames():
    global frame_count, last_faces

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Run emotion detection occasionally
        if frame_count % FRAME_SKIP == 0:
            last_faces = []

            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )

            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.6:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    try:
                        result = DeepFace.analyze(
                            face,
                            actions=["emotion"],
                            enforce_detection=False
                        )
                        last_faces.append((x1, y1, x2, y2, result[0]))
                    except:
                        pass

        # Draw cached results
        for (x1, y1, x2, y2, res) in last_faces:
            emotion = res["dominant_emotion"]
            conf = res["emotion"][emotion]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion.upper()} ({conf:.1f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
