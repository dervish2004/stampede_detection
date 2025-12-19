import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")  # COCO model includes 'person'

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Load video
video_path = "videos/your_video.mp4"
cap = cv2.VideoCapture(video_path)

CROWD_THRESHOLD = 15     # number of people
SPEED_THRESHOLD = 10.0   # average pixels/frame

frame_num = 0
track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    results = model(frame, verbose=False)[0]

    detections = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls.item()) == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    person_count = 0
    total_speed = 0.0

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx, cy = int((l + r) / 2), int((t + b) / 2)

        person_count += 1

        # Calculate speed
        prev_pos = track_history.get(track_id, None)
        if prev_pos:
            dx = cx - prev_pos[0]
            dy = cy - prev_pos[1]
            speed = (dx**2 + dy**2) ** 0.5
        else:
            speed = 0

        track_history[track_id] = (cx, cy)
        total_speed += speed

        # Draw bounding box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    avg_speed = total_speed / max(1, person_count)

    # Trigger logic
    if person_count > CROWD_THRESHOLD and avg_speed > SPEED_THRESHOLD:
        cv2.putText(frame, "🚨 Stampede Alert!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        print(f"[ALERT] Frame {frame_num}: Stampede Detected - Count={person_count}, Speed={avg_speed:.2f}")

    # Overlay info
    cv2.putText(frame, f"People: {person_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Avg Speed: {avg_speed:.2f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Stampede Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
