import cv2

def draw_detections_on_frame(frame, results):
    detections = results.boxes
    count = 0

    if detections is not None:
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0:  # Assuming class 0 is 'stampede'
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Stampeede {conf:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw threat level banner
    if count > 30:
        threat = "HIGH THREAT"
        color = (0, 0, 255)
    elif count >= 10:
        threat = "MEDIUM THREAT"
        color = (0, 255, 255)
    else:
        threat = "LOW THREAT"
        color = (0, 255, 0)

    cv2.putText(frame, f'Threat: {threat} ({count} detections)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame
