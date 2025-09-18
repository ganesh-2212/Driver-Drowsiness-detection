import time

import cv2
from sound_manager import CHANNEL_YOLO1, sound_yolo1

alert_classes = {"Drowsy", "Yawning"}
drowsy_start_time = None
DROWSY_THRESHOLD = 5  

def play_alert():
    if not CHANNEL_YOLO1.get_busy():
        CHANNEL_YOLO1.play(sound_yolo1)

def process_yolo1(frame, yolo_model):
    global drowsy_start_time

    results = yolo_model(frame, imgsz=640, conf=0.25, verbose=True)
    annotated_frame = results[0].plot()

    detected_classes = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls.cpu().item())
            conf = float(box.conf.cpu().item())
            label = results[0].names[cls]
            detected_classes.append(label)
            print(f"YOLO1 detection: {label} ({conf:.2f})")
    else:
        print("YOLO1: no boxes detected")

 
    if any(cls in {"Drowsy", "Yawning"} for cls in detected_classes):
        play_alert()
        if "Drowsy" in detected_classes:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time >= DROWSY_THRESHOLD:
                cv2.putText(annotated_frame, "🚨 DROWSY ALERT 🚨", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            drowsy_start_time = None
    else:
        drowsy_start_time = None

    return annotated_frame, detected_classes