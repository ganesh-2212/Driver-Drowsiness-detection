from sound_manager import CHANNEL_YOLO2, sound_yolo2

alert_classes = {"rub_eye", "look_away", "phone"}

def play_alert():
    if not CHANNEL_YOLO2.get_busy():
        CHANNEL_YOLO2.play(sound_yolo2)
def process_yolo2(frame, yolo_model):
    results = yolo_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    detected_classes = []
    if results[0].boxes is not None and results[0].boxes.cls is not None:
        detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

    if any(cls in {"rub_eye", "look_away", "phone"} for cls in detected_classes):
        play_alert()

    return annotated_frame, detected_classes