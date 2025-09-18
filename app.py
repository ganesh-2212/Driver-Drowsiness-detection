import matplotlib
matplotlib.use("Agg")  

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import threading
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

from yolo1_alert import process_yolo1
from yolo2_alert import process_yolo2
from ultralytics import YOLO

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

yolo_model_1 = YOLO("content/runs/obb/train/weights/best.pt").to(device)
yolo_model_2 = YOLO("driver_drowsiness_train/weights/best.pt").to(device)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
mobilevit = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").to(device)
mobilevit.eval()

class VideoCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        thread = threading.Thread(target=self.update)
        thread.daemon = True
        thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

camera = VideoCamera(0)

detection_counts = {}

def update_counts(detected_classes):
    global detection_counts
    for cls in detected_classes:
        detection_counts[cls] = detection_counts.get(cls, 0) + 1

import io

def draw_bar_chart():
    global detection_counts
    if not detection_counts:
        blank = np.ones((480, 400, 3), dtype=np.uint8) * 255
        return blank

    classes = list(detection_counts.keys())
    counts = list(detection_counts.values())

    fig, ax = plt.subplots(figsize=(4, 6), facecolor="white")
    ax.barh(classes, counts, color="wheat")
    ax.set_xlabel("Count")
    ax.set_title("Detections")
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    chart_img = np.array(Image.open(buf).convert('RGB'))
    buf.close()
    plt.close(fig)

    return chart_img

def add_padding_and_stack(video, chart, pad=30, canvas_width=1230):
    target_height = video.shape[0]
    chart = cv2.resize(chart, (400, target_height))

    h = target_height
    min_width = video.shape[1] + chart.shape[1] + pad * 2
    w = max(canvas_width, min_width)
    combined = np.ones((h, w, 3), dtype=np.uint8) * 255

    combined[0:video.shape[0], 0:video.shape[1]] = video
    chart_x = w - chart.shape[1]
    combined[0:chart.shape[0], chart_x:chart_x + chart.shape[1]] = chart

    return combined

@app.get("/video_feed")
async def video_feed():
    def generate():
        frame_count = 0
        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_count += 1
            small_frame = cv2.resize(frame, (640, 480))

            annotated_frame, yolo1_classes = process_yolo1(frame, yolo_model_1)
            update_counts(yolo1_classes)

            if frame_count % 3 == 0:
                annotated_frame, yolo2_classes = process_yolo2(small_frame, yolo_model_2)
                update_counts(yolo2_classes)

            if frame_count % 5 == 0:
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
                    inputs = feature_extractor(images=pil_img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = mobilevit(**inputs)
                        pred_id = outputs.logits.argmax(-1).item()
                        label = mobilevit.config.id2label[pred_id]

                    cv2.putText(annotated_frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    update_counts([label])
                except Exception as e:
                    print(f"MobileViT error: {e}")

            chart_img = draw_bar_chart()
            combined = add_padding_and_stack(annotated_frame, chart_img, pad=40, canvas_width=1280)

            ret, buffer = cv2.imencode('.jpg', combined)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')
