import streamlit as st
import numpy as np
import cv2
import cvzone
import math
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tempfile

# Class labels for YOLO object detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load models
yolo_model = YOLO("yolov8n-seg.pt")  # YOLOv8 for object detection
nationality_model = load_model("Race.h5")
age_model = load_model("Age.h5")
emotion_model = load_model("Facial_emotionl.h5")

# Labels for predictions
nationality_labels = ["African", "American", "Indian"]
emotion_labels = ["Anger", "Happiness", "Neutral", "Sadness", "Surprise"]
age_labels = ["10-19", "20-29", "30-39", "40-49", "50-59"]

# Function to process person image
def process_person(img):
    img_resized = cv2.resize(img, (150, 150))
    img_resized = img_to_array(img_resized) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    nationality_pred = nationality_model.predict(img_resized)
    nationality_idx = np.argmax(nationality_pred)
    nationality = nationality_labels[nationality_idx]
    nationality_conf = round(nationality_pred[0][nationality_idx] * 100, 2)

    emotion_pred = emotion_model.predict(img_resized)
    emotion_idx = np.argmax(emotion_pred)
    emotion = emotion_labels[emotion_idx]
    emotion_conf = round(emotion_pred[0][emotion_idx] * 100, 2)

    age = "N/A"
    age_conf = None
    if nationality in ["Indian", "American"]:
        age_pred = age_model.predict(img_resized)
        age_idx = np.argmax(age_pred)
        age = age_labels[age_idx]
        age_conf = round(age_pred[0][age_idx] * 100, 2)

    return nationality, nationality_conf, age, age_conf, emotion, emotion_conf

# Streamlit App
st.set_page_config(page_title="VisionAI: Advanced Image & Video Analysis", layout="wide")
st.title("VisionAI: Advanced Image & Video Analysis")
st.sidebar.title("Upload Options")
upload_option = st.sidebar.selectbox("Choose input type", ["Image", "Video"])

# Function to resize media
def resize_media(image, target_width=800):
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(image, (target_width, new_height))

if upload_option == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = resize_media(img)

        results = yolo_model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls)
                conf = math.ceil(box.conf * 100) / 100
                currentClass = classNames[cls]

                if currentClass == "person" and conf > 0.3:
                    person_img = img[y1:y2, x1:x2]
                    nationality, nationality_conf, age, age_conf, emotion, emotion_conf = process_person(person_img)
                    display_text = [
                        f"{nationality} ({nationality_conf}%)",
                        f"{emotion} ({emotion_conf}%)"
                    ]
                    if age != "N/A":
                        display_text.append(f"Age: {age} ({age_conf}%)")
                    
                    # Calculate width and height of the bounding box
                    w = x2 - x1
                    h = y2 - y1
                    
                    cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=5, rt=2, colorR=(0, 255, 0), colorC=(255, 0, 255))
                    
                    for i, text in enumerate(display_text):
                        y_offset = y1 + i * 30
                        cvzone.putTextRect(img, text, (x1, y_offset), scale=1, thickness=2, offset=10)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

elif upload_option == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = resize_media(frame)
            results = yolo_model(frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls)
                    conf = math.ceil(box.conf * 100) / 100
                    currentClass = classNames[cls]

                    if currentClass == "person" and conf > 0.3:
                        person_img = frame[y1:y2, x1:x2]
                        nationality, nationality_conf, age, age_conf, emotion, emotion_conf = process_person(person_img)
                        display_text = [
                            f"{nationality} ({nationality_conf}%)",
                            f"{emotion} ({emotion_conf}%)"
                        ]
                        if age != "N/A":
                            display_text.append(f"Age: {age} ({age_conf}%)")
                        
                        # Calculate width and height of the bounding box
                        w = x2 - x1
                        h = y2 - y1
                        
                        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=5, rt=2, colorR=(0, 255, 0), colorC=(255, 0, 255))

                        for i, text in enumerate(display_text):
                            y_offset = y1 + i * 30
                            cvzone.putTextRect(frame, text, (x1, y_offset), scale=1, thickness=2, offset=10)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        cap.release()
