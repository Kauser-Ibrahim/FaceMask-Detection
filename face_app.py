import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np

# Load the model and Haar Cascade
model = load_model("mobileNetV2_50data.h5")
haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_face(img):
    coods = haar.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return coods

def detect_face_mask(img):
    img = cv2.resize(img, (224, 224)) / 255.0
    y_pred = model.predict(img.reshape(1, 224, 224, 3), verbose=False)
    prediction = (y_pred > 0.5).astype(int)
    return int(prediction[0][0])

def draw_label(frame, label, position, color):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, label, font=font, fill=color)
    return frame_pil

# Streamlit app
st.title("Face Mask Detection")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Failed to open video capture device")
else:
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    while True:
        if stop_button:
            break
        
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break

        img = cv2.resize(frame, (224, 224))
        y_pred = detect_face_mask(img)

        # Detect faces and draw rectangles
        coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for x, y, w, h in coods:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Draw mask or no mask label
        if y_pred == 0:
            frame_pil = draw_label(frame, "Mask", (30, 30), (0, 255, 0))
        else:
            frame_pil = draw_label(frame, "No Mask", (30, 30), (255, 0, 0))

        # Convert the frame to RGB for Streamlit
        buf = BytesIO()
        frame_pil.save(buf, format="JPEG")
        byte_img = buf.getvalue()

        # Update the Streamlit placeholder with the new image
        frame_placeholder.image(byte_img, use_column_width=True)
    
    cap.release()
