from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("mobileNetV2_50data.h5")


def detect_face_mask(img):
    img = img.resize((224, 224))  # Resize image for the model
    img_array = np.array(img) / 255.0  # Normalize
    y_pred = model.predict(img_array.reshape(1, 224, 224, 3), verbose=False)
    prediction = (y_pred > 0.5).astype(int)
    return int(prediction[0][0])


def draw_label(frame, label, position, color):
    # Create drawing context
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, label, font=font, fill=color)
    return frame


# Initialize Streamlit and camera
st.title("Face Mask Detection")
st.write("Please allow access to your webcam.")

# Create a placeholder for the video feed
frame_placeholder = st.empty()

# Video capture
camera = st.camera_input("Capture image")

if camera:
    # Open the video feed
    video_feed = camera.read()

    if video_feed is not None:
        # Process the video feed
        img = Image.open(camera)

        # Detect mask
        y_pred = detect_face_mask(img)

        # Draw mask or no mask label
        if y_pred == 0:
            img = draw_label(img, "Mask", (30, 30), (0, 255, 0))
        else:
            img = draw_label(img, "No Mask", (30, 30), (255, 0, 0))

        # Convert the PIL image to bytes
        buf = BytesIO()
        img.save(buf, format="JPEG")
        byte_img = buf.getvalue()

        # Update the Streamlit placeholder with the new image
        frame_placeholder.image(byte_img, use_column_width=True)
