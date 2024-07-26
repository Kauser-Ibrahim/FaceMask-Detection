from tensorflow.keras.models import load_model
import cv2
import streamlit as st
from PIL import Image
from io import BytesIO

model = load_model("mobileNetV2_50data.h5")
haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect_face(img):
    coods = haar.detectMultiScale(img)
    return coods


def detect_face_mask(img):
    # Ensure img is resized and normalized
    img = cv2.resize(img, (224, 224)) / 255.0

    # Predict using the model
    y_pred = model.predict(img.reshape(1, 224, 224, 3), verbose=False)

    # Binary classification, so use a threshold
    prediction = (y_pred > 0.5).astype(int)

    # Return class label
    return int(prediction[0][0])


def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
st.title("Face Mask Detection")

if "stop_button" not in st.session_state:
    st.session_state.stop_button = False


def toggle_stop():
    st.session_state.stop_button = not st.session_state.stop_button


stop_button = st.button("Stop", key="stop_but", on_click=toggle_stop)

frame_placeholder = st.empty()


while not st.session_state.stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break

    img = cv2.resize(frame, (224, 224))
    y_pred = detect_face_mask(img)

    # Draw face rectangles
    coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for x, y, w, h in coods:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Draw mask or no mask label
    if y_pred == 0:
        draw_label(frame, "Mask", (30, 30), (0, 255, 0))
    else:
        draw_label(frame, "No Mask", (30, 30), (0, 0, 255))

    # Convert the frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_img = buf.getvalue()

    # Update the Streamlit placeholder with the new image
    frame_placeholder.image(byte_img, use_column_width=True)

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
