import numpy as np
import cv2
from PIL import Image
import pytesseract
from gtts import gTTS
import IPython.display as ipd
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-']
thai_characters = ['ก', 'ป', 'อ']
english_characters = ['A', 'E']

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a video file path

while True:
    ret, frame = cap.read()  # Read a frame from the video stream

    # Detect objects in the frame using YOLO
    results = model(frame)
    for result in results.xyxy[0]:
        box = result.tolist()
        x_min, y_min, x_max, y_max = map(int, box[:4])

        # Crop the region of interest (ROI) from the frame
        roi = frame[y_min:y_max, x_min:x_max]

        # Perform OCR on the ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config_psm = '--psm 13'
        text = pytesseract.image_to_string(thresh, config=config_psm)
        new_text = ''.join([i for i in text if i in number_list or i in thai_characters or i in english_characters])

        # Generate audio from OCR result
        audio_text = "รถเมล์หมายเลข " + new_text
        language = 'th'
        audio_obj = gTTS(text=audio_text, lang=language, slow=False)
        audio_obj.save("bus_number.mp4")  # Save the audio to a file

        # Display bounding boxes around detected objects (optional)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the processed frame (optional)
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
