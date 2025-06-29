import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow.lite as tflite

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector (detects only one hand)
detector = HandDetector(maxHands=1)

# Load TFLite model
interpreter = tflite.Interpreter(model_path=r"C:\Users\ACER\Desktop\sign_git\Model\model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Expected input size for TFLite model
model_input_size = (224, 224)  # Update this if your model has a different input size

# Padding and image size configuration
offset = 20
imgSize = 300

# Define gesture labels (Ensure they match the training labels)
labels = ["Hello", "Thank you", "Yes"]  # Update this if you have more gestures

while True:
    success, img = cap.read()
    if not success:
        continue
    
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image (300x300)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop hand region from the image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # **Fix: Resize to match model input size (224x224)**
        imgInput = cv2.resize(imgWhite, model_input_size)

        # Convert to RGB (TFLite models typically require RGB format)
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)

        # Expand dimensions (TFLite expects batch dimension)
        imgInput = np.expand_dims(imgInput, axis=0).astype(np.float32)

        # Normalize (Uncomment if required by your model)
        imgInput = imgInput / 255.0  # Normalizing to range [0,1]

        # **Run inference**
        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Get predicted class index
        index = np.argmax(prediction)

        # Draw label and bounding box
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                      (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show cropped images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
