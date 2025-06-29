import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\ACER\Desktop\sing_language\Sign-Language-detection\Data\Yes"

if not cap.isOpened():
    print("Error: Camera not found")
    exit()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the bounding box is within valid image boundaries
        if x >= 0 and y >= 0 and (x + w) < img.shape[1] and (y + h) < img.shape[0]:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        else:
            continue  # Skip this frame if bounding box is out of bounds

        # Check if the cropped image is empty
        if imgCrop.size == 0:
            continue  # Skip this frame if the crop operation results in an empty image

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Resize the image based on the aspect ratio
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{int(time.time())}.jpg', imgWhite)
        print(counter)

    # Close on pressing ESC
    if key == 27:  # 27 is the ASCII value for the Escape key
        break

cap.release()
cv2.destroyAllWindows()
