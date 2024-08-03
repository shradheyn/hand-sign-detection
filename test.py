import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imgsize = 300
labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']


        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
        imgcrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgcropShape = imgcrop.shape
        aspectRatio = h/w

        if aspectRatio >1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgcrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            #imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False )
            print(prediction, index)


        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            # imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.rectangle(imgOutput, (x - offset, y - offset), (x - offset+150, y - offset-50+50), (0, 0, 0), cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w+offset, y + h+offset), (0, 0, 0), 4)
        cv2.imshow("ImageCrop", imgcrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)






