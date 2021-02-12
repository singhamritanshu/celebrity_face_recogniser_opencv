import os
import cv2 as cv 
import numpy as np 
celebrities=[]
for char in os.listdir("dataset"):
    celebrities.append(char)


haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread("test_images/kohli_2.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Person",gray)

face_rect = haar_cascade.detectMultiScale(gray,1.1,8)
for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print("Name of the person in the image", celebrities[label]," with confidence of",confidence)
    cv.putText(img,str(celebrities[label]),(70,70),cv.FONT_HERSHEY_COMPLEX, 1.0,255,2)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow("Detected image", img)

cv.waitKey(0)