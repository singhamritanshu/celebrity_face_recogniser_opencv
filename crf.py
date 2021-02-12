import cv2 as cv 
import os 
import numpy as np

celebrities = []
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
for char in os.listdir("dataset"):
    celebrities.append(char)

features=[]
labels=[]
DIR = r'dataset'

def collect_feature():
    for celebrity in celebrities:
        path = os.path.join(DIR,celebrity)
        label = celebrities.index(celebrity)
        if os.path.isdir(path):
            for img in os.listdir(path):
                img_path = os.path.join(path,img)
                img_array = cv.imread(img_path)
                gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
                face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)
                for (x,y,w,h) in face_rect:
                    face_roi = gray[y:y+h,x:x+w]
                    features.append(face_roi)
                    labels.append(label)

collect_feature()
print("Training done")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Train the recognizer on the features list and the label
features = np.array(features,dtype='object')
labels = np.array(labels)
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml') # Saving the face trained model so that we can use it.
# Saving the features and the labels
np.save("features.npy",features)
np.save("labels.npy",labels)
