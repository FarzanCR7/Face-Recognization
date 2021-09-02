# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 21:45:51 2021

@author: farzt
"""

import cv2
import numpy as np
import os
#%%
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
cascadePath= "FaceRecognition.xml"
faceCascade=cv2.CascadeClassifier(cascadePath)
#%%
font = cv2.FONT_HERSHEY_SIMPLEX
ids = 1
names = ['','Farzan','Unknown']

#%%
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minw = 0.1*cam.get(3)
minh = 0.1*cam.get(4)
#%%
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors = 5, minSize = (int(minw),int(minh)),)
    for(x,y,w,h ) in faces:
        cv2.rectangle(img, (x,y), (w + x, h + y), (0,255,0),2)
        ids,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if confidence<100:
            ids =names[1]
            confidence = "{0}%".format(round(100-confidence))
        else:
            ids = "Unknown"
            confidence = "{0}%".format(round(100-confidence))
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0,255,255),2)
        cv2.putText(img, str(ids)+"  OPEN", (x+5, y -5), font, 1, (0,255,255),2)
        cv2.imshow("Farzan", img)
       
    if cv2.waitKey(100) & 0xFF == ord("q"):
       break
print("Exit Program")
cam.release()
cv2.destroyAllWindows()
