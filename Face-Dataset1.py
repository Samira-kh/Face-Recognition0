import cv2 as cv
import os
import numpy as np
import glob

cam = cv.VideoCapture(0)

face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')           

FaceId= input('\n enter user id end press <return> ==>  ')                                # For each person, enter one numeric face id
print("\n  Initializing face capture. Look the camera and wait ...")     

count = 0                                                                                   # Initialize individual sampling face count
while(True):
    ret, image = cam.read()
    images = glob.glob("dataset/Samples"+ '.' +  str(FaceId) + '.' +  str(count) + ".jpg")
    imgs = []
    for fname in images:
     image = cv.imread(fname)
     imgs.append(image)

    dst = cv.fastNlMeansDenoisingColoredMulti(imgs, 3, 7, None, 10, 10, 7, 21)
    gray = cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(imgs, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv.imwrite("dataset/Samples"+ '.' +  str(FaceId) + '.' +  str(count) + ".jpg", gray[y:y+h,x:x+w])  # Save the captured image into the datasets folder
        cv.imshow('image', imgs)
    if cv.waitKey(1) & 0xff == ord('q'):                                                       #the 'q' is set as the qutting button
        break
    elif count >= 30:                                                                          # Take 100 face sample and stop video
         break
cam.release()
cv.destroyAllWindows()