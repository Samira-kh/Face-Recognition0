import cv2 as cv
import numpy as np
import os 
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascadePath)
font = cv.FONT_HERSHEY_SIMPLEX
# id = 0                                                           #iniciate id counter
names  = ['Jamali','Sayyad','Samira']                                  # names related to ids

cam = cv.VideoCapture(0)                                          # Initialize and start realtime video capture

cam.set(3, 640)                                                   # set video widht
cam.set(4, 480)                                                   # set video height
minW = 0.1*cam.get(3)                                             # Define min window size to be recognized as a face
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    Gblr49 = cv.GaussianBlur(img,( 7, 7), 0)
    Gblur = np.vstack([np.hstack ([img, Gblr49])])
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 10,
        minSize = (int(minW), int(minH)),)
    for(x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
         
        if (confidence < 82):                                    # If confidence is less them 100 ==> "0" : perfect match
            id1 = names[Id]
            confidence = "  {0}%".format(round(100 - confidence))
            print(confidence)
        else:
            id1 = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv.putText(
                    img, 
                    str(id1), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv.imshow('camera',img) 
    if cv.waitKey(1) & 0xff == ord('q'):                                    #the 'q' is set as the qutting button
        break
print("\n Exiting Program and cleanup stuff")
cam.release()
cv.destroyAllWindows()