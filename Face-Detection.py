import cv2 as cv
  

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')                                 # load the required trained XML classifiers


cam = cv.VideoCapture(0)                                                                                   #define a video capture object
  
while True: 
  
    result , frame  = cam.read()                                                                           #capture the video frame by frame 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                                                           #converting color image to gray_scale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                                                    #detects faces of different sizes in the input image

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)                                                  #drawing a rectangle around a face 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
  
    cv.imshow('WebCam',frame)                                                                              #display the resulting frame
    cv.imwrite('WebCam Image.jpg', frame)                                                                  #saving image in local storage

    if cv.waitKey(1) & 0xff == ord('q'):                                                                   #the 'q' is set as the qutting button
        break
  

cam.release()                                                                                              # Close the window
cv.destroyAllWindows() 