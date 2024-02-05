import cv2 as cv
import numpy as np
from PIL import Image
import os
path = r'C:\Users\khand\Desktop\Total\Project\VSC\Facedetection\dataset'                            # Path for face image database
recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
abs_path = r'C:\Users\khand\Desktop\Total\Project\VSC\Facedetection\dataset'

def getImagesAndLabels(path):
    imagePaths = [os.path.join(abs_path, x) for x in os.listdir(path) if x.endswith("jpg")]
    print(imagePaths)
    faceSamples = []                                                                                # create empth face list
    Ids = []                                                                                        # create empty ID list
   
    for imagePath in imagePaths:                                                                    # now looping through all the image paths and loading the Ids and the images
        
      pilImage = Image.open(imagePath).convert('L')                                                 # loading the image and converting it to gray scale
        
      imageNp = np.array(pilImage, 'uint8')                                                         # Now we are converting the PIL image into numpy array
       
      print(imagePath)
      Id = int((imagePath).split(".")[1])                                          # getting the Id from the image
      print(Id)
      faces = detector.detectMultiScale(imageNp)                                                    # extract the face from the training image sample
                              
      for (x, y, w, h) in faces:                                                                    # If a face is there then append that in the list as well as Id of it
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
            print(Id)
    return faceSamples, Ids 
print ("\n Training faces. It will take a few seconds. Wait ...")
faces,Ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('trainer/trainer.yml')                                                            # Save the model into trainer/trainer.yml 
print("\n  {0} Faces trained. Exiting Program".format(len(np.unique(Ids))))                        # Print the numer of faces trained and end program

