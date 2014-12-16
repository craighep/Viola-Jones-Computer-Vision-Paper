import numpy as np
import cv2
import time
import glob

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

path = 'D:\Dropbox\Assignments\Computer Vision\BioID-FaceDatabase-V1.2\*.jpg'   
files = glob.glob(path)  
count = 0
faces_found = 0 
test_num = 18
for name in files:
	image = cv2.imread(name, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(image, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),3)
		roi_gray = image[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		faces_found += 1
		
	cv2.imwrite( name+".jpg", image );
	count += 1
	if (count > test_num ):
		break

perc = 100/ test_num * faces_found
print ""
print ("Faces found: " + str(faces_found))
print ( str(perc) +"% Accuracy" )

