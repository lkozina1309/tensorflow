#Script image_face_detection.py is used for face detection on an image. It uses image 'chelsea.jpeg' which can be found indata folder.

import cv2
from mtcnn.mtcnn import MTCNN
 
detector = MTCNN()
img = cv2.imread('chelsea.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(img)

for (x, y , w ,h) in faces:
	cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
	
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
