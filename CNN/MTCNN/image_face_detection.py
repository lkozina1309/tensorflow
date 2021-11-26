#Script image_face_detection.py is used for face detection on an image. It uses image 'chelsea.jpeg' which can be found in data folder.

import cv2
from mtcnn.mtcnn import MTCNN
 

img = cv2.imread("chelsea.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector = MTCNN()
faces = detector.detect_faces(img)

for i in range(len(faces)):
	x1, y1, width, height = faces[i]['box']
	x2, y2 = x1 + width, y1 + height
	cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0 , 0), 3)
	
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
