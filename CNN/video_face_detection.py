import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture('/home/marija/OpenCV/data/Megamind.avi')

while cap.isOpened():
	_, img = cap.read()
	faces = detector.detect_faces(img)
	
	for result in faces:
		x, y, w, h = result['box']
		x1, y1 = x + w, y + h
		cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
		
	cv2.imshow('img', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap.release()
