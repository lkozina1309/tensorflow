#Script image_face_detection.py is used for face detection on an image. It uses image 'chelsea.jpeg' which can be found indata folder.

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
 
def rectangle(filename, result_list):
	data = pyplot.imread(filename)
	pyplot.imshow(data)
	ax = pyplot.gca()

	for result in result_list:
		x, y, width, height = result['box']
		rect = Rectangle((x, y), width, height, fill=False, color='green')
		ax.add_patch(rect)
	pyplot.show()
 
image = 'chelsea.jpeg'
read = pyplot.imread(image)
detector = MTCNN()
faces = detector.detect_faces(read)
rectangle(image, faces)
