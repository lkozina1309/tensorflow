from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
 
def draw_image_with_boxes(filename, result_list):
	data = pyplot.imread(filename)
	pyplot.imshow(data)
	ax = pyplot.gca()

	for result in result_list:
		x, y, width, height = result['box']
		rect = Rectangle((x, y), width, height, fill=False, color='green')
		ax.add_patch(rect)
	pyplot.show()
 
image = '/home/marija/tensorflow/CNN/my/chelsea.jpeg'
read = pyplot.imread(image)
detector = MTCNN()
faces = detector.detect_faces(read)
draw_image_with_boxes(image, faces)
