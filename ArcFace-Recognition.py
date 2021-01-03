import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import os

KNOWN_FACES_DIR = '../Dataset'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

def compute_sim(emb1, emb2):
	emb1 = emb1.flatten()
	emb2 = emb2.flatten()
	from numpy.linalg import norm
	sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
	    
	return sim

#Detection Model
print('Loading and preparing Detection model')
detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
detector.prepare(ctx_id = -1, nms=0.4)

#Recognition Model
print('Loading and preparing Recognition Model')
recognizer = insightface.model_zoo.get_model('arcface_r100_v1')
recognizer.prepare(ctx_id = -1)


print('Loading faces from dataset...')
known_embeddings = []
known_names = []

#extracting embeddings for faces
for person_name in os.listdir(KNOWN_FACES_DIR):
	for file_name in os.listdir(KNOWN_FACES_DIR + '/' + person_name):
		img = cv2.imread(KNOWN_FACES_DIR + '/' + person_name + '/' + file_name)
		bbox, landmark = detector.detect(img, threshold=0.5, scale=1.0)
		COLORS = np.random.uniform(0, 255, size=(10, 3))
		startX = int(bbox[0][0])
		startY = int(bbox[0][1])
		endX = int(bbox[0][2])
		endY = int(bbox[0][3])
		face = img[startY:endY, startX:endX]
		face = cv2.resize(face, dsize=(112,112))
		#cv2.imwrite(KNOWN_FACES_DIR + '/' + person_name + '/cropped_' + file_name, face)
		embedding = recognizer.get_embedding(face)
		known_names.append(person_name)
		known_embeddings.append(embedding)


#testing on test image
test_img = cv2.imread('../test.jpg')
bbox, landmark = detector.detect(test_img, threshold=0.5, scale=1.0)

for detection in bbox:
	startX = int(detection[0])
	startY = int(detection[1])
	endX = int(detection[2])
	endY = int(detection[3])
	face = test_img[startY:endY, startX:endX]
	face = cv2.resize(face, dsize=(112,112))
	embedding = recognizer.get_embedding(face)

	print(embedding.shape)

	sim = 0
	label = ''
	for idx, emb in enumerate(known_embeddings):
		temp = compute_sim(emb, embedding)
		if temp > sim:
			sim = temp
			label = known_names[idx]

	cv2.rectangle(test_img, (startX, startY), (endX, endY), (255,0,0), 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv2.putText(test_img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

cv2.imwrite('../output.png', test_img)
