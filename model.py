import csv
import cv2
import numpy as np

lines = []
with open('../simulator_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

imgs = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('\\')[-1]
		current_path = '../simulator_data/IMG/' + filename
		image = cv2.imread(current_path)
		measurement = float(line[3])
		flipped_image = np.fliplr(image)
		correction = 0.2
		if i == 0:
			pass
		elif i == 1:
				measurement += correction
		else:
				measurement -= correction
		imgs.append(image)
		measurements.append(measurement)
		imgs.append(flipped_image)
		measurements.append(measurement*(-1.0))



X_train = np.array(imgs)
y_train = np.array(measurements)
print(X_train.shape, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Convolution2D
from keras.layers.pooling import MaxPooling2D
in_size = X_train[0].shape
print(X_train[0].shape)
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = in_size))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.save('model.h5')
exit()

