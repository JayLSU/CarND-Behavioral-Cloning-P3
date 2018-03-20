import csv
import cv2
import numpy as np
import sklearn
samples = []
with open('./simulator_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size = 64):
	num_samples = len(samples)
	while True:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			imgs = []
			measurements = []
			for batch_sample in batch_samples:
				for i in range(3):
					source_path = line[i]
					filename = source_path.split('\\')[-1]
					current_path = './simulator_data/IMG/' + filename
					image = cv2.imread(current_path)
					measurement = float(line[3])
					flipped_image = np.fliplr(image)
					correction = 0.25
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
			yield sklearn.utils.shuffle(X_train, y_train)


print('Train samples: {}.'.format(len(train_samples)))
print('Validation samples: {}.'.format(len(validation_samples)))

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = \
	6*len(train_samples), validation_data = validation_generator, \
	nb_val_samples=len(validation_samples), nb_epoch = 3)
model.save('model.h5')

print(history_object.history.keys())
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()

exit()