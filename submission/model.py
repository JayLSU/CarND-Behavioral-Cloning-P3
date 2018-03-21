import csv
import cv2
import numpy as np
import sklearn
import random 
random.seed(10)
samples = []

cvs_path = './simulator_data_MORE/driving_log.csv'


with open(cvs_path) as csvfile:
	reader = csv.reader(csvfile, skipinitialspace=True)
	for line in reader:
		samples.append(line)


import matplotlib.pyplot as plt
steering_angles = []

for row in samples:
	steering_angles.append(float(row[3]))


steering_angles = np.array(steering_angles)

num_bins = 25
avg_samples_per_bin = len(steering_angles)/num_bins
hists, bins = np.histogram(steering_angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hists, align='center', width=width)
plt.plot((np.min(steering_angles), np.max(steering_angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k--')
plt.show()

keep_probabilities = []
target = 0.7*avg_samples_per_bin
for i in range(num_bins):
    if hists[i] < target:
        keep_probabilities.append(1.)
    else:
        keep_probabilities.append(1./(hists[i]/target))
rm_list = []
for i in range(len(steering_angles)):
    for j in range(num_bins):
        if steering_angles[i] > bins[j] and steering_angles[i] <= bins[j+1]:
            if np.random.rand() > keep_probabilities[j]:
                rm_list.append(i)
samples = np.delete(samples, rm_list, axis=0)
steering_angles = np.delete(steering_angles, rm_list)

hists, bins = np.histogram(steering_angles, num_bins)
plt.bar(center, hists, align='center', width=width)
plt.plot((np.min(steering_angles), np.max(steering_angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k--')
plt.show()

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.1)

def generator(samples, batch_size = 128):
	num_samples = len(samples)
	while True:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			imgs = []
			measurements = []
			for batch_sample in batch_samples:
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('\\')[-1]
					current_path = './simulator_data_MORE/IMG/' + filename
					image = cv2.imread(current_path)
					measurement = float(batch_sample[3])
					correction = 0.25
					if i == 0:
						pass
					elif i == 1:
							measurement += correction
					else:
							measurement -= correction
					if np.random.randint(1) == 0:
						image = np.fliplr(image)
						measurement = measurement * -1.0

					imgs.append(image)
					measurements.append(measurement)


			X_train = np.array(imgs)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

print('Total samples: {}.'.format(3*len(samples)))
print('Training samples: {}.'.format(3*len(train_samples)))
print('Validation samples: {}.'.format(3*len(validation_samples)))

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = \
	3*len(train_samples), validation_data = validation_generator, \
	nb_val_samples=3*len(validation_samples), nb_epoch = 3, verbose=1)
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
