import csv
import cv2
import numpy as np
from scipy import ndimage

lines = []
# with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
# print(lines[0:3])
for line in lines:
    if line[0] == 'center':
        continue
    source_path_c = line[0]
    source_path_l = line[1]
    source_path_r = line[2]
    filename_c = source_path_c.split('/')[-1]
    filename_l = source_path_l.split('/')[-1]
    filename_r = source_path_r.split('/')[-1]
    
#     print(filename)
    current_path_c = './data/IMG/' + filename_c
    current_path_l = './data/IMG/' + filename_l
    current_path_r = './data/IMG/' + filename_r
    
   
    image_c = ndimage.imread(current_path_c)
    image_l = ndimage.imread(current_path_l)
    image_r = ndimage.imread(current_path_r)
    images.append(image_c)
    images.append(image_l)
    images.append(image_r)
    
    correction = 0.1
    measurement_c = float(line[3])
    measurement_l = measurement_c + correction
    measurement_r = measurement_c - correction
    measurements.append(measurement_c)
    measurements.append(measurement_l)
    measurements.append(measurement_r)
    
# print(measurements)
# print(y_train[1249:1260])

#Augment data
# Flip images and measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image,1))
#     augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5 , input_shape=(160,320,3))) #(x/255 -0.5)
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')
