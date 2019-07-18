from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten
import scipy.io as sio
import numpy as np
import random
import model_plot as pt
import matplotlib.pyplot as plt


# read data from matlab
x_input=sio.loadmat('input_x.mat')
y_input=sio.loadmat('input_dishwasher.mat')
x_train = x_input.get('input_x')
y_train = y_input.get('input_app5')
x_train = np.transpose(x_train)
x_train = x_train[:,:,np.newaxis]
y_train = np.transpose(y_train)

# model building
model = Sequential()
model.add(Conv1D(16, kernel_size=8, activation='relu', input_shape=(99,1), name='conv1'))
model.add(Conv1D(32, kernel_size=16, activation='relu', name='conv2'))
model.add(Conv1D(64, kernel_size=16, activation='relu', name='conv3'))
model.add(Conv1D(16, kernel_size=8, activation='relu', name='conv4'))
model.add(Conv1D(8, kernel_size=4, activation='relu', name='conv5'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='linear'))

# model setting
model.compile(optimizer='adam', loss='mse',metrics=['mae'])
model.summary()



# test data:80%, valid data:20%
x_valid = x_train[int(0.2*len(x_train)):,:,:]
y_valid = y_train[int(0.2*len(y_train)):,:]
x_train = x_train[0:int(0.8*len(x_train)),:,:]
y_train = y_train[0:int(0.8*len(y_train)),:]

# random the dataset
index = [i for i in range(len(x_train))]
random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]


# model saving
hist=model.fit(x_train, y_train, batch_size=32, epochs=30,validation_data=(x_valid,y_valid))
model.save('CNN_model_dishwasher.h5')
pt.plot_history(hist)
