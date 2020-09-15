#Importing required libraries, i.e. OpenCV, Numpy and Tensor Flow
import cv2 as cv
import numpy as np
import tensorflow as tf

#importing the dataset form mnist
mnist=tf.keras.datasets.mnist
#splitting the data in training and testing datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scaling down the training and test datasets
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#defining the model, which'll have a input layer, two hidden layers and an output layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  #flatten means it's a simple feet forwaed neural network
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  #dense means all the neurons are connected to
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  #previous and the next layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
