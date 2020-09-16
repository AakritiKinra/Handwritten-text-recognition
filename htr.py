#Importing required libraries, i.e. OpenCV, Numpy and Tensor Flow
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#importing the dataset form mnist
mnist=tf.keras.datasets.mnist
#splitting the data in training and testing datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scaling down the training and test datasets
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#defining the model, which will have a input layer, two hidden layers and an output layer
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),   #flatten means it's a simple feet forwaed neural network
    tf.keras.layers.Dense(units=128,activation=tf.nn.relu),   #dense means all the neurons are connected to
    tf.keras.layers.Dense(units=128,activation=tf.nn.relu),   #previous and the next layer
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('digits.model')


for x in range(1,6):
    img = cv.imread(f'{x}.jpeg')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'Probably the result is: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
