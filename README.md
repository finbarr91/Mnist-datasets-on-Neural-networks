# Mnist-datasets-on-Neural-networks
keras, python, tensorflow
# Importing the library
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dense, Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Loading the dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(len(train_labels)):
    print(class_names[train_labels[i]])

# To visualize the shapes of the datasets
print(train_images.shape)
print(test_images.shape)


# To visualize our datasets
i = 2000
plt.imshow(train_images[i])
plt.title(class_names[train_labels[i]])
plt.show()

# creating the image matrix
ncol = 15
nrows = 15

fig,axes = plt.subplots(ncol,nrows, figsize=(25,25))
axes = axes.ravel()
training_data = len(train_images)

for i in np.arange(0,ncol*nrows):
    index = np.random.randint(0,training_data)
    axes[i].imshow(train_images[index])
    axes[i].set_title(class_names[train_labels[index]], size= 5)
    axes[i].axis('off')

plt.subplots_adjust(hspace=2)
plt.show()

# Data preparation

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# converting the train and test labels to categorical data
classes = 10
train_labels = keras.utils.to_categorical(train_labels, classes)
test_labels = keras.utils.to_categorical(test_labels,classes)

# To normalize the data
train_images = train_images/255.0
test_images = test_images/255.0


# Building the CNN model
model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units= 512, activation='relu'))
model.add(Dense(units= 512, activation= 'relu'))
model.add(Dense(units=10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr=0.001), metrics=['acc'])
histroy = model.fit(train_images, train_labels, batch_size= 32, epochs =10, shuffle =True)

# Evaluation of our model
'''
Remark:
In the evaluation of the model, the model is evaluated using the testing data
 '''
evaluation= model.evaluate(test_images,test_labels)
print('Test Accuracy : {}'. format(evaluation[1]))

prediction= model.predict_classes(test_images)
print('Predicted classes of X_test\n', prediction)

'''
We need to return our y_test from binary to decimal(integer) values so as to make a good comparison of 
the predicted classes of x_test with the actual classes
'''
y_test = test_labels.argmax(1)
print('y_test in decimal\n',y_test)

nrows = 7
ncols = 7

fig,axes = plt.subplots(nrows,ncols,figsize=(12,12))
axes = axes.ravel()

for i in np.arange(0, nrows*ncols):
    axes[i].imshow(test_images[i])
    axes[i].set_title('Prediction ={}\n True ={}'.format(prediction[i],y_test[i]), size=5)
    axes[i].axis('off')

plt.subplots_adjust(wspace=1, hspace=2)
plt.show()

# Confusion Matrix : This is used to summarize all our results in one matrix
cm = confusion_matrix(y_test, prediction)
print('Confusion Matrix\n', cm)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True)
plt.show()


# To save the model
directory = os.path.join(os.getcwd(), 'saved_models') # get the current directory and a file name called saved_model

if not os.path.isdir(directory): # If there is no folder called saved model in the directory, then create one
    os.makedirs(directory)

model_path = os.path.join(directory, 'keras_Mnist_trained_model.h5') # creating the model path
model.save(model_path) # saving the model path
