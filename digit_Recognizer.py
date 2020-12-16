# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:40:19 2020

@author: agni1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("train.csv")
print(train.shape)
train.head()

test=pd.read_csv("test.csv")
print(test.shape)
test.head()

y_train=train['label']
x_train=train.drop(labels=['label'],axis=1)



g=sns.countplot(y_train,palette='icefire')
plt.title('Number of Digit Classes')
plt.show()



plt.imshow(x_train.iloc[0].to_frame().to_numpy().reshape((28,28)),cmap='gray')
plt.title(x_train.iloc[0,0])
plt.axis('off')
plt.show()




plt.imshow(x_train.iloc[3].to_frame().to_numpy().reshape((28,28)),cmap='gray')
plt.title(x_train.iloc[0,0])
plt.axis('off')
plt.show()


x_train=x_train/255.0
test=test/255.0

x_train.shape
test.shape

#Reshape
x_train=x_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
print('X train shape',x_train.shape)
print('test Shape', test.shape)

#Label Encoding
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train,num_classes=10)

#train & Test set split

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=2)
print('x train shape',x_train.shape)
print('x val shape',x_val.shape)
print('y train shape',y_train.shape)
print('y tval shape',y_val.shape)


plt.imshow(x_train[2][:,:,0],cmap='gray')
plt.show()


from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau



model=Sequential()
model.add(Conv2D(filters=8,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=8,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

#define the optimizer
optimizer=Adam(lr=0.001,beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

epochs=10
batch_size=250

#data augumentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                                         epochs=epochs,validation_data=(x_val,y_val),
                                         steps_per_epoch=x_train.shape[0]//batch_size
                                         )

#plot loss and accuracy curve of training and validation
plt.plot(history.history['val_loss'],color='b',label='Validation Loss')
plt.title('Test Loss')
plt.xlabel('Number of epolchs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#confusion matrix
y_pred=model.predict(x_val)
y_pred_classes=np.argmax(y_pred,axis=1)
y_true_classes=np.argmax(y_val,axis=1)
confusion_mtx=confusion_matrix(y_true_classes,y_pred_classes)
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx,annot=True,linewidths=0.01,cmap='Greens',linecolor='gray', fmt='.1f',ax=ax)
plt.xlabel('Prediction Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()