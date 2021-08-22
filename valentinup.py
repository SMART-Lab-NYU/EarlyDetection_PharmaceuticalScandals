# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 16:24:03 2021

@author: hm2801
"""

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import glob
import cv2
from sklearn.metrics import accuracy_score
####################################################################
fdconso = open('./Results/125_w-_34.csv','w')
fdconso.write("All\n")
fdconso.write("File,Accuracy\n")
path='./NewData/125_w-_34/all/'
tb_dir='./tb'
mod_dir='./model'
nb_classes=2
img_channels=3
nb_filters=32
listing=os.listdir(path)
matdata=[]
label=[]
for file in listing:
    fname =glob.glob(path+'/'+file+'/*.png')
    for fn in fname:
        im=cv2.resize(cv2.imread(fn),(100,100)).reshape(-1)
        matdata.append(np.r_[int(file),im])
data = np.array(matdata, dtype='uint8')
X=data[:,1:]
y=data[:,0]
model=Sequential()
model.add(Convolution2D(32, (5,5), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
fd = open('./Results/125_w-_34all.csv','w')
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=4)
org=np.array([], dtype=np.float64)
pred=np.array([], dtype=np.float64)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train=X_train.reshape(X_train.shape[0],100,100,3)
    X_test=X_test.reshape(X_test.shape[0],100,100,3)
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')
    X_train/=255
    X_test/=255
    print('X_train shape:',X_train.shape)
    Y_train=np_utils.to_categorical(Y_train,2)
    Y_test=np_utils.to_categorical(Y_test,2)
    model.fit(X_train, Y_train,20,100)
    md=model.predict_proba(X_test)
    md1=model.predict_classes(X_test)
    cd=np.column_stack((md,y_test))
    cd=np.column_stack((cd,md1))
    org=np.concatenate((org, y_test), axis=0)
    pred=np.concatenate((pred, md1), axis=0)
    np.savetxt(fd, cd, delimiter=",")
fd.close()
fdconso.write("all,"+str(accuracy_score(org,pred))+"\n")

