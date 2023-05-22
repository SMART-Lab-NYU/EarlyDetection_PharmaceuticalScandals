import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#one hot encoding function
def one_hot_encoder(df_name, df_column_name, suffix=''):
    temp = pd.get_dummies(df_name[df_column_name]) #get dummies is used to create dummy columns
    df_name = df_name.join(temp, lsuffix=suffix) #join the newly created dummy columns to original dataframe
    df_name = df_name.drop(df_column_name, axis=1) #drop the old column used to create dummy columnss
    return df_name

#function to draw confusion matrix
def draw_confusion_matrix(true,preds):
    conf_matx = confusion_matrix(true, preds)
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.show()
    #return conf_matx
    
def cnn_model(size, num_cnn_layers):
    NUM_FILTERS = 32
    KERNEL = (3, 3)
    #MIN_NEURONS = 20
    MAX_NEURONS = 120
    model = Sequential()
    model.add(Conv2D(128, (6,6), input_shape=(100,100,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (6,6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (6,6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    return model

def fit_and_evaluate(tr_x, ts_x, tr_y, ts_y, tro_y, tso_y, EPOCHS=100, BATCH_SIZE=50):
    model = None
    model = cnn_model(IMAGE_SIZE, 2)
    results = model.fit(tr_x, tr_y, epochs=EPOCHS, validation_split = 0.2, batch_size=BATCH_SIZE,  verbose=1)  
    print("Val Score: ", model.evaluate(ts_x, ts_y))
    cc=model.predict_classes(ts_x)
    del model
    return cc
#callbacks=[early_stopping, model_checkpoint],

path='./Data'
fdal = open('./Results/Consolidated.txt','w')
fname =glob.glob(path+'/'+'*.csv')
for fn in fname:
    print(fn)
    train_images = pd.read_csv(fn)
    train_images_x = train_images.iloc[:,1:]
    train_images_array = train_images_x.values
    train_x = train_images_array.reshape(train_images_array.shape[0], 100, 100, 3)
    train_x_scaled = train_x/255
    IMAGE_SIZE = (100, 100, 3)
    train_images_y = train_images[['0']]
    #do one hot encoding with the earlier created function
    train_images_y_encoded = one_hot_encoder(train_images_y, '0', 'lab')
    #get the labels as an array
    train_images_y_encoded = train_images_y_encoded.values
    train_images_y=train_images.iloc[:,0].values
    # Defining the CNN Architecture
    model = cnn_model(IMAGE_SIZE, 2)
    model.summary()
    pat = 5 
    early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
    model_checkpoint = ModelCheckpoint('pharmacoModel.h5', verbose=1, save_best_only=True)
    n_folds=5
    epochs=100
    batch_size=50
    model_history = [] 
    logo = StratifiedKFold(n_splits=n_folds)
    subject_index = 1
    preds=[]
    trues=[]
    for train_index, test_index in logo.split(train_x_scaled, train_images_y, train_images_y_encoded):
        train_x, test_x = train_x_scaled[train_index], train_x_scaled[test_index]
        train_y,test_y= train_images_y_encoded[train_index], train_images_y_encoded[test_index]
        train_y_org,test_y_org=train_images_y[train_index], train_images_y[test_index]
        model_checkpoint = ModelCheckpoint('.\Models\\pharmacoModel_'+str(subject_index)+'.h5', verbose=1, save_best_only=True)
        ft=fit_and_evaluate(train_x, test_x, train_y, test_y,train_y_org,test_y_org, epochs, batch_size)
        print("======="*12, end="\n\n\n")
        subject_index = subject_index + 1
        preds=np.concatenate((preds,ft),axis=0)
        trues=np.concatenate((trues,test_y_org))
    cr=classification_report(trues,preds)
    cnf=confusion_matrix(trues, preds)
    print(cr)
    print(cnf)
    print(fn)
    op=fn.split('/')[2].split('.')[0]
    fd = open('./Results/'+op+'.txt','w')
    np.savetxt(fd, cnf, delimiter=",")
    fd.close()
    fdal.write(op)
    fdal.write('.txt\n')
    fdal.write(str(accuracy_score(trues,preds)))
    fdal.write('\n')
fdal.close()