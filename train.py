import numpy as np
import cv2 
import os 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import tensorflow as tf
import keras

'''
TODO

Сделать xml или таблицу эксель где будет рахграничения по цифрам и буквам 
10 = А
11 = Б
12 = В
и тд.


'''

##############################################

path = 'myData'
path = 'E:\\Python Projects\\Licence_plate_recognition\\train\\TestData\\'

max_sizeW = 32
max_sizeH = 32
testRatio = 0.2
valRatio = 0.2
imageDimensions = (max_sizeW,max_sizeH,3)


###############################################

#symbolsList = "0,1,2,3,4,5,6,7,8,9,A,Б,В,Г,Д,Е"

count=0
images = []
classNo = []
myList = os.listdir(path)
print("Total No of Classes Detected ",len(myList))
print("Importing Calsses...")
noOfClasses = len(myList)

#symbols = symbolsList.split(',')
#print(symbols)

#for x in range (0,noOfClasse):
#    myPicList = os.listdir(path+"/"+str(symbols[x]))
#    #myPicList = os.listdir(path+"/"+str(x))
#    for y in myPicList:
#        #print(y)
#        #curImg = cv2.imread(path+"/"+str(x)+"/"+y)
#        curImg = cv2.imread(path+"/"+str(symbols[x])+"/"+ y)
##        if symbols[x] == 'А':
# #           print(symbols[x])
# #           print(curImg)
#            #cv2.imshow("Test",curImg)
#  #      curImg = cv2.resize(curImg, ( max_size, max_size))
#   #     images.append(curImg)
#    #    classNo.append(str(x))
#   # print(x)


for x in range (0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, ( max_sizeW, max_sizeH))
        images.append(curImg)
        classNo.append(x)
    print(count, end=" ")
    count+=1
print("\nDone ")
print(" ")
print("Total Images in Images List = ",len(images))
print("Total IDS in classNo List= ",len(classNo))
 
#### CONVERT TO NUMPY ARRAY
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
 
#### SPLITTING THE DATA
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

#### PLOT BAR CHART FOR DISTRIBUTION OF IMAGES
numOfSamples = []
for x in range(0,noOfClasses):
    #print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)


# plt.figure(figsize =(10,5))
# plt.bar(range(0,noOfClasses),numOfSamples)
# plt.title("Number of images of each class")
# plt.xlabel("Class ID")
# plt.ylabel("Number of Images")
#plt.show()


def preProcessing (img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img =img / 255
    return img 

#img = preProcessing(X_train[30])
#img = cv2.resize(img,(300,300))
#cv2.imshow("Preprocessed IMG",img)
#cv2.waitKey(0)


X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


# Normalize
# X_train = X_train.astype(np.float32)
# X_train /= 255.0
# X_test = X_test.astype(np.float32)
# X_test /= 255.0
# X_validation = X_validation.astype(np.float32)
# X_validation /= 255.0




#### IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range = 0.2,
                             shear_range = 0.2,
                             rotation_range = 10)
dataGen.fit(X_train)
 
#### ONE HOT ENCODING OF MATRICES
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)


#### CREATING THE MODEL
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (3, 3)
    sizeOfFilter2 = (2, 2)
    sizeOfPool = (2, 2)
    noOfNodes = 500 # 70 - Test Accuracy = 0.9120879173278809 № 
    path_save ='E:\\Python Projects\\Licence_plate_recognition\\train\\TrainModel\\'

    best_w =keras.callbacks.ModelCheckpoint(path_save + 'fcn_best.h5',
                                            monitor= 'val_loss',
                                            verbose=0,
                                            save_best_only= True,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1)
                                            
    last_w =keras.callbacks.ModelCheckpoint(path_save + 'fcn_last.h5',
                                            monitor= 'val_loss',
                                            verbose=0,
                                            save_best_only= False,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1)
                                        
    best_res = keras.callbacks.ModelCheckpoint(path_save + 'fcn_best_res_09.h5',
                                        monitor= 'val_loss',
                                        verbose=0,
                                        save_best_only= True,
                                        save_weights_only= False,
                                        mode='auto',
                                        period=1)
    # Set a learning rate reduction
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', patience = 2, verbose = 1, mode='auto', factor=0.5, min_lr=0.0000001)

    callbacks =[learning_rate_reduction, best_w, last_w, best_res]
    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))


    
    model.add((Conv2D(noOfFilters//3, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//3, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
 
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model, callbacks


batchSizeVal = 64 ## 64 32 
epochsVal = 150
stepsPerEpochVal = 2000
max_epoch = 11;

path_save_model ='E:\\Python Projects\\Licence_plate_recognition\\train\\TrainModel\\model_trained_123.h5'
path_learn_model ='E:\\Python Projects\\Licence_plate_recognition\\train\\TrainModel\\model_trained_15.h5'

min_lern_Rate = 0.1

'''
for i in range (0,max_epoch):

    model = myModel(min_lern_Rate)
    print(min_lern_Rate)
    min_lern_Rate = min_lern_Rate / 10

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_save_model,
                                                 save_weights_only=False,
                                                 verbose = 1,
                                                 period = 5)

    #model = keras.models.load_model(path_learn_model)

    #### STARTING THE TRAINING PROCESS
    history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size = batchSizeVal),
                                 steps_per_epoch = stepsPerEpochVal,
                                 epochs = epochsVal,
                                 validation_data = (X_validation,y_validation),
                                 shuffle = 1,
                                 callbacks=[cp_callback])
'''
#print(model.summary())
model, callbacks = myModel()

path_save_model_1 ='E:\\Python Projects\\Licence_plate_recognition\\train\\TrainModel\\'

#model = keras.models.load_model(path_save_model_1 + 'fcn_best_res.h5' )


history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size = batchSizeVal),
                                 steps_per_epoch = stepsPerEpochVal,
                                 epochs = epochsVal,
                                 validation_data = (X_validation,y_validation),
                                 #shuffle = 1,
                                 verbose = 1,
                                 callbacks = callbacks)


#### PLOT THE RESULTS 



#### SAVE THE TRAINED MODEL

#model.save(path_save_model + "model_trained.p")
model.save( path_save_model_1 +'train_all_test_005.h5')

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])
 



#pickle_out = open(path_save_model+"model_trained.p", "wb")
#pickle_out = path_save_model+"model_trained.p"
#pickle.dump(model,open(pickle_out,'wb'))
#pickle_out.close()

# print(X_train.shape)
# img = X_train[30]
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocessed IMG",img)
# cv2.waitKey(0)