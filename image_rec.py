from PIL import Image
from os import listdir
from os import walk
from os.path import isfile, isdir, join
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

#anticipation
class LossHistory(keras.callbacks.Callback):
    def __init__(self,model,x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.i=0
    def on_train_end(self, logs=None):
        y_test_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        error=0
        for i in range(len(y_test)):
            if(y_test_pred[i]!=y_test[i]):
                error+=1
        print("\nTest Accurance: ",(len(y_test)*1.0-(error*1.0))/len(y_test))
        
#set up path
train_path = "./image"
files = listdir(train_path)
x_data=np.array([])
data_number=0

#Read all in the folder
for root, dirs, files in walk(train_path):
    for f in files:
        if data_number==0:
            fullpath = join(root, f)
            im = Image.open(fullpath)
            x_data = (np.array(im) / 255).reshape(1,28,28)  
            data_number += 1
        else:
            fullpath = join(root, f)
            im = Image.open(fullpath) 
            im = (np.array(im)/255).reshape(1,28,28)
            x_data = np.vstack((x_data,im)) 
            data_number += 1
x_data=x_data.reshape(data_number,28,28,1) #adjust data

#build label
y_data=[]
for k in range(0,43,1):
    for i in range(0,10,1):
        for j in range(0,5,1):
            y_data.append(i)

#read test data
test_path = "./test_image"
files = listdir(test_path)
x_test=[]
test_number=0
for root, dirs, files in walk(test_path):
    for f in files:
        fullpath = join(root, f)
        im = Image.open(fullpath)
        p = np.array(im)/255
        x_test.append(p)
        test_number+=1
x_test=np.array(x_test)
x_test=x_test.reshape(test_number,28,28,1)

#build up test label
y_test=[]
for k in range(0,6,1):
    for i in range(0,10,1):
        for j in range(0,5,1):
            y_test.append(i)

#one hot encoding
y_train = np_utils.to_categorical(y_data)

#buile up model
model = Sequential()
model.add(Conv2D(25, kernel_size=(3, 3),input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=[2, 2]))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

#set up the parameter of model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#setup record
history = LossHistory(model, x_test, y_test)

#start training
train_history = model.fit(x_data, y_train,epochs=10,verbose=1,shuffle=True,callbacks=[history])

#plot:to visualize the analysis
plt.plot(train_history.history['loss'])   
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss'], loc='upper left')  
plt.show() 

#save model
model.save('my_model.h5')
del model