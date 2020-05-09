import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import joblib

from keras import backend as K

# fix random seed for reproducibility
seed = 7


colnames=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

iris = pd.read_csv("./iris.data", names=colnames, header=None)
iris.head()


X = iris.drop('species' ,axis=1)
y = iris['species']

encoder = LabelBinarizer()
y = encoder.fit_transform(y)

scaler = MinMaxScaler()

scaled_X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=seed)

model = Sequential()
model.add(Dense(units=64, input_shape=[4], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16,activation='relu'))
# Last layer for multi-class classification of 3 species
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(scaled_X,y, validation_data= (X_test, y_test), epochs=150)
#validation_data= (X_test, y_test)

model.save('final_iris_model.h5')

joblib.dump(scaler, 'iris_scaler.pkl')

###############
from keras.models import load_model

from tensorflow.keras.models import load_model
model = load_model('final_iris_model.h5')


K.clear_session()

##
##plt.plot(history.history['acc'])
##plt.plot(history.history['val_acc'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()
### summarize history for loss
##plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()





