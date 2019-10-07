import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta, SGD
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score as PS
from sklearn.metrics import recall_score as RS
from sklearn.metrics import f1_score as F1

def prepare_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(60000, 28,28,1)
  x_test = x_test.reshape(10000, 28,28,1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_test = to_categorical(y_test)
  y_train = to_categorical(y_train)
  return x_train, y_train, x_test, y_test

def build_network():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))
  model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
  model.add(MaxPool2D((2, 2)))
  model.add(Dropout(0.20))
  model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
  model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])
  return model

def amax(y):
  return np.argmax(y, axis=1)

def evaluation(y_test, y_pred):
    print("test precision:", PS(amax(y_test),amax(y_pred),  labels=range(10), average="macro"))
    print("test recall:", RS(amax(y_test),amax(y_pred),  labels=range(10), average="macro"))
    print("test f1 score:", F1(amax(y_test),amax(y_pred),  labels=range(10), average="macro"))
