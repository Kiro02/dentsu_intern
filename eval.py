import keras
from mltools import evaluation, prepare_data

x_train, y_train, x_test, y_test = prepare_data()
model = keras.models.load_model("model.h5")

y_pred = model.predict(x_test)
evaluation(y_test, y_pred)
