from mltools import *

x_train, y_train, x_test, y_test = prepare_data()
model = build_network()

model.fit(x_train, y_train,
              validation_split=0.2,
              batch_size=128, epochs=50)

print("test accuracy:", model.evaluation(x_test, y_test)[1])
model.save("model.h5")
