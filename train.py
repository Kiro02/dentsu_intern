from mltools import *
import pandas as pd
#compile_args = {"optimizer": Adam(), "loss":"categorical_crossentropy", "metrics":["accuracy"]}

study_name = "cnn_mnist"

ok = OptKeras(study_name=study_name,
              monitor='val_acc',
              save_weights_only=True,
              direction='maximize')

x_train, y_train, x_test, y_test = prepare_data()

def train(compile_args=None, fit_args=None, optuna=True, ok=ok):
  def objective(trial):

    global model
    lr = trial.suggest_loguniform("lr", 1e-4, 0.5)
    algo = trial.suggest_categorical("optimizer", ["adam","sgd", "rmsprop", "adagrad", "adadelta"])
    model = build_network(optimizer=get_optimizer(name=algo, lr=lr))
    model.fit(x_train, y_train,
              validation_split=0.2,
              batch_size=128, epochs=5,
              callbacks=ok.callbacks(trial),
              verbose=ok.keras_verbose )
    return ok.trial_best_value

  if optuna:
    history = {"lr":[],"best_values":[],"optimizer":[]}
    for i in range(10):
      ok.optimize(objective, timeout = 2*60)
      history["lr"].append(ok.best_trial.params["lr"])
      history["optimizer"].append(ok.best_trial.params["optimizer"])
      history["best_values"].append(ok.best_trial.value)
      trial_num = ok.best_trial.number
      weight_file = ''.join([study_name,"_model_",
                        '*' if trial_num is None else '{:06d}'.format(trial_num),
                       ".h5" ])
      model.load_weights(weight_file)

  #else:
    #model.compile(**compile_args)
    #model.fit(**fit_args)
  return model, history

def main():
    trained_model, history = train(ok=ok)
    df = pd.DataFrame.from_dict(history)
    df.to_csv("history.csv")
    trained_model.save("mnist_cnn.h5")
    print("evaluation")
    trained_model.evaluate(x_test,y_test)

if __name__ == "__main__":
     main()
