""" 
Usage: simulation_train <network_json> [<n_trials>] [-p]



Options:

  <n_trials>  Number of runs of the net [default: 50].
  -p          Print the best network


"""




import random
import numpy as np

import re
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# disabled to run in cluster


import docopt
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Dropout, MaxPool1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import Constant
import keras.backend as K
from keras.engine.topology import Layer
from core import MotifMirrorGradientBleeding, CustomSumPool, MCRCDropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import tflearn
from data_utils import one_hot, train_test_val_split, reverse_complement, load_recomb_data
from keras.models import Model
from keras.layers import GlobalMaxPool1D, GaussianDropout, Lambda
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score
import gzip as gz
import types
from keras.layers import BatchNormalization
from sim_train import augment_data



def predict_mc(self, X_pred, n_preds=100):
    return np.mean([self.predict(X_pred) for i in range(n_preds)], axis=0)
    




def generate_model(epochs, nn_params):


    bigger_model = Sequential()    
    
    input_height = 4
    input_length=997

    bigger_model.add(Conv1D(input_shape=(input_length, input_height), #but one channel in the one hot encoding of the genome
                         filters=nn_params["input_filters"],
                         kernel_size=nn_params["filter_length"],
                         strides=1,
                         padding="valid",
                           activation=nn_params["activation"],
                         kernel_regularizer=l2(nn_params["reg"])

                        ))

    if nn_params["batch_norm"]:
        bigger_model.add(BatchNormalization())

    if nn_params["use_dropout"]:
         if nn_params["mc_dropout"]:    
              if nn_params["apply_rc"]:
                  bigger_model.add(MCRCDropout(0.1))
              else:
                  bigger_model.add(Lambda(lambda x: K.dropout(x, level=0.1)))
         else:
               bigger_model.add(Dropout(0.1))


    bigger_model.add(MaxPool1D(pool_size=8))

    bigger_model.add(Conv1D(
                         filters=nn_params["input_filters"],
                         kernel_size=4,
                         strides=1,
                         padding="valid",
                          activation=nn_params["activation"],
                         kernel_regularizer=l2(nn_params["reg"])
                        ))


    if nn_params["batch_norm"]:
        bigger_model.add(BatchNormalization())

    if nn_params["use_dropout"]:
         if nn_params["mc_dropout"]:
              if nn_params["apply_rc"]: 
                  bigger_model.add(MCRCDropout(0.1))
              else:
                  bigger_model.add(Lambda(lambda x: K.dropout(x, level=0.1)))
         else:
              bigger_model.add(Dropout(0.1))


    if nn_params["apply_rc"]:
        bigger_model.add(CustomSumPool())
        divisor = 2
    else:
        divisor = 1


    bigger_model.add(GlobalMaxPool1D(
                           ))

    if nn_params["custom_init"]:
        bigger_model.add(Dense(2, activation="softmax", kernel_initializer=Constant(np.array([[1]*(nn_params["input_filters"]//divisor), [1]*(nn_params["input_filters"]//divisor)])),
        bias_initializer=Constant(np.array([1, -1]))))
    else:
        bigger_model.add(Dense(2, activation="softmax"))


    lrate = 0.01
    decay = lrate/epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    bigger_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    bigger_model.predict_mc = types.MethodType(predict_mc, bigger_model)
    return bigger_model


def binary_accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)))


def train(nn_params, n_trials, save_best):
    rocs = []
    res = []
    best_acc = 0
    val_accs = []
    raw = []


    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_recomb_data(False)
    
    if nn_params["augment_data"]:
        X_train, Y_train = augment_data(X_train, Y_train)

    ### removing training data optionally to test how this impacts results
    if "data_frac" in nn_params:
        n_data = int(float(nn_params["data_frac"]) * len(X_train))
        X_train = X_train[:n_data]
        Y_train = Y_train[:n_data]

    for trail in range(n_trials):
        epochs=50
        model=generate_model(epochs, nn_params)
        mm_0 = MotifMirrorGradientBleeding(0,assign_bias=True)
        mm_1 = MotifMirrorGradientBleeding(2,assign_bias=True)

        es = EarlyStopping(monitor='val_acc', patience=4)
        if nn_params["apply_rc"]:
            callbacks=[es, mm_0, mm_1]
        else:
            callbacks=[es]

        model.fit(X_train, tflearn.data_utils.to_categorical(Y_train,2),
              validation_data=(X_val, tflearn.data_utils.to_categorical(Y_val,2)),
              epochs=epochs, batch_size=64, callbacks=callbacks)

        if nn_params["mc_dropout"] and nn_params["use_dropout"]:
            predictions = model.predict_mc(X_test)
        else:
            predictions = model.predict_mc(X_test, n_preds=1)

        acc = binary_accuracy(tflearn.data_utils.to_categorical(Y_test,2), predictions)
        raw.append((Y_test.tolist(), predictions.tolist()))
        if save_best:
             if acc > best_acc:
                 best_acc = acc
                 from data_utils import save_model_yaml
                 save_model_yaml(model,"SavedModel")

        res.append(acc)

       
    with open(nn_params["output_prefix"]+".json","w") as outfile:
        json.dump(raw, outfile)
        
    

#print (rocs)

if __name__ == "__main__":

    args = docopt.docopt(__doc__)
    import json
    nn_args = json.load(open(args["<network_json>"]))
    if args["<n_trials>"] is None:
        args["<n_trials>"] = 50

    
    # not that this is currently incompatible with equivariance, as trials on asymmetric data didn't prove 
    # useful in these small networks 
    if "batch_norm" not in nn_args:
        nn_args["batch_norm"] = 0

    if "augment_data" not in nn_args:
        nn_args["augment_data"] = 0
   
    print (nn_args)
    train(nn_args, int(args["<n_trials>"]), args["-p"])

