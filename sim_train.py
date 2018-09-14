
"""
Usage: simulation_train <network_json> [<n_trials>]



Options:

  <n_trials>  Number of runs of the net [default: 50].



"""

import docopt



import random
import numpy as np
import re
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import keras
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import Constant
import keras.backend as K
from keras.engine.topology import Layer
import keras.backend as K
from core import MotifMirrorGradientBleeding, CustomSumPool, CustomMeanPool, MCRCDropout
import gzip as gz
from data_utils import one_hot, train_test_val_split, reverse_complement
from keras.layers import BatchNormalization
from keras.layers import Lambda


def augment_data(sequences, responses):
    # we use this method as we want to augment just the training data
    s_new, resp_new = [], []
    for seq, resp in zip(sequences, responses):
        s_new.append(seq)
        s_new.append(seq[::-1,::-1])# this assumes the correct encoding 
        resp_new.append(resp)
        resp_new.append(resp)

    return np.array(s_new), np.array(resp_new)


def load_data(aug):
    sequences = []
    responses = []
    with gz.open("additive_training_dat.gz") as training_dat:
        for line in training_dat:
            line = line.decode("ascii")
            seq, resp = line.strip().split(" ")
            sequences.append(one_hot(seq))

            responses.append(int(resp))
            if aug:
                sequences.append(one_hot(reverse_complement(seq)))
                responses.append(int(resp))


    X_train, X_val, X_test, Y_train, Y_val, Y_test = train_test_val_split(np.array(sequences), np.array(responses))
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


import types

def predict_mc(self, X_pred, n_preds=100):
    return np.mean([self.predict(X_pred) for i in range(n_preds)], axis=0)


def generate_model(nn_params):
    K.clear_session()
    tf_classifier = Sequential()

    tf_classifier.add(Conv1D(input_shape=(1000, 4),
                            filters=nn_params["input_filters"],
                            kernel_size = (nn_params["filter_length"]),
                            padding = "valid",
                            activation = nn_params["activation"],
                            kernel_regularizer=l2(nn_params["reg"])))
   

    if nn_params["apply_rc"]:
        divisor = 2
        tf_classifier.add(CustomMeanPool())
    else:
        divisor = 1
     
    if nn_params["batch_norm"]:
        tf_classifier.add(BatchNormalization())
    # batch norm never actually used with dropout or Equivariance

   
    if nn_params["use_dropout"]:
         if nn_params["mc_dropout"]:
              tf_classifier.add(Lambda(lambda x: K.dropout(x, level=0.1)))
         else: #don't need mcrc dropout as it appears after sum pool
              tf_classifier.add(Dropout(0.1))
    

    tf_classifier.add(GlobalMaxPool1D())

    if nn_params["custom_init"]:
        tf_classifier.add(Dense(2,activation="softmax", kernel_initializer=Constant(np.array([[1]*(nn_params["input_filters"]//divisor), [1]*(nn_params["input_filters"]//divisor) ])), bias_initializer=Constant(np.array([1, -1]))))
    else:
        tf_classifier.add(Dense(2,activation="softmax"))
    epochs = 50
    lrate = 0.01
    decay = lrate/epochs
    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)


    tf_classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    tf_classifier.predict_mc = types.MethodType(predict_mc, tf_classifier)
    return tf_classifier

def train(nn_params, n_trials):
    batch_size=32
    from keras.callbacks import EarlyStopping
    import tflearn
    results = []

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(False)
    if nn_params["augment_data"]:
        X_train, Y_train = augment_data(X_train, Y_train)

    ### removing training data optionally to test how this impacts results
    if "data_frac" in nn_params:
        n_data = int(float(nn_params["data_frac"]) * len(X_train))
        X_train = X_train[:n_data]
        Y_train = Y_train[:n_data]

    for trial in range(n_trials):

        tf_classifier = generate_model(nn_params)

        es = EarlyStopping(patience=4, monitor='val_acc')
        mrc = MotifMirrorGradientBleeding(0,assign_bias=True)
        if nn_params["apply_rc"]:
            callbacks = [es,mrc]
        else:
            callbacks=[es]

        tf_classifier.fit(X_train, tflearn.data_utils.to_categorical(Y_train,2), validation_data=(X_val, tflearn.data_utils.to_categorical(Y_val,2)),
                 epochs=50, batch_size=batch_size, callbacks=callbacks,verbose=True)
        
        if nn_params["mc_dropout"] and nn_params["use_dropout"]:
            predictions = tf_classifier.predict_mc(X_test)
        else:
            predictions = tf_classifier.predict_mc(X_test, n_preds=1)

        results.append((Y_test.tolist(), predictions.tolist()))


    with open(nn_params["output_prefix"]+".json","w") as outfile:
        json.dump(results, outfile)






if __name__ == "__main__":

    args = docopt.docopt(__doc__)
    import json
    nn_args = json.load(open(args["<network_json>"]))
    if args["<n_trials>"] is None:
        args["<n_trials>"] = 50

    if "batch_norm" not in nn_args:
        nn_args["batch_norm"] = 0

    if "augment_data" not in nn_args:
        nn_args["augment_data"] = 0
    print (nn_args)
    train(nn_args, int(args["<n_trials>"]))

