"""
Tests on individual layers to make sure they preserve the required Equivariance
"""

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
from keras.layers import GlobalMaxPool1D, GaussianDropout, Lambda, Input
import numpy as np
from sim_train import load_data

def rc(tensor):
    return tensor[::,::-1,::-1]


dummy_input = np.array([[[1,2],[3,4],[5,6],[7,8]],])


dummy_input_2 = np.array([[[1,2,3,4],[5,6,7,8]],])
dummy_input_2_multiple_rows = np.array([[[1,2,3,4],[5,6,7,8]], [[1,2,2,4],[3,1,4,5]]])


# checking that filter sum pooling is invariant under reverse complementing

inp = Input(shape=(4,2))
inp2 = Input(shape=(2,4))
csumout = CustomSumPool()(inp)
csumout_2 = CustomSumPool()(inp2)


csumpool = Model(inputs=inp, outputs=csumout)
csumpool.compile("Adam",loss='categorical_crossentropy') # this is a bit silly

csumpool2 = Model(inputs=inp2, outputs=csumout_2)
csumpool2.compile("Adam",loss='categorical_crossentropy') # this is a bit silly


assert np.allclose(csumpool.predict(rc(dummy_input)), csumpool.predict(dummy_input))
assert np.allclose(csumpool2.predict(rc(dummy_input_2)), csumpool2.predict(dummy_input_2))
assert np.allclose(csumpool2.predict(rc(dummy_input_2_multiple_rows)), csumpool2.predict(dummy_input_2_multiple_rows))


## checking that convolutional filters commute after training
tf_inp = Input(shape=(1000,4))
conv_out = Conv1D(input_shape=(1000, 4),
                            filters=12,
                            kernel_size = (14),
                            padding = "valid",
                            activation = "elu")(tf_inp)

out = CustomSumPool()(conv_out)
out = GlobalMaxPool1D()(out)
out = Dense(2,activation="softmax")(out)


tf_classifier = Model(inputs=tf_inp, outputs=out)
tf_conv_out= Model(inputs=tf_inp, outputs=conv_out)



epochs = 4
lrate = 0.01
decay = lrate/epochs
adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
tf_classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
tf_conv_out.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(False)

es = EarlyStopping(patience=4, monitor='val_acc')
mrc = MotifMirrorGradientBleeding(0, assign_bias=True)
tf_classifier.fit(X_train, tflearn.data_utils.to_categorical(Y_train,2), validation_data=(X_val, tflearn.data_utils.to_categorical(Y_val,2)),
                 epochs=epochs, batch_size=32, callbacks=[es, mrc],verbose=True)



assert np.allclose(rc(tf_conv_out.predict(X_test)), tf_conv_out.predict(rc(X_test)), atol=1e-5)
