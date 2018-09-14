# testing that the predictions of the reverse complements are the same as forward strands for the entire 
# dataset



from data_utils import load_model_yaml, load_recomb_data
from core import MCRCDropout, CustomSumPool
from recombination_train import predict_mc
import types
import numpy as np
import importlib


import reproduce

recombination_model = load_model_yaml("SavedModel", cust_objects={"MCRCDropout":MCRCDropout, "CustomSumPool": CustomSumPool})
recombination_model.predict_mc = types.MethodType(predict_mc, recombination_model)


_, _, x_test, _, _, _ = load_recomb_data(False)
x_test = x_test
x_rc_test = x_test[:,::-1,::-1]

res1 = recombination_model.predict_mc(x_rc_test, n_preds=10)[:,1]
importlib.reload(reproduce) # set that rng for the next batch of predictions

recombination_model = load_model_yaml("SavedModel", cust_objects={"MCRCDropout":MCRCDropout, "CustomSumPool": CustomSumPool})
recombination_model.predict_mc = types.MethodType(predict_mc, recombination_model)



res2 = recombination_model.predict_mc(x_test, n_preds=10)[:,1]

diff = res1 - res2

print ("{} training examples".format(len(diff)) )
print ("Max diff: {}".format(max(diff)))
assert (max(diff) < 10**-5)
# This runs perfectly on a dual core cpu. This may generalise to GPU computing, however if 
# you are using a cudnn backend, some of the algorithms are non deterministic. This could impact the
# seeding of MCRCDropout, and cause a failure of this script.

