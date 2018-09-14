import keras
from keras.initializers import Constant
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class CustomSumPool(Layer):
    def __init__(self, **kwargs):
        super(CustomSumPool, self).__init__(**kwargs)

    def call(self, x, mask=None):
        n_units = x.get_shape()[2]//2
        fwd = x[::,::,:n_units]
        rev = x[::,::,n_units:]
        comb = fwd + rev[::,::-1,::-1]
        return comb

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] //2

class CustomMeanPool(Layer):
    def __init__(self, **kwargs):
        super(CustomMeanPool, self).__init__(**kwargs)

    def call(self, x, mask=None):
        n_units = x.get_shape()[2]//2
        fwd = x[::,::,:n_units]
        rev = x[::,::,n_units:]
        comb = fwd + rev[::,::-1,::-1]
        return 0.5*comb

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] //2



class MotifMirrorGradientBleeding(keras.callbacks.Callback):
    def __init__(self, weight_layer, initial_gradient=0.5, gradient_increment=0.0003, assign_bias=False):
        self.weight_layer = weight_layer
        self.constrained=None
        self.unconstrained=None
        self.placeholder=None
        self.ass = None
        self.rxc = None
        self.constrained_rxc = None
        self.i=initial_gradient
        self.x = tf.placeholder(tf.float32, shape=[])
        self.m = tf.constant(0.5)
        self.bleed=None
        self.ass_b = None
        self.gradient_increment = gradient_increment
        self.assign_bias = assign_bias
    def cached_constrained(self):
        if self.constrained is None:

            self.constrained = self.model.weights[self.weight_layer][:,:,self.split_ix:]
        return self.constrained

    def cached_unconstrained(self):
        if self.unconstrained is None:

            self.unconstrained = self.model.weights[self.weight_layer][:,:,:self.split_ix]
        return self.unconstrained

    def assign_constrained(self):
        if self.ass is None:
            self.ass = self.model.weights[self.weight_layer][:,:,self.split_ix:].assign(self.cached_rc())
        return self.ass

    def cached_rc(self):
        if self.rxc is None:
            self.rxc = self.cached_unconstrained()[::-1,::-1,::-1]
        return self.rxc

    def ass_bias(self):
        if self.ass_b is None:
            self.ass_b = self.model.weights[self.weight_layer+1][self.split_ix:].assign(self.model.weights[self.weight_layer+1][:self.split_ix][::-1])
        return self.ass_b

    def cached_constrained_rc(self):
        if self.constrained_rxc is None:
            self.constrained_rxc = self.cached_constrained()[::-1,::-1,::-1]
        return self.constrained_rxc

    def bleed_rc_gradient(self):
        if self.bleed is None:
            self.bleed = self.model.weights[self.weight_layer][:,:,:self.split_ix].assign(tf.add(tf.scalar_mul(tf.constant(1.0) - tf.minimum(self.x, self.m),self.cached_constrained_rc())
                                                                                                    ,tf.scalar_mul(tf.minimum(self.x, self.m), self.cached_unconstrained())))
        self.i+=self.gradient_increment
        return self.bleed

    def on_batch_end(self, batch, logs={}):
        assert self.model.weights[self.weight_layer].get_shape().as_list()[2] % 2 == 0, "Input must have an even number of filters"
        self.split_ix = self.model.weights[self.weight_layer].get_shape().as_list()[2] // 2
        import tensorflow as tf
        sess = K.get_session()
        sess.run(self.bleed_rc_gradient(), feed_dict={self.x:self.i})
        sess.run(self.assign_constrained())
        if self.assign_bias:
            sess.run(self.ass_bias())

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numbers
from tensorflow.python.framework import tensor_util
def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape

class MCRCDropout(Layer):
    """Applies MC Dropout to the input.
       The applied noise vector is symmetric to reverse complement symmetry
       Class structure only slightly adapted 

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Remains active ative at test time so sampling is required

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(MCRCDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            import numpy as np
            noise_shape = self._get_noise_shape(inputs)
            x = inputs
            seed = self.seed
            keep_prob = 1. - self.rate
            if seed is None:
                seed = np.random.randint(10e6)
            # the dummy 1. works around a TF bug
            # (float32_ref vs. float32 incompatibility)
            x= x*1
            name = None
            with ops.name_scope(name, "dropout", [x]) as name:
                x = ops.convert_to_tensor(x, name="x")
                if not x.dtype.is_floating:
                    raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
                if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
                    raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
                keep_prob = ops.convert_to_tensor(
                             keep_prob, dtype=x.dtype, name="keep_prob")
                keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

                # Do nothing if we know keep_prob == 1
                if tensor_util.constant_value(keep_prob) == 1:
                    return x

                noise_shape = _get_noise_shape(x, noise_shape)
                # uniform [keep_prob, 1.0 + keep_prob)
                random_tensor = keep_prob
                random_tensor += random_ops.random_uniform(
                noise_shape, seed=seed, dtype=x.dtype)
               
                # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                binary_tensor = math_ops.floor(random_tensor)
                dim = binary_tensor.shape[2]//2

                symmetric_binary= tf.concat([binary_tensor[:,:,dim:],binary_tensor[:,:,dim:][::,::-1,::-1]], 2)
                ret = math_ops.div(x, keep_prob) * symmetric_binary
                
                return ret


    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(MCRCDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape




def generate_motif(dna_string):
    eps = 0.3
    motif = eps* np.array([one_hot_conv[base] for base in dna_string])
    return motif

