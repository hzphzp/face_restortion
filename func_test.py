import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn
from tensorflow.contrib.framework.python.ops import add_arg_scope
def my_gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters)
    if a is not None:  # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)