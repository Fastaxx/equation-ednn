# Louis Libat MSME - Gustave Eiffel
# Adapté de Yifan Du Tamer A. Zaki (Johns Hokpins University)

#  Costs Functions for Neural Networks

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard

# -----------------------------------------------------------------------------
# Loss Functions to enforce
# -----------------------------------------------------------------------------

# MSE : differentiable
mse = [tf.reduce_mean(tf.square(Ypred[i] - Y_batch[i,0])) for i in range(len(Ypred))]

# MAE : not differentiable at zero
mae = [tf.reduce_mean(Ypred[i]-Y_batch[i,0]) for i in range(len(Ypred))]

# RMSE : 
rmse = [tf.sqrt(tf.reduce_mean(tf.square(Ypred[i] - Y_batch[i,0]))) for i in range(len(Ypred))]

# MSLE : 
msle = [tf.reduce_mean(tf.square(tf.math.log(Y_batch[i,0]+1)-tf.math.log(Ypred[i]+1))) for i in range(len(Ypred))]

# Log Cosh Loss :
logcosh = [tf.reduce_mean(tf.math.log(tf.math.cosh(Ypred[i]-Y_batch[i,0]))) for i in range(len(Ypred))]

# Modified Cosine Similarity : 
cossim = [tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(Ypred[i]))) * tf.sqrt(tf.reduce_sum(tf.square(Y_batch[i,0])))) for i in range(len(Ypred))]

