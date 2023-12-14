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
def mse_loss(Ypred,Y_batch):
	"""
	Compute the MSE Loss
	"""
	return [tf.reduce_mean(tf.square(Ypred[i] - Y_batch[i,:])) for i in range(len(Ypred))]

# MAE : not differentiable at zero
def mae_loss(Ypred,Y_batch):
	"""
	Compute the MAE Loss
	"""
	return [tf.reduce_mean(tf.abs(Ypred[i] - Y_batch[i,:])) for i in range(len(Ypred))]

# RMSE : 
def rmse_loss(Ypred,Y_batch):
	"""
	Compute the RMSE Loss
	"""
	return [tf.sqrt(tf.reduce_mean(tf.square(Ypred[i] - Y_batch[i,:]))) for i in range(len(Ypred))]

# MSLE : 
def msle_loss(Ypred,Y_batch):
	"""
	Compute the MSLE Loss
	"""
	return [tf.reduce_mean(tf.square(tf.math.log1p(Ypred[i])-tf.math.log1p(Y_batch[i,:]))) for i in range(len(Ypred))]

# Log Cosh Loss :
def logcosh_loss(Ypred,Y_batch):
	"""
	Compute the Log-Cosh Loss
	"""
	return [tf.reduce_mean(tf.math.log(tf.math.cosh(Ypred[i]-Y_batch[i,:]))) for i in range(len(Ypred))]

# MAPE Loss :
def mape_loss(Ypred,Y_batch):
	"""
	Compute the MAPE Loss
	"""
	return [100*tf.reduce_mean(tf.abs(Ypred[i] - Y_batch[i,:])/(Y_batch[i,:]+tf.keras.backend.epsilon())) for i in range(len(Ypred))]

# MSE + Sobolev :
def mse_loss_sobolev(Ypred,Y_batch,dY,Y_der):
	"""
	Compute the MSE Loss
	"""
	return [tf.reduce_mean(tf.square(Ypred[i] - Y_batch[i,:]) + tf.reduce_sum(tf.square(dY[i] - Y_der[i,:]))) for i in range(len(Ypred))]



