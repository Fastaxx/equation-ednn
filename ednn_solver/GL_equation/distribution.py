# Louis Libat MSME - Gustave Eiffel
# Adapté de Yifan Du Tamer A. Zaki (Johns Hokpins University)

#  Points distribution for Neural Networks

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard
from pyDOE import lhs
from sobol_seq import i4_sobol_generate

# Return a Sobol Distribution
def sobol_distribution(lower_bound,upper_bound,array_shape):
	sobol_samples = i4_sobol_generate(1, array_shape)
	sobol_array = lower_bound + sobol_samples.flatten() * (upper_bound - lower_bound)
	X = np.float32(np.sort(sobol_array))
	return X

# Return a Latin Hypercube Distribution
def lhs_distribution(lower_bound,upper_bound,array_shape):
	lhs_samples = lhs(1, samples=array_shape, criterion="maximin",)
	lhs_array = lower_bound + lhs_samples * (upper_bound - lower_bound)
	lhs_array = lhs_array.flatten()
	X = np.float32(np.sort(lhs_array))
	return X

# Return a triangular Random Distribution
def triangular_distribution(lower_bound,upper_bound,array_shape):
	triangular_array = np.random.triangular(lower_bound, 0, upper_bound, array_shape)
	X = np.float32(np.sort(triangular_array))
	return X

def uniform_distribution(lower_bound,upper_bound,array_shape):
	random_array = np.linspace(lower_bound+1, upper_bound-1, array_shape,dtype=np.float32)
	random_array += np.random.uniform(-1, 1, array_shape)
	X = np.sort(random_array)
	return X

def grid_distribution(lower_bound,upper_bound,array_shape):
	X  = np.linspace(lower_bound,upper_bound,num=array_shape, dtype=np.float32)
	return X



