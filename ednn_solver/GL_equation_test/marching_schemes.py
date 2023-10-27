# Louis Libat MSME - Gustave Eiffel
# Adapté de Yifan Du Tamer A. Zaki (Johns Hokpins University)

# Time marching functions for EDNN

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard
# -----------------------------------------------------------------------------
# Time marching schemes 
# -----------------------------------------------------------------------------
def Forward_Euler(w, eval_rhs, Input,params):
    dt = params[0]
    nbatch = params[1]
    dwdt = eval_rhs(Input,nbatch)
    
    dw = dt * dwdt
    w += dw 
    return w

def Runge_Kutta(w,eval_rhs,Input,params):
    dt = params[0]
    nbatch = params[1]
    c = [1.0/8.0,3.0/8.0,3.0/8.0,1.0/8.0]

    k1 = eval_rhs(Input,nbatch)
    k2 = eval_rhs(Input,nbatch,w+k1*dt/3.0)
    k3 = eval_rhs(Input,nbatch,w-k1*dt/3.0+k2*dt)
    k4 = eval_rhs(Input,nbatch,w+k1*dt-k2*dt+k3*dt)
    for ce,k in zip(c,[k1,k2,k3,k4]): 
        w += ce*k*dt
    return w

    

