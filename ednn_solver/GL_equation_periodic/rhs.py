# Louis Libat MSME - Gustave Eiffel
# Adapté de Yifan Du Tamer A. Zaki (Johns Hokpins University)

#  Rhs of various evolutional PDEs

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard
# -----------------------------------------------------------------------------
# Equations to enforce
# -----------------------------------------------------------------------------


# RHS of Kuramoto-Sivashinsky equation
@tf.function
def rhs_Kuramoto_Sivashinsky(output, coords, params):
    with tf.GradientTape(persistent=True) as tape4:
        tape4.watch(coords) 
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(coords)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(coords)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(coords)
                    u = output(coords)[0]
                dudx = tape1.gradient(u,coords)[:,0]
                del tape1
            d2udx2 = tape2.gradient(dudx,coords)[:,0]
            del tape2
        d3udx3 = tape3.gradient(d2udx2,coords)[:,0]
        del tape3
    d4udx4 = tape4.gradient(d3udx3,coords)[:,0]
    del tape4
    rhs = -u*dudx - d2udx2 - d4udx4   
    return rhs

# RHS of 2D advection diffusion equation. Also needed for 
# Incompressible Navier Stokes equation 
@tf.function
def rhs_2d_adv_diff_eqs(output, coords, params):
    dout = 1
    # params[0] : viscosity
    # params[1] : strength of sinusoidal forcing
    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(coords)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(coords)
            [U,V] = output(coords)
        dU = tape2.gradient(U,coords)
        dV = tape2.gradient(V,coords)
        dUdX = dU[:,0]
        dUdY = dU[:,1]
        dVdX = dV[:,0]
        dVdY = dV[:,1]
        del tape2
    ddUdX = tape3.gradient(dUdX, coords)
    ddUdY = tape3.gradient(dUdY, coords)
    ddVdX = tape3.gradient(dVdX, coords)
    ddVdY = tape3.gradient(dVdY, coords)
    d2UdX2 = ddUdX[:,0]
    d2UdY2 = ddUdY[:,1]
    d2VdX2 = ddVdX[:,0]
    d2VdY2 = ddVdY[:,1]
    del tape3
    return [(- (U*dUdX+V*dUdY) + params[0]*(d2UdX2+d2UdY2)) + params[1]*tf.sin(tf.constant(4.0) * coords[:,1]),
            (- (U*dVdX+V*dVdY) + params[0]*(d2VdX2+d2VdY2))]


# RHS of 2-D heat equation
def rhs_2d_heat_eqs(output, coords, coords_boundary, params):
    dout = 1
    # params[0]: heat diffusivity
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            U = output(coords, coords_boundary)
        dU = tape1.gradient(U, coords)
        dUdx = dU[:,0]
        dUdy = dU[:,1]
        del tape1
    ddUdx = tape2.gradient(dUdx,coords)
    ddUdy = tape2.gradient(dUdy,coords)
    d2Udx2 = ddUdx[:,0]
    d2Udy2 = ddUdy[:,1]
    del tape2
    return params[0]*(d2Udx2+d2Udy2)


# RHS of viscous Burgers equation. 
@tf.function
def rhs_Burgers(output, coords, params):
    nu = params[0]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            u = output(coords)[0]
        dudx = tape1.gradient(u,coords)[:,0]
        del tape1
    d2udx2 = tape2.gradient(dudx,coords)[:,0]
    del tape2
    rhs = -u*dudx + nu *  d2udx2
    return rhs

#  RHS of GL equation. 
@tf.function
def rhs_gl(output, coords, params):
    # params[0] : U
    # params[1] : cu
    # params[2] : cd
    # params[3] : mu0
    # params[4] : mu2
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            [U,V] = output(coords)
        dU = tape1.gradient(U,coords)
        dV = tape1.gradient(V,coords)
        dUdX = dU[:,0]
        dVdX = dV[:,0]
        del tape1
    ddUdX = tape2.gradient(dUdX, coords)
    ddVdX = tape2.gradient(dVdX, coords)
    d2UdX2 = ddUdX[:,0]
    d2VdX2 = ddVdX[:,0]
    del tape2
    rhs_real = (-params[0]*dUdX+d2UdX2)+(2*params[1]*dVdX-params[2]*d2VdX2)+(params[3]-params[4]**2+(params[4]/2)*coords[:,0]**2)*U
    rhs_img =(-params[0]*dVdX+d2VdX2)+(-2*params[1]*dUdX+params[2]*d2UdX2)+(params[3]-params[4]**2+(params[4]/2)*coords[:,0]**2)*V
    return [rhs_real,rhs_img]
