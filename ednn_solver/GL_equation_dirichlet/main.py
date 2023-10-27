# Louis Libat MSME - Gustave Eiffel
# Adapté de Yifan Du Tamer A. Zaki (Johns Hokpins University)

# EDNN solver of 1D Ginzburg Landau equation
# Solves the differential system real part imaginary part by EDNN approach 

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard
from ednn import EvolutionalDNN
from marching_schemes import *
from rhs import *
from tensorflow.keras.optimizers.legacy import Adam, SGD
import time

# U is for the real part / V is for the imaginary part
def Kflowinit(X):
    U =   np.exp(-X*X)
    V =   np.exp(-X*X)
    return [U,V]

def main():
    # -----------------------------------------------------------------------------
    # Parameters for simulation configuration
    # -----------------------------------------------------------------------------
    # NN and solution directory
    case_name = "GL_NN/"
    # Numer of collocation points
    Nx = 1000
    Ny = 1
    # if Initial == True, train the neural network for initial condition
    # if Initial == False, march the initial network stored in case_name
    if sys.argv[1] == '0':
        Initial = True
    elif sys.argv[1] == '1':
        Initial = False
    else:
        sys.exit("Wrong flag specified")
    # Physical domain
    x1 = -20.0
    x2 = 20

    # Other parameters
    U = 2.0
    cu = 0.2
    cd = -1.0
    mu0 = 0.38
    mu2 = -0.01


    Nt = 1000
    dt = 1e-2
    tot_eps = 10000

    # ------------------------------------------------------------------------------
    # Generate the collocation points and initial condition array
    # ------------------------------------------------------------------------------
    X  = np.linspace(x1,x2,num=Nx, dtype=np.float32)
    Xi = X[1:-1]

    usq,vsq = Kflowinit(Xi)
    u = usq.reshape((Nx-2),1)
    v = vsq.reshape((Nx-2),1)

    Input = X.reshape(Nx,-1)
    InputInterior = Xi.reshape((Nx-2),-1)
    InitU = u.reshape((Nx-2),-1)
    InitV = v.reshape((Nx-2),-1)

    Index = np.arange(Nx).reshape(Nx)
    IE = (0.0 * Index + Index[-1]).astype(int).reshape((Nx),-1)
    IW = (0.0 * Index + Index[0]).astype(int).reshape((Nx),-1)
    BI = np.concatenate((IW,IE),axis = 1)

    #Extract the index of boundary points for the enforcement of B.C. 
    IEInterior = (0.0 * Index[1:-1] + Index[-1]).astype(int).reshape((Nx-2),-1)
    IWInterior = (0.0 * Index[1:-1] + Index[0]).astype(int).reshape((Nx-2),-1)
    BIInterior = np.concatenate((IWInterior,IEInterior),axis = 1)

    try: 
        nrestart = int(np.genfromtxt(case_name + 'nrestart'))
    except OSError: 
        nrestart = 0
    
    # -----------------------------------------------------------------------------
    # Initialize EDNN
    # -----------------------------------------------------------------------------
    lr = keras.optimizers.schedules.ExponentialDecay(1e-4, 10000000, 0.9)
    layers  =[2] + 3*[20] + [2]
    
    EDNN = EvolutionalDNN(layers,
                             rhs = rhs_gl, 
                             marching_method = Forward_Euler,
                             dest=case_name,activation = 'tanh',
                             optimizer=Adam(lr),    
                             eq_params=[U,cu,cd,mu0,mu2],
                             restore=True)
    print('Learning rate:', EDNN.optimizer._decayed_lr(tf.float32))
    
    #print('Learning rate:', PINN.optimizer._decayed_lr(tf.float32))
    
    
    if Initial: 
        t0 = time.time()
        # Train the initial condition tot_eps epochs,
        start = time.time() 
        for i in range(tot_eps):
            InputInteriorBoundary = Input[BIInterior]
            #print(InputInterior.shape)
            EDNN.train(InputInterior, InputInteriorBoundary, InitU, InitV, epochs=1,
                   batch_size=100, verbose=False, timer=False)
        # Evaluate and output the initial condition 
        InputBoundary = tf.convert_to_tensor(Input[BI])
        Input = tf.convert_to_tensor(Input)
        [U,V] = EDNN.output(Input,InputBoundary)
        end = time.time()
        print(end - start)
        U = U.numpy().reshape(Nx)
        V = V.numpy().reshape(Nx)
        X.dump(case_name+'X')
        U.dump(case_name+'U')
        V.dump(case_name+'V')
    
    
    else:
        InputInteriorBoundary = tf.convert_to_tensor(Input[BIInterior])
        InputBoundary = tf.convert_to_tensor(Input[BI])
        InputInterior = tf.convert_to_tensor(InputInterior)
        Input = tf.convert_to_tensor(Input)
    
        nbatch = 100
        params_marching = [dt,nbatch]
        # March the EDNN class till Nt time steps. 
        for n in range(nrestart+1,Nt):
            print('time step', n)
            EDNN.Marching(InputInterior,InputInteriorBoundary,params_marching)

            # The real and imaginary field is stored every time step.
            [U,V] = EDNN.output(Input,InputBoundary)
            U = U.numpy().reshape(Nx)
            V = V.numpy().reshape(Nx)
    
            X.dump(case_name+'X'+str(n))
            U.dump(case_name+'U'+str(n))
            V.dump(case_name+'V'+str(n))

            EDNN.save_NN()
            
            np.savetxt(case_name+'nrestart',np.array([n]))

if __name__ == "__main__":
    main()

