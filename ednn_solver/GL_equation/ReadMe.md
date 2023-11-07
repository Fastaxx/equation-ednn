# Solve Ginzburg-Landau equation with EDNN

$$\frac{\partial q}{\partial t} = - \nu \frac{\partial q}{\partial x} + \gamma \frac{\partial^2 q}{\partial x^2} + \mu q$$
avec $\nu = U+2i c_u$ et $\gamma =1+ic_d$ et $\mu=\mu_0-c_u^2+\mu_2 \frac{x^2}{2}$

1. Add related folder : ./GL_NN/
2. Train NN on initial condition : `python main.py 0`
3. March NN through time space : `python main.py 1`

./param : multiple test with different configurations
## Installation
1. Clone : `git clone`

2. Create virtual environnement

3. `pip install -r requirements.txt`

4. Use code

## Settings

- Define Initial condition : Kflowinit
- Collocation points distribution : Nx
- Physical domain : x1, x2
- Parameters : U, cu, cd, mu0, mu2
- Numbers of time steps : Nt
- Time step : dt
- Number of epochs : tot_eps
- Marching Method : Runge_Kutta, Forward_Euler
- Activation function : tanh, relu, adaptive_global
- Optimizer : Adam, SGD, ...
- Loss Functions

## Model creation 

```# Input definition
        coords = keras.layers.Input(self.din, name='coords')

        # Normalzation
        hidden  = coords

        # Hidden layers
        for ii in range(depth):
            hidden = keras.layers.Dense(width)(hidden)
            if activation=='adaptive_layer':
                self.act_fn = AdaptiveAct()
            hidden = self.act_fn(hidden)

        # Output definition
        fields = keras.layers.Dense(self.dout, name='fields')(hidden)

        cte   = keras.layers.Lambda(lambda x: 0*x[:,0:1]+1)(coords)
        dummy = keras.layers.Dense(1, use_bias=False)(cte)
        self.inv_outputs = [dummy]

        # Create model
        model = keras.Model(inputs=coords, outputs=[fields]+self.inv_outputs)
        self.model = model
```

## RHS

```
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

```

