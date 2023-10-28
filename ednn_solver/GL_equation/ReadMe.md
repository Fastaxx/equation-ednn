# Solve Ginzburg-Landau equation with EDNN

$$\frac{\partial q}{\partial t} = - \nu \frac{\partial q}{\partial x} + \gamma \frac{\partial^2 q}{\partial x^2} + \mu q$$
avec $\nu = U+2i c_u$ et $\gamma =1+ic_d$ et $\mu=\mu_0-c_u^2$

1. Add related folder : ./GL_NN/
2. Train NN on initial condition : `python main.py 0`
3. March NN through time space : `python main.py 1`

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


