# Solve Ginzburg-Landau equation with EDNN

$$\frac{\partial q}{\partial t} = - \nu \frac{\partial q}{\partial x} + \gamma \frac{\partial^2 q}{\partial x^2} + \mu q$$

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
- Collocation points distribution
- Physical domain
- Parameters
- Numbers of time steps
- Time step
- Number of epochs
- Marching Method
- Activation function
- Optimizer
