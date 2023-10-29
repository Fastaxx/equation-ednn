# EDNN
Le répertoire de travail.

## Arborescence
Les dossiers contiennent différentes implémentation d'EDP
- 1D_Burgers_Equation : Résolution de l'équation 1D de Burgers avec conditions périodiques
- 2D_Navier_Stokes_Periodic : Résolution de l'équation 2D de Navier Stokes avec conditions périodiques
- GL_Equation : Résolution de l'équation de Ginzburg-Landau complexe 1D sans conditions limites

## Installation
1. Clone 
`git clone`

2. Create virtual environnement

3. `pip install -r requirements.txt`

4. Use code

## Fonctionnement
1. Train NN on initial condition : `python main.py 0`
2. March NN through time space : `python main.py 1`

## Settings

- Define Initial condition
- Collocation points distribution
- Physical domain
- Parameters
- Numbers of time steps
- Time step
- Number of epochs
- Marching Method
- Activation function
- Optimizer