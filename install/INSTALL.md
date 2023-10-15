# Installation
Code dans répertoire travail (`$HOME/work/notebooks`) et accessible dans le répertoire du container `/tf/notebooks`

## Pyenv Virtualenv

Installer pyenv : https://github.com/pyenv/
```brew update
brew install pyenv
```
+ Env Variables (.bash_profile)

Installation de python
`pyenv install <version-python>`

Création de l'environnement
`pyenv virtualenv <version-python> <nom-venv>`

Dans un nouveau shell, activer
`pyenv activate <nom-venv>`

Installation des requirements.txt
```
scipy
matplotlib
torch
numpy
tqdm
```
`pip install -r requirements.txt`

`pip freeze -l > requirements.txt`

 
## Docker

### Tensorflow
https://www.tensorflow.org/install/docker?hl=fr
https://hub.docker.com/r/tensorflow/tensorflow/

Lancement du container Tensorflow avec Jupyter Notebook + Récupérer l'url de connexion avec le token dans les logs de démarage.
```
docker run -v $(pwd)/notebooks:/tf/notebooks -it --rm -p 8888:8888 tensorflow/tensorflow:nightly-jupyter
```
Le fichier code sont dans ./notebooks

### Pytorch

Lancement du container Pytorch avec Jupyter Notebook + Récupérer l'url de connexion avec le token dans les logs de démarage.

```
docker run -v $(pwd)/notebooks:/tf/notebooks -p 8888:8888 -it --rm pytorch/pytorch 
# Dans le container, on execute
pip install jupyterlab
cd /tf/notebooks/
jupyter lab --allow-root --no-browser --port=8888 --ip=0.0.0.0
```

Todo : 
- Faire un container contenant Tensorflow, Jupyterlab, Matplotlib, Scipy, ...

### Keras
https://hands-on.cloud/custom-keras-docker-container/



