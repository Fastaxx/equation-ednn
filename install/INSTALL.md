# Installation

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



