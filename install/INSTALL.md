# Installation

## Docker
https://www.tensorflow.org/install/docker?hl=fr
https://hub.docker.com/r/tensorflow/tensorflow/

Lancement du container Tensorflow avec Jupyter Notebook + Récupérer l'url de connexion avec le token dans les logs de démarage.
```
docker run -v $(pwd)/notebooks:/tf/notebooks -it -p 8888:8888 tensorflow/tensorflow:nightly-jupyter
```
Le fichier code sont dans ./notebooks