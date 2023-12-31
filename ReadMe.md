# Résolution d’Équations aux Dérivées Partielles par Réseaux de Neurones Artificiels

## Contexte
Résoudre des EDP via l'utilisation de réseaux de neurones artificiels. Alternatives aux méthodes classiques (éléments finies, volumes finis).
Variante : PINNS (Physics- Informed Neural Networks).

MAIS Pb: 
- Data hungry : gros volumes de données d'entrainement
- Dégradation des performances lorsque les entrées s’éloignent des conditions d’apprentissage.

## Nouvelle approche
Evolutional Deep Neural Networks (EDNN, Du and Zaki (2021)).
Remplace la discrétisation spatiale par le réseau de neurone. Mise à jour des paramètres du réseau au cours du temps

Ce nouvel usage des réseaux de neurones est au centre des problématiques actuelles, qui cherchent à combiner le meilleur de la mécanique classique et du Machine Learning.

## Références
- Bagheri, S., D. S. Henningson, J. Hoepffner, and P. J. Schmid. 2009. “Input-Output Analysis and Control Design Applied to a Linear Model of Spatially Developing Flows.” Applied Mechanics Reviews.
- Du, Y., and T. A. Zaki. 2021. “Evolutional Deep Neural Network.” Physical Review E.

## Arborescence

### ./docs
Les références du projets

### ./ednn_solver
Le projet