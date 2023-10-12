# OC/DS Projet 6 : Classifiez automatiquement des biens de consommation
Formation OpenClassrooms - Parcours data scientist - Projet Professionnalisant (Février - Avril 2023)

## Secteur : 
Commerce et distribution

## Technologies utilisées : 
  * Jupyter Notebook
  * Python
    - Scikit-learn : ACP, SVD, t-SNE
    - HyperOpt
    - Keras, Tensorflow
    - Nltk
    - OpenCV, PIL
    - Wordcloud
      
## Mots-clés : 
NLP, computer vision, classification supervisée, t-SNE, CNN, transfer learning, data augmentation, Deep Learning

## Le contexte : 
Le client cherche à lancer une market place e-commerce et souhaite automatiser la catégorisation des articles vendus.

## La mission : 
* Tester la faisabilité d’un moteur de classification des articles en différentes catégories à partir des descriptions ou des images postées par les vendeurs.
* Si faisable, réaliser une classification supervisée à partir des images.
* Tester une API qui permet de collecter des informations sur des produits d’alimentation.

## Livrables :
* notebook_faisabilite_texte.ipynb : notebook contenant le prétraitement et la feature extraction des données textes ainsi que l’étude de faisabilité
* notebook_faisabilite_image.ipynb : notebook contenant le prétraitement et la feature extraction des données images ainsi que l’étude de faisabilité
* notebook_classification_images.ipynb : notebook de classification supervisée des images
* notebook_api.ipynb : notebook de test de l’API
* presentation.pdf : support de présentation pour la soutenance détaillant le travail réalisé

## Algorithme retenu : 
* USE pour classification de données textes
* ResNet50 pour classification de données images 

## Méthodologie suivie (test de faisabilité) : 
La méthodologie suivie pour tester la faisabilité d’un moteur de classification est la même que l’on travaille sur les images ou les descriptions. 

1. Prétraitement des données :
* analyse exploratoire pour comprendre les spécificités du jeu de données
* nettoyage pour améliorer la qualité des données et ainsi faciliter le travail des algorithmes utilisés à l’étape suivante.
	- textes : tokenisation, stopwords, lemmatization, racinisation ...
	- images : passage au gris, re-dimension, amélioration contraste ...

2. Extraction des features :
* transformation des données textes ou images en nombres qui fassent sens afin que les algorithmes de machine learning puissent les exploiter.
  - textes : bag-of-words (tf, tf-idf), word-embeddings (word2vec, Bert, USE)
  - images : bag-of-features (SIFT), transfer-learning sur CNN (VGG16, RESNet50)

3. Réduction de dimensions :
But = Réduire le nombre de variables utilisées pour représenter les données, en 2 étapes :
*  application d'une technique linéaire afin de ne garder que les variables pertinentes (ACP ou  SVD)
*  application d'une technique non-linéaire pour visualiser nos données en 2D (t-SNE)

4. Clustering :
*  application de l'algorithme du Kmeans à 7 clusters (correspondants à nos 7 catégories de produits) sur nos données en 2D

5. Analyse graphique et indice de rand ajusté :
* comparaison de deux graphiques type «nuage de points», le premier coloré selon les vraies étiquettes, le second coloré selon les clusters trouvés par l’algorithme Kmeans à l’étape précédente.
* calcul d'un score de similarité, l'indice de rand ajusté, entre la partition originale et la partition proposée par l'algorithme Kmeans. Plus il sera proche de 1, plus le clustering correspondra parfaitement à la partition initiale et donc plus le projet de création d’un moteur de classification automatique sera faisable.

## Méthodologie suivie (classification) :

1. Transfer Learning : 
* Charger le modèle VGG16 pré-entrainé sans les couches fully connected
* Ajouter de nouvelle couches fully connected pour classer les images dans nos 7 catégories d’articles
* Entrainer seulement les nouvelles couches ajoutées, sur notre collection d’images

2. Data Augmentation (pour limiter le surapprentissage)

3. Optimisation des hyperparamètres (pour améliorer la performance):
* optimisation baysienne avec la librairie HyperOpt

4. Comparaison des performances avec/sans optimisation :
* matrice de confusion
* mesures de performance : Recall, Precision, F-mesure

## Compétences acquises :  
* Prétraiter des données image pour obtenir un jeu de données exploitable
* Prétraiter des données texte pour obtenir un jeu de données exploitable
* Représenter graphiquement des données à grandes dimensions
* Mettre en oeuvre des techniques de réduction de dimension
* Utiliser des techniques d’augmentation des données
* Définir la stratégie d’élaboration d’un modèle d'apprentissage profond
*  Évaluer la performance des modèles d’apprentissage profond selon différents critères
* Définir la stratégie de collecte de données en recensant les API disponibles

## Data source : 
* Images et descriptions :  site de e-commerce indien Flipkart
* API testée : https://rapidapi.com/edamam/api/edamam-food-and-grocery-database
