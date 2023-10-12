import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from enum import Enum

from PIL import Image, ImageOps, ImageFilter
import cv2

from wordcloud import WordCloud
from nltk.corpus import stopwords, words
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

import gensim
from gensim.models.keyedvectors import KeyedVectors 

import sklearn
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import transformers
from transformers import AutoTokenizer, BertTokenizerFast, TFAutoModel, AutoModelForSequenceClassification

import ssl
from skimage import io
from math import ceil
from hyperopt import Trials

import nltk
nltk.download('stopwords')


class LinearDimReduction(Enum):
    TRUNCATED_SVD = 0
    PCA = 1
    

class BertMode(Enum):
    TENSORFLOW_HUB = 0
    HUGGINGFACE = 1

    
def display_image_from_url(url: str, title: str, fig_size: tuple):
    """
    Affiche une image à partir de son url

    Positional arguments : 
    -------------------------------------
    url : str : url de l'image à afficher 
    title : str : titre à afficher au dessus de l'image
    figsize : tuple : taille de la zone d'affichage de l'image (largeur, hauteur)
    """
    
    ssl._create_default_https_context = ssl._create_unverified_context
    img = io.imread(url)
    plt.figure(figsize=fig_size)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20, fontname='Corbel', pad=20)
    plt.imshow(img)

    plt.show()
    
    
def display_images_from_url(img_path: str, img_names: pd.Series, title: str, col_n: int, figsize: tuple,
                            top=0.9, wspace=0.1, hspace=0.7):
    """
    Affiche plusieurs images à partir de leur url

    Positional arguments : 
    -------------------------------------
    img_path : str : emplacement des images
    img_names : pd.Series : noms des images à afficher 
    title : str : titre principal de la zone de graphique
    col_n : int : nombre d'images à afficher par ligne
    figsize : tuple : taille de la zone d'affichage (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    sns.set_theme(style='white')
    rgb_text = sns.color_palette('Greys', 15)[12]
    
    fig, axes = plt.subplots(ceil(img_names.shape[0] / col_n), col_n, figsize=figsize)
    fig.tight_layout()
    fig.suptitle(title, fontname='Corbel', fontsize=40, color=rgb_text)
    
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)
    
    (l, c) = (0, 0)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    for name in img_names:
        img = io.imread(img_path + name)
        axes[l, c].imshow(img)
        axes[l, c].axis('off')
        
        (c, l) = (0, l + 1) if c == col_n - 1 else (c + 1, l)
        
    plt.show()


def display_image(img, title: str, fig_size: tuple):
    """
    Affiche une image à partir de son url

    Positional arguments : 
    -------------------------------------
    url : str : jurl de l'image à afficher 
    title : str : titre à afficher au dessus de l'image
    figsize : tuple : taille de la zone d'affichage de l'image (largeur, hauteur)
    """
    
    plt.figure(figsize=fig_size)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20, fontname='Corbel', pad=20)
    plt.imshow(img.astype(np.uint8))

    plt.show()
    

def plot_donut(dataset: pd.DataFrame, categ_var: str, title: str, figsize: tuple, text_color='#595959',
               colors={'outside': sns.color_palette('Set2')}, nested=False, sub_categ_var=None, labeldistance=1.1,
               textprops={'fontsize': 20, 'color': '#595959', 'fontname': 'Open Sans'}):
    """
    Affiche un donut de la répartition d'une variable qualitative

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative
    
    palette : strings : nom de la palette seaborn à utiliser
    title : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optionnal arguments : 
    -------------------------------------
    text_color : str : couleur du texte
    colors : dict : couleurs du donut extérieur et couleurs du donut intérieur
    nested : bool : créer un double donut ou non
    sub_categ_var : str : nom de la colonne contenant les catégories à afficher dans le donut intérieur
    labeldistance : float : distance à laquelle placer les labels du donut extérieur
    textprops : dict : personnaliser les labels du donut extérieur (position, couleur ...)
    """
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontname='Corbel', fontsize=30)
        plt.rcParams.update(
            {'axes.labelcolor': text_color, 'axes.titlecolor': text_color, 'legend.labelcolor': text_color,
             'axes.titlesize': 16, 'axes.labelpad': 10})

    pie_series = dataset[categ_var].value_counts(sort=False, normalize=True)
    patches, texts, autotexts = ax.pie(pie_series, labels=pie_series.index, autopct='%.0f%%', pctdistance=0.85,
                                       colors=colors['outside'], labeldistance=labeldistance,
                                       textprops=textprops,
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    centre_circle = plt.Circle((0, 0), 0.7, fc='white')

    if nested:
        inside_pie_series = dataset[sub_categ_var].value_counts(sort=False, normalize=True)
        patches_sub, texts_sub, autotexts_sub = ax.pie(inside_pie_series, autopct='%.0f%%', pctdistance=0.75,
                                                       colors=colors['inside'], radius=0.7,
                                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

        for autotext in autotexts_sub:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        plt.legend(patches_sub, inside_pie_series.index, title=sub_categ_var, fontsize=14, title_fontsize=16, loc=0)
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')

    ax.axis('equal')
    ax.add_artist(centre_circle)

    plt.tight_layout()
    plt.show()
    

def plot_boxplot(dataset: pd.DataFrame, numeric_var: str, title: str, figsize: tuple, categ_var=None, palette='Set2'):
    """
    Affiche un graphique avec un ou plusieurs boxplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    numeric_var : str : nom de la colonne contenant les valeurs dont on veut étudier la distribution
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
   
    Optional arguments : 
    -------------------------------------
    categ_var : str : nom de la colonne contenant les catégories (si on souhaite regrouper les variables numériques par
    catégorie)
    palette : str or list of strings : nom de la palette seaborn utilisée ou liste de couleurs personnalisées
    """ 
    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]
    
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)
    plt.rcParams['axes.labelpad'] = "30"
    
    ax = sns.boxplot(data=dataset, x=numeric_var, y=categ_var,
                     orient='h', palette=palette, saturation=0.95,
                     showfliers=False, 
                     medianprops={"color": "#c2ecff", 'linewidth': 3.0},
                     showmeans=True, 
                     meanprops={'marker': 'o', 'markeredgecolor': 'black',
                                'markerfacecolor': '#c2ecff', 'markersize': 10},
                     boxprops={'edgecolor': 'black', 'linewidth': 1.5},
                     capprops={'color': 'black', 'linewidth': 1.5},
                     whiskerprops={'color': 'black', 'linewidth': 1.5})
    
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    
    plt.title(title, fontname='Corbel', color=rgb_text, fontsize=26, pad=20)
    plt.xlabel(numeric_var, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.ylabel(categ_var, fontsize=20, fontname='Corbel', color=rgb_text)
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor=rgb_text)
    
    plt.show()
    

def display_distribution(dataset: pd.DataFrame, numeric_features: [str], column_n: int, figsize: tuple, top=0.85,
                         wspace=0.2, hspace=1.8, suptitle='Distribution variables numériques'):
    """
    Affiche la distribution de chaque variable de la liste.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    numeric_features : list of strings : liste des variables numériques dont on souhaite afficher la distribution
    column_n : int : nombre de graphique à afficher par ligne
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    suptitle : str : titre principal de la zone de graphique
    """
    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid', palette='Set2')

    fig = plt.figure(figsize=figsize)
    fig.tight_layout()

    fig.suptitle(suptitle, fontname='Corbel', fontsize=20, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=top, wspace=wspace, hspace=hspace)

    for i, feature in enumerate(numeric_features):
        sub = fig.add_subplot(ceil(len(numeric_features) / column_n), column_n, i + 1)
        sub.set_xlabel(feature, fontsize=14, fontname='Corbel', color=rgb_text)
        sub.set_title(feature, fontsize=16, fontname='Corbel', color=rgb_text)

        sns.histplot(dataset, x=feature)
        sub.grid(False, axis='x')
        sub.tick_params(axis='both', which='major', labelsize=14, labelcolor=rgb_text)

    plt.show()
    
    
def count_vector_to_df(vectorized_doc: np.array, top_n: int, features: np.array):
    """
    Renvoie un dataframe contenant les "top_n" tokens qui apparaissent le plus dans un document

    Positional arguments : 
    -------------------------------------
    vectorized_doc : np.array : document sous forme de bag of words
    top_n : int : nombre de tokens à renvoyer
    features : np.array : tokens (sous forme de chaine de caractères)
    """
    sorted_index = vectorized_doc.argsort()  
    counts = vectorized_doc
    counts.sort()
    
    df = pd.DataFrame({'count': counts[:-(top_n+1):-1],
                       'token': pd.Series([features[i] for i in sorted_index[:-(top_n+1):-1]])})
    
    return df


def display_barplot(data: pd.DataFrame, x: str, y: str, color: tuple, titles: dict, figsize: tuple, edgecolor: tuple):
    """
    Affiche un barplot
 
    Positional arguments : 
    -------------------------------------
    data : pd.DataFrame : jeu de données à afficher dans le graph
    x : str : nom de la colonne contenant les données à afficher en abscisse
    y : str : nom de la colonne contenant les données à afficher en ordonnée
    color : tuple : couleur des barres
    titles : dict : titres du graphique et des axes - ex: {'chart_title': 'c', 'y_title': 'b', 'x_title': 'a'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    edgecolor : tuple : couleur du contour des barres
    """
    plt.figure(figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]

    with sns.axes_style('white'):

        ax = sns.barplot(data=data, x=x, y=y, edgecolor=edgecolor, linewidth=3, facecolor=color, width=.6)
        ax.set(xticklabels=[])
        sns.despine(bottom=True)

    for container in ax.containers:
        ax.bar_label(container, size=18, fontname='Open Sans', padding=5)

    plt.title(titles['chart_title'], size=24, fontname='Corbel', pad=40, color=rgb_text)
    plt.ylabel(titles['y_title'], fontsize=20, fontname='Corbel', color=rgb_text)
    ax.set_xlabel(titles['x_title'], rotation=0, labelpad=20, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.show()


def conf_mat_transform(y_true: pd.Series, y_pred: pd.Series):
    """
    Renvoie les labels correspondants à la prédiction
 
    Positional arguments : 
    -------------------------------------
    y_true : pd.Series : cibles
    y_pred : pd.Seried : prédictions
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    
    corresp = np.argmax(conf_mat, axis=0)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x: corresp[x])
    
    return labels['y_pred_transform']

    
def plot_confusion_matrix(y: pd.DataFrame, figsize: tuple, add_label=False, title=''):
    """
    Affiche une matrice de confusion
 
    Positional arguments : 
    -------------------------------------
    y : pd.DataFrame : tableau contenant les cibles et les prédictions
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    add_label : boolean : ajouter les labels de la cible
    
    """
    if add_label:
        label_encoder = LabelEncoder()
        y["label"] = label_encoder.fit_transform(y["target"])
    
    cls_labels_transform = conf_mat_transform(y['label'], y['pred'])
    conf_mat = confusion_matrix(y['label'], cls_labels_transform, labels=np.sort(y['label'].unique()))
    
    labels = np.sort(y['target'].unique())
    df_cm = pd.DataFrame(conf_mat, index=labels, columns=labels)

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)
    
    ax = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", 
                     annot_kws={"fontsize": 16, 'fontname': 'Open Sans'},
                     linewidth=1, linecolor='w')
    ax.xaxis.set_label_position('top')
    plt.title('Matrice de confusion' + title, fontsize=35, pad=30, fontname='Corbel')
    plt.xlabel('Catégories prédites', fontsize=25, fontname='Corbel', labelpad=20)
    plt.ylabel('Vraies catégories', fontsize=25, fontname='Corbel', labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14, labeltop=True,  labelbottom=False)

    plt.show()
    

def build_callbacks(save_path: str):
    """
    Création de callbacks
 
    Positional arguments : 
    -------------------------------------
    save_path : str : chemin de sauvegarde de l'historique du modèle
    """
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, es]
    
    return callbacks_list
 

def fit_transform_tsne(data, n_components: int, init='random'):
    """
    Renvoie les données auxquelles on a appliqué une réduction de dimension t-SNE
 
    Positional arguments : 
    -------------------------------------
    data :  : jeu de données
    n_components : int : nombre de dimensions réduites
    
    Optional arguments :
    -------------------------------------
    init : str :
    """
    tsne = manifold.TSNE(n_components=n_components, init=init, random_state=8)
    transformed_data = tsne.fit_transform(data)
    
    return transformed_data


def fit_transform_pca(data, n_components: float, verbose=False):
    """
    Renvoie les données auxquelles on a appliqué une réduction de dimension linéaire ACP
 
    Positional arguments : 
    -------------------------------------
    data :  : jeu de données
    n_components : float : nombre de dimensions réduites ou pourcentage de la variance à expliquer
    
    Optional arguments :
    -------------------------------------
    verbose : bool : afficher ou non le pourcentage de variance expliquée par les composantes
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(scaled_data)
      
    if verbose:
        explained_variance_cum = pca.explained_variance_ratio_.cumsum()
        print('{:.2f}% de variance expliquée par les {} composantes ensembles'.format(
                np.max(explained_variance_cum) * 100, len(explained_variance_cum)))

    return transformed_data


def fit_transform_truncated_svd(data, n_components: int, verbose=False, thresh=0.9):
    """
    Renvoie les données auxquelles on a appliqué une réduction de dimension linéaire SVD
 
    Positional arguments : 
    -------------------------------------
    data :  : jeu de données
    n_components : int : nombre de dimensions réduites
    
    Optional arguments :
    -------------------------------------
    verbose : bool : afficher ou non le pourcentage de variance expliquée par les composantes
    """
    svd = TruncatedSVD(n_components=n_components, random_state=8)
    transformed_data = svd.fit_transform(data)

    if verbose:
        explained_variance_cum = svd.explained_variance_ratio_.cumsum()
        components = [component for component, explainedvar in enumerate(explained_variance_cum)
                      if explainedvar >= thresh]
        
        if len(components) > 0:
            print('{:.2f}% de variance expliquée par les {} premières composantes ensembles'.format(
                        explained_variance_cum[components[0]] * 100, components[0] + 1))
        else:
            print('{:.2f}% de variance expliquée par toutes les composantes ensembles'.format(
                np.max(explained_variance_cum) * 100))

    return transformed_data


def fit_kmeans(data, k: int, n_init=10):
    """
    Renvoie un modèle Kmeans entrainé
 
    Positional arguments : 
    -------------------------------------
    data :  : jeu de données d'entrainement
    k : int : nombre de clusters
    
    Optional arguments :
    -------------------------------------
    n_init : int : nombre d'initialisations 
    """
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=8)
    kmeans.fit(data)
    
    return kmeans


def build_df_vizu(data, ldr: LinearDimReduction, targets: pd.Series, n_components: float, k_cluster: int, 
                  var_explained=0.9, verbose=False):
    """
    Renvoie un dataframe contenant le jeu de données réduit en 2D par une réduction dimensionelle linéaire (ACP ou SVD) 
    et une réduction non linéaire (t-SNE), ainsi que les clusters trouvés par un algorithme KMeans appliqué aux données
    réduites.

    Positional arguments : 
    -------------------------------------
    data : : matrice de features à réduire
    ldr : LinearDimReduction : énumération permettant de choisir entre ACP et SVD
    targets : pd.Series : vraies étiquettes
    n_components : float : nombre de composantes pour la réduction dimensionnelle linéaire
    k_cluster : int : nombre de clusters (hyperparamètre de l'algo KMeans)

    Optional arguments :
    -------------------------------------
    
    var_explained_svd : float : afficher le nombre de composantes SVD qui expliquent ce pourcentage de variance
    verbose : bool : afficher ou pas le pourcentage de variance expliqué par les composantes du modèle 
    de réduction dimensionnelle linéaire choisi
    """
    if ldr.name == 'PCA':
        transformed_data = fit_transform_pca(data, n_components, verbose=verbose)
    else:
        transformed_data = fit_transform_truncated_svd(data, n_components, thresh=var_explained, verbose=verbose)
        
    transformed_data = fit_transform_tsne(transformed_data, 2)
    fitted_kmeans = fit_kmeans(transformed_data, k_cluster)

    data_df = pd.DataFrame(transformed_data, columns=['ax' + str(n) for n in range(1, 3)])
    data_df['target'] = targets
    data_df['cluster'] = fitted_kmeans.labels_
    
    return data_df


def display_vizu_many(ldr: LinearDimReduction, models: list, targets: pd.Series, figsize: tuple, suptitle: str,
                      k_clusters=7, var_explained_svd=0.9, top=0.9, wspace=0.1, hspace=0.7, verbose=False):
    """
    Affiche deux graphiques type "nuage de points" pour chaque modèle, l'un coloré selon les vraies étiquettes, 
    le second coloré selon des clusters trouvés par un algorithme KMeans appliqué au jeu de données réduit en 2D par 
    une réduction dimensionelle linéaire (ACP ou SVD) et une réduction non linéaire (t-SNE).
    Renvoie un dataframe contenant les données réduites, les clusters et l'indice de rand ajusté entre la partition
    originale et la partition proposée par l'algorithme KMeans.

    Positional arguments : 
    -------------------------------------
    ldr : LinearDimReduction : énumération permettant de choisir entre ACP et SVD
    models : list : liste de dictionnaires contenant des infos sur chaque modèle 
    (données, titre à afficher, nom du modèle, nombre de composantes pour la réduction dimensionnelle linéaire)
    ex : {'data': dataset, 'title': 'descriptions', 'model': 'model from tensorflow_hub', 'n_components': 100},
    targets : pd.Series : vraies étiquettes
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    suptitle : str : titre principale de la zone de graphique

    Optional arguments :
    -------------------------------------
    k_cluster : int : nombre de clusters (hyperparamètre de l'algo KMeans)
    var_explained_svd : float : afficher le nombre de composantes SVD qui expliquent ce pourcentage de variance
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    verbose : bool : afficher ou pas le pourcentage de variance expliqué par les composantes du modèle 
    de réduction dimensionnelle linéaire choisi
    """
    sns.set_theme(style='white')
    rgb_text = sns.color_palette('Greys', 15)[12]
    title = 'Représentation des {} par {} \n({}) \nARI : {}'

    fig, axes = plt.subplots(len(models), 2, figsize=figsize, sharey=True)
    fig.tight_layout()
    fig.suptitle(ldr.name + ' + TSNE + KMeans' + suptitle,
                 fontname='Corbel', fontsize=40, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    l = 0
    transformed_data_all = {}
    for model in models:
        transformed_data_df = build_df_vizu(model['data'], ldr, targets, model['n_components'],
                                            k_clusters, var_explained_svd, verbose=verbose)

        ari = adjusted_rand_score(transformed_data_df['target'],
                                  transformed_data_df['cluster'])
        transformed_data_df['ari'] = ari
        transformed_data_all[model['model']] = transformed_data_df

        for param in [{'c': 0, 'hue': 'target'}, {'c': 1, 'hue': 'cluster'}]:

            sns.scatterplot(data=transformed_data_df, x='ax1', y='ax2',
                            hue=param['hue'], palette='bright',
                            ax=axes[l, param['c']])

            axes[l, param['c']].set_title(title.format(model['title'], param['hue'], model['model'], round(ari, 2)),
                                          fontname='Corbel', fontsize=30, color=rgb_text, pad=50)
            axes[l, param['c']].set_xticks([])
            axes[l, param['c']].set_yticks([])
            axes[l, param['c']].set_xlabel(None)
            axes[l, param['c']].set_ylabel(None)

        l += 1

    plt.show()

    return transformed_data_all


def display_vizu_one(ldr: LinearDimReduction, model: dict, targets: pd.Series, figsize: tuple, suptitle: str,
                     k_clusters=7, var_explained_svd=0.9, top=0.9, wspace=0.1, hspace=0.7, verbose=False):
    """
    Affiche deux graphiques type "nuage de points", l'un coloré selon les vraies étiquettes, 
    le second coloré selon des clusters trouvés par un algorithme KMeans appliqué au jeu de données réduit en 2D par 
    une réduction dimensionelle linéaire (ACP ou SVD) et une réduction non linéaire (t-SNE).
    Renvoie un dataframe contenant les données réduites, les clusters et l'indice de rand ajusté entre la partition
    originale et la partition proposée par l'algorithme KMeans.

    Positional arguments : 
    -------------------------------------
    ldr : LinearDimReduction : énumération permettant de choisir entre ACP et SVD
    model : dict : dictionnaire contenant des infos sur le modèle testé
    (données, titre à afficher, nom du modèle, nombre de composantes pour la réduction dimensionnelle linéaire)
    ex : {'data': dataset, 'title': 'descriptions', 'model': 'model from tensorflow_hub', 'n_components': 100},
    targets : pd.Series : vraies étiquettes
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    suptitle : str : titre principale de la zone de graphique

    Optional arguments :
    -------------------------------------
    k_cluster : int : nombre de clusters (hyperparamètre de l'algo KMeans)
    var_explained_svd : float : afficher le nombre de composantes SVD qui expliquent ce pourcentage de variance
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    verbose : bool : afficher ou pas le pourcentage de variance expliqué par les composantes du modèle 
    de réduction dimensionnelle linéaire choisi
    """
    sns.set_theme(style='white')
    rgb_text = sns.color_palette('Greys', 15)[12]
    title = 'Représentation des {} par {} \nARI : {}'

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.tight_layout()
    fig.suptitle(ldr.name + ' + TSNE + KMeans' + suptitle,
                 fontname='Corbel', fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    transformed_data_df = build_df_vizu(model['data'], ldr, targets, model['n_components'],
                                        k_clusters, var_explained_svd, verbose=verbose)

    ari = adjusted_rand_score(transformed_data_df['target'],
                              transformed_data_df['cluster'])

    transformed_data_df['ari'] = ari

    for param in [{'c': 0, 'hue': 'target'}, {'c': 1, 'hue': 'cluster'}]:

        sns.scatterplot(data=transformed_data_df, x='ax1', y='ax2',
                        hue=param['hue'], palette='bright',
                        ax=axes[param['c']])

        axes[param['c']].set_title(title.format(model['title'], param['hue'], round(ari, 2)),
                                   fontname='Corbel', fontsize=20, color=rgb_text)
        axes[param['c']].set_xticks([])
        axes[param['c']].set_yticks([])
        axes[param['c']].set_xlabel(None)
        axes[param['c']].set_ylabel(None)

    plt.show()

    return transformed_data_df


def build_trial_df(trials: Trials, loss: str):
    """
    Retourne un dataframe contenant des informations sur les itérations de l'optimisation réalisée avec hyperopt 
    (score, paramètres testés)

    Positional arguments : 
    -------------------------------------
    trials : hyperopt.Trials : objet Trials contenant les informations sur chaque itération 
    loss : str : score à minimiser lors de l'optimisation
    """ 
    trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(lambda row: row[0]) for t in trials])
    trials_df[loss] = [t["result"]["loss"] for t in trials]
    trials_df["trial_number"] = trials_df.index
    
    return trials_df


def tokenize_text(doc: str):
    """
    Retourne une liste de tokens

    Positional arguments : 
    -------------------------------------
    doc : str : document à tokeniser
    """ 
    text = doc.lower()
    tokens = RegexpTokenizer(r'[a-zA-Z0-9]{2,}').tokenize(text)

    return tokens


def remove_stop_words(tokens: [str], language='english', verbose=True):
    """
    Retourne une liste de tokens sans les mots trop génériques pour être informatifs

    Positional arguments : 
    -------------------------------------
    tokens : list of str : liste de tokens à filtrer
    
    Optional arguments :
    -------------------------------------
    language : str : langue des tokens
    verbose : bool : afficher le nombre de tokens supprimés ou non
    """ 
    stop_words = set(stopwords.words(language))
    voc = [w for w in tokens if w not in stop_words]

    if verbose:
        voc_n = len(voc)
        print("{:_} tokens supprimés, il reste {:_} tokens".format(
            len(tokens)-voc_n, voc_n))

    return voc


def lemmatize_tokens(tokens: [str], verbose=True):
    """
    Retourne une liste de tokens sous leur forme canonique

    Positional arguments : 
    -------------------------------------
    tokens : list of str : liste de tokens à lemmatiser
    
    Optional arguments :
    -------------------------------------
    verbose : bool : afficher le nombre de tokens uniques ou non
    """ 
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(i) for i in tokens]

    if verbose:
        print('Avec la lemmatisation, il reste {:_} tokens uniques'.format(
            len(set(lemmatized_text))))

    return lemmatized_text


def stem_tokens(tokens: [str], verbose=True):
    """
    Retourne une liste de racines de tokens

    Positional arguments : 
    -------------------------------------
    tokens : list of str : liste de tokens à raciniser
    
    Optional arguments :
    -------------------------------------
    verbose : bool : afficher le nombre de tokens uniques ou non
    """ 
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(i) for i in tokens]

    if verbose:
        print('Avec la racinisation, il reste {:_} tokens uniques'.format(
            len(set(stemmed_text))))

    return stemmed_text


def display_wordcloud_by_label(products_dataset: pd.DataFrame, labels: str, text: str, column_nb: int, figsize: tuple,
                               title: str, max_words=50, stop_words=set(stopwords.words('english')), top=0.9,
                               wspace=0.1, hspace=0.7):
    """
    Affiche une représentation graphique des tokens les plus utilisés pour chaque catégorie listée

    Positional arguments : 
    -------------------------------------
    products_datase : pd.DataFrame : jeu de données contenant les tokens
    labels : str : nom de la colonne contenant les catégories
    text : str : nom de la colonne contenant les tokens
    column_nb : int : nombre de graphiques à afficher par ligne
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    title : str : titre principal de la zone de graphique
    
    Optional arguments :
    -------------------------------------
    max_words : int : nombre maximum de tokens à afficher par catégorie
    stop_words : set : set de mots trop génériques pour être informatifs
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """ 

    rgb_text = sns.color_palette('Greys', 15)[12]
    fig, axes = plt.subplots(
        ceil(len(labels) / column_nb), column_nb, figsize=figsize)
    fig.tight_layout()
    suptitle_text = 'Mots les plus utilisés ' + title
    fig.suptitle(suptitle_text, fontname='Corbel', fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    (l, c) = (0, 0)

    for label in products_dataset[labels].unique():
        products = products_dataset[products_dataset[labels] == label]
        corpus = " ".join(products[text].values)
        voc = tokenize_text(corpus)

        wordcloud = WordCloud(background_color='white',
                              stopwords=stop_words,
                              max_words=max_words).generate(" ".join(voc))
        axes[l, c].imshow(wordcloud)
        axes[l, c].set_title(label, fontname='Corbel',
                             fontsize=20, pad=50, color=rgb_text)
        axes[l, c].axis('off')

        (c, l) = (0, l + 1) if c == column_nb - 1 else (c + 1, l)

    plt.show()
    

def display_wordcloud(tokens: [str], title: str, figsize=(12, 6), max_words=30, stopwords=[]):
    """
    Affiche une représentation graphique des tokens les plus utilisés

    Positional arguments : 
    -------------------------------------
    tokens : list of strings : liste de tokens  
    title : str : titre du graphique
    
    Optional arguments :
    -------------------------------------
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    max_words : int : nombre maximum de tokens à afficher par catégorie
    stop_words : list : liste de mots trop génériques pour être informatifs
    """ 
    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='white', palette='Set2')
    plt.figure(figsize=figsize)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=max_words).generate(" ".join(tokens))
    plt.imshow(wordcloud)
    plt.title(title, fontname='Corbel', fontsize=20, pad=50, color=rgb_text)
    plt.axis('off')
    plt.show()
    

def get_most_used_words(n: int, thresh: int, products_dataset: pd.DataFrame(), labels: str, text: str):
    """
    Renvoie les tokens parmi les "n" tokens les plus utilisés dans plus de "thresh" classes

    Positional arguments : 
    -------------------------------------
    n : int : n premiers tokens les plus utilisés dans chaque classe
    thresh : int : nombre de classes
    products_dataset : pd.DataFrame : jeu de données contenant les tokens
    labels : str : nom de la colonne contenant les classes
    text : str : nom de la colonne contenant les tokens
    """ 
    most_used_words_df = pd.DataFrame()

    for label in products_dataset[labels].unique():
        products = products_dataset.loc[products_dataset[labels] == label]
        corpus = " ".join(products[text].values)
        voc = tokenize_text(corpus)
        voc = remove_stop_words(voc, verbose=False)
        voc = lemmatize_tokens(voc, verbose=False)
        voc = stem_tokens(voc, verbose=False)
        voc_counts = pd.Series(voc).value_counts().head(n)

        most_used_words_df = pd.concat([most_used_words_df, voc_counts])

    most_used_words_df = most_used_words_df.index.value_counts()
    t = most_used_words_df[most_used_words_df >
                           thresh].sort_values(ascending=False)

    print('{} tokens parmi les {} tokens les plus utilisés dans plus de {} classes'.format(
        len(t), n, thresh))

    return pd.DataFrame({'words': t.index, 'labels_n': t.values})


def get_uncommon_tokens(vocabulary: [str], n_min: int):
    """
    Renvoie les tokens les moins utilisés

    Positional arguments : 
    -------------------------------------
    vocabulary: list of strings : list de tokens à filtrer
    n_min : int : nombre minimum d'apparitions du mot dans le vocabulaire brut
    """ 
    value_counts = pd.Series(vocabulary).value_counts()
    uncommon_tokens = value_counts[value_counts <= n_min]

    uncommon_tokens = list(uncommon_tokens.index)
    uncommon_tokens_n = len(uncommon_tokens)

    print('{:_} tokens utilisés {} fois ou moins'.format(
        uncommon_tokens_n, n_min))

    return uncommon_tokens


def plot_screeplot(model, n_components: int, figsize: tuple, titles: dict, color_bar: str,
                   legend_x: float, legend_y: float):
    """
    Affiche l'éboulis des valeurs propres avec la courbe de la somme cumulée des inertie

    Positional arguments : 
    -------------------------------------
    model : : modèle de réduction dimensionnelle non linéaire (ACP ou SVD déjà entrainé)
    n_components : int : nombre d'axes d'inertie
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    titles : dict : titres du graphique et des axes - ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'a'}
    color_bar : str : couleur utilisée pour le diagramme à bar
    """
    scree = (pca.explained_variance_ratio_ * 100).round(2)
    scree_cum = scree.cumsum().round(2)
    x_list = range(1, n_components + 1)

    rgb_text = sns.color_palette('Greys', 15)[12]

    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 18})
        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(x_list, scree, color=color_bar)
        ax.set_xticks(x_list)
        ax.plot(x_list, scree_cum, color='coral', marker='o', markerfacecolor='white', markeredgecolor='coral',
                markersize=18, markeredgewidth=2)
        ax.text(legend_x, legend_y, "variance cumulée", fontsize=20, color='coral', fontname='Corbel')

    plt.title(titles['chart_title'], fontname='Corbel', fontsize=23, pad=20, color=rgb_text)
    plt.ylabel(titles['y_label'], color=rgb_text, fontsize=18)
    plt.xlabel(titles['x_label'], color=rgb_text, fontsize=18)
    plt.grid(False, axis='x')

    plt.show()
    

def train_w2vec_model(sentences: list, min_count: int, window: int, vector_size: int, tokenize=False, epochs=100):
    """
    Renvoie un modèle Word2Vec entrainé avec la méthode Continuous bag of words

    Positional arguments : 
    -------------------------------------
    sentences : list : jeu d'entrainement (liste de documents brutes)
    min_count : int : les tokens qui apparaissent moins de min_count dans le corpus sont supprimés
    window : int : distance max entre un mot et le mot prédit lors de l'entrainement 
    vector_size : int : nombre de word embeddings assignés à chaque token du vocabulaire

    Optional arguments :
    -------------------------------------
    tokenize : bool : tokeniser ou non les documents du jeu d'entrainement
    epochs : int : nombre maximum d'itérations autorisé lors de la descente de gradient
    """
    if tokenize:
        sentences = [gensim.utils.simple_preprocess(
            text) for text in sentences]

    w2v_model = gensim.models.Word2Vec(min_count=min_count,
                                       window=window,
                                       vector_size=vector_size,
                                       seed=8,
                                       workers=1)
    w2v_model.build_vocab(sentences)
    w2v_model.train(
        sentences, total_examples=w2v_model.corpus_count, epochs=epochs)

    return w2v_model


def get_w2v_vocab(w2v_vectors: KeyedVectors, print_info=True, sample_word=None):
    """
    Renvoie le vocabulaire associé à un modèle Word2Vec déjà entrainé et affiche des informations supplémentaires

    Positional arguments : 
    -------------------------------------
    w2v_vectors : KeyedVectors : vecteurs d'embeddings retournés par un modèle entrainé

    Optional arguments :
    -------------------------------------
    print_info : bool : afficher des informations sur le vocabulaire et un mot en particulier
    sample_word : str : mot pour lequel on souhaite afficher des informations supplémentaires
    """
    w2v_words = w2v_vectors.index_to_key

    if print_info:
        w2v_words_n = len(w2v_words)
        print("Modèle Word2Vec : \n")
        print("Nombre de tokens dans le vocabulaire: {:_}".format(w2v_words_n))

        if sample_word is None:
            sample_word = w2v_words[randint(0, w2v_words_n + 1)]

        print('Mot du vocabulaire : ', sample_word)
        print('Dimensions du vecteur associé à ce mot: ',
              w2v_vectors[sample_word].shape)
        print('10 mots les plus similaires à ce mot:')
        display(w2v_vectors.most_similar(sample_word, topn=10))

    return w2v_words


def tokens_to_int(sentences: list, pad: bool, sentence_max_len=1_000, verbose=True):
    """
    Remplace les tokens d'un corpus par des entiers positifs 
    et renvoie le modèle entrainé et utilisé pour transformer les tokens en entiers + les tokens transformés

    Positional arguments : 
    -------------------------------------
    sentences : list : liste de documents (corpus)
    pad : bool : faire en sorte que tous les documents aient le même nombre de tokens ou non

    Optional arguments :
    -------------------------------------
    sentence_max_len : int : nombre maximum de tokens par documents
    verbose : bool : afficher le nombre de tokens uniques identifiés dans le corpus
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sentences_int = tokenizer.texts_to_sequences(sentences)

    if pad:
        sentences_int = pad_sequences(sentences_int,
                                      maxlen=sentence_max_len,
                                      padding='post')
    if verbose:
        num_words = len(tokenizer.word_index)
        print("Nombre de tokens uniques: {:_}".format(num_words))

    return tokenizer, sentences_int


def build_embedding_matrix(embeddings_n: int, w2v_words: [str], w2v_vectors: KeyedVectors, word_index: dict,
                           verbose=True):
    """
    Renvoie la matrice contenant les embeddings word2vec assignés à chaque token du corpus
    (une ligne = un token unique du corpus, une colonne = un embedding calculé avec le modèle word2vec)

    Positional arguments : 
    -------------------------------------
    embeddings_n : int : nombre d'embeddings assignés à chaque token
    w2v_words : list of str : vocabulaire associé à un modèle Word2Vec entrainé 
    (i.e. tokens pour lesquels un vecteur d'embeddings existe)
    w2v_vectors : KeyedVectors : vecteurs issus d'un modèle word2Vec entrainé 
    (i.e. matrice de poids entre l'input et la fonction d'activation du réseau de neurone Word2Vec)
    word_index : dict : dictionnaire contenant les tokens du corpus (clé=token, valeur=entier positif)

    Optional arguments :
    -------------------------------------
    verbose : bool : afficher, ou pas, le format de la matrice d'embeddings
    et le taux de word embeddings, i.e. le rapport : 
    (nombre de tokens retenus lors du word embedding)/(nombre de tokens uniques dans le corpus)
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embeddings_n))
    (i, j) = (0, 0)
    for word, idx in word_index.items():
        i += 1
        if word in w2v_words:
            j += 1
            embedding_vector = w2v_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = w2v_vectors[word]

    if verbose:
        word_rate = np.round(j/i, 4)
        print("Word embedding rate : ", word_rate)
        print("Format de la matrice d'embeddings: %s\n" %
              str(embedding_matrix.shape))

    return embedding_matrix


def build_embedding_model(doc_max_len: int, embedding_matrix: np.ndarray, vocab_size: int, embeddings_n: int):
    """
    Renvoie un réseau de neurones construit pour transformer une liste de documents chacun représenté par une liste
    d'entiers positifs (représentant des tokens eux-mêmes associés à un nombre "embeddings_n" d'embeddings)
    en une liste de documents chacun représenté par une liste d'embeddings 
    obtenus en faisant la moyenne des tokens sur chaque dimension d'embedding.
    i.e. schématiquement on réduit la dimension d'une matrice : (docs, tokens, embeddings) -> (docs, embeddings)

    Positional arguments : 
    -------------------------------------
    doc_max_len : int : nombre maximum de tokens par document du corpus
    embedding_matrix : np.ndarray : matrice contenant les embeddings word2vec assignés à chaque token du corpus
    i.e. (une ligne = un token)
    vocab_size : int : nombre de tokens dans le corpus
    embeddings_n : int : nombre d'embeddings assignés à chaque token
    """
    doc_input = Input(shape=(doc_max_len,), dtype='float64')

    word_embedding = Embedding(input_dim=vocab_size,
                               output_dim=embeddings_n,
                               weights=[embedding_matrix],
                               input_length=doc_max_len)(doc_input)

    word_vec = GlobalAveragePooling1D()(word_embedding)
    model = Model([doc_input], word_vec)

    model.summary()

    return model


def get_embeddings_w2v(corpus: list, doc_max_len: int, embeddings_n: int, w2v_vocab: [str], w2v_vectors: KeyedVectors):
    """
    Renvoie une matrice de documents chacun représenté par une liste d'embeddings.

    Positional arguments : 
    -------------------------------------
    corpus : list : liste de documents à représenter par des nombres faisant sens (embeddings)
    doc_max_len : int : nombre maximum de tokens par document du corpus
    embeddings_n : int : nombre d'embeddings à assigner à chaque document
    w2v_vocab : list of str : vocabulaire associé à un modèle Word2Vec entrainé 
    (i.e. tokens pour lesquels un vecteur d'embeddings existe)
    w2v_vectors : KeyedVectors : vecteurs issus d'un modèle word2Vec entrainé 
    (i.e. matrice de poids entre l'input et la fonction d'activation du réseau de neurone Word2Vec)
    """
    tokenizer, corpus_int = tokens_to_int(corpus, True, doc_max_len, False)
    embedding_matrix = build_embedding_matrix(embeddings_n,
                                              w2v_vocab,
                                              w2v_vectors,
                                              tokenizer.word_index
                                              )
    embedding_model = build_embedding_model(doc_max_len,
                                            embedding_matrix,
                                            len(tokenizer.word_index) + 1,
                                            embeddings_n
                                            )
    embeddings = embedding_model.predict(corpus_int)

    return embeddings


def get_inputs_bert(corpus: [str], tokenizer: BertTokenizerFast, doc_max_length=512):
    """
    Renvoie la liste des tokens, chacun représenté par un entier positif correspondant à son indice dans le vocabulaire, 
    la liste des embeddings de segment, qui indiquent dans quel segment (phrase) du document se trouve chaque token
    (ex : [0, 0, 1] -> le premier token se trouve dans la première phrase, le troisième token dans la deuxième phrase), 
    la liste attention_mask, qui indique si le token a été ajouté au moment du padding pour que toutes les phrases ou
    documents fassent la même taille (et donc qu'il ne faut pas y faire attention) ou non


    Positional arguments : 
    -------------------------------------
    corpus : list of str : liste de documents
    tokenizer : BertTokenizerFast : objet permettant de tokeniser les documents

    Optionnal arguments : 
    -------------------------------------
    doc_max_lenght : int : taille maximum d'un document
    """
    input_ids = []
    token_type_ids = []
    attention_mask = []

    for doc in corpus:
        bert_input = tokenizer(doc,
                               add_special_tokens=True,
                               max_length=doc_max_length,
                               padding='max_length',
                               return_attention_mask=True,
                               return_token_type_ids=True,
                               truncation=True,
                               return_tensors="tf")

        input_ids.append(bert_input['input_ids'][0])
        token_type_ids.append(bert_input['token_type_ids'][0])
        attention_mask.append(bert_input['attention_mask'][0])

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)

    return input_ids, token_type_ids, attention_mask


def get_embeddings_bert(model, model_type: str, corpus: [str], doc_max_length: int, batch_size: int, mode: BertMode):
    """
    Renvoie les embeddings du corpus obtenus à partir des embeddings d'un modèle BERT pré-entrainé

    Positional arguments : 
    -------------------------------------
    model :  : modèle Bert pré-entrainé
    model_type : str : type de modèle bert (ex: 'bert-base-uncased')
    corpus : list of str : liste de documents
    doc_max_length : int : taille maximum d'un document
    batch_size : int : nombre de documents par batch
    mode : BertMode : énumération qui indique depuis quelle source le modèle a été chargé 
    """
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    input_ids, token_type_ids, attention_mask = get_inputs_bert(
        corpus, tokenizer, doc_max_length)

    last_hidden_states = np.array([]).reshape(0, doc_max_length, 768)
    for batch in range(len(corpus)//batch_size):
        idx = batch*batch_size

        if mode.name == 'TENSORFLOW_HUB':
            text_preprocessed = {"input_word_ids": input_ids[idx:idx+batch_size],
                                 "input_mask": attention_mask[idx:idx+batch_size],
                                 "input_type_ids": token_type_ids[idx:idx+batch_size]}

            outputs = model(text_preprocessed)
            last_hidden_states = np.concatenate(
                (last_hidden_states, outputs['sequence_output']))

        elif mode.name == 'HUGGINGFACE':
            outputs = model([input_ids[idx:idx+batch_size],
                             attention_mask[idx:idx+batch_size]
                             ]
                            )
            last_hidden_states = np.concatenate(
                (last_hidden_states, outputs.last_hidden_state))

    embeddings_matrix = np.array(last_hidden_states).mean(axis=1)

    return embeddings_matrix


def get_embeddings_use(model, corpus: [str], batch_size: int):
    """
    Renvoie les embeddings du corpus obtenus à partir des poids d'un modèle USE pré-entrainé

    Positional arguments : 
    -------------------------------------
    model : : modèle Universal Sentence Encoder pré-entrainé
    corpus : list of str : liste de documents
    batch_size : int : nombre de documents par batch
    """
    for step in range(len(corpus)//batch_size):
        idx = step*batch_size
        feat = model(corpus[idx:idx+batch_size])

        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

    return features


def display_image_by_label(data: pd.DataFrame, label_col: str, img_col: str, img_n: int, figsize: tuple,
                           img_path: str, top=0.9, wspace=0.1, hspace=0.7):
    """
    Affiche un certain nombre d'images pour chaque catégorie

    Positional arguments : 
    -------------------------------------
    data : pd.DataFrame : jeu de données contenant les noms des images à afficher et leur étiquettes
    label_coll : str : nom de la colonne contenant les catégories d'images
    img_col : str : nom de la colonne contenant les noms des images
    img_n : int : nombre d'images à afficher par catégorie
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    img_path : str : chemin du dossier contenant les images

    Optionnal arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    label_n = len(data[label_col].unique())
    fig, axes = plt.subplots(label_n, img_n, figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]
    fig.tight_layout()
    fig.suptitle('Images par label', fontname='Corbel',
                 fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    subset = data.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), ceil(img_n))))
    for l, label in enumerate(subset[label_col].unique()):
        for c, img in enumerate(subset.loc[subset[label_col] == label, img_col].values):
            axes[l, c].imshow(Image.open(img_path + img))
            axes[l, c].set_title(label, fontname='Corbel',
                                 fontsize=20, pad=30, color=rgb_text)
            axes[l, c].axis('off')

    plt.show()
    
    
def display_image_transformation(original, transformed, title: str, figsize: tuple,
                                 top=0.6, wspace=0.1, hspace=0.7, axis_off=True):
    """
    Affiche une image avant et après transformation

    Positional arguments : 
    -------------------------------------
    original : : image avant transformation
    transformed : : image après transformation
    tile : str : titre principal de la zone de graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optionnal arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    axis_off : bool : afficher ou non la graduation sur les axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]
    fig.tight_layout()
    fig.suptitle(title, fontname='Corbel', fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    params = [{'c': 0, 'img': original, 'title': 'Image Originale'},
              {'c': 1, 'img': transformed, 'title': 'Image Transformée'}]

    for param in params:
        axes[param['c']].imshow(param['img'])
        axes[param['c']].set_title(
            param['title'], fontname='Corbel', fontsize=20, pad=20, color=rgb_text)
        if axis_off: 
            axes[param['c']].axis('off')

    plt.show()
    
    
def display_histo_transformation(img_original, img_transformed, title: str, figsize: tuple,
                                 top=0.9, wspace=0.1, hspace=0.7):
    """
    Affiche une image et son histogramme avant et après transformation

    Positional arguments : 
    -------------------------------------
    img_original : : image avant transformation
    img_transformed : : image après transformation
    tile : str : titre principal de la zone de graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optionnal arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]
    fig.tight_layout()
    fig.suptitle(title, fontname='Corbel', fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    params = [{'c': 0, 'img': img_original, 'title': 'Image Originale'},
              {'c': 1, 'img': img_transformed, 'title': 'Image Transformée'}]

    for param in params:
        axes[0, param['c']].imshow(param['img'], cmap='gray')
        axes[0, param['c']].set_title(
            param['title'], fontname='Corbel', fontsize=20, pad=30, color=rgb_text)
        axes[0, param['c']].axis('off')

    axes[1, 0].hist(np.array(img_original).flatten(), bins=range(256))
    axes[1, 1].hist(np.array(img_transformed).flatten(), bins=range(256))

    for c in range(0, 2):
        axes[1, c].set_title('Histogramme ' + params[c]['title'],
                             fontname='Corbel',
                             fontsize=20, pad=30,
                             color=rgb_text)
        axes[1, c].set_xlabel(
            'niveau de gris', fontname='Corbel', fontsize=15, color=rgb_text)
        axes[1, c].set_ylabel('nombre de pixels',
                              fontname='Corbel', fontsize=15, color=rgb_text)

    plt.show()
    
    
def get_keypoint_descriptors_sift(sift, img: np.ndarray):
    """
    Renvoie les points d'intérêts identifiés par un algo SIFT dans une image 
    ainsi que les vecteurs associés qui décrivent leur voisinage (descripteurs)

    Positional arguments : 
    -------------------------------------
    sift : : objet SIFT
    img : np.ndarray : image 
    """
    if (img.shape[0] > 4_000) or (img.shape[1] > 4_000):
        img = cv2.resize(img, (0, 0),
                         fx=.1, fy=.1,
                         interpolation=cv2.INTER_CUBIC
                         )

    img = cv2.equalizeHist(img)

    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def show_key_points_sift(image_path: str, figsize: tuple, print_descriptors=False):
    """
    Affiche les points d'intérêts d'une image extraits par un algo SIFT

    Positional arguments : 
    -------------------------------------
    image_path : str : emplacement de l'image 
    figsize : tuple : taille de la zone d'affichage de l'image (largeur, hauteur)

    Optionnal arguments : 
    -------------------------------------
    print_descriptors : bool : affiche ou non la matrice des descripteurs associés aux points d'intérêts
    """
    sift = cv2.SIFT_create()
    image = cv2.imread(image_path, 0)
    kp, des = get_keypoint_descriptors_sift(sift, image)

    img = cv2.drawKeypoints(image, kp, image)

    plt.figure(figsize=figsize)
    title = '{:_} points d\'intérêts extraits par SIFT'
    plt.title(title.format(len(kp)), fontsize=20, pad=10)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    if print_descriptors:
        print("\nDescripteurs  associés aux features: \n")
        print(des)
        print('\nMatrice de dimensions :', des.shape)
        
        
def get_descriptors_sift(img_collection: pd.Series, img_path: str, n_features: int, verbose=False):
    """
    Renvoie une liste contenant pour chaque image de la collection une matrice de descripteurs calculés par un algo SIFT

    Positional arguments : 
    -------------------------------------
    img_collection : pd.Series : noms des images de la collection
    image_path : str : emplacement des images
    n_features : int : nombre de meilleurs descripteurs à retenir

    Optionnal arguments : 
    -------------------------------------
    verbose : bool : toutes les 100 images, afficher ou non l'indice de l'image en cours de traitement 
    """
    descriptors_all = []
    sift = cv2.SIFT_create(n_features)

    for idx, img_name in enumerate(img_collection):
        if verbose and idx % 100 == 0:
            print('image n°', idx + 1)

        img = cv2.imread(img_path + img_name, 0)
        kp, des = get_keypoint_descriptors_sift(sift, img)
        descriptors_all.append(des)

    return descriptors_all


def build_histogram(kmeans: MiniBatchKMeans, descriptors: np.ndarray, image_idx: int):
    """
    Renvoie, pour une image donnée, un histogramme indiquant la fréquence d'apparition de chacun des visual words

    Positional arguments : 
    -------------------------------------
    kmeans : MiniBatchKMeans : modèle kMeans entrainé pour associer chaque descripteur à un visual word
    descriptors : np.ndarray : descripteurs des features de l'image
    image_idx : int : indice de l'image dans la collection
    """
    nb_descriptors = len(descriptors)
    hist = np.zeros(len(kmeans.cluster_centers_))

    if nb_descriptors == 0:
        print("Aucun descripteur pour l'image n°", image_idx)
        return hist

    visual_words = kmeans.predict(descriptors)
    for visual_word in visual_words:
        hist[visual_word] += 1.0/nb_descriptors

    return hist


def fit_visual_words_model(sift_descriptors: list, verbose=False):
    """
    Renvoie un modèle KMeans entrainé pour regrouper des descripteurs SIFT de features en un nombre limité de visual
    words (visual words = centres des clusters trouvés)

    Positional arguments : 
    -------------------------------------
    sift_descriptors : list : liste de descripteurs SIFT

    Optionnal arguments : 
    -------------------------------------
    verbose : bool : afficher ou non le nombre de visual words
    """
    sift_descriptors_all = np.concatenate(sift_descriptors, axis=0)
    clusters_n = int(round(np.sqrt(len(sift_descriptors_all)), 0))

    if verbose:
        print('Nombre de visual words : ', clusters_n)

    kmeans = MiniBatchKMeans(n_clusters=clusters_n,
                             init_size=3 * clusters_n,
                             random_state=8,
                             n_init=3,
                             batch_size=1024)

    kmeans.fit(sift_descriptors_all)

    return kmeans


def build_bag_of_features(sift_descriptors: list, verbose=False):
    """
    Renvoie une matrice de bag_of_features à partir des descripteurs SIFT d'une collection d'images

    Positional arguments : 
    -------------------------------------
    sift_descriptors : list : liste des matrices de descripteurs des features d'une collection d'images

    Optionnal arguments : 
    -------------------------------------
    verbose : bool : toutes les 100 images, afficher ou non l'indice de l'image en cours de traitement 
    """
    kmeans = fit_visual_words_model(sift_descriptors, True)

    hist_vectors = []
    sift_descriptors_by_img = np.asarray(sift_descriptors, dtype=object)

    for idx, descriptors in enumerate(sift_descriptors_by_img):
        if verbose and idx % 100 == 0:
            print('image n°', idx)
        hist = build_histogram(kmeans, descriptors, idx)
        hist_vectors.append(hist)
    bag_of_features = np.asarray(hist_vectors)

    return bag_of_features


def get_features_transfer_learning(model, img_collection: pd.Series, img_path: str, verbose=False):
    """
    Renvoie les features d'une collection d'images en utilisant les features d'un réseau de neurones 
    convolutionnel pré-entrainé

    Positional arguments : 
    -------------------------------------
    model : : réseau de neurone convolutionnel pré-entrainé pour classer des images
    img_collection : pd.Series : noms des images
    img_path : str : emplacement des images

    Optionnal arguments : 
    -------------------------------------
    verbose : bool : toutes les 100 images, afficher ou non l'indice de l'image en cours de traitement 
    """
    images_features = []

    for idx, img_name in enumerate(img_collection):
        if verbose and idx % 100 == 0:
            print('image n°', idx + 1)
        image = load_img(img_path + img_name, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        images_features.append(model.predict(image, verbose=0)[0])

    images_features = np.asarray(images_features)

    return images_features


def preprocess_img_vgg16(img_collection: pd.Series, img_path: str, target_size=(224, 224)):
    """
    Renvoie des images qui respectent les spécifications des images en entrée du modèle vgg-16

    Positional arguments : 
    -------------------------------------
    img_collection : pd.Series : noms des images de la collection
    img_path : str : emplacement des images

    Optionnal arguments : 
    -------------------------------------
    target_size : tuple : nouvelles dimensions des images
    """
    preprocessed_img = []
    for img_name in img_collection:
        img = load_img(img_path + img_name, target_size=target_size)
        img = img_to_array(img)
        img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)

        preprocessed_img.append(img)
    preprocessed_img = np.array(preprocessed_img)

    return preprocessed_img
