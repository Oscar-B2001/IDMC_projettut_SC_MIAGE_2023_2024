###### Méthode d'évaluation  ######
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression

from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.special import kl_div


#df_merged = df_syn.append(def_rel, ignore_index=True)
#df_merged.sort_values(by=['temps])
#df_merged --> objet+nom+verbe

def pca_temp(df_merged, m):
    """
    Cette fonction effectue une ACP temporel sur nos données avec la méthode de la fenetre temporelle.

    Paramètres:
     - df_merged -> data frame, concaténation du jeu de donnée synthétique et du jeu de donnée réelle triée par code temporel sans le code temporel
     - m -> int, nombre de fenetres temporelles

    Return : 
     - df_pca -> data frame, jeu de donnée projeter en 2 dimention
    """
    df_merged.drop(['time'], axis=1, inplace=True)

    X_pca = pd.DataFrame(columns=['c1', 'c2'])
    n_length = len(df_merged)//m
    for i in range(m):
        if i!=m-1:
            mypca = PCA(n_components=2).fit(df_merged[m*n_length:(1+m)*n_length])
        elif i==m-1 :
             mypca = PCA(n_components=2).fit(df_merged[m*n_length:len(df_merged)])
        X_pca = X_pca.append(mypca, ignore_index=True)
    
    return X_pca

def mds_temp(df_merged, m, n):
    """
    Cette fonction effectue une MDS linéaire temporel sur nos données avec la méthode de la fenetre temporelle.

    Paramètres:
     - df_merged -> data frame, concaténation du jeu de donnée synthétique et du jeu de donnée réelle triée par code temporel sans le code temporel
     - m -> int, nombre de fenetres temporelles
     - n -> int, taille des overlaps

    Return : 
     - df_mds -> data frame, jeu de donnée projeter en 1 dimention puis exprimer avec le time stamp
    """
    T = df_merged(['time'])
    df_merged.drop(['time'], axis=1, inplace=True)

    X_mds = pd.DataFrame(columns=['c1'])
    n_length = len(df_merged)//m
    for i in range(m):
        if i!=m-1:
            mymds = MDS(n_components=1).fit(df_merged[m*n_length:(1+m)*n_length])
        elif i==m-1 :
            mymds = MDS(n_components=1).fit(df_merged[m*n_length:len(df_merged)])
        X_mds = X_mds.append(mymds, ignore_index=True)
    X_mds['time'] = T
    return X_mds


def t_sne(df_merged, perplexity): 
    df_merged.drop(['time'], axis=1, inplace=True)
    tsne = TSNE(n_components=2, perplexity=perplexity).fit(df_merged)
    return tsne

################# MSE / DTW ##########################

def mse(s_1, s_2):
    """
    Cette fonction évalue la MSE entre deux série temoprelle de dimension >=1

    Paramètres:
    s_1, s_2  --> list/numpy/tuple, deux série temporelles de dimensions >=1

    Return :
    int, distance euclédienne carrée entre deux série temporel 
    """
    if len(s_1) != len(s_2) :
        raise ValueError('len(df_rel) != len(df_syn)')
    else :
        return np.mean(np.square(np.subtract(s_1, s_2)))


def dtw(s_1, s_2):
    """
    Calcule la distance DTW entre deux séries temporelles de dimension >=1 s_1 et s_2.

    Paramètres:
    s_1, s_2  --> list/numpy/tuple, deux série temporelles de dimension >=1

    Return :
    dtw_matrix[len_s1, len_s2] --> int, distance DTW entre s_1 et s_2
    """
    # Calculer la taille des séries temporelles
    len_s1, len_s2 = len(s_1), len(s_2)

    # Initialiser la matrice de coût DTW avec des valeurs infinies
    dtw_matrix = np.zeros((len_s1 + 1, len_s2 + 1))
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, :] = np.inf
    dtw_matrix[0, 0] = 0

    # Remplir la matrice de coût DTW
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = euclidean(s_1[i - 1], s_2[j - 1])  # Calcul du coût local
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Coût de déplacement vers le haut
                                           dtw_matrix[i, j - 1],    # Coût de déplacement vers la gauche
                                           dtw_matrix[i - 1, j - 1])  # Coût de déplacement en diagonale

    # Retourner la distance DTW entre les deux séries temporelles
    return dtw_matrix[len_s1, len_s2]


def mesure_larges(df_syn, df_rel):
    """
    retourn dic des distances + kullback leibler
    """
    X_syn = df_syn.drop(['time'], axis=1)
    X_rel = df_rel.drop(['time'], axis=1)
    dist = {'MSE' : mse(X_rel, X_syn), 'DTW' : dtw(X_rel, X_syn), 'KL' : kl_div(X_rel, X_syn)}
    return dist

def Mont_carlo():
    pass

def IO(df_syn, df_rel):
    for i in np.unique(df_rel['action']):
        syn_it = df_syn[df_syn['action'] == i]
        rel_it = df_rel[df_rel['action'] == i]
        y_syn = df_syn['time']
        y_rel = df_rel['time']
        X_syn = df_syn.drop(['time'], axis=1)
        X_rel = df_rel.drop(['time'], axis=1)
        reg_syn = LinearRegression().fit(X_syn, y_syn)
        reg_rel = LinearRegression().fit(X_rel, y_rel)

    pass