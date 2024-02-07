###### Méthode d'évaluation  ######
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

#df_merged = df_sync.append(def_rel, ignore_index=True)
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

