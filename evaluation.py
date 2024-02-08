###### Méthode d'évaluation  ######
import pandas as pd
import numpy as np
from math import sqrt

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression

from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.special import kl_div
from scipy.stats import f
from scipy.stats import multivariate_t



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

################# Mesure étroite / IO, EO #################################

def Mont_carlo_io(loc, scale, int_1, int_2, n=5000):
    """
    Calcule la proba qu'un point appartenant à l'intervalle 1 appartiennent a l'intervalle 2 en suivant une loi normal
    """
    l=[]
    while len(l)<n:
        tirage = np.random.normal(loc, scale)
        if tirage >= int_1[0] & tirage<=int_1[1]:
            l.append(tirage)
    P = np.sum((l >= int_2[0]) & (l <= int_2[1]))
    return P/n


def IO(df_syn, df_rel):
    """
    Calcule le % de chevauchement d'intervalle de confiance entre données réelles et données synthétiques

    Paramètres :
    df_syn : data frame -> jeu de donnée synthétique
    df_rel : data frame -> jeu de donnée réelle

    Retourne :
    I/p : int -> probabilitée de chevauchement d'intervale calculer avec une regression linaire et MontCarlos    
    """
    I = 0 #résultat final
    p = 0 #compteur de tours
    for k in np.unique(df_rel['action']):
        p +=1

        #def de X et y en fonction de l'action regarder
        syn_it = df_syn[df_syn['action'] == k]
        rel_it = df_rel[df_rel['action'] == k]
        y_syn = syn_it['time']
        y_rel = rel_it['time']
        X_syn = syn_it.drop(['time'], axis=1)
        X_rel = rel_it.drop(['time'], axis=1)
        
        #recupération des coéfficents directeur du verbe pour une action k en fonction du temps
        reg_syn = LinearRegression().fit(X_syn, y_syn)
        reg_rel = LinearRegression().fit(X_rel, y_rel)
        syn_coef = reg_syn.coef_[0]
        rel_coef = reg_rel.coef_[0]

        #intervalle de confiance a 95%
        int_syn = [syn_coef - 1.96* sqrt(np.var(X_syn)/len(X_syn)), syn_coef + 1.96* sqrt(np.var(X_syn)/len(X_syn))]
        int_rel = [rel_coef - 1.96* sqrt(np.var(X_rel)/len(X_rel)), rel_coef + 1.96* sqrt(np.var(X_rel)/len(X_rel))]

        #Mont Carlo pour supperposition (non nécéssaire lol)
        Ik = 1/2 * (Mont_carlo_io(syn_coef, sqrt(np.var(X_syn)), int_syn, int_rel), 
                    Mont_carlo_io(rel_coef, sqrt(np.var(X_rel)), int_rel, int_syn))

    I += Ik
    return I/p


def EO(df_syn, df_rel):
    """
    Calcule le % de chevauchement d'éllipsoïdes entre données réelles et données synthétiques

    Paramètres :
    df_syn : data frame -> jeu de donnée synthétique
    df_rel : data frame -> jeu de donnée réelle

    Retourne :
    1/2 * (p_1+p_2) : int -> probabilitée de chevauchement des ellispoïdes
    """
    Y_syn = df_syn['time']
    Y_rel = df_rel['time']
    X_syn = df_syn.drop(['time'], axis=1)
    X_rel = df_rel.drop(['time'], axis=1)
    
    def elipsoides(X, Y) :
        """
        Cette fonction crée l'élipsoïde correspondante aux données X et Y

        Paramètres: 
        X : array -> objet + verbe
        Y : array -> time stamp

        Retourne : 
        d : dict -> les paramètres de Bétat
        """
        #recupération des coéfficents directeur du verbe  en fonction du temps
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y #coef de la regression linéaire
        residuals = Y - X @ beta_hat
        p = X_syn.shape[1]
        n = X_syn.shape[0]
        sigma_hat_squared = np.sum(residuals ** 2) / (n - p)

        #Création des élipsoïdes
        alpha = 0.05
        f_critical = f.ppf(1 - alpha, p, n - p) #le quantil d'ordre 1-alpha d'une Fisher de paramètre (p, n-p)
        inf = beta_hat - sqrt(f_critical * p * sigma_hat_squared / np.diag(X.T @ X)) #potentiellement just (X.T @ X) en non np.diag(X.T @ X)
        sup = beta_hat + sqrt(f_critical * p * sigma_hat_squared / np.diag(X.T @ X))

        d = {'X' : X, 'Y' : Y, 'intervale' : [inf, sup], 'beta_hat' : beta_hat, 'sigma' : sigma_hat_squared, 'p' : p, 'n' : n}
        return d
    
    d_rel = elipsoides(X_rel, Y_rel)
    d_syn = elipsoides(X_syn, Y_syn)

    def Mont_carlo(X, Y, beta_hat, sigma_squared, degrees_of_freedom, num_samples=5000):
        """
        Génère num_samples données en fonction d'une loi de student multivariée (ref rapport écrit)
        """
        inv_XTX = np.invert(X.T @ X)
        cov_matrix = sigma_squared * inv_XTX
        
        #création de la distribution 
        t_dist = multivariate_t(loc=beta_hat, shape=cov_matrix, df=degrees_of_freedom)

        tirage = t_dist.rvs(size=num_samples)
        return tirage

    #nous effectuons des tirages qui suivent la loi à postérieurie de X_rel et X_syn 
    num_samples = 5000
    tirage_rel = Mont_carlo(X=d_rel['X'], Y=d_rel['Y'], beta_hat=d_rel['beta_hat'], sigma_squared=d_rel['sigma'], degrees_of_freedom=d_rel['p'] - d_rel['n'], num_samples=num_samples)
    tirage_syn = Mont_carlo(X=d_syn['X'], Y=d_syn['Y'], beta_hat=d_syn['beta_hat'], sigma_squared=d_syn['sigma'], degrees_of_freedom=d_syn['p'] - d_rel['n'], num_samples=num_samples)
    
    #Cacule du % de recouvrement dans chaque cas
    p_1 = np.sum(((tirage_rel[:,0] >= d_syn['intervale'][0][0]) & (tirage_rel[:,0] >= d_syn['intervale'][1][0])) & ((tirage_rel[:,0] >= d_syn['intervale'][0][1]) & (tirage_rel[:,0] >= d_syn['intervale'][1][1])))
    p_1 = p_1 / num_samples
    p_2 = np.sum(((tirage_syn[:,0] >= d_rel['intervale'][0][0]) & (tirage_syn[:,0] >= d_rel['intervale'][1][0])) & ((tirage_syn[:,0] >= d_rel['intervale'][0][1]) & (tirage_syn[:,0] >= d_rel['intervale'][1][1])))     
    p_2 = p_2 / num_samples

    #moyenne les deux probas
    return 1/2 * (p_1+p_2)