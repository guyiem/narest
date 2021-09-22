#coding: utf-8

"""
Ce module contient les fonctions en charge de l'etude
de la repartition des NaN dans des series temporelles.

Le principe generale sera globalement :
Les fonctions vont prendre en entree des pandas DataFrame
indexees en temps, avec des tickers en colonne. Les valeurs pourront
etre des prix, ou des hurst, etc.

Elles vont ensuite, ticker par ticker, garder les valeurs entre
le et le dernier non-NaN. Et proceder sur cette partie a une analyse
des NaN : pourcentage, pourcentage par fenetre glissante, etc.
"""

import datetime
import pandas as pds
import numpy as npy
from matplotlib import pyplot as plt
params = {
          'font.size' : 20,
          }
plt.rcParams.update(params)


__all__ = [ "rolling_mean_ts",
            "rolling_mean_df",
            "find_contiguous_nan",
            "pc_nan",
            "nan_by_window",
            "find_ffill_nan",
            "nb_valid_windows", ]

##########################################
# deux fonctions utilitaires pour la suite
##########################################
def rolling_mean_ts( ts , duration ):
    """
    Calcule la moyenne sur une fenêtre roulante d'une série temporelle
    
    Entrées :
    -------
    ts : pandas.Series
    duration : Int
    
    Output :
    ------
    pds.Series contenant ts moyenné avec une fenêtre roulante de taille duration.
    """
    output = pds.Series( index = ts.index[duration-1:], name = ts.name )
    convolution_filter = npy.ones(duration)
    convolution_filter = 1/len(convolution_filter) * convolution_filter
    ctmp = npy.convolve( ts , convolution_filter, mode="valid" )
    output.loc[:] = ctmp
    return output

def rolling_mean_df( df , duration ):
    """
    Calcule la moyenne sur une fenêtre roulante d'une pandas.DataFrame suivant l'index
    
    Entrées :
    -------
    df : pandas.DataFrame
    duration : Int
    
    Output :
    ------
    pandas.DataFrame contenant df moyenné sur une fenêtre de taille durée
    """
    output = pds.DataFrame( index = df.index, columns = df.columns, dtype=float )
    convolued = df.apply( npy.convolve, axis=0, args=(npy.ones(duration)/duration,) , mode="valid")
    output.loc[ output.index[duration-1:] ] = convolued.values
    return output
###############################################
# fin deux fonctions utilitaires pour la suite
###############################################


######################
# NAN
######################
# les premières fonctions ne font pas référence à une quelconque fenêtre.
def find_contiguous_nan(df):
    """
    Calcul le nombre de groupes de NaN contigues dans une DataFrame, en partant,
    serie par serie, de la premiere valeur non NaN a la derniere non NaN. 
    Par exemple la dataframe suivante
                 c1    c2
    date                  
    1980-12-12  0.81   NaN
    1980-12-15  1.77   NaN
    1980-12-16 -0.50   NaN
    1980-12-17   NaN  0.01
    1980-12-18   NaN -1.06
    1980-12-19 -1.17  0.62
    1980-12-22   NaN  0.55
    1980-12-23 -0.93 -0.54
    1980-12-24  1.58   NaN
    1980-12-26 -0.13  1.11

    donnera en output :
                 c1   c2
    date                
    1980-12-12  NaN  NaN
    1980-12-15  NaN  NaN
    1980-12-16  NaN  NaN
    1980-12-17  2.0  NaN
    1980-12-18  NaN  NaN
    1980-12-19  NaN  NaN
    1980-12-22  1.0  NaN
    1980-12-23  NaN  NaN
    1980-12-24  NaN  1.0
    1980-12-26  NaN  NaN

    On a deux NaN au milieu et un a la fin pour la premiere serie, les trois premiers pour la deuxieme serie
    ne sont pas comptes, car on par du premier non nan au dernier non nan.
    
    Entree :
    --------
    df : pandas DataFrame
        La dataframe dont on cherche a determiner les zones de NaN.

    Output : pandas DataFrame
        DataFrame au meme format que df. Si a l'indice I et la colonne C dans df commence une zone de n NaN,
        alors Output vaut n en (I,C). Si au couple (I,C) ne commence pas une zone de NaN, output vaut NaN, 
        ce qui permet d'utiliser la fonction dropna() si necessaire.

    """
    output = pds.DataFrame( index=df.index, data=npy.nan, columns=df.columns, dtype=float)
    for tick in df.columns:
        dftmp = df[tick]
        dftmp = dftmp.loc[ dftmp.first_valid_index():dftmp.last_valid_index() ]
        mask = dftmp.isna()
        dd = dftmp.index.to_series()[mask].groupby((~mask).cumsum()[mask]).agg(['first', 'size'])
        dd = dd.rename(columns=dict(size=dftmp.name, first='date')).reset_index(drop=True)
        dd = dd.set_index("date")
        output.loc[ dd.index, tick ] = dd.values.flatten()
    return output


def pc_nan(df):
    """
    renvoie le pourcentage de NaN pour chaque ticker. On peut travailler sur un tableau de prix, de hurst, etc.

    Entrees :
    -------
    df : pandas DataFrame, temps en index, tickers en colonnes.
    
    Outputs :
    -------
    pandas Series, indexe par les tickers. 
    En valeur, le pourcentage de NaN pour le ticker considere, entre la premiere et la derniere valeur non-NaN
    
    """
    PCN = []
    for tick in df.columns:
        stmp = df[tick]
        stmp = stmp[ stmp.first_valid_index():stmp.last_valid_index() ]
        PCN.append(stmp.isna().sum() / len(stmp) * 100)
    return pds.Series( index = df.columns , data=PCN , name = "pcn" )


# les fonctions suivantes étudient les NaN par rapport à une fenêtre roulante
def nan_by_window(df,longueur_fenetre):
    """
    Calcule le nombre de NaN dans la pandas.DataFrame df par fenetre 
    pour une fenetre glissante de taille longueur_fenetre. 
    """
    output = rolling_mean_df( df.isna(), longueur_fenetre) * longueur_fenetre # on compte les nan
    for tick in df.columns: # ticker par ticker on va supprimer le comptage hors période de cotation
        ts = df[tick]
        ts = ts[ts.first_valid_index():ts.last_valid_index()]
        try:
            i0 = ts.index[longueur_fenetre-1]
            i1 = ts.index[-1]
            output.loc[ output.index[0]:i0 , tick ] = npy.nan
            if i1<(output.index[-1]):
                output.loc[ i1:output.index[-1] , tick ] = npy.nan
        except IndexError: # on n'a pas une taille de fenêtre sur ts
            output[ tick ] = npy.nan
    return output


def nb_valid_windows(df,longueur_fenetre,modif_output=None):
    """
    Calcule les fenêtres valides dans la pandas.DataFrame df par fenetre 
    pour une fenetre glissante de taille longueur_fenetre. Le paramètre modif_output
    permet d'exprimer le résultat en pourcentage si on le souhaite
    """
    npf = nan_by_window(df,longueur_fenetre)
    if modif_output == "pc": # on retourne le pourcentage de fenêtres valides
        return (npf==0).sum()/((npf>=0).sum()) * 100
    elif modif_output is None: # sinon on retourne la position des fenêtres valides
        return npf==0
    else:
        raise ValueError(" modif_output inconnu dans nb_valid_windows ")
        
######################
# FIN NAN
######################
def find_ffill_nan(df):
    diff = df - df.shift(1)
    mask = (diff==0)
    return mask
