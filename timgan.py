import os
import pandas as pd
import json

def load_concatenate_json_files(folder_path):
    # Initialiser une liste vide pour stocker les DataFrames
    dfs = []

    # Lister tous les fichiers JSON dans le dossier
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Charger les données JSON
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Transformer en DataFrame
            df = pd.json_normalize(data)

            # Sélectionner les colonnes requises et renommer
            if 'timestamp' in df.columns and 'actor.mbox' in df.columns and \
               'verb.display.en' in df.columns and 'object.id' in df.columns:
                df = df[['timestamp', 'actor.mbox', 'verb.display.en', 'object.id']]
                df.columns = ['Timestamp', 'Actor', 'Verb', 'Object']
                dfs.append(df)

    # Concaténer tous les DataFrames
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()  # Retourner un DataFrame vide si aucun fichier JSON n'est trouvé ou transformé

# Fonction pour simplifier les données des actors
def simplify_df(df):
    # Le [0] sélectionne la première partie de la chaîne, c'est-à-dire tout ce qui précède '@'
    df['Actor'] = df['Actor'].apply(lambda x: x.split('@')[0])

    # Supprimer 'mailto:' avant l'adresse e-mail si présent et garder uniquement la partie avant '@'
    df['Actor'] = df['Actor'].apply(lambda x: x.replace('mailto:', '').split('@')[0])
    
    # Supprimer 'http://moodle-example.com/' avant l'objet
    df['Object'] = df['Object'].apply(lambda x: x.replace('http://moodle-example.com/', ''))
    
    return df

# Utiliser la fonction pour charger et concaténer les fichiers JSON
folder_path = 'Données_bruts_projet_tut'
df = load_concatenate_json_files(folder_path)

# Appel de la fonction pour simplifier les données des actors
df = simplify_df(df)

df