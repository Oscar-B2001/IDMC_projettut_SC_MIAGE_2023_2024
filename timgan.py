import os
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

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

# Partie encocodage de données 

# Apply integer encoding to 'Verb' and 'Object' columns
label_encoder_actor = LabelEncoder()
label_encoder_verb = LabelEncoder()
label_encoder_object = LabelEncoder()
df['Actor'] = label_encoder_actor.fit_transform(df['Actor'])
df['Verb'] = label_encoder_verb.fit_transform(df['Verb'])
df['Object'] = label_encoder_object.fit_transform(df['Object'])

# Convert the 'Timestamp' column to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Convert datetime timestamps to seconds since the epoch using .timestamp()
df['Timestamp'] = df['Timestamp'].apply(lambda x: x.timestamp())

# Find the earliest timestamp in seconds since the epoch
min_timestamp_seconds = df['Timestamp'].min()

# Calculate the difference in seconds from the first event
df['Timestamp'] = df['Timestamp'] - min_timestamp_seconds

# Sort the DataFrame by the 'Timestamp' column
df_sorted = df.sort_values(by='Timestamp')

# Reset the index of the sorted DataFrame to get a new index that represents the chronological order
df_sorted = df_sorted.reset_index(drop=True)

# Add the 'Chronological_Order' column at the first position (index 0)
df_sorted.insert(0, 'Chronological_Order', df_sorted.index + 1)  # Adding 1 so that the order starts from 1 instead of 0

df = df_sorted

# The dataframe is now preprocessed and ready for synthetic data generation
df_preprocessed_shape = df.shape
df_preprocessed_head = df.head()

df_preprocessed_shape, df_preprocessed_head

df
