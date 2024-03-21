import numpy as np

from time_gan_tensorflow.model import TimeGAN
from time_gan_tensorflow.plots import plot
from encoding import load_concatenate_json_files
from encoding import MinMaxScaler
from encoding import simplify_df
from encoding import encoding_data

# Generate the data
# Utiliser la fonction pour charger et concaténer les fichiers JSON
folder_path = 'Données_bruts_projet_tut'
df = load_concatenate_json_files(folder_path)

# Appel de la fonction pour simplifier les données des actors
df = simplify_df(df)

real_data = encoding_data(df)

# Déterminer la taille des ensembles d'entraînement et de test
train_size = int(0.8 * len(real_data))  # 80% pour l'entraînement
test_size = len(real_data) - train_size  # 20% pour les tests

# Diviser les données en ensembles d'entraînement et de test
#np.random.shuffle(data_matrix)  # Mélanger les données pour garantir l'ordre aléatoire
x_train = real_data[:train_size]  # 80% pour l'entraînement
x_test = real_data[train_size:]   # 20% pour les tests

# Fit the model to the training data
model = TimeGAN(
    x=x_train,
    timesteps=20,
    hidden_dim=64,
    num_layers=3,
    lambda_param=0.1,
    eta_param=10,
    learning_rate=0.001,
    batch_size=16
)

model.fit(
    epochs=500,
    verbose=True
)

# Reconstruct the test data
x_hat = model.reconstruct(x=x_test)
print('===========================================')
print(x_hat)
# Generate the synthetic data
x_sim = model.simulate(samples=len(x_test))
print('------------------------------------------')
print(x_sim)
# Plot the actual, reconstructed and synthetic data
#fig = plot(actual=x_test, reconstructed=x_hat, synthetic=x_sim)
#fig.write_image('results.png', scale=4, height=900, width=700)