# Documentation du projet VAE

# VAE MNIST Reconstruction Project

Ce projet implémente un **Autoencodeur Variationnel (VAE)** pour reconstruire des images du dataset MNIST. Il comprend des scripts pour entraîner le modèle, le convertir en format ONNX, et une API Flask pour exécuter des prédictions.

## Structure du projet

```plaintext
.
├── app/
│   ├── __init__.py                # Initialisation du module Flask
│   ├── api.py                     # Définition des routes Flask
│   ├── static/
│   │   ├── input_images/          # Images d'entrée pour les prédictions
│   │   └── output_images/         # Images reconstruites
│   └── utils/
│       ├── preprocess.py          # Pré-traitement des données
│       ├── postprocess.py         # Post-traitement des données
│       └── __init__.py            # Initialisation des utilitaires
├── data/                          # Dossier pour stocker les datasets
├── environments/                  # Environnements virtuels (non suivis par git)
├── logs/                          # Dossier pour logs d'entraînement
├── notebooks/
│   └── vae_mnist_reconstruction.ipynb # Notebook pour expérimentations
├── scripts/
│   ├── convert_to_onnx.py         # Conversion du modèle en ONNX
│   ├── test_api.py                # Script pour tester l'API
│   └── train_model.py             # Script pour entraîner le modèle
├── tests/
│   └── test_reconstruction.py     # Tests unitaires pour la reconstruction
├── requirements.txt               # Dépendances Python
├── run.py                         # Point d'entrée pour exécuter l'API Flask
└── README.md                      # Documentation du projet
```

## Installation
### Prérequis
- Python 3.9 ou supérieur
- pip installé

## Étapes

1. Clonez ce dépôt :
    ```bash
    git clone https://github.com/AntoineBendafiSchulmann/vae_model
    ```
    ```
     cd vae_model

    ```
2. Créez un environnement virtuel et activez-le :
    ```bash
    python -m venv environments/vae_env
    source environments/vae_env/bin/activate  # Sur Windows : environments\vae_env\Scripts\activate
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

4. Téléchargez le dataset MNIST

## Utilisation
### Entraîner le modèle

Pour entraîner le modèle et sauvegarder les poids :

```bash
python scripts/train_model.py
```

Convertir le modèle en format ONNX
```bash
python scripts/convert_to_onnx.py
```

### Lancer l'API Flask
```bash
python run.py
```
- Endpoint principal : http://127.0.0.1:5004/
- Endpoint pour la reconstruction : http://127.0.0.1:5004/image/reconstruct

### Tester l'API
Envoyez une image via test_api.py :

```bash
python scripts/test_api.py
```

### Tester la reconstruction
Exécutez les tests unitaires :

```bash
python -m unittest tests/test_reconstruction.py
```

### Dossiers et fichiers importants
- app/static/input_images/ : Contient les images d'entrée envoyées à l'API.
- app/static/output_images/ : Contient les images reconstruites générées par l'API.
- logs/ : Stocke les journaux d'entraînement du modèle.
