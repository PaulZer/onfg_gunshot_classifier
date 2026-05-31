FROM python:3.8-slim

# Dépendances système nécessaires (ajout de build-essential si besoin de compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Mise à jour de pip et installation des outils de build
RUN pip install --no-cache-dir -U pip setuptools wheel

# 2. Pré-installation des grosses dépendances pour éviter le backtracking de pip
# On fige des versions majeures compatibles avec l'écosystème opensoundscape 0.7.x
RUN pip install --no-cache-dir \
    "torch>=1.9.0,<1.13.0" \
    "torchvision>=0.10.0,<0.14.0" \
    "scikit-learn<1.1.0" \
    "librosa<0.10.0" \
    "pandas<2.0.0"

# 3. Installation d'opensoundscape (qui trouvera ses dépendances déjà satisfaites)
RUN pip install --no-cache-dir opensoundscape==0.7.1

# Récupération du dépôt GitHub et du script
RUN git clone https://github.com/lydiakatsis/tropical_forest_gunshot_classifier.git /data/lydiakatsis/tropical_forest_gunshot_classifier
COPY src/predict.py /app/predict.py

RUN mkdir -p /data/audio /data/results