"""
Script de diagnostic : teste le préprocesseur sur un seul fichier
et affiche l'erreur exacte.
"""
import warnings
import traceback
import torch

MODEL_PATH = ("/data/lydiakatsis/tropical_forest_gunshot_classifier/"
              "Model and code/best.model")
AUDIO_FILE = "/data/audio/gunshot_1.WAV"

print("=== DIAGNOSTIC PRÉPROCESSEUR ===\n")

# 1. Charger le modèle
from opensoundscape.torch.models.cnn import load_outdated_model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = load_outdated_model(MODEL_PATH, 'resnet18', 4.0)
model.preprocessor.pipeline.to_img.set(invert=True)
print("Modèle chargé.\n")

# 2. Afficher le pipeline complet
print("=== PIPELINE DU PRÉPROCESSEUR ===")
for name, action in model.preprocessor.pipeline.items():
    print(f"  {name}: {action}")
print()

# 3. Tenter de traiter le fichier manuellement étape par étape
print("=== TEST MANUEL DU PIPELINE ===")
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
import numpy as np

print(f"Fichier : {AUDIO_FILE}")
try:
    audio = Audio.from_file(AUDIO_FILE)
    print(f"  Audio chargé : durée={audio.duration:.2f}s, sr={audio.sample_rate}Hz")
except Exception as e:
    print(f"  ✗ Erreur chargement audio : {e}")
    traceback.print_exc()

# 4. Tester avec raise_errors=True pour avoir l'erreur exacte
print("\n=== PRÉDICTION AVEC raise_errors=True ===")
try:
    from opensoundscape.torch.datasets import AudioSplittingDataset
    dataset = AudioSplittingDataset([AUDIO_FILE], model.preprocessor)
    dataset.bypass_augmentations = True
    dataset.raise_errors = True
    print(f"Dataset créé : {len(dataset)} échantillon(s)")
    
    # Tenter d'accéder au premier élément
    print("Accès au sample[0]...")
    sample = dataset[0]
    print(f"  ✓ Sample OK, shape: {sample['X'].shape}")
except Exception as e:
    print(f"  ✗ Erreur : {e}")
    traceback.print_exc()

# 5. Inspecter les paramètres du préprocesseur
print("\n=== PARAMÈTRES CLÉS DU PRÉPROCESSEUR ===")
pp = model.preprocessor
print(f"  sample_duration : {pp.sample_duration}")
try:
    print(f"  to_img params   : {pp.pipeline['to_img'].params}")
except: pass
try:
    print(f"  bandpass params : {pp.pipeline['bandpass'].params}")
except: pass
try:
    print(f"  to_spec params  : {pp.pipeline['to_spec'].params}")
except: pass