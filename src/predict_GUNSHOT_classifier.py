#!/usr/bin/env python
# coding: utf-8

"""
predict_GUNSHOT_classifier.py — Version adaptée pour OpenSoundscape 0.7.1
========================================================================
Ajout des colonnes temporelles lisibles au format HH:MM:SS en positions 2 et 3.
"""

import os
import sys
import time
import warnings
import traceback
from datetime import date
from glob import glob
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ════════════════════════════════════════════════════════════
#  CONFIGURATION DES CHEMINS & PARAMÈTRES
# ════════════════════════════════════════════════════════════
AUDIO_DIR      = "/data/audio"
MODEL_LOCATION = "/data/lydiakatsis/tropical_forest_gunshot_classifier/Model and code/"
MODEL_PATH     = os.path.join(MODEL_LOCATION, "best.model")

d = date.today().strftime('%y%m%d')
OUTPUT_DIR     = f"/data/results/Outputs_predictions_{d}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIP_DURATION  = 4.0      
CLIP_OVERLAP   = 0.5      
THRESHOLD      = 0.808802 
BATCH_SIZE     = 64

TARGET_SR      = 8000     
MAX_F          = 2000     
WINDOW_SAMPLES = 256
OVERLAP_SAMPLES= 128
# ════════════════════════════════════════════════════════════


def fmt_time(seconds):
    """Convertit des secondes en HH:MM:SS (Format 24h)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_target_model(model_path, sample_duration=4.0):
    from opensoundscape.torch.models.cnn import load_outdated_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_outdated_model(model_path, 'resnet18', sample_duration)
    
    model.preprocessor.pipeline.to_img.set(invert=True)
    model.preprocessor.pipeline.bandpass.set(out_of_bounds_ok=True)
    return model


def audio_to_clips(filepath, clip_duration, overlap=0.0, sr=TARGET_SR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(filepath, sr=sr, mono=True)

    n_samples_clip = int(clip_duration * sr)
    hop_samples    = int(n_samples_clip * (1.0 - overlap))
    total_samples  = len(y)
    clips = []

    start_sample = 0
    while start_sample < total_samples:
        end_sample = start_sample + n_samples_clip
        chunk = y[start_sample:end_sample]

        if len(chunk) < n_samples_clip:
            chunk = np.pad(chunk, (0, n_samples_clip - len(chunk)))

        start_sec = start_sample / sr
        end_sec   = min(end_sample, total_samples) / sr
        clips.append((start_sec, end_sec, chunk))
        start_sample += hop_samples

    return clips


def clip_to_tensor_adapted(audio_array, sr, preprocessor):
    from opensoundscape.audio import Audio
    from opensoundscape.spectrogram import Spectrogram

    audio = Audio(audio_array, sample_rate=sr)
    spec = Spectrogram.from_audio(
        audio, 
        window_samples=WINDOW_SAMPLES, 
        overlap_samples=OVERLAP_SAMPLES
    )
    spec = spec.bandpass(min_f=0, max_f=MAX_F, out_of_bounds_ok=True)

    pipeline = preprocessor.pipeline
    img_params = dict(pipeline['to_img'].params)
    tensor = spec.to_image(**img_params)

    if 'rescale' in pipeline:
        from opensoundscape.preprocess.actions import scale_tensor
        tensor = scale_tensor(tensor, **dict(pipeline['rescale'].params))

    return tensor


class ClipDataset(Dataset):
    def __init__(self, clips, sr, preprocessor):
        self.clips = clips
        self.sr = sr
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        start, end, audio_array = self.clips[idx]
        tensor = clip_to_tensor_adapted(audio_array, self.sr, self.preprocessor)
        return {'X': tensor, 'start_sec': start, 'end_sec': end}


def main():
    file_list = sorted(set(
        f for pat in ["*.wav", "*.WAV"]
        for f in glob(os.path.join(AUDIO_DIR, pat))
    ))
    
    if not file_list:
        print(f"✗ Aucun fichier .WAV trouvé dans : {AUDIO_DIR}")
        sys.exit(1)
    print(f"🚀 Initialisation de l'analyse : {len(file_list)} fichier(s) détecté(s).")

    if not os.path.isfile(MODEL_PATH):
        print(f"✗ Fichier de modèle manquant : {MODEL_PATH}")
        sys.exit(1)
        
    print("Chargement de l'architecture ResNet18 Binary...", end=" ", flush=True)
    model = load_target_model(MODEL_PATH, CLIP_DURATION)
    print("✓ Prêt.\n")

    full_scores_list = []
    full_preds_list = []

    model.network.eval()
    t_start = time.time()

    for idx, filepath in enumerate(file_list):
        print(f"[{idx+1}/{len(file_list)}] Analyse de {os.path.basename(filepath)}...")
        try:
            clips = audio_to_clips(filepath, CLIP_DURATION, overlap=CLIP_OVERLAP, sr=TARGET_SR)
            if not clips:
                continue

            dataset = ClipDataset(clips, TARGET_SR, model.preprocessor)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

            with torch.no_grad():
                for batch in loader:
                    outputs = model.network(batch['X'])
                    probs = F.softmax(outputs, dim=1).cpu().numpy()

                    for i in range(len(batch['X'])):
                        start_s = float(batch['start_sec'][i])
                        end_s = float(batch['end_sec'][i])
                        
                        # Génération des formats lisibles HH:MM:SS
                        start_str = fmt_time(start_s)
                        end_str = fmt_time(end_s)
                        
                        row_index = f"{filepath}_{start_s:.2f}-{end_s:.2f}"

                        # 1. Structure pour les scores continus
                        full_scores_list.append({
                            'index': row_index,
                            'start_time': start_str,
                            'end_time': end_str,
                            'negative': round(float(probs[i, 0]), 5),
                            'positive': round(float(probs[i, 1]), 5)
                        })

                        # 2. Structure pour les décisions binaires
                        is_gunshot = 1 if probs[i, 1] >= THRESHOLD else 0
                        full_preds_list.append({
                            'index': row_index,
                            'start_time': start_str,
                            'end_time': end_str,
                            'negative': 1 - is_gunshot,
                            'positive': is_gunshot
                        })

        except Exception as e:
            print(f"  ✗ Erreur sur le fichier {os.path.basename(filepath)} : {e}")
            traceback.print_exc()

    if not full_scores_list:
        print("✗ Aucun résultat généré.")
        return

    # 4. Conversion en DataFrames
    df_scores = pd.DataFrame(full_scores_list).set_index('index')
    df_preds = pd.DataFrame(full_preds_list).set_index('index')

    # Tri global par probabilité de coup de feu décroissante (comme le script d'origine)
    df_scores.sort_values(by=['positive'], ascending=False, inplace=True)
    df_preds = df_preds.reindex(df_scores.index)

    # --- RÉORGANISATION DES COLONNES (Index, start_time, end_time, negative, positive) ---
    desired_order = ['start_time', 'end_time', 'negative', 'positive']
    df_scores = df_scores[desired_order]
    df_preds = df_preds[desired_order]

    # 5. Écritures sur le disque
    out_scores = os.path.join(OUTPUT_DIR, "GUNSHOT_scores.csv")
    out_preds = os.path.join(OUTPUT_DIR, "GUNSHOT_binary_predictions.csv")

    df_scores.to_csv(out_scores)
    df_preds.to_csv(out_preds)

    total_time = time.time() - t_start
    print("\n" + "="*50)
    print(f"✨ Analyse globale terminée avec succès en {fmt_time(total_time)} !")
    print(f" -> Scores continus sauvegardés dans : {out_scores}")
    print(f" -> Décisions binaires (Seuil={THRESHOLD}) dans : {out_preds}")
    print("="*50)


if __name__ == "__main__":
    main()