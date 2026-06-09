"""
classify_gunshots.py
====================
Classifieur de coups de feu en forêt tropicale.
Basé sur : https://github.com/lydiakatsis/tropical_forest_gunshot_classifier
Compatible opensoundscape 0.7.1

Approche : découpe manuelle des fichiers audio en clips de CLIP_DURATION
secondes via librosa, puis application directe du préprocesseur opso
sur chaque clip. Garantit le bon timecode même pour des fichiers longs.
"""

import os
import sys
import warnings
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import librosa
from glob import glob
from torch.utils.data import DataLoader, Dataset

# ════════════════════════════════════════════════════════════
#  CONFIG — à adapter
# ════════════════════════════════════════════════════════════
AUDIO_DIR     = "/data/audio"
OUTPUT_DIR    = "/data/results"
MODEL_PATH    = ("/data/lydiakatsis/tropical_forest_gunshot_classifier/"
                 "Model and code/best.model")
CLIP_DURATION = 4.0    # secondes (durée d'entraînement du modèle)
CLIP_OVERLAP  = 0.5    # chevauchement entre clips (0=aucun, 0.5=50%)
THRESHOLD     = 0.6    # seuil de détection gunshot
BATCH_SIZE    = 32
# ════════════════════════════════════════════════════════════

# Fréquence d'échantillonnage cible (celle utilisée à l'entraînement)
TARGET_SR = 22050


def fmt_time(seconds):
    """Convertit des secondes en HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_model(model_path, sample_duration=4.0):
    from opensoundscape.torch.models.cnn import load_outdated_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_outdated_model(model_path, 'resnet18', sample_duration)
    model.preprocessor.pipeline.to_img.set(invert=True)
    model.preprocessor.pipeline.bandpass.set(out_of_bounds_ok=True)
    return model


def audio_to_clips(filepath, clip_duration, overlap=0.0, sr=TARGET_SR):
    """
    Charge un fichier audio et le découpe en clips de clip_duration secondes
    avec un chevauchement optionnel (overlap en fraction, ex: 0.5 = 50%).
    Retourne une liste de (start_sec, end_sec, array_audio).
    Le dernier clip est zéro-paddé si nécessaire.
    """
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

        # Zéro-padding du dernier clip si trop court
        if len(chunk) < n_samples_clip:
            chunk = np.pad(chunk, (0, n_samples_clip - len(chunk)))

        start_sec = start_sample / sr
        end_sec   = min(end_sample, total_samples) / sr
        clips.append((start_sec, end_sec, chunk))
        start_sample += hop_samples

    return clips


def clip_to_tensor(audio_array, sr, preprocessor):
    """
    Applique le pipeline de prétraitement opensoundscape sur un array audio
    numpy et retourne un tenseur (3, 224, 224).
    Reproduit exactement ce que fait AudioSplittingDataset en interne.
    """
    from opensoundscape.audio import Audio
    from opensoundscape.spectrogram import Spectrogram

    # Reconstruire un objet Audio opensoundscape depuis l'array numpy
    audio = Audio(audio_array, sample_rate=sr)

    # Appliquer les étapes du pipeline (hors augmentations)
    pipeline = preprocessor.pipeline
    spec = Spectrogram.from_audio(audio, **dict(pipeline['to_spec'].params))

    # Bandpass
    bp = dict(pipeline['bandpass'].params)
    spec = spec.bandpass(**bp)

    # Vers image tenseur
    img_params = dict(pipeline['to_img'].params)
    tensor = spec.to_image(**img_params)

    # Rescale
    if 'rescale' in pipeline:
        from opensoundscape.preprocess.actions import scale_tensor
        tensor = scale_tensor(tensor, **dict(pipeline['rescale'].params))

    return tensor


class ClipDataset(Dataset):
    """Dataset de clips audio pré-découpés, prêts pour l'inférence."""
    def __init__(self, clips, sr, preprocessor):
        self.clips       = clips   # liste de (start, end, array)
        self.sr          = sr
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        start, end, audio_array = self.clips[idx]
        tensor = clip_to_tensor(audio_array, self.sr, self.preprocessor)
        return {'X': tensor, 'start_time': start, 'end_time': end}


def predict(model, audio_files):
    all_rows = []

    for filepath in audio_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clips = audio_to_clips(filepath, CLIP_DURATION, overlap=CLIP_OVERLAP, sr=TARGET_SR)

            duration_str = fmt_time(clips[-1][1] if clips else 0)
            print(f"  {os.path.basename(filepath)} : "
                  f"{len(clips)} clip(s), durée ~{duration_str}")

            dataset = ClipDataset(clips, TARGET_SR, model.preprocessor)
            loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

            model.network.eval()
            with torch.no_grad():
                for batch in loader:
                    probs = F.softmax(model.network(batch['X']), dim=1).cpu()
                    for i in range(len(batch['X'])):
                        row = {
                            'file':       filepath,
                            'start_time': float(batch['start_time'][i]),
                            'end_time':   float(batch['end_time'][i]),
                        }
                        for j, cls in enumerate(model.classes):
                            row[cls] = round(float(probs[i, j]), 3)
                        all_rows.append(row)

        except Exception as exc:
            print(f"  ✗ {os.path.basename(filepath)} : {exc}")
            traceback.print_exc()

    return pd.DataFrame(all_rows)


def main():
    # Collecte fichiers
    audio_files = sorted(set(
        f for pat in ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.FLAC"]
        for f in glob(os.path.join(AUDIO_DIR, pat))
    ))
    if not audio_files:
        print(f"✗ Aucun fichier audio dans : {AUDIO_DIR}"); sys.exit(1)
    print(f"{len(audio_files)} fichier(s) détecté(s)")

    # Chargement modèle
    if not os.path.isfile(MODEL_PATH):
        print(f"✗ Modèle introuvable : {MODEL_PATH}"); sys.exit(1)
    print("Chargement du modèle...", end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_model(MODEL_PATH, CLIP_DURATION)
    print("✓\n")

    # Prédiction
    print("Analyse en cours...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = predict(model, audio_files)

    if results.empty:
        print("✗ Aucun résultat produit."); sys.exit(1)

    # Renommage colonnes
    rename = {}
    for col in results.columns:
        if str(col).lower() in ('negative', 'background', 'bg'):
            rename[col] = 'background'
        elif str(col).lower() in ('positive', 'gunshot', 'gun'):
            rename[col] = 'gunshot'
    if rename:
        results = results.rename(columns=rename)

    score_col = 'gunshot' if 'gunshot' in results.columns else \
                next(c for c in results.columns
                     if c not in ('file', 'start_time', 'end_time'))

    # Filtrage + tri : par fichier (A→Z) puis start_time croissant
    filtered = (results[results[score_col] > THRESHOLD]
                .sort_values(['file', score_col], ascending=[True, False])
                .reset_index(drop=True))

    # Affichage
    print(f"\nRésultats filtrés (gunshot > {THRESHOLD}) "
          f"— {len(filtered)} segment(s) :\n")
    if filtered.empty:
        print("  Aucun segment détecté.")
        print("\n  Top-5 scores :")
        top5 = results.nlargest(5, score_col).copy()
        top5['start_time'] = top5['start_time'].apply(fmt_time)
        top5['end_time']   = top5['end_time'].apply(fmt_time)
        print(top5[['file', 'start_time', 'end_time', score_col]]
              .to_string(index=False))
    else:
        display = filtered.copy()
        display['start_time'] = display['start_time'].apply(fmt_time)
        display['end_time']   = display['end_time'].apply(fmt_time)
        print(display[['file', 'start_time', 'end_time', 'background', score_col]]
              .to_string(index=False))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, "classified_gunshots.csv")
        display.to_csv(out, index=False)
        print(f"\n✓ CSV sauvegardé : {out}")


if __name__ == "__main__":
    main()