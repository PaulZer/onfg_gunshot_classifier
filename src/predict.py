#!/usr/bin/env python
# coding: utf-8

"""
predict.py — Version adaptée pour OpenSoundscape 0.7.1
========================================================================
- Colonnes temporelles lisibles au format HH:MM:SS (relatives au fichier).
- Colonnes start_datetime / end_datetime (format Y-m-d H:M:S), calculées en
  priorité à partir de l'horodatage contenu dans le nom du fichier audio
  (repli sur la date de dernière modification si non reconnu).
- Résultats écrits SÉPARÉMENT PAR ENREGISTREUR (un sous-dossier par
  identifiant d'enregistreur, extrait du nom de fichier), pour éviter un
  fichier unique trop volumineux sur plusieurs semaines de déploiement
  avec de nombreux enregistreurs.
"""

import os
import re
import sys
import time
import warnings
import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta
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

# Format de sortie pour les colonnes datetime
DATETIME_FMT   = "%Y-%m-%d %H:%M:%S"
# ════════════════════════════════════════════════════════════


def fmt_time(seconds):
    """Convertit des secondes en HH:MM:SS (Format 24h)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_recorder_id(filepath):
    """
    Extrait l'identifiant de l'enregistreur à partir du nom de fichier, afin
    de produire des fichiers de résultats séparés par enregistreur (et non un
    seul fichier combiné, qui deviendrait trop volumineux sur plusieurs
    semaines d'enregistrement avec de nombreux appareils).

    Convention supposée : <ID_ENREGISTREUR>_<YYYYMMDD>_<HHMMSS>.wav
    ex: '2MA01481_20250423_133241.wav' -> '2MA01481'

    ⚠ Adaptez cette fonction si vos enregistreurs utilisent une autre
    convention de nommage. Doit rester cohérente avec la même fonction dans
    deduplicate_detections.py et cross_recorder_simultaneous_events.py.
    """
    name = os.path.basename(filepath)
    return name.split('_')[0]


def parse_datetime_from_filename(filepath):
    """
    Tente d'extraire la date/heure de DÉBUT d'enregistrement à partir du nom
    du fichier (convention standard des enregistreurs type AudioMoth /
    Wildlife Acoustics Song Meter : l'horodatage du nom de fichier correspond
    au DÉBUT de l'enregistrement, pas à la fin).

    Retourne un objet datetime si un format connu est reconnu quelque part
    dans le nom de fichier, sinon None.

    Formats reconnus :
      - YYYYMMDD_HHMMSS         ex: 20230615_143000.WAV  (AudioMoth par défaut)
      - YYYY-MM-DD_HH-MM-SS     ex: 2023-06-15_14-30-00.wav
      - YYYYMMDDHHMMSS          ex: SITE1_20230615143000.wav (14 chiffres collés)

    ⚠ Si vos enregistreurs utilisent une autre convention de nommage,
    ajoutez/adaptez un pattern ci-dessous.
    """
    name = os.path.basename(filepath)

    # Format 1 : YYYYMMDD_HHMMSS
    m = re.search(r'(\d{8})_(\d{6})', name)
    if m:
        date_part, time_part = m.groups()
        try:
            return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # Format 2 : YYYY-MM-DD_HH-MM-SS
    m = re.search(r'(\d{4}-\d{2}-\d{2})[_ ](\d{2}-\d{2}-\d{2})', name)
    if m:
        date_part, time_part = m.groups()
        try:
            return datetime.strptime(f"{date_part} {time_part.replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # Format 3 : YYYYMMDDHHMMSS (14 chiffres consécutifs, isolés)
    m = re.search(r'(?<!\d)(\d{14})(?!\d)', name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass

    return None


def get_recording_start_datetime(filepath, total_duration_sec):
    """
    Détermine la date/heure de DÉBUT d'enregistrement, avec deux méthodes :

      1) PRIORITAIRE : extraction depuis le nom du fichier (méthode fiable,
         car indépendante des manipulations de fichiers — copie, transfert,
         sauvegarde — qui peuvent altérer le mtime).

      2) REPLI : si le nom ne correspond à aucun format connu, on utilise la
         date de dernière modification du fichier (mtime), en supposant
         qu'elle correspond à la FIN de l'enregistrement, puis on soustrait
         la durée totale de l'audio pour obtenir le début.

    Retourne un tuple (datetime_debut, source) où source vaut "filename" ou
    "mtime_fallback", pour que l'appelant puisse signaler les cas de repli.
    """
    start_dt = parse_datetime_from_filename(filepath)
    if start_dt is not None:
        return start_dt, "filename"

    mtime = os.path.getmtime(filepath)
    recording_end_dt = datetime.fromtimestamp(mtime)
    recording_start_dt = recording_end_dt - timedelta(seconds=total_duration_sec)
    return recording_start_dt, "mtime_fallback"


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
    total_duration_sec = total_samples / sr
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

    return clips, total_duration_sec


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

    # Accumulation des résultats PAR ENREGISTREUR (et non dans une seule
    # liste globale), pour produire un jeu de fichiers de résultats distinct
    # par enregistreur -> fichiers plus légers et faciles à gérer sur des
    # semaines de déploiement.
    results_by_recorder = defaultdict(lambda: {'scores': [], 'preds': []})

    model.network.eval()
    t_start = time.time()

    for idx, filepath in enumerate(file_list):
        print(f"[{idx+1}/{len(file_list)}] Analyse de {os.path.basename(filepath)}...")
        try:
            recorder_id = extract_recorder_id(filepath)

            clips, total_duration_sec = audio_to_clips(filepath, CLIP_DURATION, overlap=CLIP_OVERLAP, sr=TARGET_SR)
            if not clips:
                continue

            # Date/heure de début d'enregistrement : priorité au nom du
            # fichier, repli sur le mtime si le nom n'est pas reconnu
            recording_start_dt, dt_source = get_recording_start_datetime(filepath, total_duration_sec)
            if dt_source == "mtime_fallback":
                print(f"  ⚠ Date/heure non reconnue dans le nom du fichier "
                      f"'{os.path.basename(filepath)}' -> repli sur la date de "
                      f"modification (mtime). Vérifiez la convention de nommage "
                      f"ou la fiabilité de cette date.")

            dataset = ClipDataset(clips, TARGET_SR, model.preprocessor)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

            with torch.no_grad():
                for batch in loader:
                    outputs = model.network(batch['X'])
                    probs = F.softmax(outputs, dim=1).cpu().numpy()

                    for i in range(len(batch['X'])):
                        start_s = float(batch['start_sec'][i])
                        end_s = float(batch['end_sec'][i])
                        
                        # Génération des formats lisibles HH:MM:SS (relatif au fichier)
                        start_str = fmt_time(start_s)
                        end_str = fmt_time(end_s)

                        # Génération des dates/heures absolues (Y-m-d H:M:S)
                        start_datetime = recording_start_dt + timedelta(seconds=start_s)
                        end_datetime = recording_start_dt + timedelta(seconds=end_s)
                        start_datetime_str = start_datetime.strftime(DATETIME_FMT)
                        end_datetime_str = end_datetime.strftime(DATETIME_FMT)
                        
                        row_index = f"{filepath}_{start_s:.2f}-{end_s:.2f}"

                        # 1. Structure pour les scores continus
                        results_by_recorder[recorder_id]['scores'].append({
                            'index': row_index,
                            'start_time': start_str,
                            'end_time': end_str,
                            'start_datetime': start_datetime_str,
                            'end_datetime': end_datetime_str,
                            'negative': round(float(probs[i, 0]), 5),
                            'positive': round(float(probs[i, 1]), 5)
                        })

                        # 2. Structure pour les décisions binaires
                        is_gunshot = 1 if probs[i, 1] >= THRESHOLD else 0
                        results_by_recorder[recorder_id]['preds'].append({
                            'index': row_index,
                            'start_time': start_str,
                            'end_time': end_str,
                            'start_datetime': start_datetime_str,
                            'end_datetime': end_datetime_str,
                            'negative': 1 - is_gunshot,
                            'positive': is_gunshot
                        })

        except Exception as e:
            print(f"  ✗ Erreur sur le fichier {os.path.basename(filepath)} : {e}")
            traceback.print_exc()

    if not results_by_recorder:
        print("✗ Aucun résultat généré.")
        return

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"✨ Analyse terminée en {fmt_time(total_time)} ! Écriture des résultats par enregistreur :")

    n_recorders = 0
    n_windows_total = 0

    for recorder_id in sorted(results_by_recorder.keys()):
        data = results_by_recorder[recorder_id]
        if not data['scores']:
            continue

        df_scores = pd.DataFrame(data['scores']).set_index('index')
        df_preds = pd.DataFrame(data['preds']).set_index('index')

        # Tri par probabilité de coup de feu décroissante (comme le script d'origine)
        df_scores.sort_values(by=['positive'], ascending=False, inplace=True)
        df_preds = df_preds.reindex(df_scores.index)

        desired_order = ['start_datetime', 'end_datetime', 'start_time', 'end_time', 'negative', 'positive']
        df_scores = df_scores[desired_order]
        df_preds = df_preds[desired_order]

        recorder_dir = os.path.join(OUTPUT_DIR, recorder_id)
        os.makedirs(recorder_dir, exist_ok=True)

        out_scores = os.path.join(recorder_dir, "GUNSHOT_scores.csv")
        out_preds = os.path.join(recorder_dir, "GUNSHOT_binary_predictions.csv")

        df_scores.to_csv(out_scores)
        df_preds.to_csv(out_preds)

        n_recorders += 1
        n_windows_total += len(df_scores)
        print(f" -> [{recorder_id}] {len(df_scores)} fenêtre(s) analysée(s) -> {recorder_dir}/")

    print("=" * 60)
    print(f"✨ {n_recorders} enregistreur(s), {n_windows_total} fenêtre(s) au total, sauvegardés dans : {OUTPUT_DIR}/<enregistreur>/")
    print(f"   Seuil de décision binaire utilisé : {THRESHOLD}")
    print("=" * 60)


if __name__ == "__main__":
    main()