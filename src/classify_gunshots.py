"""
classify_gunshots.py
====================
Classifieur de coups de feu en forêt tropicale.
Basé sur : https://github.com/lydiakatsis/tropical_forest_gunshot_classifier
Compatible opensoundscape 0.7.1
"""

import os
import sys
import warnings
import traceback
import torch
import torch.nn.functional as F
import pandas as pd
from glob import glob
from opensoundscape.torch.datasets import AudioSplittingDataset
from torch.utils.data import DataLoader

# ════════════════════════════════════════════════════════════
#  CONFIG — à adapter
# ════════════════════════════════════════════════════════════
AUDIO_DIR  = "/data/audio"
OUTPUT_DIR = "/data/results"
MODEL_PATH = ("/data/lydiakatsis/tropical_forest_gunshot_classifier/"
              "Model and code/best.model")
CLIP_DURATION = 4.0   # secondes
THRESHOLD     = 0.2   # seuil de détection
# ════════════════════════════════════════════════════════════


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


def predict(model, audio_files):
    all_rows = []
    for filepath in audio_files:
        try:
            dataset = AudioSplittingDataset(
                [filepath], model.preprocessor, final_clip='extend'
            )
            dataset.bypass_augmentations = True
            if len(dataset) == 0:
                print(f"  ⚠ {os.path.basename(filepath)} : aucun clip généré, ignoré")
                continue

            loader = DataLoader(dataset, batch_size=16, num_workers=0)
            file_logits, file_times = [], []
            model.network.eval()
            with torch.no_grad():
                for batch in loader:
                    probs = F.softmax(model.network(batch['X']), dim=1)
                    file_logits.append(probs.cpu())
                    if 'start_time' in batch:
                        for s, e in zip(batch['start_time'], batch['end_time']):
                            file_times.append((float(s), float(e)))
                    else:
                        dur = model.preprocessor.sample_duration
                        for i in range(batch['X'].shape[0]):
                            file_times.append((i * dur, (i + 1) * dur))

            all_probs = torch.cat(file_logits, dim=0).numpy()
            for i, (s, e) in enumerate(file_times):
                row = {'file': filepath, 'start_time': s, 'end_time': e}
                for j, cls in enumerate(model.classes):
                    row[cls] = round(float(all_probs[i, j]), 3)
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
    model = load_model(MODEL_PATH, CLIP_DURATION)
    print("✓")

    # Prédiction
    print("Analyse en cours...")
    warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load metadata")
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
                next(c for c in results.columns if c not in ('file', 'start_time', 'end_time'))

    # Filtrage + tri : par fichier (A→Z) puis score décroissant
    filtered = (results[results[score_col] > THRESHOLD]
                .sort_values(['file', score_col], ascending=[True, False])
                .reset_index(drop=True))

    # Affichage
    print(f"\nRésultats filtrés (gunshot > {THRESHOLD}) — {len(filtered)} segment(s) :\n")
    if filtered.empty:
        print("  Aucun segment détecté.")
        print(f"\n  Top-5 scores :")
        print(results.nlargest(5, score_col)[['file', 'start_time', 'end_time', score_col]]
              .to_string(index=False))
    else:
        display = filtered.copy()
        display['start_time'] = display['start_time'].apply(fmt_time)
        display['end_time']   = display['end_time'].apply(fmt_time)
        print(display[['file', 'start_time', 'end_time', 'background', score_col]]
              .to_string(index=False))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, "classified_gunshots.csv")
        # CSV avec temps formatés également
        filtered.to_csv(out, index=False, float_format="%.2f")
        print(f"\n✓ CSV sauvegardé : {out}")


if __name__ == "__main__":
    main()