#!/usr/bin/env python
# coding: utf-8

"""
deduplicate_detections.py — Regroupement des détections dupliquées
========================================================================
predict.py découpe chaque fichier audio en fenêtres de CLIP_DURATION
secondes qui se chevauchent (CLIP_OVERLAP), avec un décalage entre deux
fenêtres consécutives (hop) de CLIP_DURATION * (1 - CLIP_OVERLAP).
Avec les valeurs par défaut (4s, 50%), ce décalage vaut 2 secondes.

Un même coup de feu physique peut donc déclencher plusieurs détections
consécutives à haut score (une par fenêtre qui le contient).

predict.py produit désormais un GUNSHOT_scores.csv PAR ENREGISTREUR, dans
un sous-dossier <OUTPUT_DIR>/<recorder_id>/. Ce script :
  1. Trouve tous les GUNSHOT_scores.csv sous un dossier de résultats
     (un par enregistreur), ou traite un fichier unique si demandé.
  2. Pour chaque enregistreur : filtre les détections à haut score
     (SCORE_THRESHOLD), regroupe celles dont les fenêtres se chevauchent ou
     sont séparées de moins de MAX_GAP_SECONDS, et ne conserve qu'une
     occurrence par groupe (celle au score le plus élevé).
  3. Écrit un GUNSHOT_scores_deduplicated.csv dans le sous-dossier de
     chaque enregistreur.

Usage :
    # Mode automatique : traite tous les enregistreurs du dossier de
    # résultats le plus récent
    python deduplicate_detections.py

    # Mode dossier explicite : traite tous les enregistreurs sous ce dossier
    python deduplicate_detections.py /data/results/Outputs_predictions_XXXXXX

    # Mode fichier unique (rétrocompatibilité / usage manuel ponctuel)
    python deduplicate_detections.py /data/results/Outputs_predictions_XXXXXX/2MA01481/GUNSHOT_scores.csv
"""

import os
import re
import sys
from glob import glob
import pandas as pd

# ════════════════════════════════════════════════════════════
#  PARAMÈTRES (à ajuster selon vos besoins)
# ════════════════════════════════════════════════════════════
RESULTS_ROOT    = "/data/results"

# Seuil de score à partir duquel une détection est considérée comme
# "à haute confiance" pour l'analyse de doublons. Indépendant du seuil de
# décision binaire (THRESHOLD=0.808802) utilisé dans predict.py.
SCORE_THRESHOLD = 0.5

# Écart maximal toléré (en secondes) entre la fin d'une fenêtre et le début
# de la suivante pour les considérer comme un seul et même événement.
# Valeur par défaut = décalage (hop) entre deux fenêtres d'analyse
# consécutives dans predict.py : CLIP_DURATION * (1 - CLIP_OVERLAP)
# = 4.0 * (1 - 0.5) = 2.0 secondes.
MAX_GAP_SECONDS = 2.0
# ════════════════════════════════════════════════════════════

# Reconnaît le format d'index généré par predict.py :
# "<filepath>_<start_sec>-<end_sec>" avec start/end au format "%.2f"
INDEX_PATTERN = re.compile(r'^(?P<filepath>.+)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)$')


def find_latest_output_dir():
    """Cherche le dossier Outputs_predictions_* le plus récent sous RESULTS_ROOT."""
    candidates = sorted(glob(os.path.join(RESULTS_ROOT, "Outputs_predictions_*")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_recorder_scores_csvs(output_dir):
    """Trouve tous les GUNSHOT_scores.csv sous <output_dir>/<recorder_id>/."""
    return sorted(glob(os.path.join(output_dir, "*", "GUNSHOT_scores.csv")))


def parse_index(index_value):
    """Extrait (filepath, start_sec, end_sec) depuis la colonne 'index'."""
    m = INDEX_PATTERN.match(index_value)
    if not m:
        raise ValueError(f"Format d'index non reconnu : {index_value}")
    return m.group('filepath'), float(m.group('start')), float(m.group('end'))


def group_detections(df, max_gap_seconds):
    """
    Regroupe les détections à haut score par fichier audio, en fusionnant
    celles dont les fenêtres se chevauchent ou sont séparées de moins de
    max_gap_seconds. Retourne un DataFrame avec une ligne par groupe
    (= un événement physique distinct).
    """
    groups = []
    group_counter = 0

    for filepath, sub in df.groupby('filepath', sort=False):
        sub = sub.sort_values('start_sec').reset_index(drop=True)

        def flush_group(rows):
            nonlocal group_counter
            group_counter += 1
            best_row = max(rows, key=lambda r: r['positive'])
            groups.append({
                'group_id': group_counter,
                'file': os.path.basename(filepath),
                'n_windows_in_group': len(rows),
                'group_start_datetime': min(r['start_datetime'] for r in rows),
                'group_end_datetime': max(r['end_datetime'] for r in rows),
                'best_start_datetime': best_row['start_datetime'],
                'best_end_datetime': best_row['end_datetime'],
                'best_start_time': best_row['start_time'],
                'best_end_time': best_row['end_time'],
                'best_score': best_row['positive'],
                'best_index': best_row['index'],
            })

        current_group_rows = [sub.iloc[0]]
        current_end = sub.iloc[0]['end_sec']

        for i in range(1, len(sub)):
            row = sub.iloc[i]
            gap = row['start_sec'] - current_end
            if gap <= max_gap_seconds:
                # Chevauchement ou écart faible -> même événement
                current_group_rows.append(row)
                current_end = max(current_end, row['end_sec'])
            else:
                # Écart trop grand -> nouvel événement
                flush_group(current_group_rows)
                current_group_rows = [row]
                current_end = row['end_sec']

        flush_group(current_group_rows)

    return pd.DataFrame(groups)


def process_one_csv(input_path, quiet_prefix=""):
    """
    Traite un GUNSHOT_scores.csv (un seul enregistreur) : filtre, regroupe,
    écrit GUNSHOT_scores_deduplicated.csv dans le même dossier.
    Retourne (n_filtered, n_groups) ou (None, None) en cas d'échec/absence.
    """
    df = pd.read_csv(input_path)

    required_cols = {'index', 'start_datetime', 'end_datetime', 'start_time', 'end_time', 'positive'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"{quiet_prefix}✗ Colonnes manquantes dans le CSV : {sorted(missing)}")
        print(f"{quiet_prefix}  Ce script nécessite un GUNSHOT_scores.csv généré par la version à jour de predict.py")
        print(f"{quiet_prefix}  (avec les colonnes start_datetime / end_datetime / start_time / end_time).")
        return None, None

    parsed = df['index'].apply(parse_index)
    df['filepath']  = parsed.apply(lambda t: t[0])
    df['start_sec'] = parsed.apply(lambda t: t[1])
    df['end_sec']   = parsed.apply(lambda t: t[2])

    n_total = len(df)
    df_filtered = df[df['positive'] >= SCORE_THRESHOLD].copy()
    print(f"{quiet_prefix}🎯 {len(df_filtered)} détection(s) à haut score (positive ≥ {SCORE_THRESHOLD}) sur {n_total} fenêtres analysées.")

    if df_filtered.empty:
        print(f"{quiet_prefix}✗ Aucune détection à haut score trouvée avec ce seuil.")
        return 0, 0

    df_groups = group_detections(df_filtered, MAX_GAP_SECONDS)
    df_groups.sort_values(by=['file', 'group_start_datetime'], inplace=True)

    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "GUNSHOT_scores_deduplicated.csv")
    df_groups.to_csv(output_path, index=False)

    n_duplicates = len(df_filtered) - len(df_groups)
    print(f"{quiet_prefix}✨ {len(df_filtered)} détections regroupées en {len(df_groups)} événement(s) distinct(s) "
          f"({n_duplicates} doublon(s) éliminé(s)) -> {output_path}")

    return len(df_filtered), len(df_groups)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Mode fichier unique : rétrocompatibilité / usage manuel sur un enregistreur précis
    if arg and os.path.isfile(arg):
        print(f"📄 Mode fichier unique : {arg}")
        process_one_csv(arg)
        return

    # Mode dossier : traite tous les enregistreurs trouvés sous ce dossier
    output_dir = arg if (arg and os.path.isdir(arg)) else find_latest_output_dir()
    if not output_dir or not os.path.isdir(output_dir):
        print("✗ Dossier de résultats introuvable.")
        print("  Précisez un dossier ou un fichier en argument, par exemple :")
        print("  python deduplicate_detections.py /data/results/Outputs_predictions_XXXXXX")
        sys.exit(1)

    recorder_csvs = find_recorder_scores_csvs(output_dir)
    if not recorder_csvs:
        print(f"✗ Aucun GUNSHOT_scores.csv trouvé sous {output_dir}/<enregistreur>/")
        print("  Vérifiez que predict.py a bien été exécuté et a produit des sous-dossiers par enregistreur.")
        sys.exit(1)

    print(f"📂 {len(recorder_csvs)} enregistreur(s) trouvé(s) sous : {output_dir}")

    total_filtered = 0
    total_groups = 0
    n_processed = 0

    for csv_path in recorder_csvs:
        recorder_id = os.path.basename(os.path.dirname(csv_path))
        print(f"\n— Enregistreur {recorder_id} —")
        n_filtered, n_groups = process_one_csv(csv_path, quiet_prefix="  ")
        if n_filtered is not None:
            total_filtered += n_filtered
            total_groups += n_groups
            n_processed += 1

    print("\n" + "=" * 60)
    print(f"✨ Regroupement terminé pour {n_processed}/{len(recorder_csvs)} enregistreur(s).")
    print(f" -> {total_filtered} détections à haut score regroupées en {total_groups} événement(s) distinct(s) au total.")
    print("=" * 60)


if __name__ == "__main__":
    main()