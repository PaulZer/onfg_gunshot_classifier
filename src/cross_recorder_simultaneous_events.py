#!/usr/bin/env python
# coding: utf-8

"""
cross_recorder_simultaneous_events.py — Détections simultanées multi-enregistreurs
========================================================================
Un même coup de feu peut être capté par PLUSIEURS enregistreurs à la fois
si le réseau est suffisamment dense (~1 km d'écart). Ce script recherche,
parmi tous les fichiers audio analysés (potentiellement issus de dizaines
d'enregistreurs différents), les détections à haut score dont les fenêtres
temporelles ABSOLUES (start_datetime / end_datetime) se chevauchent ou sont
très proches, MAIS qui proviennent d'enregistreurs DIFFÉRENTS.

Comme la distance entre enregistreurs n'est pas connue, on ne calcule pas
un délai de propagation théorique précis : on utilise à la place une
tolérance temporelle réglable (MAX_TIME_DIFF_SECONDS) qui doit couvrir à la
fois :
  - le temps de propagation du son entre les enregistreurs les plus
    éloignés pouvant capter le même événement (à 343 m/s, ~1 km ≈ 2.9 s,
    2 km ≈ 5.8 s, etc.) ;
  - l'imprécision / le désynchronisation des horloges internes des
    enregistreurs (souvent plus impactante que le délai sonore si les
    appareils ne sont pas synchronisés par GPS).

⚠ IMPORTANT : un chevauchement temporel entre deux enregistreurs est un
INDICE, pas une preuve absolue qu'il s'agit du même coup de feu (deux coups
de feu distincts peuvent survenir au même moment dans des secteurs
différents). Une vérification auditive croisée reste recommandée.

Étapes :
  1. Charge un GUNSHOT_scores.csv (toutes les fenêtres, tous enregistreurs).
  2. Filtre les fenêtres à haut score (SCORE_THRESHOLD).
  3. Identifie l'enregistreur de chaque fichier (à partir du nom de fichier).
  4. Regroupe d'abord les fenêtres qui se chevauchent AU SEIN d'un même
     fichier (déduplication "verticale", comme deduplicate_detections.py),
     pour obtenir un événement par détection réelle et par enregistreur.
  5. Regroupe ensuite ces événements ENTRE enregistreurs différents,
     lorsqu'ils tombent dans la fenêtre de tolérance MAX_TIME_DIFF_SECONDS
     (déduplication "horizontale" / recherche de simultanéité).
  6. Ne conserve que les groupes impliquant AU MOINS 2 enregistreurs
     distincts, et les exporte pour vérification.

Usage :
    python cross_recorder_simultaneous_events.py [chemin_vers_GUNSHOT_scores.csv]

Si aucun chemin n'est fourni, cherche le GUNSHOT_scores.csv le plus récent
dans /data/results/Outputs_predictions_*/
"""

import os
import re
import sys
from glob import glob
from datetime import timedelta
import pandas as pd

# ════════════════════════════════════════════════════════════
#  PARAMÈTRES (à ajuster selon votre déploiement)
# ════════════════════════════════════════════════════════════
RESULTS_ROOT    = "/data/results"

# Seuil de score pour considérer une fenêtre comme "à haute confiance"
SCORE_THRESHOLD = 0.5

# Tolérance pour fusionner des fenêtres AU SEIN d'un même fichier (même
# enregistreur) - identique à deduplicate_detections.py. Correspond au
# décalage (hop) entre deux fenêtres d'analyse consécutives de predict.py :
# CLIP_DURATION * (1 - CLIP_OVERLAP) = 4.0 * (1 - 0.5) = 2.0 s par défaut.
MAX_GAP_WITHIN_RECORDER_SECONDS = 2.0

# Tolérance pour considérer deux événements de DEUX ENREGISTREURS
# DIFFÉRENTS comme potentiellement simultanés (même coup de feu perçu par
# plusieurs appareils). À calibrer : distance max plausible entre
# enregistreurs pouvant capter le même son / vitesse du son (~343 m/s) +
# marge pour désynchronisation d'horloge. Valeur de départ prudente : 15 s.
MAX_TIME_DIFF_SECONDS = 15.0

# Corrections d'horloge connues par enregistreur (en secondes), si vous les
# avez mesurées (ex: via un test de synchronisation avec un son de
# référence). Ajoutée à l'heure déduite du nom de fichier.
# Exemple : {'2MA01481': 0, '2MA01482': -3.5}
CLOCK_OFFSETS = {}
# ════════════════════════════════════════════════════════════

INDEX_PATTERN = re.compile(r'^(?P<filepath>.+)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)$')


def find_latest_scores_csv():
    candidates = sorted(glob(os.path.join(RESULTS_ROOT, "Outputs_predictions_*", "GUNSHOT_scores.csv")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_index(index_value):
    """Extrait (filepath, start_sec, end_sec) depuis la colonne 'index'."""
    m = INDEX_PATTERN.match(index_value)
    if not m:
        raise ValueError(f"Format d'index non reconnu : {index_value}")
    return m.group('filepath'), float(m.group('start')), float(m.group('end'))


def extract_recorder_id(filepath):
    """
    Extrait l'identifiant de l'enregistreur à partir du nom de fichier.
    Convention supposée : <ID_ENREGISTREUR>_<YYYYMMDD>_<HHMMSS>.wav
    ex: '2MA01481_20250423_133241.wav' -> '2MA01481'

    ⚠ Adaptez cette fonction si vos enregistreurs utilisent une autre
    convention de nommage.
    """
    name = os.path.basename(filepath)
    return name.split('_')[0]


def dedup_within_recorder(df, max_gap_seconds):
    """
    Étape 1 : fusionne, pour chaque FICHIER, les fenêtres qui se
    chevauchent ou sont séparées de moins de max_gap_seconds.
    Retourne un DataFrame avec une ligne par événement détecté
    (un enregistreur / un fichier donné peut avoir plusieurs événements).
    """
    events = []
    event_counter = 0

    for filepath, sub in df.groupby('filepath', sort=False):
        sub = sub.sort_values('start_sec').reset_index(drop=True)
        recorder_id = extract_recorder_id(filepath)
        offset = CLOCK_OFFSETS.get(recorder_id, 0)

        def flush(rows):
            nonlocal event_counter
            event_counter += 1
            best_row = max(rows, key=lambda r: r['positive'])
            start_dt = min(pd.to_datetime(r['start_datetime']) for r in rows) + timedelta(seconds=offset)
            end_dt = max(pd.to_datetime(r['end_datetime']) for r in rows) + timedelta(seconds=offset)
            events.append({
                'event_id': event_counter,
                'recorder_id': recorder_id,
                'file': os.path.basename(filepath),
                'n_windows_in_group': len(rows),
                'start_datetime': start_dt,
                'end_datetime': end_dt,
                'best_score': best_row['positive'],
                'best_index': best_row['index'],
            })

        current_rows = [sub.iloc[0]]
        current_end = sub.iloc[0]['end_sec']

        for i in range(1, len(sub)):
            row = sub.iloc[i]
            gap = row['start_sec'] - current_end
            if gap <= max_gap_seconds:
                current_rows.append(row)
                current_end = max(current_end, row['end_sec'])
            else:
                flush(current_rows)
                current_rows = [row]
                current_end = row['end_sec']

        flush(current_rows)

    return pd.DataFrame(events)


def cluster_across_recorders(events_df, max_time_diff_seconds):
    """
    Étape 2 : regroupe les événements (potentiellement issus de plusieurs
    enregistreurs) dont les fenêtres temporelles ABSOLUES se chevauchent ou
    sont séparées de moins de max_time_diff_seconds, indépendamment du
    fichier/enregistreur d'origine.
    Retourne le même DataFrame avec une colonne 'cluster_id' ajoutée.
    """
    events_df = events_df.sort_values('start_datetime').reset_index(drop=True)

    cluster_ids = [None] * len(events_df)
    cluster_counter = 0
    current_end = None

    for i, row in events_df.iterrows():
        if current_end is None:
            cluster_counter += 1
            current_end = row['end_datetime']
        else:
            gap = (row['start_datetime'] - current_end).total_seconds()
            if gap > max_time_diff_seconds:
                cluster_counter += 1
                current_end = row['end_datetime']
            else:
                current_end = max(current_end, row['end_datetime'])
        cluster_ids[i] = cluster_counter

    events_df['cluster_id'] = cluster_ids
    return events_df


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_scores_csv()
    if not input_path or not os.path.isfile(input_path):
        print("✗ Fichier GUNSHOT_scores.csv introuvable.")
        print("  Précisez le chemin en argument, par exemple :")
        print("  python cross_recorder_simultaneous_events.py /data/results/Outputs_predictions_XXXXXX/GUNSHOT_scores.csv")
        sys.exit(1)

    print(f"📄 Lecture de : {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {'index', 'start_datetime', 'end_datetime', 'positive'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"✗ Colonnes manquantes dans le CSV : {sorted(missing)}")
        print("  Ce script nécessite un GUNSHOT_scores.csv généré par la version à jour de predict.py")
        print("  (avec les colonnes start_datetime / end_datetime, calculées à partir du nom de fichier).")
        sys.exit(1)

    parsed = df['index'].apply(parse_index)
    df['filepath']  = parsed.apply(lambda t: t[0])
    df['start_sec'] = parsed.apply(lambda t: t[1])
    df['end_sec']   = parsed.apply(lambda t: t[2])

    n_total = len(df)
    df_filtered = df[df['positive'] >= SCORE_THRESHOLD].copy()
    print(f"🎯 {len(df_filtered)} fenêtre(s) à haut score (positive ≥ {SCORE_THRESHOLD}) sur {n_total} analysées.")

    if df_filtered.empty:
        print("✗ Aucune détection à haut score trouvée avec ce seuil.")
        return

    n_recorders_total = df_filtered['filepath'].apply(extract_recorder_id).nunique()
    print(f"📡 {n_recorders_total} enregistreur(s) distinct(s) détecté(s) dans les données.")

    # Étape 1 : un événement par détection réelle, par enregistreur
    events_df = dedup_within_recorder(df_filtered, MAX_GAP_WITHIN_RECORDER_SECONDS)
    print(f"🔗 {len(df_filtered)} fenêtres regroupées en {len(events_df)} événement(s) par enregistreur.")

    # Étape 2 : regroupement entre enregistreurs différents
    events_df = cluster_across_recorders(events_df, MAX_TIME_DIFF_SECONDS)

    # On ne garde que les clusters impliquant au moins 2 enregistreurs distincts
    cluster_recorder_counts = events_df.groupby('cluster_id')['recorder_id'].nunique()
    multi_recorder_clusters = cluster_recorder_counts[cluster_recorder_counts >= 2].index

    simultaneous_df = events_df[events_df['cluster_id'].isin(multi_recorder_clusters)].copy()

    if simultaneous_df.empty:
        print("\n✗ Aucune détection simultanée entre plusieurs enregistreurs trouvée avec ces paramètres.")
        print(f"  (tolérance actuelle : {MAX_TIME_DIFF_SECONDS} s — essayez de l'augmenter si vous pensez")
        print("   qu'il devrait y avoir des correspondances, par ex. en cas de désynchronisation d'horloge.)")
        return

    # Ajout d'informations de synthèse par cluster (nb d'enregistreurs, liste)
    recorders_per_cluster = simultaneous_df.groupby('cluster_id')['recorder_id'].apply(
        lambda s: ",".join(sorted(s.unique()))
    )
    simultaneous_df['n_recorders_in_cluster'] = simultaneous_df['cluster_id'].map(cluster_recorder_counts)
    simultaneous_df['recorders_in_cluster'] = simultaneous_df['cluster_id'].map(recorders_per_cluster)

    simultaneous_df.sort_values(by=['cluster_id', 'start_datetime'], inplace=True)

    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "SIMULTANEOUS_detections.csv")

    out_cols = [
        'cluster_id', 'n_recorders_in_cluster', 'recorders_in_cluster',
        'recorder_id', 'file', 'start_datetime', 'end_datetime',
        'best_score', 'n_windows_in_group', 'best_index'
    ]
    simultaneous_df[out_cols].to_csv(output_path, index=False)

    n_clusters = len(multi_recorder_clusters)
    print("\n" + "=" * 60)
    print("✨ Recherche de simultanéité terminée.")
    print(f" -> {n_clusters} groupe(s) de détections simultanées entre {len(simultaneous_df)} événements,")
    print(f"    impliquant plusieurs enregistreurs à la fois (tolérance : {MAX_TIME_DIFF_SECONDS} s).")
    print(f" -> Résultats sauvegardés dans : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()