#!/usr/bin/env python
# coding: utf-8

"""
run_pipeline.py — Exécute la chaîne d'analyse complète en une seule commande
========================================================================
Enchaîne automatiquement les 3 étapes, sans avoir à les lancer une par une :

  1. predict.py
     -> Détection brute des coups de feu sur tous les fichiers .WAV,
        écrite SÉPARÉMENT PAR ENREGISTREUR dans <OUTPUT_DIR>/<recorder_id>/
        (GUNSHOT_scores.csv, GUNSHOT_binary_predictions.csv)

  2. deduplicate_detections.py
     -> Pour CHAQUE enregistreur, regroupe les fenêtres qui se chevauchent
        (GUNSHOT_scores_deduplicated.csv dans chaque sous-dossier)

  3. cross_recorder_simultaneous_events.py
     -> Charge TOUS les enregistreurs et recherche les détections
        simultanées entre enregistreurs différents
        (SIMULTANEOUS_detections.csv à la racine du dossier de résultats)

Usage :
    docker compose exec classifier python run_pipeline.py

Chaque étape tourne dans un sous-processus séparé, avec son affichage en
direct (comme si vous la lanciez manuellement). Le pipeline s'arrête
proprement dès qu'une étape échoue, sans lancer les suivantes.

Prérequis : les 4 scripts (celui-ci + les 3 étapes) doivent se trouver dans
le même dossier (src/, monté sur /app dans le conteneur).
"""

import os
import sys
import subprocess
from glob import glob
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICT_SCRIPT        = os.path.join(SCRIPT_DIR, "predict.py")
DEDUPLICATE_SCRIPT    = os.path.join(SCRIPT_DIR, "deduplicate_detections.py")
CROSS_RECORDER_SCRIPT = os.path.join(SCRIPT_DIR, "cross_recorder_simultaneous_events.py")

# Doit correspondre à OUTPUT_DIR dans predict.py
RESULTS_ROOT = "/data/results"


def check_scripts_exist():
    for path in [PREDICT_SCRIPT, DEDUPLICATE_SCRIPT, CROSS_RECORDER_SCRIPT]:
        if not os.path.isfile(path):
            print(f"✗ Script manquant : {path}")
            print("  Vérifiez que les 4 scripts se trouvent bien dans le même dossier.")
            sys.exit(1)


def run_step(step_num, total_steps, title, cmd):
    print("\n" + "═" * 70)
    print(f"ÉTAPE {step_num}/{total_steps} — {title}")
    print("═" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n✗ L'étape {step_num} ({title}) a échoué (code retour {result.returncode}).")
        print("  Arrêt du pipeline — les étapes suivantes ne sont pas lancées.")
        sys.exit(result.returncode)


def get_today_output_dir():
    """Reconstruit le chemin du dossier de résultats généré par predict.py (basé sur la date du jour)."""
    d = date.today().strftime('%y%m%d')
    return os.path.join(RESULTS_ROOT, f"Outputs_predictions_{d}")


def main():
    check_scripts_exist()

    # ── Étape 1 : détection brute ───────────────────────────────────────
    run_step(1, 3, "Détection des coups de feu (predict.py)",
              [sys.executable, PREDICT_SCRIPT])

    output_dir = get_today_output_dir()
    recorder_csvs = glob(os.path.join(output_dir, "*", "GUNSHOT_scores.csv"))

    if not recorder_csvs:
        print(f"\n✗ Aucun GUNSHOT_scores.csv trouvé sous {output_dir}/<enregistreur>/")
        print("  Le pipeline ne peut pas continuer sans ce fichier.")
        sys.exit(1)
    print(f"\n📡 {len(recorder_csvs)} enregistreur(s) trouvé(s) sous : {output_dir}")

    # ── Étape 2 : déduplication au sein de chaque enregistreur ──────────
    run_step(2, 3, "Regroupement des doublons par enregistreur (deduplicate_detections.py)",
              [sys.executable, DEDUPLICATE_SCRIPT, output_dir])

    # ── Étape 3 : détections simultanées entre enregistreurs ────────────
    run_step(3, 3, "Recherche de détections simultanées entre enregistreurs (cross_recorder_simultaneous_events.py)",
              [sys.executable, CROSS_RECORDER_SCRIPT, output_dir])

    # ── Récapitulatif final ───────────────────────────────────────────
    print("\n" + "═" * 70)
    print("✅ Pipeline complet terminé avec succès.")
    print(f"   Tous les résultats se trouvent dans : {output_dir}")
    print("═" * 70)

    print("\nFichiers générés par enregistreur :")
    recorder_dirs = sorted(set(os.path.dirname(p) for p in recorder_csvs))
    for recorder_dir in recorder_dirs:
        recorder_id = os.path.basename(recorder_dir)
        print(f"  📁 {recorder_id}/")
        for fname in ["GUNSHOT_scores.csv", "GUNSHOT_binary_predictions.csv", "GUNSHOT_scores_deduplicated.csv"]:
            fpath = os.path.join(recorder_dir, fname)
            status = "✓" if os.path.isfile(fpath) else "✗ (non généré)"
            print(f"      {status}  {fname}")

    print("\nFichier de synthèse multi-enregistreurs :")
    sim_path = os.path.join(output_dir, "SIMULTANEOUS_detections.csv")
    status = "✓" if os.path.isfile(sim_path) else "— (non généré, ex: aucune détection simultanée trouvée)"
    print(f"  {status}  {sim_path}")


if __name__ == "__main__":
    main()