import os
import pandas as pd

CSV_PATH = "/data/results/predictions_coups_de_feu.csv"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Erreur : Le fichier {CSV_PATH} n'existe pas.")
        return

    # Chargement correct avec index_col
    df = pd.read_csv(CSV_PATH, index_col=["file", "start_time", "end_time"])
    
    # Filtrage : On cherche où la colonne '1' (coup de feu) est supérieure à la colonne '0'
    # Note : Selon la manière dont Pandas a écrit le CSV, les noms de colonnes peuvent être 
    # des chaînes de caractères "0" et "1". On utilise des chaînes pour être sûr.
    tirs_detectes = df[df['1'] > df['0']]
    
    print("--- Rapport d'analyse bioacoustique ---")
    print(f"Nombre total de segments analysés : {len(df)}")
    print(f"Nombre de tirs suspects détectés : {len(tirs_detectes)}")
    print("-" * 39)
    
    if not tirs_detectes.empty:
        print("\nListe des détections (Tirs suspects) :")
        # On affiche le résultat trié du score le plus fort au plus faible
        tirs_tries = tirs_detectes.sort_values(by='1', ascending=False)
        print(tirs_tries[['0', '1']])
    else:
        print("\nAucun coup de feu détecté sur les fichiers analysés.")

if __name__ == "__main__":
    main()