import os
import torch
import pandas as pd
from glob import glob
from opensoundscape.torch.models.cnn import CNN

AUDIO_DIR = "/data/audio"
OUTPUT_DIR = "/data/results"
MODEL_PATH = "/data/lydiakatsis/tropical_forest_gunshot_classifier/Model and code/best.model"

def main():
    print("--- Démarrage du classifier (Tri par fichier & Seuil > 0.2) ---")
    
    # 1. Collecte des fichiers audio
    audio_files = glob(os.path.join(AUDIO_DIR, "*.wav")) + glob(os.path.join(AUDIO_DIR, "*.WAV")) + glob(os.path.join(AUDIO_DIR, "*.mp3"))
    if not audio_files:
        print(f"Erreur : Aucun fichier trouvé dans {AUDIO_DIR}")
        return
    print(f"{len(audio_files)} fichier(s) audio détecté(s).")

    # 2. Chargement et conversion du modèle (Compatibilité v0.7.1)
    loaded_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    raw_state_dict = loaded_dict['model_state_dict'] if isinstance(loaded_dict, dict) and 'model_state_dict' in loaded_dict else loaded_dict

    cleaned_state_dict = {}
    for key, value in raw_state_dict.items():
        new_key = key.replace("feature.", "") if key.startswith("feature.") else key
        if key.startswith("classifier."):
            new_key = key.replace("classifier.", "fc.")
        cleaned_state_dict[new_key] = value

    model = CNN(architecture='resnet18', classes=['0', '1'], sample_duration=5.0) 
    model.network.load_state_dict(cleaned_state_dict)

    df = pd.DataFrame(index=audio_files)

    print("Analyse et calcul des probabilités (Softmax)...")
    try:
        result = model.predict(
            df, 
            batch_size=64, 
            num_workers=2,
            activation_layer='softmax'
        )
        
        # Extraction du DataFrame des prédictions
        predictions_df = result[0] if isinstance(result, tuple) else result
        
        # Passer de l'index multi-niveau à des colonnes standards
        final_table = predictions_df.reset_index()
        
        # Renommer les colonnes pour plus de clarté
        final_table = final_table.rename(columns={
            'index_file': 'file',
            '0': 'prob_pas_de_tir',
            '1': 'prob_coup_de_feu'
        })

        # 3. FILTRAGE : On garde uniquement les scores de coups de feu > 0.2
        final_table = final_table[final_table['prob_coup_de_feu'] > 0.2]

        if final_table.empty:
            print("\n[Info] Aucun segment n'a dépassé le seuil de 0.2 de probabilité.")
            print("Le fichier CSV ne sera pas généré car il est vide.")
            return

        # 4. TRI : Par nom de fichier (A-Z) puis par probabilité de coup de feu (descendante)
        final_table = final_table.sort_values(by=['file', 'prob_coup_de_feu'], ascending=[True, False])

        # 5. Sauvegarde du fichier CSV filtré
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "classified_gunshots.csv")
        
        final_table.to_csv(output_file, index=False)
        
        print(f"\nSuccès ! Le fichier filtré et trié a été sauvegardé.")
        print(f"Chemin du fichier : {output_file}")
        print(f"Nombre de lignes retenues (> 0.2) : {len(final_table)}")
        
        # Aperçu rapide dans le terminal
        print("\nAperçu des premiers résultats enregistrés :")
        print(final_table[['file', 'start_time', 'end_time', 'prob_coup_de_feu']].head(10))
            
    except Exception as e:
        print(f"Une erreur est survenue lors de la prédiction : {e}")

if __name__ == "__main__":
    main()