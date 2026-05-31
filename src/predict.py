import os
import torch
import pandas as pd
from glob import glob
from opensoundscape.torch.models.cnn import CNN
import warnings
warnings.filterwarnings("ignore", category=Warning)

AUDIO_DIR = "/data/audio"
OUTPUT_DIR = "/data/results"
MODEL_PATH = "/data/lydiakatsis/tropical_forest_gunshot_classifier/Model and code/best.model"

def main():
    print("--- Démarrage du classifier de coups de feu (OpenSoundscape v0.7.1 - Alignement des clés) ---")
    
    audio_files = glob(os.path.join(AUDIO_DIR, "*.wav")) + glob(os.path.join(AUDIO_DIR, "*.WAV")) + glob(os.path.join(AUDIO_DIR, "*.mp3"))
    if not audio_files:
        print(f"Erreur : Aucun fichier .wav ou .mp3 trouvé dans {AUDIO_DIR}")
        return
    print(f"{len(audio_files)} fichier(s) audio détecté(s).")

    print("Chargement et conversion du modèle...")
    
    # 1. Charger le dictionnaire d'origine
    loaded_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # Extraire le state_dict brut si encapsulé
    if isinstance(loaded_dict, dict) and 'model_state_dict' in loaded_dict:
        raw_state_dict = loaded_dict['model_state_dict']
    else:
        raw_state_dict = loaded_dict

    # 2. Recréer un dictionnaire propre en renommant les clés pour la v0.7.1
    cleaned_state_dict = {}
    for key, value in raw_state_dict.items():
        new_key = key
        # Supprime le préfixe "feature."
        if key.startswith("feature."):
            new_key = key.replace("feature.", "")
        # Renomme la couche finale "classifier" en "fc"
        if key.startswith("classifier."):
            new_key = key.replace("classifier.", "fc.")
            
        cleaned_state_dict[new_key] = value

    # 3. Créer l'architecture CNN vierge pour la v0.7.1
    # Le modèle de Lydia Katsis possède 2 classes : [0, 1] ou ['gunshot', 'background']
    # On initialise avec 2 classes pour correspondre aux poids fc.weight (taille 2)
    model = CNN(architecture='resnet18', classes=['0', '1'], sample_duration=5.0) 

    # 4. Charger le dictionnaire nettoyé dans le réseau ResNet
    model.network.load_state_dict(cleaned_state_dict)
    print("Modèle converti et chargé avec succès !")

    df = pd.DataFrame(index=audio_files)

    print("Analyse en cours (cela peut prendre du temps)...")
    try:
        # On récupère le résultat global
        result = model.predict(
            df, 
            batch_size=1, 
            num_workers=0
        )
        
        # Sécurité : Si c'est un tuple (peu importe sa taille : 2, 3 ou plus), 
        # on prend le tout premier élément qui est TOUJOURS le DataFrame des prédictions.
        if isinstance(result, tuple):
            predictions_df = result[0]
        else:
            predictions_df = result
        
        # Sauvegarde du DataFrame de prédictions
        output_file = os.path.join(OUTPUT_DIR, "predictions_coups_de_feu.csv")
        predictions_df.to_csv(output_file)
        print(f"Analyse terminée ! Les résultats ont été sauvegardés dans : {output_file}")
        
    except Exception as e:
        print(f"Une erreur est survenue lors de la prédiction : {e}")

if __name__ == "__main__":
    main()