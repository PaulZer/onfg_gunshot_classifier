# ONF Guyane - Gunshot classifier

## Introduction

Cet outil analyse automatiquement des enregistrements audio réalisés en forêt et identifie les moments où un coup de feu a été tiré. Il est conçu pour aider à mesurer la pression de chasse dans des zones ciblées, à partir d'enregistreurs autonomes posés sur le terrain.

---

## Comment ça fonctionne ?

### L'idée générale

Le programme écoute vos enregistrements et, pour chaque passage de 4 secondes, répond à la question : *"Est-ce qu'on entend un coup de feu ici ?"* Il produit ensuite un tableau indiquant les moments suspects, avec un score de confiance entre 0 et 1.

### Le modèle d'intelligence artificielle

Au cœur de l'outil se trouve un **modèle entraîné**, c'est-à-dire un programme qui a appris à reconnaître le son d'un coup de feu en écoutant des milliers d'exemples sonores étiquetés — des enregistrements de coups de feu réels en forêt tropicale, mélangés à des sons de fond (pluie, insectes, oiseaux, vent...). Ce modèle a été développé par [Lydia Katsis et al.](https://github.com/lydiakatsis/tropical_forest_gunshot_classifier) et publié en accès libre.

### Ce que le programme fait concrètement, étape par étape

1. **Découpe** — Le fichier audio est découpé en petites fenêtres de 4 secondes qui se chevauchent légèrement (par défaut de 50%, soit un décalage de 2 secondes entre chaque fenêtre). Ce chevauchement évite qu'un coup de feu tombant entre deux fenêtres soit raté.

2. **Conversion en image** — Chaque fenêtre de 4 secondes est transformée en une image appelée **spectrogramme** : une représentation visuelle du son où l'axe horizontal est le temps, l'axe vertical est la fréquence, et la couleur indique l'intensité. Un coup de feu produit une signature visuelle reconnaissable sur ce type d'image.

3. **Analyse par l'IA** — Chaque image est soumise au modèle, qui lui attribue un **score entre 0 et 1** indiquant la probabilité qu'il s'agisse d'un coup de feu. Un score proche de 1 signifie que le modèle est très confiant ; proche de 0, qu'il s'agit probablement d'un bruit de fond.

4. **Filtrage** — Seules les fenêtres dont le score dépasse un seuil réglable (par défaut 0.5) sont conservées dans les résultats.

5. **Export** — Les résultats sont affichés à l'écran et sauvegardés dans un fichier tableur (`.csv`) que vous pouvez ouvrir avec Excel ou LibreOffice.

---

## Prérequis

L'outil fonctionne sur un conteneur Docker sous Linux disposant de Python 3, et de toutes les dépendances nécessaires et du fichier du modèle `best.model`, disponible dans le dépôt GitHub du projet original.

---

## Utilisation

### 1. Préparer vos fichiers

Placez vos enregistrements audio (formats acceptés : `.wav`, `.WAV`, `.mp3`, `.flac`) dans le dossier `/audio_to_analyze`, à la racine du projet.

### 2. Configurer le script (optionnel)

Ouvrez le fichier `classify_gunshots.py` avec un éditeur de texte. En haut du fichier, modifiez les trois lignes suivantes pour qu'elles correspondent à votre situation :

```python
AUDIO_DIR  = "/data/audio"          # Dossier contenant vos enregistrements
OUTPUT_DIR = "/data/results"        # Dossier où sera sauvegardé le tableau de résultats
MODEL_PATH = "/data/.../best.model" # Chemin vers le fichier du modèle
```

Vous pouvez également ajuster ces deux paramètres selon vos besoins :

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `THRESHOLD` | `0.5` | Seuil de détection. Augmenter réduit les fausses alarmes ; diminuer permet de ne rien rater. |
| `CLIP_OVERLAP` | `0.5` | Chevauchement entre fenêtres (50%). Augmenter améliore la détection mais allonge le temps d'analyse. |

### 3. Lancer l'analyse

Dans un terminal, exécutez :

```
python classify_gunshots.py
```

L'avancement s'affiche en temps réel. Pour un enregistrement d'une heure, comptez environ 5 à 15 minutes d'analyse selon votre matériel.

### 4. Lire les résultats

Un fichier `classified_gunshots.csv` est créé dans votre dossier de résultats. Chaque ligne correspond à une fenêtre de 4 secondes détectée comme coup de feu potentiel :

| Colonne | Signification |
|---|---|
| `file` | Nom du fichier audio analysé |
| `start_time` | Heure de début de la fenêtre dans l'enregistrement (format HH:MM:SS) |
| `end_time` | Heure de fin de la fenêtre |
| `background` | Score de probabilité que ce soit un bruit de fond (entre 0 et 1) |
| `gunshot` | Score de probabilité que ce soit un coup de feu (entre 0 et 1) |

Les résultats sont triés par fichier, puis par score décroissant : les détections les plus certaines apparaissent en premier.

**Exemple de sortie :**

```
                        file  start_time  end_time  background  gunshot
/data/audio/terrain_01.WAV    00:14:32    00:14:36       0.001    0.999
/data/audio/terrain_01.WAV    00:14:30    00:14:34       0.014    0.986
/data/audio/terrain_01.WAV    00:52:08    00:52:12       0.038    0.962
```

> **Conseil de lecture :** Plusieurs lignes consécutives avec des timecodes proches correspondent souvent au même coup de feu détecté dans des fenêtres qui se chevauchent. Il s'agit donc d'un seul événement, pas de plusieurs tirs.

---

## Limites et précautions d'usage

- Le modèle a été entraîné sur des coups de feu en **forêt tropicale**. Ses performances peuvent varier selon le type d'arme, la distance, ou les conditions acoustiques de votre site.
- Un score élevé n'est pas une certitude absolue : une vérification manuelle des détections à score intermédiaire (entre 0.5 et 0.8) est recommandée.
- La qualité de l'enregistreur et le niveau de bruit ambiant influencent significativement les résultats.
- Les résultats permettent d'estimer une **pression de chasse relative** (comparaison entre sites ou entre périodes) plus que de compter précisément le nombre de coups de feu tirés.


## Ressources

- [tropical_forest_gunshot_classifier Github repository](https://github.com/lydiakatsis/tropical_forest_gunshot_classifier)
- [Original classifying script on Google Colab](https://colab.research.google.com/github/lydiakatsis/tropical_forest_gunshot_classifier/blob/main/Gunshot%20classification%20GUI/Gunshot_classifier_colab.ipynb)

