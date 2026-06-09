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

## Guide d'utilisation

### 📂 Étape 1 : Où déposer vos enregistrements audio ?

Avant de lancer l'analyse, vous devez copier vos fichiers audio au format `.WAV` dans le dossier partagé de votre ordinateur prévu à cet effet :

* **Dossier d'entrée :** `audio_a_analyser` (ou le nom défini sur votre machine).
* *Note : Veillez à ce que les extensions de vos fichiers soient bien en majuscules ou minuscules (`.wav`, `.WAV`).*

---

### 🚀 Étape 2 : Lancer l'analyse

L'analyse se lance très simplement depuis le terminal (invite de commandes) grâce à Docker. 

Tapez ou copiez-collez la commande suivante et appuyez sur **Entrée** :

```bash
docker compose exec classifier python predict_GUNSHOT_classifier.py
```
Le programme va afficher sa progression fichier par fichier (ex: `[1/12] Analyse de Rec_20260609.WAV`...).

### 📊 Étape 3 : Comprendre et lire les résultats
Une fois l'analyse terminée, un nouveau dossier apparaît sur votre ordinateur. Son nom contient la date du jour (par exemple : `Outputs_predictions_260609`).

À l'intérieur, vous trouverez deux fichiers que vous pouvez ouvrir directement avec Excel, LibreOffice ou Google Sheets :

#### 1. `GUNSHOT_scores.csv` (Le plus utile)
Ce fichier classe tous les morceaux analysés, du plus suspect au moins suspect. Le premier morceau de la liste est celui où le système est le plus certain d'avoir entendu un coup de feu.

Voici à quoi ressemblent les colonnes :

- **index** : Le nom du fichier audio suivi de la seconde exacte du morceau (ex: `0.00-4.00` pour les 4 premières secondes).

- **start_time** : L'heure de début du morceau dans le fichier (au format Heure:Minute:Seconde).

- **end_time** : L'heure de fin du morceau dans le fichier.

- **negative** : La confiance du modèle que ce soit un bruit de fond normal (oiseau, pluie, vent...).

- **positive** : La confiance du modèle que ce soit un coup de feu (un score de `0.950` signifie plus ou moins 95% de certitude).

💡 Pour l'analyse, commencez par le haut du fichier, regardez la colonne `positive`, qui est triée par scores décroissants, et concentrez-vous sur les scores les plus proches de `1.0`. 

#### 2. `GUNSHOT_binary_predictions.csv`
Ce fichier applique une décision stricte ("Oui" ou "Non") selon un seuil scientifique pré-configuré (environ 80% de certitude).

- Si la colonne `positive` affiche `1`, le système considère qu'il s'agit officiellement d'un coup de feu.

- Si elle affiche `0`, le bruit est classé comme bruit de fond.

### 🔍 Étape 4 : Contrôler les résultats

Grâce aux colonnes `start_time` et `end_time`, vous disposez de la fenêtre de détection exacte d'un coup de feu supposé. Vous pouvez alors ouvrir le fichier audio original dans un logiciel de visualisation sonore comme **Raven Lite** ou **Audacity** et écouter le passage indiqué pour confirmer par vous-même si le coup de feu est positif (réellement détecté), négatif (bruit de fond), ou s'il s'agit d'un faux positif (erreur du modèle).

---

## Limites et précautions d'usage

- Le modèle a été entraîné sur des coups de feu en **forêt tropicale**. Ses performances peuvent varier selon le type d'arme, la distance, ou les conditions acoustiques de votre site.
- Un score élevé n'est pas une certitude absolue : une vérification manuelle des détections à score intermédiaire (entre 0.5 et 0.8) est recommandée.
- La qualité de l'enregistreur et le niveau de bruit ambiant influencent significativement les résultats.
- Les résultats permettent d'estimer une **pression de chasse relative** (comparaison entre sites ou entre périodes) plus que de compter précisément le nombre de coups de feu tirés.


## Ressources

- [tropical_forest_gunshot_classifier Github repository](https://github.com/lydiakatsis/tropical_forest_gunshot_classifier)
- [Original classifying script on Google Colab](https://colab.research.google.com/github/lydiakatsis/tropical_forest_gunshot_classifier/blob/main/Gunshot%20classification%20GUI/Gunshot_classifier_colab.ipynb)

