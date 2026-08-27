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

6. **Élimination des doublons** — Comme les fenêtres se chevauchent, un même coup de feu déclenche souvent plusieurs détections consécutives. Une étape de regroupement fusionne ces détections pour ne garder qu'une ligne par événement réel.

7. **Recherche de recoupements entre enregistreurs** — Sur un réseau de plusieurs enregistreurs, une dernière étape recherche les cas où un même coup de feu a été capté par plusieurs appareils à la fois, ce qui permet de confirmer certaines détections.

---

## Guide d'utilisation


### 📂 Étape 1 : Où déposer vos enregistrements audio ?

Avant de lancer l'analyse, vous devez copier vos fichiers audio au format `.WAV` dans le dossier partagé de votre ordinateur prévu à cet effet :

* **Dossier d'entrée :** `audio_to_analyze` (ou le nom défini sur votre machine).
* *Note : Veillez à ce que les extensions de vos fichiers soient bien `.wav` ou `.WAV`.*

#### ⚠️ Convention de nommage requise

Pour que les horodatages des détections soient corrects, le nom de chaque fichier audio doit contenir la **date et l'heure de DÉBUT d'enregistrement**, sous l'un des formats suivants :

| Format du nom de fichier | Exemple |
|---|---|
| `..._YYYYMMDD_HHMMSS...wav` | `2MA01481_20250423_133241.wav` |
| `..._YYYY-MM-DD_HH-MM-SS...wav` | `2MA01481_2025-04-23_13-32-41.wav` |
| `...YYYYMMDDHHMMSS...wav` (14 chiffres collés) | `2MA01481-20250423133241.wav` |

C'est le format par défaut de la plupart des enregistreurs autonomes (AudioMoth, Wildlife Acoustics Song Meter, etc.). Si aucun de ces formats n'est reconnu dans le nom d'un fichier, le programme se rabat automatiquement sur la date de dernière modification du fichier (`mtime`) — une méthode moins fiable, car cette date peut être modifiée lors d'une copie ou d'un transfert. Un avertissement s'affiche alors dans le terminal pour ce fichier.

* **Identifiant de l'enregistreur :** pour le repérage des détections simultanées entre plusieurs enregistreurs (voir plus bas), le programme suppose que l'identifiant de l'enregistreur est la partie du nom de fichier précédant le premier `_` (ex : `2MA01481` dans `2MA01481_20250423_133241.wav`). Assurez-vous que tous vos enregistreurs suivent cette convention.

---

### 🚀 Étape 2 : Lancer l'analyse

L'analyse complète comporte plusieurs traitements successifs (détection brute, puis élimination des doublons, puis recherche de détections simultanées entre enregistreurs). Deux façons de les lancer :

#### Option recommandée : tout en une seule commande

```bash
docker compose exec onfg_gunshot_classifier python run_pipeline.py
```

Cette commande enchaîne automatiquement les 3 étapes de traitement (voir détail ci-dessous) et affiche la progression de chacune. Si une étape échoue, le pipeline s'arrête proprement sans lancer les suivantes, et affiche un récapitulatif des fichiers générés à la fin.

#### Option manuelle : étape par étape

Utile si vous voulez inspecter les résultats intermédiaires ou relancer une seule étape (par exemple, réessayer le regroupement des doublons avec un seuil différent, sans refaire toute l'analyse audio) :

```bash
# 1. Détection brute des coups de feu sur tous les fichiers .WAV
docker compose exec onfg_gunshot_classifier python predict.py

# 2. Regroupement des fenêtres en doublon au sein d'un même enregistreur
docker compose exec onfg_gunshot_classifier python deduplicate_detections.py

# 3. Recherche de détections simultanées entre enregistreurs différents
docker compose exec onfg_gunshot_classifier python cross_recorder_simultaneous_events.py
```

Les étapes 2 et 3 recherchent automatiquement le `GUNSHOT_scores.csv` le plus récent — vous pouvez aussi leur passer un chemin explicite en argument, par exemple :
```bash
docker compose exec onfg_gunshot_classifier python deduplicate_detections.py /data/results/Outputs_predictions_260609/GUNSHOT_scores.csv
```

Dans tous les cas, le programme affiche sa progression fichier par fichier (ex: `[1/12] Analyse de 2MA01481_20250423_133241.WAV`...).

---

### 📊 Étape 3 : Comprendre et lire les résultats
Une fois l'analyse terminée, un nouveau dossier apparaît dans le dossier `results`. Son nom contient la date du jour (par exemple : `Outputs_predictions_260609`).

À l'intérieur, vous trouverez jusqu'à quatre fichiers que vous pouvez ouvrir directement avec Excel, LibreOffice ou Google Sheets :

#### 1. `GUNSHOT_scores.csv` (Le plus complet)
Ce fichier classe tous les morceaux analysés, du plus suspect au moins suspect. Le premier morceau de la liste est celui où le système est le plus certain d'avoir entendu un coup de feu.

Voici à quoi ressemblent les colonnes :

- **index** : Le nom du fichier audio suivi de la seconde exacte du morceau (ex: `0.00-4.00` pour les 4 premières secondes).

- **start_datetime** / **end_datetime** : La date et l'heure **absolues** de début et de fin du morceau (format `AAAA-MM-JJ HH:MM:SS`), déduites de l'horodatage contenu dans le nom du fichier audio (voir la convention de nommage à l'Étape 1). C'est la colonne à utiliser pour comparer des détections entre plusieurs enregistreurs, ou pour situer précisément un événement dans le temps.

- **start_time** / **end_time** : L'heure de début et de fin du morceau **relative au fichier audio** (au format Heure:Minute:Seconde depuis le début du fichier, et non l'heure réelle).

- **negative** : La confiance du modèle que ce soit un bruit de fond normal (oiseau, pluie, vent...).

- **positive** : La confiance du modèle que ce soit un coup de feu (un score de `0.950` signifie plus ou moins 95% de certitude).

💡 Pour l'analyse, commencez par le haut du fichier, regardez la colonne `positive`, qui est triée par scores décroissants, et concentrez-vous sur les scores les plus proches de `1.0`. 

#### 2. `GUNSHOT_binary_predictions.csv`
Ce fichier applique une décision stricte ("Oui" ou "Non") selon un seuil scientifique pré-configuré (environ 80% de certitude). Mêmes colonnes que `GUNSHOT_scores.csv`.

- Si la colonne `positive` affiche `1`, le système considère qu'il s'agit officiellement d'un coup de feu.

- Si elle affiche `0`, le bruit est classé comme bruit de fond.

#### 3. `GUNSHOT_scores_deduplicated.csv` (généré par `deduplicate_detections.py`)
Comme les fenêtres d'analyse se chevauchent (chevauchement de 50% par défaut), un même coup de feu déclenche souvent **plusieurs détections consécutives à haut score**. Ce fichier les regroupe pour ne garder qu'**une seule ligne par événement réel**, au sein d'un même enregistreur.

Colonnes principales :

- **group_id** : Identifiant du groupe (= de l'événement regroupé).
- **file** : Le fichier audio concerné.
- **n_windows_in_group** : Le nombre de fenêtres d'analyse qui ont été fusionnées dans ce groupe (un nombre élevé peut indiquer un son prolongé, ou plusieurs tirs très rapprochés).
- **group_start_datetime** / **group_end_datetime** : L'étendue temporelle totale du groupe.
- **best_start_datetime** / **best_end_datetime** / **best_score** / **best_index** : La fenêtre du groupe ayant obtenu le meilleur score — c'est celle à écouter en priorité pour une vérification manuelle.

Deux paramètres réglables en tête du script permettent d'ajuster le comportement sans relancer l'analyse audio :
- `SCORE_THRESHOLD` (par défaut `0.5`) : seuil de score à partir duquel une détection est prise en compte.
- `MAX_GAP_SECONDS` (par défaut `2.0`) : écart maximal toléré, en secondes, entre deux fenêtres pour les considérer comme le même événement.

#### 4. `SIMULTANEOUS_detections.csv` (généré par `cross_recorder_simultaneous_events.py`)
Ce fichier recherche les cas où **plusieurs enregistreurs différents** ont capté, à peu de choses près, le même coup de feu au même moment — un signe de recoupement utile pour confirmer une détection ou reconstituer un secteur d'activité.

Colonnes principales :

- **cluster_id** : Identifiant du groupe de détections jugées simultanées.
- **n_recorders_in_cluster** / **recorders_in_cluster** : Le nombre et la liste des enregistreurs impliqués dans ce groupe.
- **recorder_id** / **file** : L'enregistreur et le fichier concernés par cette ligne.
- **start_datetime** / **end_datetime** / **best_score** : La fenêtre et le score de l'événement, pour cet enregistreur.

⚠️ Le fichier n'est généré que s'il existe au moins un groupe impliquant 2 enregistreurs ou plus. Un chevauchement temporel entre deux enregistreurs est un **indice à vérifier**, pas une preuve absolue : deux coups de feu distincts peuvent survenir au même moment dans des secteurs différents. Deux paramètres réglables en tête du script :
- `MAX_TIME_DIFF_SECONDS` (par défaut `15.0`) : tolérance temporelle entre enregistreurs pour juger deux détections "simultanées". Cette valeur doit couvrir à la fois le délai de propagation du son (~3 secondes pour 1 km) et surtout l'imprécision de synchronisation des horloges internes des enregistreurs, souvent le facteur dominant si vos appareils ne sont pas synchronisés par GPS.
- `CLOCK_OFFSETS` : permet de renseigner, enregistreur par enregistreur, une correction d'horloge connue (en secondes) si vous l'avez mesurée sur le terrain.

---

### 🔍 Étape 4 : Contrôler les résultats

Grâce aux colonnes `start_time` et `end_time` (position dans le fichier audio) ou `start_datetime`/`end_datetime` (date et heure réelles), vous disposez de la fenêtre de détection exacte d'un coup de feu supposé. Vous pouvez alors ouvrir le fichier audio original dans un logiciel de visualisation sonore comme **Raven Lite** ou **Audacity** et écouter le passage indiqué pour confirmer par vous-même si le coup de feu est positif (réellement détecté), négatif (bruit de fond), ou s'il s'agit d'un faux positif (erreur du modèle).

Pour les événements listés dans `SIMULTANEOUS_detections.csv`, pensez à écouter le passage correspondant sur **chacun des enregistreurs impliqués** (colonne `recorders_in_cluster`) afin de confirmer qu'il s'agit bien du même coup de feu, et non d'une coïncidence temporelle entre deux événements distincts.

---

## Limites et précautions d'usage

- Le modèle a été entraîné sur des coups de feu en **forêt tropicale**. Ses performances peuvent varier selon le type d'arme, la distance, ou les conditions acoustiques de votre site.
- Un score élevé n'est pas une certitude absolue : une vérification manuelle des détections à score intermédiaire (entre 0.5 et 0.8) est recommandée.
- La qualité de l'enregistreur et le niveau de bruit ambiant influencent significativement les résultats.
- Les résultats permettent d'estimer une **pression de chasse relative** (comparaison entre sites ou entre périodes) plus que de compter précisément le nombre de coups de feu tirés.
- Les colonnes `start_datetime`/`end_datetime` dépendent de la convention de nommage des fichiers audio (voir Étape 1). Si le nom d'un fichier ne contient pas d'horodatage reconnu, le programme se rabat sur la date de modification du fichier — moins fiable, car altérable par une copie ou un transfert.
- La recherche de détections simultanées entre enregistreurs (`cross_recorder_simultaneous_events.py`) suppose que les horloges internes de vos enregistreurs sont raisonnablement synchronisées entre elles. Sans synchronisation GPS, un décalage d'horloge non corrigé peut faire manquer de vraies détections simultanées, ou au contraire en suggérer de fausses si la tolérance (`MAX_TIME_DIFF_SECONDS`) est réglée trop large.


## Ressources

- [tropical_forest_gunshot_classifier Github repository](https://github.com/lydiakatsis/tropical_forest_gunshot_classifier)
- [Original classifying script on Google Colab](https://colab.research.google.com/github/lydiakatsis/tropical_forest_gunshot_classifier/blob/main/Gunshot%20classification%20GUI/Gunshot_classifier_colab.ipynb)