# 🦉 Guide d'utilisation — Détecteur de Coups de Feu (Gunshot Classifier)

Ce système automatique permet d'analyser des enregistrements audio de longue durée pour y détecter d'éventuels **coups de feu**. Il découpe les fichiers audio en morceaux de 4 secondes et calcule pour chacun d'eux un score de confiance (entre 0 et 1) de la présence d'un tir.

---

## 📂 Étape 1 : Où déposer vos enregistrements audio ?

Avant de lancer l'analyse, vous devez copier vos fichiers audio au format `.WAV` dans le dossier partagé de votre ordinateur prévu à cet effet :

* **Dossier d'entrée :** `audio_a_analyser` (ou le nom défini sur votre machine).
* *Note : Veillez à ce que les extensions de vos fichiers soient bien en majuscules ou minuscules (`.wav`, `.WAV`).*

---

## 🚀 Étape 2 : Lancer l'analyse

L'analyse se lance très simplement depuis le terminal (invite de commandes) grâce à Docker. 

Tapez ou copiez-collez la commande suivante et appuyez sur **Entrée** :

```bash
docker compose exec classifier python predict_GUNSHOT_classifier.py
```
Le programme va afficher sa progression fichier par fichier (ex: `[1/12] Analyse de Rec_20260609.WAV`...).

## 📊 Étape 3 : Comprendre et lire les résultats
Une fois l'analyse terminée, un nouveau dossier apparaît sur votre ordinateur. Son nom contient la date du jour (par exemple : `Outputs_predictions_260609`).

À l'intérieur, vous trouverez deux fichiers que vous pouvez ouvrir directement avec Excel, LibreOffice ou Google Sheets :

### 1. `GUNSHOT_scores.csv` (Le plus utile)
Ce fichier classe tous les morceaux analysés, du plus suspect au moins suspect. Le premier morceau de la liste est celui où le système est le plus certain d'avoir entendu un coup de feu.

Voici à quoi ressemblent les colonnes :

- **index** : Le nom du fichier audio suivi de la seconde exacte du morceau (ex: `0.00-4.00` pour les 4 premières secondes).

- **start_time** : L'heure de début du morceau dans le fichier (au format Heure:Minute:Seconde).

- **end_time** : L'heure de fin du morceau dans le fichier.

- **negative** : La confiance du modèle que ce soit un bruit de fond normal (oiseau, pluie, vent...).

- **positive** : La confiance du modèle que ce soit un coup de feu (un score de `0.950` signifie plus ou moins 95% de certitude).

💡 Pour l'analyse, commencez par le haut du fichier, regardez la colonne `positive`, qui est triée par scores décroissants, et concentrez-vous sur les scores les plus proches de `1.0`. 

### 2. `GUNSHOT_binary_predictions.csv`
Ce fichier applique une décision stricte ("Oui" ou "Non") selon un seuil scientifique pré-configuré (environ 80% de certitude).

- Si la colonne `positive` affiche `1`, le système considère qu'il s'agit officiellement d'un coup de feu.

- Si elle affiche `0`, le bruit est classé comme bruit de fond.

## 🔍 Étape 4 : Contrôler les résultats

Grâce aux colonnes `start_time` et `end_time`, vous disposez de la fenêtre de détection exacte d'un coup de feu supposé. Vous pouvez alors ouvrir le fichier audio original dans un logiciel de visualisation sonore comme **Raven Lite** ou **Audacity** et écouter le passage indiqué pour confirmer par vous-même si le coup de feu est positif (réellement détecté), négatif (bruit de fond), ou s'il s'agit d'un faux positif (erreur du modèle).

