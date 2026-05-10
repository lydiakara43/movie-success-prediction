# Movie Success Prediction

Projet de machine learning pour prédire le revenu et le succès commercial d'un film à partir de caractéristiques telles que le budget, la durée, l'année de sortie et les genres.

## Objectifs

- Prédire le revenu ajusté d'un film avec des modèles de régression.
- Prédire si un film est un succès commercial avec des modèles de classification.
- Comparer plusieurs algorithmes : baseline, régression linéaire, Lasso, KNN, régression logistique, Random Forest.
- Explorer les profils de films avec PCA et K-means.
- Produire un compte rendu LaTeX rigoureux avec résultats, figures, métriques et interprétation.

## Structure

```text
.
├── CODE.ipynb
├── build_project.py
├── movie.csv
├── requirements.txt
├── rapport_movie_success.tex
├── Compte rendu.pdf
├── figures/
└── outputs/
```

## Résultats principaux

- Films initiaux : 10 866.
- Films exploitables après nettoyage : 3 854.
- Taux de succès : 52.4 %.
- Meilleur modèle de régression : Random Forest Regressor.
- Meilleur modèle de classification : Random Forest Classifier.
- Accuracy classification : 0.636.
- Balanced accuracy classification : 0.631.
- ROC-AUC classification : 0.667.

## Installation

Créer un environnement virtuel puis installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

Ouvrir et exécuter :

```text
CODE.ipynb
```

Le script suivant permet de reconstruire le notebook à partir du générateur :

```bash
python build_project.py
```

Les graphiques sont générés dans `figures/` et les métriques dans `outputs/`.

## Rapport

Le rapport source est disponible dans :

```text
rapport_movie_success.tex
```

Le PDF compilé est disponible dans :

```text
Compte rendu.pdf
```

Pour compiler le rapport sur Overleaf, importer `rapport_movie_success.tex` et le dossier `figures/`.

## Remarque méthodologique

Les variables `revenue`, `revenue_adj` et `roi` ne sont pas utilisées comme variables explicatives du modèle de classification, car elles définissent directement la cible de succès. Cela évite une fuite de données.
