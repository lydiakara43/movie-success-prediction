from pathlib import Path

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "CODE.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        r"""
# Prediction du succes et du revenu des films

Ce notebook suit la roadmap du projet : donnees, nettoyage, analyse exploratoire,
modelisation supervisee, evaluation, PCA/K-means, interpretation et generation
du compte rendu LaTeX.

Deux objectifs sont traites :

1. **Regression** : predire le revenu ajuste d'un film.
2. **Classification** : predire si un film est un succes commercial.

Le point methodologique important est d'eviter la fuite de donnees : la variable
`success` est construite a partir du revenu, donc les variables `revenue`,
`revenue_adj` et `ROI` ne doivent jamais etre utilisees comme entrees du modele.
"""
    ),
    md(
        r"""
## 0. Importation des bibliotheques et configuration

On fixe un `random_state` pour rendre les resultats reproductibles. Les figures
sont sauvegardees automatiquement dans le dossier `figures/`, et les tableaux de
resultats dans `outputs/`.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / "movie.csv"
FIG_DIR = PROJECT_DIR / "figures"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")

print(f"Dossier projet : {PROJECT_DIR}")
print(f"Dataset trouve : {DATA_PATH.exists()} -> {DATA_PATH.name}")
"""
    ),
    md(
        r"""
## 1. Chargement et comprehension des donnees

Le fichier contient des informations issues de films : budget, revenu, popularite,
 genres, votes, duree et annee de sortie. On commence par observer la taille,
les colonnes et quelques lignes.
"""
    ),
    code(
        r"""
raw = pd.read_csv(DATA_PATH)

print(f"Dimensions brutes : {raw.shape[0]:,} lignes x {raw.shape[1]} colonnes")
display(raw.head())

columns_overview = pd.DataFrame({
    "colonne": raw.columns,
    "type_initial": raw.dtypes.astype(str).values,
    "valeurs_manquantes": raw.isna().sum().values,
    "taux_manquant": (raw.isna().mean().values * 100).round(2),
})
display(columns_overview)
"""
    ),
    md(
        r"""
## 2. Diagnostic qualite : valeurs manquantes et zeros

Dans ce dataset, une valeur `0` pour `budget` ou `revenue` ne signifie pas que
le film a vraiment coute ou rapporte 0 : c'est generalement une information
absente. Pour une prediction de revenu, ces lignes doivent etre traitees avec
prudence.
"""
    ),
    code(
        r"""
quality_cols = ["budget", "revenue", "runtime", "budget_adj", "revenue_adj"]
quality = pd.DataFrame({
    "valeurs_manquantes": raw[quality_cols].isna().sum(),
    "zeros": (raw[quality_cols] == 0).sum(),
    "taux_zero_%": ((raw[quality_cols] == 0).mean() * 100).round(2),
})
display(quality)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
missing_plot = raw.isna().sum().sort_values(ascending=False)
missing_plot = missing_plot[missing_plot > 0]
sns.barplot(x=missing_plot.values, y=missing_plot.index, ax=axes[0], color="#4C78A8")
axes[0].set_title("Valeurs manquantes par variable")
axes[0].set_xlabel("Nombre")
axes[0].set_ylabel("")

zero_plot = (raw[quality_cols] == 0).sum().sort_values(ascending=False)
sns.barplot(x=zero_plot.values, y=zero_plot.index, ax=axes[1], color="#F58518")
axes[1].set_title("Zeros structurels")
axes[1].set_xlabel("Nombre")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_missing_zero_counts.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 3. Nettoyage et feature engineering

On applique les decisions suivantes :

- supprimer les doublons sur l'identifiant du film ;
- convertir les colonnes numeriques ;
- utiliser `budget_adj` et `revenue_adj`, car elles corrigent l'effet de l'inflation ;
- supprimer les lignes sans budget/revenu/duree/genre exploitable ;
- creer le ROI, la cible de succes, les informations de date et les variables de genres.

Definition de la cible de classification :

$$
ROI_i = \frac{revenue\_adj_i}{budget\_adj_i}
$$

$$
success_i =
\begin{cases}
1 & \text{si } revenue\_adj_i > 2 \times budget\_adj_i \\
0 & \text{sinon}
\end{cases}
$$
"""
    ),
    code(
        r"""
df = raw.copy()
rows_initial = len(df)

df = df.drop_duplicates(subset="id").copy()
rows_after_duplicates = len(df)

numeric_cols = [
    "popularity", "budget", "revenue", "runtime", "vote_count",
    "vote_average", "release_year", "budget_adj", "revenue_adj",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

df_model = df[
    (df["budget_adj"] > 0)
    & (df["revenue_adj"] > 0)
    & (df["runtime"] > 0)
    & df["genres"].notna()
    & df["release_year"].notna()
].copy()

df_model["roi"] = df_model["revenue_adj"] / df_model["budget_adj"]
df_model["success"] = (df_model["revenue_adj"] > 2 * df_model["budget_adj"]).astype(int)
df_model["log_budget_adj"] = np.log1p(df_model["budget_adj"])
df_model["log_revenue_adj"] = np.log1p(df_model["revenue_adj"])

df_model["genre_list"] = df_model["genres"].str.split("|")
df_model["genre_count"] = df_model["genre_list"].apply(len)
df_model["main_genre"] = df_model["genre_list"].str[0]

df_model["release_month"] = df_model["release_date"].dt.month
df_model["release_month"] = df_model["release_month"].fillna(0).astype(int)

def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    if month in [9, 10, 11]:
        return "autumn"
    return "unknown"

df_model["release_season"] = df_model["release_month"].apply(month_to_season)

all_genres = sorted({genre for genres in df_model["genre_list"] for genre in genres})
genre_features = []
for genre in all_genres:
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", genre).strip("_").lower()
    col = f"genre_{safe_name}"
    df_model[col] = df_model["genre_list"].apply(lambda values, g=genre: int(g in values))
    genre_features.append(col)

cleaning_summary = pd.DataFrame({
    "etape": [
        "lignes initiales",
        "apres suppression doublons id",
        "apres budget/revenu/duree/genre valides",
        "films succes",
        "films non succes",
    ],
    "nombre": [
        rows_initial,
        rows_after_duplicates,
        len(df_model),
        int(df_model["success"].sum()),
        int((1 - df_model["success"]).sum()),
    ],
})
cleaning_summary["proportion_%"] = (cleaning_summary["nombre"] / len(df_model) * 100).round(2)
cleaning_summary.loc[:1, "proportion_%"] = np.nan
display(cleaning_summary)

print(f"Genres detectes : {len(all_genres)}")
print(all_genres)
"""
    ),
    md(
        r"""
## 4. Analyse exploratoire des donnees

L'objectif de l'EDA est de comprendre les ordres de grandeur, les relations entre
budget et revenu, et les differences entre genres. On travaille souvent en
echelle logarithmique parce que les budgets et revenus de films sont tres
asymetriques.
"""
    ),
    code(
        r"""
eda_cols = [
    "budget_adj", "revenue_adj", "roi", "runtime", "popularity",
    "vote_count", "vote_average", "release_year", "genre_count",
]
summary_stats = df_model[eda_cols].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
).T
display(summary_stats)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df_model["log_revenue_adj"], bins=40, kde=True, ax=axes[0], color="#4C78A8")
axes[0].set_title("Distribution de log(1 + revenu ajuste)")
axes[0].set_xlabel("log(1 + revenue_adj)")

sns.scatterplot(
    data=df_model,
    x="log_budget_adj",
    y="log_revenue_adj",
    hue="success",
    palette={0: "#E45756", 1: "#54A24B"},
    alpha=0.65,
    ax=axes[1],
)
axes[1].set_title("Budget ajuste vs revenu ajuste")
axes[1].set_xlabel("log(1 + budget_adj)")
axes[1].set_ylabel("log(1 + revenue_adj)")
axes[1].legend(title="success", loc="lower right")

plt.tight_layout()
plt.savefig(FIG_DIR / "02_budget_revenue_distribution.png", dpi=180, bbox_inches="tight")
plt.show()

genre_rows = []
for genre in all_genres:
    mask = df_model["genre_list"].apply(lambda values, g=genre: g in values)
    subset = df_model.loc[mask]
    genre_rows.append({
        "genre": genre,
        "n": len(subset),
        "success_rate": subset["success"].mean(),
        "median_revenue_musd": subset["revenue_adj"].median() / 1e6,
        "median_roi": subset["roi"].median(),
    })

genre_stats = pd.DataFrame(genre_rows).sort_values("n", ascending=False)
display(genre_stats.head(15))

top_genres = genre_stats.head(12).copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
sns.barplot(data=top_genres, y="genre", x="n", ax=axes[0], color="#4C78A8")
axes[0].set_title("Genres les plus frequents")
axes[0].set_xlabel("Nombre de films")
axes[0].set_ylabel("")

sns.barplot(
    data=top_genres.sort_values("success_rate", ascending=False),
    y="genre",
    x="success_rate",
    ax=axes[1],
    color="#54A24B",
)
axes[1].set_title("Taux de succes par genre frequent")
axes[1].set_xlabel("Taux de succes")
axes[1].set_xlim(0, 1)
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "03_genre_success.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 5. Preparation des variables pour le machine learning

Pour garder une prediction honnete, on construit un modele **pre-sortie** :
il utilise seulement des informations plausiblement connues avant ou au moment
de la sortie : budget, duree, annee, mois/saison et genres.

Variables volontairement exclues du modele principal :

- `revenue`, `revenue_adj`, `roi` : elles definissent directement la cible ;
- `vote_count`, `vote_average` : disponibles apres reaction du public ;
- `popularity` : variable ambigue, souvent mesuree apres exposition du film.

On conserve cependant ces variables pour l'analyse descriptive.
"""
    ),
    code(
        r"""
numeric_features = [
    "budget_adj",
    "runtime",
    "release_year",
    "release_month",
    "genre_count",
] + genre_features

categorical_features = ["main_genre", "release_season"]
feature_columns = numeric_features + categorical_features

X = df_model[feature_columns].copy()
y_reg = df_model["revenue_adj"].copy()
y_clf = df_model["success"].copy()

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X,
    y_reg,
    y_clf,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_clf,
)

def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

print(f"Features utilisees : {len(feature_columns)}")
print(f"Train : {len(X_train):,} lignes | Test : {len(X_test):,} lignes")
print(f"Taux de succes train : {y_clf_train.mean():.3f} | test : {y_clf_test.mean():.3f}")
"""
    ),
    md(
        r"""
## 6. Modelisation de la regression

On compare plusieurs modeles :

- **DummyRegressor** : baseline naive, utile pour savoir si les vrais modeles apprennent quelque chose.
- **Linear Regression** : modele lineaire sans regularisation.
- **LassoCV** : regression lineaire regularisee par penalisation L1.
- **Random Forest** : modele non lineaire de reference.

Comme les revenus sont tres asymetriques, les modeles apprennent
`log(1 + revenue_adj)` via `TransformedTargetRegressor`, puis les predictions
sont retransformees dans l'echelle monetaire.

Metriques :

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2}
$$

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat{y_i})^2}{\sum_i(y_i-\bar{y})^2}
$$

On ajoute aussi le **RMSLE**, plus stable pour les ordres de grandeur :

$$
RMSLE = \sqrt{\frac{1}{n}\sum_i(\log(1+y_i)-\log(1+\hat{y_i}))^2}
$$
"""
    ),
    code(
        r"""
def positive_predictions(values):
    return np.maximum(np.asarray(values, dtype=float), 0)

def rmsle(y_true, y_pred):
    y_pred = positive_predictions(y_pred)
    return root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred))

def regression_metrics(y_true, y_pred):
    y_pred = positive_predictions(y_pred)
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, np.maximum(y_pred, 1)),
    }

def make_log_target_model(estimator):
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

regression_models = {
    "Dummy median": DummyRegressor(strategy="median"),
    "Linear Regression": make_log_target_model(LinearRegression()),
    "LassoCV": make_log_target_model(
        LassoCV(
            alphas=np.logspace(-4, -1, 25),
            cv=5,
            max_iter=50000,
            random_state=RANDOM_STATE,
        )
    ),
    "Random Forest": make_log_target_model(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    ),
}

regression_results = []
fitted_regressors = {}

for name, model in regression_models.items():
    model.fit(X_train, y_reg_train)
    predictions = model.predict(X_test)
    metrics = regression_metrics(y_reg_test, predictions)
    regression_results.append({"model": name, **metrics})
    fitted_regressors[name] = model

regression_metrics_df = pd.DataFrame(regression_results)
regression_metrics_df["RMSE_M_USD"] = regression_metrics_df["RMSE"] / 1e6
regression_metrics_df["MAE_M_USD"] = regression_metrics_df["MAE"] / 1e6
regression_metrics_df = regression_metrics_df.sort_values("RMSLE").reset_index(drop=True)
display(regression_metrics_df)
regression_metrics_df.to_csv(OUT_DIR / "regression_metrics.csv", index=False)

best_reg_name = regression_metrics_df.iloc[0]["model"]
best_reg_model = fitted_regressors[best_reg_name]
best_reg_pred = positive_predictions(best_reg_model.predict(X_test))

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(np.log1p(y_reg_test), np.log1p(best_reg_pred), alpha=0.6, color="#4C78A8")
lims = [
    min(np.log1p(y_reg_test).min(), np.log1p(best_reg_pred).min()),
    max(np.log1p(y_reg_test).max(), np.log1p(best_reg_pred).max()),
]
ax.plot(lims, lims, color="#E45756", linewidth=2)
ax.set_title(f"Regression : observe vs predit ({best_reg_name})")
ax.set_xlabel("log(1 + revenu observe)")
ax.set_ylabel("log(1 + revenu predit)")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_regression_observed_vs_predicted.png", dpi=180, bbox_inches="tight")
plt.show()

print(f"Meilleur modele de regression selon RMSLE : {best_reg_name}")
"""
    ),
    md(
        r"""
### Interpretation des variables de regression

Pour disposer d'une interpretation non lineaire simple, on regarde l'importance
des variables du Random Forest. L'importance n'est pas une preuve causale, mais
elle indique quelles variables reduisent le plus l'erreur dans ce modele.
"""
    ),
    code(
        r"""
rf_reg = fitted_regressors["Random Forest"].regressor_
rf_feature_names = rf_reg.named_steps["preprocess"].get_feature_names_out()
rf_importances = pd.Series(
    rf_reg.named_steps["model"].feature_importances_,
    index=rf_feature_names,
).sort_values(ascending=False)

top_rf_importances = rf_importances.head(15).reset_index()
top_rf_importances.columns = ["feature", "importance"]
display(top_rf_importances)
top_rf_importances.to_csv(OUT_DIR / "regression_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5.4))
sns.barplot(data=top_rf_importances, x="importance", y="feature", ax=ax, color="#72B7B2")
ax.set_title("Importance des variables - Random Forest regression")
ax.set_xlabel("Importance")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_regression_feature_importance.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 7. Modelisation de la classification

On compare :

- **DummyClassifier** : baseline majoritaire ;
- **Logistic Regression** : modele probabiliste lineaire ;
- **KNN** : classification par voisins, sensible a la normalisation ;
- **Random Forest** : modele non lineaire.

Metriques :

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Precision = \frac{TP}{TP + FP}, \quad
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

On surveille aussi la **balanced accuracy**, car un modele qui predit toujours
la classe majoritaire peut obtenir un F1 trompeur si le recall est artificiellement
egal a 1.
"""
    ),
    code(
        r"""
classification_models = {
    "Dummy majority": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ]),
    "KNN": GridSearchCV(
        Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", KNeighborsClassifier()),
        ]),
        param_grid={
            "model__n_neighbors": [5, 9, 15, 21, 31, 45],
            "model__weights": ["uniform", "distance"],
        },
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    ),
    "Random Forest": Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ]),
}

def classification_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics

classification_results = []
fitted_classifiers = {}

for name, model in classification_models.items():
    model.fit(X_train, y_clf_train)
    pred = model.predict(X_test)
    proba = None
    scoring_model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
    if hasattr(scoring_model, "predict_proba"):
        proba = scoring_model.predict_proba(X_test)[:, 1]
    row = {"model": name, **classification_metrics(y_clf_test, pred, proba)}
    if isinstance(model, GridSearchCV):
        row["best_params"] = str(model.best_params_)
    else:
        row["best_params"] = ""
    classification_results.append(row)
    fitted_classifiers[name] = model

classification_metrics_df = pd.DataFrame(classification_results)
classification_metrics_df["selection_score"] = np.where(
    classification_metrics_df["model"].eq("Dummy majority"),
    -np.inf,
    classification_metrics_df["balanced_accuracy"],
)
classification_metrics_df = classification_metrics_df.sort_values(
    ["selection_score", "roc_auc", "f1"],
    ascending=False,
).reset_index(drop=True)
display(classification_metrics_df.drop(columns=["selection_score"]))
classification_metrics_df.drop(columns=["selection_score"]).to_csv(OUT_DIR / "classification_metrics.csv", index=False)

best_clf_name = classification_metrics_df[classification_metrics_df["model"] != "Dummy majority"].iloc[0]["model"]
best_clf_raw = fitted_classifiers[best_clf_name]
best_clf_model = best_clf_raw.best_estimator_ if isinstance(best_clf_raw, GridSearchCV) else best_clf_raw
best_clf_pred = best_clf_model.predict(X_test)

cm = confusion_matrix(y_clf_test, best_clf_pred)
fig, ax = plt.subplots(figsize=(5.8, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["non succes", "succes"],
    yticklabels=["non succes", "succes"],
    ax=ax,
)
ax.set_title(f"Matrice de confusion - {best_clf_name}")
ax.set_xlabel("Prediction")
ax.set_ylabel("Observation")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_classification_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()

print(f"Meilleur modele supervise de classification selon balanced accuracy : {best_clf_name}")
"""
    ),
    md(
        r"""
### Interpretation de la classification

La regression logistique donne des coefficients interpretables : un coefficient
positif augmente le log-odds de succes, un coefficient negatif le diminue,
toutes choses egales par ailleurs.

$$
P(success=1 \mid x) = \sigma(\beta_0 + x^\top\beta)
$$
"""
    ),
    code(
        r"""
logit = fitted_classifiers["Logistic Regression"]
logit_feature_names = logit.named_steps["preprocess"].get_feature_names_out()
logit_coefs = pd.Series(
    logit.named_steps["model"].coef_[0],
    index=logit_feature_names,
).sort_values()

coef_plot = pd.concat([logit_coefs.head(10), logit_coefs.tail(10)])
coef_df = coef_plot.reset_index()
coef_df.columns = ["feature", "coefficient"]
display(coef_df)
coef_df.to_csv(OUT_DIR / "classification_logistic_coefficients.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6.2))
colors = ["#E45756" if value < 0 else "#54A24B" for value in coef_df["coefficient"]]
sns.barplot(data=coef_df, x="coefficient", y="feature", palette=colors, ax=ax)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Coefficients principaux - regression logistique")
ax.set_xlabel("Coefficient")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_logistic_coefficients.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    md(
        r"""
## 8. PCA et K-means

La PCA projette les films dans un espace 2D qui conserve autant de variance que
possible :

$$
w_1 = \arg\max_{\|w\|=1} Var(Xw)
$$

K-means cherche ensuite des groupes en minimisant l'inertie intra-cluster :

$$
\min_{C_1,\dots,C_k} \sum_{j=1}^{k}\sum_{x_i \in C_j}\|x_i-\mu_j\|^2
$$

Ce n'est pas un modele predictif supervise : il sert a explorer les profils de
films.
"""
    ),
    code(
        r"""
cluster_preprocessor = make_preprocessor()
X_prepared = cluster_preprocessor.fit_transform(X)

silhouette_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_prepared)
    silhouette_rows.append({"k": k, "silhouette": silhouette_score(X_prepared, labels)})

silhouette_df = pd.DataFrame(silhouette_rows)
display(silhouette_df)
silhouette_df.to_csv(OUT_DIR / "kmeans_silhouette.csv", index=False)

chosen_k = int(silhouette_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
kmeans = KMeans(n_clusters=chosen_k, n_init=50, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(X_prepared)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(X_prepared)

clustered = df_model.copy()
clustered["cluster"] = clusters
clustered["pca_1"] = pca_coords[:, 0]
clustered["pca_2"] = pca_coords[:, 1]

def dominant_genre(series):
    counts = series.value_counts()
    return counts.index[0] if len(counts) else "NA"

cluster_profile = clustered.groupby("cluster").agg(
    n=("id", "count"),
    success_rate=("success", "mean"),
    median_budget_musd=("budget_adj", lambda s: s.median() / 1e6),
    median_revenue_musd=("revenue_adj", lambda s: s.median() / 1e6),
    median_roi=("roi", "median"),
    dominant_main_genre=("main_genre", dominant_genre),
).reset_index()

display(cluster_profile)
cluster_profile.to_csv(OUT_DIR / "cluster_profile.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
sns.lineplot(data=silhouette_df, x="k", y="silhouette", marker="o", ax=axes[0], color="#4C78A8")
axes[0].set_title("Choix de k par silhouette")
axes[0].set_xlabel("Nombre de clusters")
axes[0].set_ylabel("Silhouette")

sns.scatterplot(
    data=clustered,
    x="pca_1",
    y="pca_2",
    hue="cluster",
    style="success",
    palette="tab10",
    alpha=0.75,
    ax=axes[1],
)
axes[1].set_title(f"PCA 2D + K-means (k={chosen_k})")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
axes[1].legend(loc="best", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "08_pca_kmeans.png", dpi=180, bbox_inches="tight")
plt.show()

print(f"k retenu automatiquement par silhouette : {chosen_k}")
print(f"Variance expliquee par les deux premieres composantes : {pca.explained_variance_ratio_.sum():.3f}")
"""
    ),
    md(
        r"""
## 9. Synthese des resultats exportes

Cette cellule rassemble les indicateurs importants dans un fichier JSON. Il sera
utilise pour le compte rendu LaTeX et facilite la verification du projet.
"""
    ),
    code(
        r"""
def to_float(value):
    if pd.isna(value):
        return None
    return float(value)

summary_payload = {
    "n_raw": int(len(raw)),
    "n_clean": int(len(df_model)),
    "n_features": int(len(feature_columns)),
    "success_rate": float(df_model["success"].mean()),
    "best_regression_model": str(best_reg_name),
    "best_regression_rmsle": to_float(regression_metrics_df.iloc[0]["RMSLE"]),
    "best_regression_rmse_musd": to_float(regression_metrics_df.iloc[0]["RMSE_M_USD"]),
    "best_regression_r2": to_float(regression_metrics_df.iloc[0]["R2"]),
    "best_classification_model": str(best_clf_name),
    "best_classification_accuracy": to_float(classification_metrics_df.iloc[0]["accuracy"]),
    "best_classification_balanced_accuracy": to_float(classification_metrics_df.iloc[0]["balanced_accuracy"]),
    "best_classification_precision": to_float(classification_metrics_df.iloc[0]["precision"]),
    "best_classification_recall": to_float(classification_metrics_df.iloc[0]["recall"]),
    "best_classification_f1": to_float(classification_metrics_df.iloc[0]["f1"]),
    "best_classification_roc_auc": to_float(classification_metrics_df.iloc[0]["roc_auc"]),
    "chosen_k": int(chosen_k),
    "pca_variance_2d": float(pca.explained_variance_ratio_.sum()),
}

with open(OUT_DIR / "summary_results.json", "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2, ensure_ascii=False)

display(summary_payload)
"""
    ),
    md(
        r"""
## 10. Generation du compte rendu LaTeX

Le rapport est produit automatiquement a partir des resultats du notebook. Il se
trouve dans `rapport_movie_success.tex`.
"""
    ),
    code(
        r"""
def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)

def latex_table(df, columns, headers, formats=None):
    formats = formats or {}
    align = "l" + "r" * (len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if col in formats and pd.notna(value):
                value = formats[col](value)
            values.append(latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)

reg_report = regression_metrics_df[["model", "RMSE_M_USD", "MAE_M_USD", "R2", "RMSLE", "MAPE"]].copy()
clf_report = classification_metrics_df[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
cluster_report = cluster_profile.copy()

reg_table = latex_table(
    reg_report,
    ["model", "RMSE_M_USD", "MAE_M_USD", "R2", "RMSLE", "MAPE"],
    ["Modele", "RMSE (M USD)", "MAE (M USD)", "R2", "RMSLE", "MAPE"],
    {
        "RMSE_M_USD": lambda x: f"{x:.2f}",
        "MAE_M_USD": lambda x: f"{x:.2f}",
        "R2": lambda x: f"{x:.3f}",
        "RMSLE": lambda x: f"{x:.3f}",
        "MAPE": lambda x: f"{x:.3f}",
    },
)

clf_table = latex_table(
    clf_report,
    ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"],
    ["Modele", "Accuracy", "Balanced acc.", "Precision", "Recall", "F1", "ROC-AUC"],
    {
        "accuracy": lambda x: f"{x:.3f}",
        "balanced_accuracy": lambda x: f"{x:.3f}",
        "precision": lambda x: f"{x:.3f}",
        "recall": lambda x: f"{x:.3f}",
        "f1": lambda x: f"{x:.3f}",
        "roc_auc": lambda x: f"{x:.3f}" if pd.notna(x) else "NA",
    },
)

cluster_table = latex_table(
    cluster_report,
    ["cluster", "n", "success_rate", "median_budget_musd", "median_revenue_musd", "median_roi", "dominant_main_genre"],
    ["Cluster", "n", "Succes", "Budget median", "Revenu median", "ROI median", "Genre dominant"],
    {
        "success_rate": lambda x: f"{x:.3f}",
        "median_budget_musd": lambda x: f"{x:.2f}",
        "median_revenue_musd": lambda x: f"{x:.2f}",
        "median_roi": lambda x: f"{x:.2f}",
    },
)

report_path = PROJECT_DIR / "rapport_movie_success.tex"

report = rf'''
\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[french]{{babel}}
\usepackage{{amsmath, amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{margin=2.3cm}}

\title{{Prediction du succes et du revenu des films}}
\author{{Projet Machine Learning}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Objectif}}

Ce projet construit un pipeline complet de data science pour predire deux grandeurs :
le revenu ajuste d'un film, par regression, et son succes commercial, par classification.
Le succes est defini par un seuil de rentabilite simple : un film est considere comme un succes si son revenu ajuste depasse deux fois son budget ajuste.

\section{{Donnees et nettoyage}}

Le dataset initial contient {len(raw):,} films et {raw.shape[1]} variables. Apres suppression des doublons et des observations dont le budget, le revenu, la duree ou le genre sont absents ou nuls, l'echantillon modelisable contient {len(df_model):,} films.
Le taux de succes observe dans cet echantillon est de {df_model["success"].mean():.3f}.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{figures/01_missing_zero_counts.png}}
\caption{{Diagnostic des valeurs manquantes et des zeros structurels.}}
\end{{figure}}

\section{{Feature engineering}}

Les variables monetaires ajustees sont utilisees pour limiter le biais lie a l'inflation :
\[
ROI_i = \frac{{revenue\_adj_i}}{{budget\_adj_i}}.
\]
La cible de classification est :
\[
success_i =
\begin{{cases}}
1, & \text{{si }} revenue\_adj_i > 2 \times budget\_adj_i,\\
0, & \text{{sinon.}}
\end{{cases}}
\]

Pour eviter la fuite de donnees, les variables \texttt{{revenue}}, \texttt{{revenue\_adj}} et \texttt{{roi}} ne sont pas utilisees comme entrees. Le modele principal utilise {len(feature_columns)} variables construites a partir du budget, de la duree, de l'annee, du mois, de la saison et des genres.

\section{{Analyse exploratoire}}

Les revenus et budgets presentent une distribution tres asymetrique. L'echelle logarithmique rend la relation budget--revenu plus lisible et montre que le budget est informatif, mais insuffisant pour expliquer seul le succes.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{figures/02_budget_revenue_distribution.png}}
\caption{{Distribution du revenu ajuste et relation budget--revenu en echelle logarithmique.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{figures/03_genre_success.png}}
\caption{{Frequence et taux de succes des principaux genres.}}
\end{{figure}}

\section{{Methodologie algorithmique}}

Pour la regression, les modeles apprennent la cible transformee :
\[
z_i = \log(1 + revenue\_adj_i).
\]
La prediction finale est obtenue par transformation inverse :
\[
\hat{{y}}_i = \exp(\hat{{z}}_i) - 1.
\]

La regression lineaire estime :
\[
\hat{{y}} = \beta_0 + X\beta,
\]
tandis que le Lasso resout :
\[
\min_\beta \frac{{1}}{{2n}}\|y-X\beta\|_2^2 + \lambda\|\beta\|_1.
\]

Pour la classification, la regression logistique modelise :
\[
P(success=1 \mid x)=\sigma(\beta_0+x^\top\beta),
\quad
\sigma(t)=\frac{{1}}{{1+\exp(-t)}}.
\]
Le KNN attribue la classe majoritaire parmi les voisins les plus proches apres normalisation des variables numeriques.

\section{{Resultats de regression}}

\begin{{table}}[H]
\centering
\small
\caption{{Performances des modeles de regression sur le jeu de test.}}
{reg_table}
\end{{table}}

Le meilleur modele selon le RMSLE est \textbf{{{latex_escape(best_reg_name)}}}, avec un RMSLE de {summary_payload["best_regression_rmsle"]:.3f}, un RMSE de {summary_payload["best_regression_rmse_musd"]:.2f} millions USD et un $R^2$ de {summary_payload["best_regression_r2"]:.3f}.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.68\linewidth]{{figures/04_regression_observed_vs_predicted.png}}
\caption{{Revenus observes et predits par le meilleur modele de regression.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{figures/05_regression_feature_importance.png}}
\caption{{Importance des variables dans le Random Forest de regression.}}
\end{{figure}}

\section{{Resultats de classification}}

\begin{{table}}[H]
\centering
\small
\caption{{Performances des modeles de classification sur le jeu de test.}}
{clf_table}
\end{{table}}

Le meilleur modele supervise selon la balanced accuracy est \textbf{{{latex_escape(best_clf_name)}}}. Il obtient une accuracy de {summary_payload["best_classification_accuracy"]:.3f}, une balanced accuracy de {summary_payload["best_classification_balanced_accuracy"]:.3f}, une precision de {summary_payload["best_classification_precision"]:.3f}, un recall de {summary_payload["best_classification_recall"]:.3f}, un F1 de {summary_payload["best_classification_f1"]:.3f} et un ROC-AUC de {summary_payload["best_classification_roc_auc"]:.3f}. Le baseline majoritaire est conserve dans le tableau, mais il n'est pas retenu comme meilleur modele car il predit une seule classe.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.62\linewidth]{{figures/06_classification_confusion_matrix.png}}
\caption{{Matrice de confusion du meilleur modele de classification.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{figures/07_logistic_coefficients.png}}
\caption{{Coefficients les plus influents de la regression logistique.}}
\end{{figure}}

\section{{PCA et K-means}}

La PCA et K-means servent a explorer la structure des films. Le nombre de clusters retenu par silhouette est $k={chosen_k}$.
Les deux premieres composantes principales expliquent {pca.explained_variance_ratio_.sum():.3f} de la variance transformee.

\begin{{table}}[H]
\centering
\small
\caption{{Profil des clusters K-means. Les montants medians sont en millions USD.}}
{cluster_table}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{figures/08_pca_kmeans.png}}
\caption{{Selection de k par silhouette et projection PCA des clusters.}}
\end{{figure}}

\section{{Conclusion}}

Le projet montre qu'il est possible de construire une prediction exploitable du revenu et du succes d'un film a partir de variables simples connues avant ou au moment de la sortie. Le budget ajuste est une variable centrale, mais les genres, la duree et la temporalite apportent aussi de l'information.

La regression reste difficile car les revenus de films sont extremement disperses : quelques blockbusters dominent les ordres de grandeur. La classification est plus stable car elle transforme le probleme en decision binaire autour d'un seuil de rentabilite.

La principale limite vient de la disponibilite temporelle des variables. Pour un vrai systeme de prediction avant sortie, il faudrait enrichir le dataset avec des variables disponibles ex ante : franchise, studio, taille de distribution, pays, date exacte de sortie, concurrence au box-office, acteurs, realisateur et budget marketing.

\end{{document}}
'''

report_path.write_text(report, encoding="utf-8")
print(f"Rapport LaTeX genere : {report_path}")
"""
    ),
    md(
        r"""
## 11. Livrables

Le notebook a produit :

- les figures dans `figures/` ;
- les tableaux de metriques dans `outputs/` ;
- le rapport LaTeX `rapport_movie_success.tex`.
"""
    ),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "movie-success-venv",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
)

nbf.write(nb, NOTEBOOK_PATH)
print(f"Notebook written to {NOTEBOOK_PATH}")
