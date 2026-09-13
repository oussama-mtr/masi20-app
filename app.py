"""
================================================================================
 APPLICATION STREAMLIT - RECOMMANDATIONS PORTEFEUILLE ML MASI20
 (reprend la logique de pipeline_ml_masi20_v3.py, sans feuille Macro)
================================================================================
Lancer en local :  streamlit run app.py
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    PACKAGES_OK = True
except ImportError as e:
    PACKAGES_OK = False
    IMPORT_ERROR = str(e)

st.set_page_config(page_title="Recommandations Portefeuille ML - MASI20", layout="wide")

# ==============================================================================
# PARAMETRES PAR DEFAUT (modifiables dans la barre laterale)
# ==============================================================================
TARGET_COL = "Rendement M+1 (Cible)"
FEATURES_TECHNIQUE = [
    "Momentum_3M", "Mom_6M", "Mom_12M",
    "MA3_ratio", "MA6_ratio", "MA12_ratio",
    "RSI_6M", "RSI_12M",
    "EMA_fast3_ratio", "EMA_slow6_ratio",
    "MACD_line_pct", "MACD_signal_pct", "MACD_hist_pct",
    "RollMax12_ratio", "Drawdown", "MaxDD_12M",
]
FEATURES_FOND = ["ROA %", "ROE %", "Marge Nette %", "Marge EBITDA %",
                  "Croissance CA %", "P/E", "P/B", "Croissance RNPG", "Croissance EBITDA"]


def construire_modeles():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.02, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1, importance_type="gain"),
        "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, random_seed=42, verbose=0),
        "SVR": SVR(kernel="rbf", C=1, epsilon=0.001, gamma="scale"),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.15),
    }


MODELES_ARBRES = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
MODELES_A_SCALER = ["SVR", "MLP"]

# ==============================================================================
# PREPARATION DES DONNEES (mise en cache : ne se relance que si le fichier change)
# ==============================================================================
@st.cache_data(show_spinner="Preparation des donnees...")
def preparer_donnees(fichier_bytes):
    xls = pd.ExcelFile(io.BytesIO(fichier_bytes))
    df = pd.read_excel(xls, "Technique")
    df_fond = pd.read_excel(xls, "Fondamental")

    for d in (df, df_fond):
        d.columns = d.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # Nettoyage Fondamental : retirer les lignes de notes/sources (pas des donnees ticker)
    motif_periode_valide = df_fond["Période"].astype(str).str.match(r"^S[12]\s\d{4}$", na=False)
    df_fond = df_fond[motif_periode_valide].reset_index(drop=True)

    df["Ticker"] = df["Ticker"].ffill()
    tickers_source = sorted(df["Ticker"].unique())

    # Detection automatique du bug de date (jour = mois reel) au lieu de le supposer
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    jours, mois = df["Date"].dt.day, df["Date"].dt.month
    bug_detecte = (mois.nunique() == 1) and jours.between(1, 12).all() and jours.nunique() > 1
    if bug_detecte:
        df["Date"] = pd.to_datetime(dict(year=df["Date"].dt.year, month=df["Date"].dt.day, day=1))
    else:
        df["Date"] = df["Date"].values.astype("datetime64[M]")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Annee"] = df["Date"].dt.year
    df["Mois"] = df["Date"].dt.month

    # Cible
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = df.groupby("Ticker")["Price"].shift(-1) / df["Price"] - 1

    # Normalisation technique (ratio / prix)
    price_safe = df["Price"].replace(0, np.nan)
    df["MA3_ratio"] = df["Moyenne Mobile_3M"] / price_safe
    df["MA6_ratio"] = df["MM_6M"] / price_safe
    df["MA12_ratio"] = df["MM12M"] / price_safe
    df["EMA_fast3_ratio"] = df["EMA_fast3"] / price_safe
    df["EMA_slow6_ratio"] = df["EMA_slow6"] / price_safe
    df["MACD_line_pct"] = df["MACD_line"] / price_safe
    df["MACD_signal_pct"] = df["MACD_signal3"] / price_safe
    df["MACD_hist_pct"] = df["MACD_hist"] / price_safe
    df["RollMax12_ratio"] = df["RollMax_12"] / price_safe

    # Fondamental (semestriel, decale d'une periode pour eviter le look-ahead)
    def parse_semestre(s):
        sem, annee = s.split()
        return int(annee), sem

    def semestre_suivant(annee, sem, n=1):
        total = annee * 2 + (0 if sem == "S1" else 1) + n
        return total // 2, ("S1" if total % 2 == 0 else "S2")

    df_fond[["Annee_pub", "Sem_pub"]] = df_fond["Période"].apply(lambda s: pd.Series(parse_semestre(s)))
    df_fond[["Annee_use", "Sem_use"]] = df_fond.apply(lambda r: pd.Series(semestre_suivant(r["Annee_pub"], r["Sem_pub"])), axis=1)
    df["Semestre"] = np.where(df["Mois"] <= 6, "S1", "S2")

    df = df.merge(
        df_fond[["Ticker", "Annee_use", "Sem_use"] + FEATURES_FOND],
        left_on=["Ticker", "Annee", "Semestre"], right_on=["Ticker", "Annee_use", "Sem_use"], how="left",
    ).drop(columns=["Annee_use", "Sem_use"])

    all_features = FEATURES_TECHNIQUE + FEATURES_FOND
    required_cols = all_features + [TARGET_COL]
    df_ml = df.dropna(subset=required_cols).copy()

    tickers_fond = sorted(df_fond["Ticker"].unique())
    tickers_ml = sorted(df_ml["Ticker"].unique())
    tickers_exclus = sorted(set(tickers_source) - set(tickers_ml))

    return df_ml, all_features, tickers_source, tickers_fond, tickers_exclus

# ==============================================================================
# SELECTION DU MEILLEUR MODELE (train <= annee_train_max, validation = annee_validation)
# ==============================================================================
@st.cache_data(show_spinner="Comparaison des modeles sur la validation...")
def selectionner_modele(df_ml, all_features, annee_train_max, annee_validation):
    train_mask = df_ml["Annee"] <= annee_train_max
    val_mask = df_ml["Annee"] == annee_validation

    X_train, y_train = df_ml.loc[train_mask, all_features], df_ml.loc[train_mask, TARGET_COL]
    X_val, y_val = df_ml.loc[val_mask, all_features], df_ml.loc[val_mask, TARGET_COL]

    scaler = StandardScaler().fit(X_train)
    X_train_sc, X_val_sc = scaler.transform(X_train), scaler.transform(X_val)

    resultats, modeles_entraines = [], {}
    for nom, modele in construire_modeles().items():
        if nom in MODELES_A_SCALER:
            modele.fit(X_train_sc, y_train)
            y_pred = modele.predict(X_val_sc)
        else:
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_val)

        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        correlation = float(np.corrcoef(y_val, y_pred)[0, 1]) if len(y_val) > 1 else np.nan
        dir_acc = float((np.where(y_val >= 0, 1, -1) == np.where(y_pred >= 0, 1, -1)).mean())

        resultats.append({"Modele": nom, "RMSE": rmse, "MAE": mae, "R2": r2,
                           "Correlation": correlation, "Directional_Accuracy": dir_acc})
        modeles_entraines[nom] = modele

    df_resultats = pd.DataFrame(resultats).set_index("Modele")
    best_name = df_resultats["Correlation"].idxmax()

    # Feature importance (modeles entraines sur train uniquement)
    lignes_fimp = []
    for nom in MODELES_ARBRES:
        raw_imp = np.asarray(modeles_entraines[nom].feature_importances_, dtype=float)
        imp_pct = 100 * raw_imp / raw_imp.sum() if raw_imp.sum() > 0 else np.zeros_like(raw_imp)
        for feat, val in zip(all_features, imp_pct):
            lignes_fimp.append({"Modele": nom, "Feature": feat, "Importance_%": val})
    df_fimp = pd.DataFrame(lignes_fimp)
    df_fimp_moyenne = df_fimp.groupby("Feature")["Importance_%"].mean().sort_values(ascending=False)

    return best_name, df_resultats, df_fimp_moyenne


# ==============================================================================
# BACKTEST WALK-FORWARD (equipondere, Top N)
# ==============================================================================
def allocation_equiponderee(n):
    return np.full(n, 1.0 / n)


@st.cache_data(show_spinner="Backtest walk-forward en cours (reentrainement mensuel)...")
def backtester_portefeuille(df_ml, all_features, best_name, annee_test, top_n, capital_initial, taux_sans_risque):
    mois_investissement = pd.date_range(f"{annee_test}-01-01", f"{annee_test}-12-01", freq="MS")
    mois_decision = mois_investissement - pd.DateOffset(months=1)

    lignes_allocation, lignes_perf, lignes_pred = [], [], []
    valeur = capital_initial

    for date_decision, date_investissement in zip(mois_decision, mois_investissement):
        decision_rows = df_ml[df_ml["Date"] == date_decision].copy()
        if decision_rows.empty:
            continue
        limite_train = date_decision - pd.DateOffset(months=1)
        pool = df_ml[df_ml["Date"] <= limite_train]
        if len(pool) < 30:
            continue

        X_pool, y_pool = pool[all_features], pool[TARGET_COL]
        X_decision = decision_rows[all_features]

        modele_mois = clone(construire_modeles()[best_name])
        if best_name in MODELES_A_SCALER:
            scaler_mois = StandardScaler().fit(X_pool)
            modele_mois.fit(scaler_mois.transform(X_pool), y_pool)
            y_pred = modele_mois.predict(scaler_mois.transform(X_decision))
        else:
            modele_mois.fit(X_pool, y_pool)
            y_pred = modele_mois.predict(X_decision)

        decision_rows = decision_rows.assign(Rendement_predit=y_pred).sort_values("Rendement_predit", ascending=False)

        for _, r in decision_rows.iterrows():
            lignes_pred.append({"Ticker": r["Ticker"], "Date_decision": date_decision,
                                 "Date_investissement": date_investissement,
                                 "Rendement_M1_reel": r[TARGET_COL], "Rendement_M1_predit": r["Rendement_predit"]})

        if len(decision_rows) < top_n:
            continue

        topn = decision_rows.head(top_n).reset_index(drop=True)
        poids = allocation_equiponderee(top_n)

        capital_debut = valeur
        rendement_reel_pf = float(np.dot(poids, topn[TARGET_COL].values))
        rendement_predit_pf = float(np.dot(poids, topn["Rendement_predit"].values))

        for i in range(top_n):
            lignes_allocation.append({
                "Date_decision": date_decision, "Date_investissement": date_investissement,
                "Ticker": topn.loc[i, "Ticker"], "Rang": i + 1,
                "Rendement_predit_M1": topn.loc[i, "Rendement_predit"], "Allocation": poids[i],
                "Capital_alloue_MAD": capital_debut * poids[i], "Rendement_reel_M1": topn.loc[i, TARGET_COL],
                "Contribution_portefeuille": poids[i] * topn.loc[i, TARGET_COL], "Modele_utilise": best_name,
            })

        valeur = capital_debut * (1 + rendement_reel_pf)
        lignes_perf.append({"Date_decision": date_decision, "Date_investissement": date_investissement,
                             "Capital_debut_MAD": capital_debut, "Rendement_predit_portefeuille": rendement_predit_pf,
                             "Rendement_reel_portefeuille": rendement_reel_pf,
                             "Gain_perte_MAD": capital_debut * rendement_reel_pf, "Valeur_portefeuille_MAD": valeur})

    df_allocation = pd.DataFrame(lignes_allocation)
    df_perf = pd.DataFrame(lignes_perf)
    df_pred = pd.DataFrame(lignes_pred)
    if len(df_perf):
        df_perf["Rendement_cumule"] = df_perf["Valeur_portefeuille_MAD"] / capital_initial - 1

    metriques = {}
    if len(df_perf):
        rm = df_perf["Rendement_reel_portefeuille"].values
        courbe = np.concatenate([[capital_initial], df_perf["Valeur_portefeuille_MAD"].values])
        sommet = np.maximum.accumulate(courbe)
        dd = courbe / sommet - 1
        ecart_type = np.std(rm, ddof=1) if len(rm) > 1 else np.nan
        sharpe = ((np.mean(rm) - taux_sans_risque / 12) / ecart_type * np.sqrt(12)) if ecart_type and ecart_type > 0 else np.nan
        metriques = {
            "Capital_final_MAD": valeur,
            "Rendement_cumule": valeur / capital_initial - 1,
            "Volatilite_annualisee": float(ecart_type * np.sqrt(12)) if ecart_type == ecart_type else np.nan,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": float(dd.min()),
            "Mois_positifs": int((rm > 0).sum()),
            "Mois_negatifs": int((rm < 0).sum()),
        }

    return df_allocation, df_perf, df_pred, metriques

# ==============================================================================
# RECOMMANDATION ACTUELLE (derniere observation disponible par ticker)
# ==============================================================================
def recommandation_actuelle(df_ml, all_features, best_name, top_n):
    modele = clone(construire_modeles()[best_name])
    if best_name in MODELES_A_SCALER:
        scaler = StandardScaler().fit(df_ml[all_features])
        modele.fit(scaler.transform(df_ml[all_features]), df_ml[TARGET_COL])
    else:
        modele.fit(df_ml[all_features], df_ml[TARGET_COL])

    dernieres_lignes = df_ml.sort_values("Date").groupby("Ticker").tail(1).copy()
    X_dernier = dernieres_lignes[all_features]
    X_dernier_t = scaler.transform(X_dernier) if best_name in MODELES_A_SCALER else X_dernier
    dernieres_lignes["Rendement_predit"] = modele.predict(X_dernier_t)
    dernieres_lignes = dernieres_lignes.sort_values("Rendement_predit", ascending=False)

    top = dernieres_lignes.head(top_n).reset_index(drop=True)
    top["Allocation"] = 1.0 / top_n
    return top[["Ticker", "Date", "Rendement_predit", "Allocation"]]


# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================
st.title("Recommandations de portefeuille ML - Actions MASI20")
st.caption("Cette application execute le pipeline (donnees -> modeles -> backtest walk-forward -> recommandation) "
           "a chaque nouveau fichier charge, sans passer par Jupyter.")

if not PACKAGES_OK:
    st.error(f"Package manquant : {IMPORT_ERROR}. Ajoutez-le a requirements.txt.")
    st.stop()

with st.sidebar:
    st.header("1. Donnees")
    fichier = st.file_uploader("Fichier Excel (feuilles 'Technique' et 'Fondamental')", type=["xlsx"])

    st.header("2. Parametres")
    top_n = st.slider("Nombre d'actions selectionnees (Top N)", 3, 15, 10)
    capital_initial = st.number_input("Capital initial (MAD)", value=1_000_000, step=100_000)
    taux_sans_risque = st.number_input("Taux sans risque annuel (%)", value=0.0, step=0.5) / 100

    st.header("3. Periodes")
    annee_train_max = st.number_input("Training jusqu'a (inclus)", value=2023, step=1)
    annee_validation = st.number_input("Annee de validation (selection du modele + 1er backtest)", value=2024, step=1)
    annee_test = st.number_input("Annee de test final (backtest hors-echantillon)", value=2025, step=1)
    st.caption("Le modele final (onglet Prevision) est reentraine sur TOUTES les annees disponibles "
               "(training + validation + test), une fois ces 3 etapes terminees.")

    lancer = st.button("Lancer l'analyse", type="primary", disabled=fichier is None)

if fichier is None:
    st.info("Chargez un fichier Excel dans la barre laterale pour commencer.")
    st.stop()

if not lancer and "resultats_prets" not in st.session_state:
    st.info("Reglez les parametres puis cliquez sur 'Lancer l'analyse'.")
    st.stop()

if lancer:
    st.session_state["resultats_prets"] = True

# ---- Pipeline ----
df_ml, all_features, tickers_source, tickers_fond, tickers_exclus = preparer_donnees(fichier.getvalue())
best_name, df_resultats, df_fimp_moyenne = selectionner_modele(df_ml, all_features, annee_train_max, annee_validation)

# Backtest walk-forward SEPARE sur la validation (reporting) et sur le test final (hors-echantillon)
df_alloc_val, df_perf_val, df_pred_val, metriques_val = backtester_portefeuille(
    df_ml, all_features, best_name, annee_validation, top_n, capital_initial, taux_sans_risque
)
df_alloc_test, df_perf_test, df_pred_test, metriques_test = backtester_portefeuille(
    df_ml, all_features, best_name, annee_test, top_n, capital_initial, taux_sans_risque
)

# Recommandation / prevision : modele reentraine sur TOUT l'historique disponible (train+validation+test)
recommandation = recommandation_actuelle(df_ml, all_features, best_name, top_n)
annee_prevision = int(df_ml["Annee"].max()) + 1
derniere_obs = df_ml["Date"].max().date()

tickers_test = sorted(df_ml.loc[df_ml["Annee"] == annee_test, "Ticker"].unique())

# ---- Affichage ----
onglet_reco, onglet_val, onglet_test, onglet_modeles, onglet_diag = st.tabs(
    [f"Prevision {annee_prevision}", f"Backtest Validation {annee_validation}",
     f"Backtest Test {annee_test}", "Modeles & features", "Diagnostic donnees"]
)

with onglet_reco:
    st.subheader(f"Top {top_n} recommande a partir de la derniere observation disponible ({derniere_obs})")
    st.caption(f"Modele final reentraine sur TOUT l'historique disponible ({int(df_ml['Annee'].min())}-{int(df_ml['Annee'].max())}). "
               f"Modele choisi : {best_name} (selectionne sur la validation {annee_validation}, jamais sur le test {annee_test}).")
    aff = recommandation.copy()
    aff["Rendement_predit"] = (aff["Rendement_predit"] * 100).round(2).astype(str) + " %"
    aff["Allocation"] = (aff["Allocation"] * 100).round(1).astype(str) + " %"
    st.dataframe(aff, use_container_width=True, hide_index=True)
    st.warning(f"Ceci est une PREVISION prospective pour {annee_prevision} : aucun rendement reel n'existe encore "
               "pour la comparer. Ce n'est pas un backtest, et ce n'est pas un conseil en investissement.")

def afficher_backtest(df_alloc, df_perf, metriques, annee_cible, capital_initial):
    st.subheader(f"Performance simulee sur {annee_cible} (walk-forward, equipondere, Top {top_n})")
    if metriques:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital final", f"{metriques['Capital_final_MAD']:,.0f} MAD")
        c2.metric("Rendement cumule", f"{metriques['Rendement_cumule']:.1%}")
        c3.metric("Sharpe Ratio", f"{metriques['Sharpe_Ratio']:.2f}")
        c4.metric("Max Drawdown", f"{metriques['Max_Drawdown']:.1%}")
        if len(df_perf) < 12:
            st.info(f"Seulement {len(df_perf)}/12 mois disponibles pour {annee_cible} - "
                    "verifiez que votre fichier couvre bien toute l'annee.")

        courbe = pd.concat([
            pd.DataFrame({"Date": [df_perf['Date_decision'].iloc[0]], "Valeur": [capital_initial]}),
            df_perf[["Date_investissement", "Valeur_portefeuille_MAD"]].rename(
                columns={"Date_investissement": "Date", "Valeur_portefeuille_MAD": "Valeur"})
        ])
        st.line_chart(courbe.set_index("Date"))
        st.dataframe(df_alloc, use_container_width=True, hide_index=True)
    else:
        st.warning("Backtest impossible avec ces parametres (pas assez de donnees pour cette annee).")

with onglet_val:
    st.caption("Sert a choisir le meilleur modele ET donne une 1ere estimation de performance - "
               "a considerer comme du reporting complementaire, pas la preuve finale.")
    afficher_backtest(df_alloc_val, df_perf_val, metriques_val, annee_validation, capital_initial)

with onglet_test:
    st.caption(f"Backtest hors-echantillon : le modele {best_name} n'a jamais ete choisi ni ajuste sur {annee_test}.")
    afficher_backtest(df_alloc_test, df_perf_test, metriques_test, annee_test, capital_initial)

with onglet_modeles:
    st.subheader(f"Comparaison des modeles (validation {annee_validation} uniquement)")
    st.dataframe(df_resultats.style.highlight_max(subset=["Correlation"], color="lightgreen"), use_container_width=True)
    st.subheader("Importance moyenne des variables")
    st.bar_chart(df_fimp_moyenne)

with onglet_diag:
    st.subheader("Univers de tickers")
    st.write(f"{len(tickers_source)} tickers dans la source -> {len(tickers_test)} utilisables en {annee_test}.")
    st.write(f"Tickers exclus : {', '.join(tickers_exclus) if tickers_exclus else 'aucun'}")
    extremes = df_ml[df_ml[TARGET_COL].abs() > 0.50][["Ticker", "Date", "Price", TARGET_COL]]
    st.subheader("Rendements mensuels extremes (> 50%, a verifier manuellement)")
    st.dataframe(extremes, use_container_width=True, hide_index=True)

# ==============================================================================
# EXPORT EXCEL - MISE EN FORME (colonnes ajustees, pourcentages, MAD, en-tetes)
# ==============================================================================
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")

def ecrire_feuille_formatee(wb, nom_feuille, df, pct_cols=(), mad_cols=()):
    """
    Ecrit un DataFrame dans une nouvelle feuille avec :
    - en-tetes en gras, fond colore, centres, ligne figee (freeze panes)
    - colonnes de pourcentage formatees en % (0.00%) - inutile de renommer l'en-tete
    - colonnes MAD formatees avec le suffixe monetaire et separateur de milliers
    - largeur de CHAQUE colonne ajustee automatiquement au contenu le plus long
    - filtre automatique sur l'en-tete
    """
    ws = wb.create_sheet(nom_feuille[:31])  # 31 caracteres max pour un nom de feuille Excel
    if df is None or df.empty:
        ws["A1"] = "(aucune donnee disponible)"
        return ws

    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=1, column=j, value=str(col))
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center")

    for i, (_, row) in enumerate(df.iterrows(), start=2):
        for j, col in enumerate(df.columns, start=1):
            val = row[col]
            if isinstance(val, pd.Timestamp):
                val = val.to_pydatetime()
            cell = ws.cell(row=i, column=j, value=val)
            if col in pct_cols:
                cell.number_format = "0.00%"
            elif col in mad_cols:
                cell.number_format = '#,##0 "MAD"'
            elif col == "Date" or "Date" in str(col):
                cell.number_format = "dd/mm/yyyy"

    # Ajustement automatique de la largeur de chaque colonne au texte le plus long
    for j, col in enumerate(df.columns, start=1):
        contenu_max = df[col].astype(str).map(len).max() if len(df) else 0
        largeur = max(len(str(col)), contenu_max) + 3
        ws.column_dimensions[get_column_letter(j)].width = min(largeur, 45)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
    ws.row_dimensions[1].height = 20
    return ws


# ---- Export Excel ----
st.divider()
wb_export = Workbook()
wb_export.remove(wb_export.active)

ecrire_feuille_formatee(wb_export, "Comparaison_Modeles", df_resultats.reset_index(),
                         pct_cols=["Directional_Accuracy"])
ecrire_feuille_formatee(wb_export, f"Alloc_Valid_{annee_validation}", df_alloc_val,
                         pct_cols=["Rendement_predit_M1", "Allocation", "Rendement_reel_M1", "Contribution_portefeuille"],
                         mad_cols=["Capital_alloue_MAD"])
ecrire_feuille_formatee(wb_export, f"Perf_Valid_{annee_validation}", df_perf_val,
                         pct_cols=["Rendement_predit_portefeuille", "Rendement_reel_portefeuille", "Rendement_cumule"],
                         mad_cols=["Capital_debut_MAD", "Gain_perte_MAD", "Valeur_portefeuille_MAD"])
ecrire_feuille_formatee(wb_export, f"Alloc_Test_{annee_test}", df_alloc_test,
                         pct_cols=["Rendement_predit_M1", "Allocation", "Rendement_reel_M1", "Contribution_portefeuille"],
                         mad_cols=["Capital_alloue_MAD"])
ecrire_feuille_formatee(wb_export, f"Perf_Test_{annee_test}", df_perf_test,
                         pct_cols=["Rendement_predit_portefeuille", "Rendement_reel_portefeuille", "Rendement_cumule"],
                         mad_cols=["Capital_debut_MAD", "Gain_perte_MAD", "Valeur_portefeuille_MAD"])
ecrire_feuille_formatee(wb_export, f"Prevision_{annee_prevision}", recommandation,
                         pct_cols=["Rendement_predit", "Allocation"])

buffer = io.BytesIO()
wb_export.save(buffer)
st.download_button("Telecharger le rapport Excel", data=buffer.getvalue(),
                    file_name="Resultats_ML_Portefeuille.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
