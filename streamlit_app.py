"""
Streamlit Front-End pour l'API Modal de Prévision
Se connecte à l'API Modal déployée pour faire des prévisions
"""
from __future__ import annotations
import io
import numpy as np
import plotly.graph_objects as go
import logging
from pathlib import Path
import requests
import streamlit as st
import pandas as pd

# Configuration
TEMP_DIR = Path("cache")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Logger
LOG_PATH = TEMP_DIR / "streamlit_app.log"
logger = logging.getLogger("StreamlitApp")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# =========================
# Configuration API Modal
# =========================

# IMPORTANT: Remplacez cette URL par votre URL Modal après déploiement
# Format: https://YOUR-USERNAME--forecast-api-predict-api.modal.run
MODAL_API_URL = st.secrets.get("MODAL_API_URL", "http://localhost:8000")  # À configurer dans .streamlit/secrets.toml

# =========================
# Fonctions utilitaires
# =========================

def prepare_daily_df(df,
                     col_article="Description article",
                     col_date="Date de livraison",
                     col_qte="Quantite"):
    """Prépare un DataFrame avec 1 ligne par (article, date) et quantités = 0 si absence."""

    # Créer une copie pour éviter de modifier l'original
    df = df.copy()

    # Détecter automatiquement les colonnes si les noms par défaut n'existent pas
    original_article = col_article
    original_date = col_date
    original_qte = col_qte

    if col_article not in df.columns:
        # Chercher des colonnes similaires (priorité à Description, pas ItemCode)
        description_col = None
        item_col = None

        for col in df.columns:
            col_lower = col.lower()
            # Priorité 1: Description
            if any(keyword in col_lower for keyword in ['description', 'dscription', 'libelle', 'libellé']):
                description_col = col
                break
            # Priorité 2: Article/Produit
            elif any(keyword in col_lower for keyword in ['article', 'produit']) and 'code' not in col_lower:
                if description_col is None:
                    description_col = col
            # Dernière priorité: ItemCode (seulement si rien d'autre)
            elif any(keyword in col_lower for keyword in ['item', 'code']) and item_col is None:
                item_col = col

        # Utiliser description en priorité, sinon item code
        col_article = description_col if description_col else item_col

        if col_article is None:
            raise ValueError(f"❌ Colonne article non trouvée. Colonnes disponibles: {df.columns.tolist()}")

    if col_date not in df.columns:
        # Chercher des colonnes de date
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'docdate', 'livraison', 'delivery']):
                col_date = col
                break
        if col_date == original_date:
            raise ValueError(f"❌ Colonne date non trouvée. Colonnes disponibles: {df.columns.tolist()}")

    if col_qte not in df.columns:
        # Chercher des colonnes de quantité
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['quantite', 'quantity', 'qte', 'sum']):
                col_qte = col
                break
        if col_qte == original_qte:
            raise ValueError(f"❌ Colonne quantité non trouvée. Colonnes disponibles: {df.columns.tolist()}")

    df[col_date] = pd.to_datetime(df[col_date], dayfirst=True, errors="coerce")

    df[col_qte] = (
        df[col_qte]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .astype(float)
    )

    grouped = (
        df.groupby([col_article, col_date], as_index=False)[col_qte]
          .sum()
          .rename(columns={col_qte: "Quantité_totale"})
    )

    all_dates = pd.date_range(
        start=grouped[col_date].min(),
        end=grouped[col_date].max(),
        freq="D",
    )

    all_articles = grouped[col_article].unique()

    full_index = pd.MultiIndex.from_product(
        [all_articles, all_dates],
        names=[col_article, col_date],
    )

    result = (
        grouped
        .set_index([col_article, col_date])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Renommer les colonnes aux noms standards attendus par le reste du code
    result = result.rename(columns={
        col_article: "Description article",
        col_date: "Date de livraison"
    })

    return result


def aggregate_quantities(df_daily, freq="D"):
    """Agrège les quantités par article sur la fréquence donnée."""

    if freq == "D":
        out = df_daily.copy()
        out = out.rename(columns={"Date de livraison": "Période"})
        return out

    agg = (
        df_daily
        .groupby(
            [
                "Description article",
                pd.Grouper(key="Date de livraison", freq=freq),
            ]
        )["Quantité_totale"]
        .sum()
        .reset_index()
        .rename(columns={"Date de livraison": "Période"})
    )
    return agg


def call_modal_api(product_name: str, series: list, dates: list, horizon: int):
    """
    Appelle l'API Modal pour obtenir des prévisions

    Args:
        product_name: Nom du produit
        series: Liste des valeurs historiques
        dates: Liste des dates (format ISO)
        horizon: Horizon de prévision

    Returns:
        dict: Résultats de l'API
    """
    payload = {
        "product_name": product_name,
        "series": series,
        "dates": dates,
        "horizon": horizon
    }

    try:
        response = requests.post(MODAL_API_URL, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur API: {e}")
        return {"success": False, "error": str(e)}


# =========================
# Interface Streamlit
# =========================

st.set_page_config(page_title="Prévisions IA - Modal API", layout="wide")
st.title("Prévisions IA avec Modal API")

# Afficher le statut de l'API
st.sidebar.header("Configuration")
api_url_input = st.sidebar.text_input(
    "URL de l'API Modal",
    value=MODAL_API_URL,
    help="URL de votre API Modal déployée"
)
MODAL_API_URL = api_url_input

# Test de connexion
if st.sidebar.button("Tester la connexion API"):
    with st.spinner("Test de connexion..."):
        try:
            # Essayer d'appeler le health endpoint
            health_url = MODAL_API_URL.replace("predict-api", "health")
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                st.sidebar.success("API connectée!")
                st.sidebar.json(response.json())
            else:
                st.sidebar.error(f"Erreur: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"Connexion échouée: {e}")

st.markdown(
    "Importe ton fichier (CSV ou Excel) contenant : "
    "`Description article`, `Date de livraison`, `Quantite`."
)

uploaded_file = st.file_uploader("Choisis ton fichier", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lecture du fichier
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file, sep=";")
    else:
        # Pour les fichiers Excel, essayer de lire le bon onglet
        try:
            # Essayer d'abord l'onglet "Historiques livraisons FFC"
            df_raw = pd.read_excel(uploaded_file, sheet_name="Historiques livraisons FFC")
        except:
            # Sinon, lire le premier onglet
            df_raw = pd.read_excel(uploaded_file)

    st.success("Fichier chargé")
    st.write("Aperçu des premières lignes :")
    st.dataframe(df_raw.head())

    # Afficher les colonnes détectées
    st.write("📊 Colonnes détectées:", df_raw.columns.tolist())

    # Préparation du DataFrame journalier
    try:
        df_daily = prepare_daily_df(df_raw)
        st.info(f"✅ Colonnes mappées automatiquement")
    except Exception as e:
        st.error(f"❌ Erreur lors de la préparation des données: {e}")
        st.write("Colonnes attendues: Description article, Date de livraison, Quantite")
        st.stop()

    # ==========
    # Ranking des produits
    # ==========
    st.subheader("Classement des produits par quantité mensuelle")

    df_monthly_all = aggregate_quantities(df_daily, freq="ME")

    ranking = (
        df_monthly_all
        .groupby("Description article")["Quantité_totale"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Quantité_totale": "Quantité_mensuelle_cumulée"})
    )

    st.dataframe(ranking)

    # ==========
    # Visualisation détaillée
    # ==========
    st.subheader("Visualisation et prévision par article")

    articles_sorted = ranking["Description article"].tolist()

    # Recherche d'article
    search_text = st.text_input(
        "🔎 Rechercher un article :",
        value="",
        placeholder="Ex : VINAIGRETTE, LINDT, PATES..."
    )

    if search_text:
        filtered_articles = [
            a for a in articles_sorted
            if search_text.lower() in a.lower()
        ]
    else:
        filtered_articles = articles_sorted

    if not filtered_articles:
        st.warning("Aucun article ne correspond à ta recherche.")
        st.stop()

    selected_article = st.selectbox(
        "Article :",
        filtered_articles,
    )

    freq_label = st.radio(
        "Fréquence d'agrégation :",
        ("Jour", "Semaine", "Mois"),
        horizontal=True,
    )

    if freq_label == "Jour":
        freq = "D"
    elif freq_label == "Semaine":
        freq = "W-MON"
    else:
        freq = "ME"

    df_agg = aggregate_quantities(df_daily, freq=freq)

    df_article = df_agg[df_agg["Description article"] == selected_article].copy()
    df_article = df_article.sort_values("Période")

    # Trimming
    nonzero_mask = df_article["Quantité_totale"] != 0
    if nonzero_mask.any():
        first_idx = df_article.index[nonzero_mask][0]
        last_idx = df_article.index[nonzero_mask][-1]
        df_article = df_article.loc[first_idx:last_idx]

    st.write(f"Article sélectionné : **{selected_article}**")
    st.write(f"Points utilisés : {len(df_article)}")

    st.dataframe(df_article)

    # ==========
    # Graphique historique
    # ==========
    st.subheader("Historique des quantités")

    series_hist = df_article.set_index("Période")["Quantité_totale"]

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Scatter(
            x=series_hist.index,
            y=series_hist.values,
            mode="lines",
            name="Historique",
            line=dict(color="black", width=1.5),
        )
    )

    fig_hist.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Date",
        yaxis_title="Quantité",
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    # Téléchargement historique
    hist_buffer = io.BytesIO()
    series_hist.to_frame(name="Quantité_totale").to_excel(
        hist_buffer,
        sheet_name="Historique",
    )
    hist_buffer.seek(0)

    st.download_button(
        label="📥 Télécharger les données historiques (Excel)",
        data=hist_buffer,
        file_name=f"historique_{selected_article}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ==========
    # Prévision via API Modal
    # ==========
    st.subheader("Prévision IA via Modal")

    horizon_choice = st.selectbox(
        "Horizon de prévision :",
        ["Aucune", "1 step", "7 steps", "30 steps", "60 steps", "90 steps"],
        index=0,
    )

    if horizon_choice == "Aucune":
        forecast_horizon = None
    else:
        horizon_map = {
            "1 step": 1,
            "7 steps": 7,
            "30 steps": 30,
            "60 steps": 60,
            "90 steps": 90
        }
        forecast_horizon = horizon_map[horizon_choice]

    run_forecast = st.button("🚀 Lancer la prévision IA")

    if forecast_horizon is not None and run_forecast:
        # Préparer les données pour l'API
        series = series_hist.values.tolist()
        dates = series_hist.index.strftime("%Y-%m-%d").tolist()

        # Appel à l'API Modal
        with st.spinner("🔄 Appel à l'API Modal en cours..."):
            result = call_modal_api(
                product_name=selected_article,
                series=series,
                dates=dates,
                horizon=forecast_horizon
            )

        if result.get("success"):
            st.success(f"✅ Prévision réussie avec le modèle : **{result['model_used']}**")

            # Afficher les diagnostics de routing
            st.caption("Diagnostics du routage intelligent :")
            routing_df = pd.DataFrame([result["routing_info"]])
            st.dataframe(routing_df)

            # Récupérer les prévisions
            predictions = result["predictions"]
            lower = result["lower_bound"]
            upper = result["upper_bound"]
            simulated_path = result["simulated_path"]
            median_predictions = result.get("median_predictions")

            # Construire l'index futur
            if isinstance(series_hist.index, pd.DatetimeIndex):
                inferred_freq = pd.infer_freq(series_hist.index)
                if inferred_freq is None:
                    inferred_freq = "D"
                start_future = series_hist.index[-1] + pd.tseries.frequencies.to_offset(inferred_freq)
                future_index = pd.date_range(
                    start=start_future,
                    periods=forecast_horizon,
                    freq=inferred_freq,
                )
            else:
                last_idx = series_hist.index[-1]
                future_index = np.arange(last_idx + 1, last_idx + 1 + forecast_horizon)

            # Graphique avec toutes les prévisions
            fig_pred = go.Figure()

            # Historique
            fig_pred.add_trace(
                go.Scatter(
                    x=series_hist.index,
                    y=series_hist.values,
                    mode="lines",
                    name="Historique",
                    line=dict(color="black", width=1.5),
                )
            )

            # Bande de confiance
            fig_pred.add_trace(
                go.Scatter(
                    x=future_index,
                    y=upper,
                    mode="lines",
                    line=dict(color="rgba(31, 119, 180, 0.0)"),
                    showlegend=False,
                )
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=future_index,
                    y=lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(31, 119, 180, 0.25)",
                    line=dict(color="rgba(31, 119, 180, 0.0)"),
                    name="IC 95%",
                )
            )

            # Prévision moyenne
            fig_pred.add_trace(
                go.Scatter(
                    x=future_index,
                    y=predictions,
                    mode="lines",
                    name="Prévision moyenne",
                    line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
                )
            )

            # Prévision médiane (si disponible)
            if median_predictions is not None:
                fig_pred.add_trace(
                    go.Scatter(
                        x=future_index,
                        y=median_predictions,
                        mode="lines",
                        name="Prévision médiane",
                        line=dict(color="rgba(124, 252, 0, 0.8)", width=2),
                    )
                )

            # Trajectoire simulée
            if result["model_used"] == "BayesianLSTM":
                label = "Trajectoire simulée (MC Dropout)"
                color = "rgba(124, 252, 0, 0.9)"
            else:
                label = "Scénario simulé 0/spikes"
                color = "rgba(255, 0, 0, 0.9)"

            fig_pred.add_trace(
                go.Scatter(
                    x=future_index,
                    y=simulated_path,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.8, dash="dot"),
                )
            )

            # Ligne verticale passé/futur
            fig_pred.add_vline(
                x=series_hist.index[-1],
                line_dash="dash",
                line_color="grey",
                line_width=1,
            )

            fig_pred.update_layout(
                template="plotly_white",
                height=500,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis_title="Temps",
                yaxis_title="Quantité",
                legend=dict(x=0.01, y=0.99),
                title=f"Historique et prévisions (H={forecast_horizon}, {result['model_used']})",
                plot_bgcolor='white',
                paper_bgcolor='white',
            )

            st.plotly_chart(fig_pred, use_container_width=True)

            # Afficher la granularité sélectionnée
            st.info(f"📊 **Granularité sélectionnée** : {freq_label}")

            # Créer tableau récapitulatif agrégé par granularité
            st.subheader(f"📋 Tableau récapitulatif des prévisions par {freq_label.lower()}")

            # Créer DataFrame avec toutes les prévisions
            forecast_df = pd.DataFrame({
                "date": future_index,
                "mean": predictions,
                "lower": lower,
                "upper": upper,
                "simulated_path": simulated_path,
            })

            if median_predictions is not None:
                forecast_df["median"] = median_predictions

            forecast_df = forecast_df.set_index("date")

            # Agréger par la granularité sélectionnée
            if freq == "D":
                # Pas d'agrégation pour les jours
                summary_df = forecast_df.copy()
                summary_df.index.name = "Date"
            else:
                # Agréger par semaine ou mois
                summary_df = forecast_df.resample(freq).sum()
                if freq == "W-MON":
                    summary_df.index.name = "Semaine (début)"
                else:
                    summary_df.index.name = "Mois"

            # Renommer les colonnes pour plus de clarté
            summary_df_display = summary_df.copy()
            summary_df_display = summary_df_display.rename(columns={
                "mean": "Prévision moyenne (somme)",
                "median": "Prévision médiane (somme)",
                "lower": "Borne inférieure (somme)",
                "upper": "Borne supérieure (somme)",
                "simulated_path": "Trajectoire simulée (somme)"
            })

            # Afficher le tableau avec formatage
            st.dataframe(
                summary_df_display.style.format("{:.2f}"),
                use_container_width=True
            )

            # Option de téléchargement du tableau récapitulatif
            summary_buffer = io.BytesIO()
            summary_df_display.to_excel(summary_buffer, sheet_name=f"Resume_{freq_label}")
            summary_buffer.seek(0)

            safe_name = str(selected_article).replace("/", "_").replace("\\", "_")
            st.download_button(
                label=f"📥 Télécharger le tableau récapitulatif ({freq_label})",
                data=summary_buffer,
                file_name=f"resume_{safe_name}_H{forecast_horizon}_{freq_label}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Préparer les données pour export (données brutes)
            forecast_export = forecast_df.copy()

            # Téléchargement des prévisions
            forecast_buffer = io.BytesIO()
            forecast_export.to_excel(
                forecast_buffer,
                sheet_name="Prevision",
            )
            forecast_buffer.seek(0)

            safe_name = str(selected_article).replace("/", "_").replace("\\", "_")
            st.download_button(
                label="📥 Télécharger les données de prévision (Excel)",
                data=forecast_buffer,
                file_name=f"forecast_{safe_name}_H{forecast_horizon}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Afficher les métadonnées
            with st.expander("📋 Détails de la prévision"):
                st.json(result["metadata"])
                st.json(result["routing_info"])

        else:
            st.error(f"❌ Erreur lors de la prévision : {result.get('error', 'Erreur inconnue')}")
            logger.error(f"Erreur API: {result.get('error')}")

    elif forecast_horizon is not None and not run_forecast:
        st.info("Clique sur **🚀 Lancer la prévision IA** pour obtenir les prévisions.")
