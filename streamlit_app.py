"""
Circumcision Age Predictor — DHS Cameroon 2018
==============================================
Streamlit application that mirrors the web demo and loads the real GBSA
model trained in the notebook.

Project structure expected:

    project/
    ├── streamlit_app.py
    ├── assets/
    │   └── bg-health.jpg
    └── model_artifacts/
        ├── gbsa_model.pkl
        └── preprocessor.pkl

Setup:
    python -m venv .venv
    # Windows:  .venv\\Scripts\\activate
    # macOS/Linux:  source .venv/bin/activate
    pip install  scikit-survival scikit-learn joblib

Run:
    streamlit run streamlit_app.py
"""

import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Circumcision Age Predictor — DHS Cameroon 2018",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Background image + custom CSS (image clearly visible, light overlay)
# =============================================================================
def set_background(image_path: str):
    if not Path(image_path).exists():
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(135deg, rgba(252,247,238,0.55), rgba(252,247,238,0.65)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        section[data-testid="stSidebar"] > div {{
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(10px);
        }}
        .block-container {{ padding-top: 2rem; }}
        .metric-card {{
            background: rgba(255,255,255,0.82);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 10px 30px -12px rgba(45,90,110,0.25);
        }}
        .metric-card .label {{
            font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
            color: #6b7c85;
        }}
        .metric-card .value {{
            font-size: 22px; font-weight: 700; color: #1f3b46; margin-top: 6px;
        }}
        .risk-pill {{
            display:inline-block; padding: 6px 14px; border-radius: 999px;
            font-weight: 600; font-size: 14px;
        }}
        .risk-EARLY  {{ background:#c97a4a; color:white; }}
        .risk-MEDIUM {{ background:#d9b26b; color:#1f3b46; }}
        .risk-LATE   {{ background:#b8453b; color:white; }}
        .interp-box {{
            background: rgba(255,255,255,0.88);
            border-left: 4px solid #2d5a6e;
            border-radius: 10px;
            padding: 16px 18px;
            margin-top: 10px;
            color: #1f3b46;
            line-height: 1.55;
            font-size: 15px;
        }}
        h1, h2, h3 {{ color:#1f3b46; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background("assets/bg-health.jpg")

# =============================================================================
# i18n
# =============================================================================
TRANSLATIONS = {
    "en": {
        "app_title": "Circumcision Age Predictor",
        "app_subtitle": "Survival analysis on the DHS Cameroon 2018 dataset (CMMR71FL)",
        "methodology": "Methodology",
        "methodology_text": (
            "This tool uses a Gradient Boosting Survival Analysis (GBSA) model trained "
            "on the DHS / EDS Cameroon 2018 dataset. It estimates the survival curve S(t) — "
            "the probability of remaining uncircumcised at each age — and reports the "
            "predicted median age, a ±2 year window, and a risk category."
        ),
        "tab_predict": "🔮 Predictor",
        "tab_variables": "📋 Variables",
        "tab_about": "ℹ️ About the project",
        "form_title": "Individual profile",
        "form_help": "Fill in the respondent's characteristics.",
        "current_age": "Current age (years)",
        "region": "Region", "religion": "Religion", "residence": "Residence",
        "wealth": "Wealth quintile", "education": "Education", "marital": "Marital status",
        "hiv": "Ever tested for HIV", "tv": "TV exposure", "radio": "Radio exposure",
        "predict": "Predict", "results": "Prediction",
        "median": "Median age", "window": "Window", "risk": "Risk",
        "curve_title": "Estimated survival curve",
        "curve_help": "Probability of still being uncircumcised at each age.",
        "years": "yrs", "no_result": "Click Predict to see results.",
        "EARLY": "Early", "MEDIUM": "Medium", "LATE": "Late",
        "interpretation": "💡 Plain-language interpretation",
        "interp_template": (
            "Based on this profile, the model estimates that a man with these "
            "characteristics is most likely to be circumcised around age **{median}**, "
            "typically between **{start}** and **{end}** years old. "
            "The risk category is **{risk}**, meaning circumcision tends to happen "
            "{risk_word} compared to the national pattern. "
            "The curve below shows the probability of being still uncircumcised at "
            "each age — when it drops below 0.5, half of similar individuals have "
            "already been circumcised."
        ),
        "risk_word_EARLY": "earlier (often in childhood)",
        "risk_word_MEDIUM": "around adolescence",
        "risk_word_LATE": "later (in adulthood, or not at all)",
        "variables_title": "Variables used in the model",
        "variables_intro": "All variables come from the DHS / EDS Cameroon 2018 (CMMR71FL) men's individual recode.",
        "about_title": "About this project",
        "about_summary": "📊 Project summary",
        "about_summary_text": (
            "This academic project answers the question: \"How old were you when you were "
            "circumcised?\" using the DHS Cameroon 2018 dataset. We built a survival-analysis "
            "pipeline (Kaplan-Meier, Cox PH, and a final Gradient Boosting Survival Analysis "
            "model) to estimate, for any individual profile, the most likely age at "
            "circumcision and the uncertainty around that estimate."
        ),
        "about_aim": "🎯 Aim",
        "about_aim_text": (
            "The aim is to identify the demographic, socio-economic, religious, and "
            "regional factors that influence the timing of male circumcision in Cameroon, "
            "and to provide a simple bilingual tool that public-health stakeholders and "
            "students can use to explore predictions for different population profiles."
        ),
        "about_data": "🗂️ Data",
        "about_data_text": (
            "Source: Demographic and Health Survey (DHS) — Cameroon 2018, men's recode "
            "(CMMR71FL), restricted to respondents with a valid circumcision status (MV483) "
            "and age at event (MV483A)."
        ),
        "about_model": "🧠 Model",
        "about_model_text": (
            "Final model: Gradient Boosting Survival Analysis (scikit-survival). Inputs "
            "are one-hot encoded categorical variables plus current age. Output is the "
            "estimated survival function S(t), from which we derive the median age, a "
            "±2 year window, and a risk category (Early / Medium / Late)."
        ),
        "col_code": "Code", "col_var": "Variable", "col_desc": "Description",
    },
    "fr": {
        "app_title": "Prédicteur d'âge à la circoncision",
        "app_subtitle": "Analyse de survie sur l'EDS Cameroun 2018 (CMMR71FL)",
        "methodology": "Méthodologie",
        "methodology_text": (
            "Cet outil utilise un modèle Gradient Boosting Survival Analysis (GBSA) entraîné "
            "sur l'EDS Cameroun 2018. Il estime la courbe de survie S(t) — la probabilité de "
            "rester non circoncis à chaque âge — et fournit l'âge médian prédit, une fenêtre "
            "±2 ans et une catégorie de risque."
        ),
        "tab_predict": "🔮 Prédicteur",
        "tab_variables": "📋 Variables",
        "tab_about": "ℹ️ À propos du projet",
        "form_title": "Profil individuel",
        "form_help": "Renseignez les caractéristiques du répondant.",
        "current_age": "Âge actuel (années)",
        "region": "Région", "religion": "Religion", "residence": "Résidence",
        "wealth": "Quintile de richesse", "education": "Éducation", "marital": "Statut matrimonial",
        "hiv": "Test VIH déjà fait", "tv": "Exposition TV", "radio": "Exposition radio",
        "predict": "Prédire", "results": "Prédiction",
        "median": "Âge médian", "window": "Fenêtre", "risk": "Risque",
        "curve_title": "Courbe de survie estimée",
        "curve_help": "Probabilité d'être encore non circoncis à chaque âge.",
        "years": "ans", "no_result": "Cliquez sur Prédire pour voir les résultats.",
        "EARLY": "Précoce", "MEDIUM": "Moyen", "LATE": "Tardif",
        "interpretation": "💡 Interprétation simple",
        "interp_template": (
            "D'après ce profil, le modèle estime qu'un homme avec ces caractéristiques "
            "a le plus de chances d'être circoncis vers l'âge de **{median}** ans, "
            "généralement entre **{start}** et **{end}** ans. "
            "La catégorie de risque est **{risk}**, ce qui signifie que la circoncision "
            "tend à survenir {risk_word} par rapport au schéma national. "
            "La courbe ci-dessous montre la probabilité d'être encore non circoncis à "
            "chaque âge — lorsqu'elle passe sous 0,5, la moitié des individus similaires "
            "sont déjà circoncis."
        ),
        "risk_word_EARLY": "plus tôt (souvent durant l'enfance)",
        "risk_word_MEDIUM": "vers l'adolescence",
        "risk_word_LATE": "plus tard (à l'âge adulte, ou pas du tout)",
        "variables_title": "Variables utilisées dans le modèle",
        "variables_intro": "Toutes les variables proviennent de l'EDS Cameroun 2018 (CMMR71FL), recode individuel hommes.",
        "about_title": "À propos de ce projet",
        "about_summary": "📊 Résumé du projet",
        "about_summary_text": (
            "Ce projet académique répond à la question : « Quel âge aviez-vous lors de votre "
            "circoncision ? » à partir de l'EDS Cameroun 2018. Nous avons construit un pipeline "
            "d'analyse de survie (Kaplan-Meier, Cox PH, puis Gradient Boosting Survival Analysis) "
            "pour estimer, à partir de tout profil individuel, l'âge le plus probable de "
            "circoncision et l'incertitude associée."
        ),
        "about_aim": "🎯 Objectif",
        "about_aim_text": (
            "L'objectif est d'identifier les facteurs démographiques, socio-économiques, "
            "religieux et régionaux qui influencent le moment de la circoncision masculine "
            "au Cameroun, et de fournir un outil bilingue simple permettant aux acteurs de "
            "santé publique et aux étudiants d'explorer les prédictions selon différents profils."
        ),
        "about_data": "🗂️ Données",
        "about_data_text": (
            "Source : Enquête Démographique et de Santé (EDS) — Cameroun 2018, recode hommes "
            "(CMMR71FL), restreint aux répondants avec un statut de circoncision valide (MV483) "
            "et un âge à l'événement (MV483A)."
        ),
        "about_model": "🧠 Modèle",
        "about_model_text": (
            "Modèle final : Gradient Boosting Survival Analysis (scikit-survival). Les entrées "
            "sont des variables catégorielles encodées en one-hot plus l'âge actuel. La sortie "
            "est la fonction de survie estimée S(t), d'où l'on dérive l'âge médian, une fenêtre "
            "±2 ans et une catégorie de risque (Précoce / Moyen / Tardif)."
        ),
        "col_code": "Code", "col_var": "Variable", "col_desc": "Description",
    },
}

# =============================================================================
# Variable dictionary (for the Variables tab)
# =============================================================================
VARIABLE_DICT = {
    "en": [
        ("MV483",  "Circumcision status",  "Whether the respondent is circumcised (event indicator)."),
        ("MV483A", "Age at circumcision",  "Reported age at the time of circumcision (event time)."),
        ("MV012",  "Current age",          "Current age of the respondent in completed years."),
        ("MV024",  "Region",               "Region of residence (12 administrative regions of Cameroon)."),
        ("MV025",  "Residence",            "Type of place of residence: urban or rural."),
        ("MV106",  "Education level",      "Highest level of education attained (none, primary, secondary, higher)."),
        ("MV190",  "Wealth index",         "Household wealth quintile (poorest → richest)."),
        ("MV130",  "Religion",             "Religion of the respondent (Catholic, Protestant, Muslim, Animist, …)."),
        ("MV501",  "Marital status",       "Current marital status (never in union, married, …)."),
        ("MV158",  "Radio exposure",       "Frequency of listening to the radio."),
        ("MV159",  "TV exposure",          "Frequency of watching television."),
        ("MV781",  "HIV testing",          "Whether the respondent has ever been tested for HIV."),
        ("MV005",  "Sample weight",        "Individual sample weight (used in survey-weighted analyses)."),
        ("MV021",  "PSU",                  "Primary sampling unit identifier."),
        ("MV022",  "Strata",               "Sample stratum identifier."),
    ],
    "fr": [
        ("MV483",  "Statut de circoncision", "Indique si le répondant est circoncis (indicateur d'événement)."),
        ("MV483A", "Âge à la circoncision",  "Âge déclaré au moment de la circoncision (temps de l'événement)."),
        ("MV012",  "Âge actuel",             "Âge actuel du répondant en années révolues."),
        ("MV024",  "Région",                 "Région de résidence (12 régions administratives du Cameroun)."),
        ("MV025",  "Résidence",              "Type de lieu de résidence : urbain ou rural."),
        ("MV106",  "Niveau d'éducation",     "Plus haut niveau atteint (aucun, primaire, secondaire, supérieur)."),
        ("MV190",  "Indice de richesse",     "Quintile de richesse du ménage (très pauvre → très riche)."),
        ("MV130",  "Religion",               "Religion du répondant (catholique, protestant, musulman, animiste, …)."),
        ("MV501",  "Statut matrimonial",     "Statut matrimonial actuel (jamais en union, marié, …)."),
        ("MV158",  "Exposition radio",       "Fréquence d'écoute de la radio."),
        ("MV159",  "Exposition TV",          "Fréquence de visionnage de la télévision."),
        ("MV781",  "Test VIH",               "Si le répondant a déjà fait un test VIH."),
        ("MV005",  "Poids de l'échantillon", "Poids individuel d'échantillonnage (analyses pondérées)."),
        ("MV021",  "UPE",                    "Identifiant de l'unité primaire d'échantillonnage."),
        ("MV022",  "Strate",                 "Identifiant de la strate d'échantillonnage."),
    ],
}

# =============================================================================
# Option lists (code -> label)
# =============================================================================
def options(lang):
    L = lang
    return {
        "region": [
            ("1", "Adamaoua"), ("2", "Centre"), ("3", "Douala"),
            ("4", "East" if L == "en" else "Est"),
            ("5", "Far North" if L == "en" else "Extrême-Nord"),
            ("6", "Littoral"),
            ("7", "North" if L == "en" else "Nord"),
            ("8", "North-West" if L == "en" else "Nord-Ouest"),
            ("9", "West" if L == "en" else "Ouest"),
            ("10", "South" if L == "en" else "Sud"),
            ("11", "South-West" if L == "en" else "Sud-Ouest"),
            ("12", "Yaoundé"),
        ],
        "religion": [
            ("1", "Catholic" if L == "en" else "Catholique"),
            ("2", "Protestant"),
            ("3", "Other Christian" if L == "en" else "Autre chrétien"),
            ("4", "Muslim" if L == "en" else "Musulman"),
            ("5", "Animist" if L == "en" else "Animiste"),
            ("96", "Other / None" if L == "en" else "Autre / Aucune"),
        ],
        "residence": [("1", "Urban" if L == "en" else "Urbain"), ("2", "Rural")],
        "wealth": [
            ("1", "Poorest" if L == "en" else "Très pauvre"),
            ("2", "Poorer"  if L == "en" else "Pauvre"),
            ("3", "Middle"  if L == "en" else "Moyen"),
            ("4", "Richer"  if L == "en" else "Riche"),
            ("5", "Richest" if L == "en" else "Très riche"),
        ],
        "education": [
            ("0", "No education" if L == "en" else "Aucune"),
            ("1", "Primary"      if L == "en" else "Primaire"),
            ("2", "Secondary"    if L == "en" else "Secondaire"),
            ("3", "Higher"       if L == "en" else "Supérieure"),
        ],
        "marital": [
            ("0", "Never in union"   if L == "en" else "Jamais en union"),
            ("1", "Married"          if L == "en" else "Marié(e)"),
            ("2", "Living together"  if L == "en" else "En concubinage"),
            ("3", "Widowed"          if L == "en" else "Veuf/Veuve"),
            ("4", "Divorced"         if L == "en" else "Divorcé(e)"),
            ("5", "Separated"        if L == "en" else "Séparé(e)"),
        ],
        "hiv":   [("0", "No" if L == "en" else "Non"), ("1", "Yes" if L == "en" else "Oui")],
        "media": [
            ("0", "Not at all"            if L == "en" else "Pas du tout"),
            ("1", "Less than once a week" if L == "en" else "Moins d'une fois/semaine"),
            ("2", "At least once a week"  if L == "en" else "Au moins une fois/semaine"),
            ("3", "Almost every day"      if L == "en" else "Presque tous les jours"),
        ],
    }

# =============================================================================
# Model loading & prediction
# =============================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_artifacts/gbsa_model.pkl")
    preprocessor = joblib.load("model_artifacts/preprocessor.pkl")
    return model, preprocessor

def predict_risk(model, preprocessor, payload: dict):
    df = pd.DataFrame([payload])
    for col in ["residence", "education", "wealth", "marital_status",
                "religion", "region", "tv_exposure", "radio_exposure", "hiv_testing"]:
        df[col] = df[col].astype("category")
    X = preprocessor.transform(df)
    surv_fn = model.predict_survival_function(X)[0]
    ages, probs = surv_fn.x, surv_fn.y

    by_age = {}
    for t, p in zip(ages, probs):
        a = int(round(t))
        by_age.setdefault(a, float(p))
    by_age = dict(sorted(by_age.items()))

    median = next((a for a, p in by_age.items() if p <= 0.5), max(by_age))
    if median > 17:
        risk = "LATE"
    elif median >= 14:
        risk = "MEDIUM"
    else:
        risk = "EARLY"

    window = (max(min(by_age), median - 2), min(max(by_age), median + 2))
    return {"curve": by_age, "median": median, "risk": risk, "window": window}

# =============================================================================
# Sidebar — language
# =============================================================================
with st.sidebar:
    st.markdown("### 🌍 Language / Langue")
    lang_label = st.radio(
        "lang", ["English", "Français"], horizontal=True, label_visibility="collapsed"
    )
    lang = "en" if lang_label == "English" else "fr"
    T = TRANSLATIONS[lang]
    st.markdown("---")
    st.markdown(f"**{T['methodology']}**")
    st.caption(T["methodology_text"])

T = TRANSLATIONS[lang]
OPTS = options(lang)

# =============================================================================
# Header
# =============================================================================
st.markdown(f"# 🩺 {T['app_title']}")
st.caption(T["app_subtitle"])

# =============================================================================
# Tabs
# =============================================================================
tab_predict, tab_vars, tab_about = st.tabs(
    [T["tab_predict"], T["tab_variables"], T["tab_about"]]
)

# -----------------------------------------------------------------------------
# TAB 1 — Predictor
# -----------------------------------------------------------------------------
with tab_predict:
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown(f"### {T['form_title']}")
        st.caption(T["form_help"])

        with st.form("profile"):
            current_age = st.number_input(T["current_age"], 0, 95, 25)

            def sel(key, label):
                opts = OPTS[key]
                return st.selectbox(
                    label,
                    options=[c for c, _ in opts],
                    format_func=lambda c, k=key: dict(OPTS[k])[c],
                )

            c1, c2 = st.columns(2)
            with c1:
                region    = sel("region", T["region"])
                residence = sel("residence", T["residence"])
                education = sel("education", T["education"])
                marital   = sel("marital", T["marital"])
                tv = st.selectbox(
                    T["tv"], [c for c, _ in OPTS["media"]],
                    format_func=lambda c: dict(OPTS["media"])[c],
                )
            with c2:
                religion = sel("religion", T["religion"])
                wealth   = sel("wealth", T["wealth"])
                hiv      = sel("hiv", T["hiv"])
                radio = st.selectbox(
                    T["radio"], [c for c, _ in OPTS["media"]],
                    format_func=lambda c: dict(OPTS["media"])[c],
                )

            submitted = st.form_submit_button(
                T["predict"], use_container_width=True, type="primary"
            )

    with col_res:
        st.markdown(f"### {T['results']}")

        if submitted:
            try:
                model, preprocessor = load_artifacts()
            except Exception as e:
                st.error(
                    "Could not load model artifacts. Make sure "
                    "'model_artifacts/gbsa_model.pkl' and "
                    "'model_artifacts/preprocessor.pkl' exist next to this script.\n\n"
                    f"Details: {e}"
                )
                st.stop()

            payload = {
                "current_age": float(current_age),
                "region": region, "religion": religion, "residence": residence,
                "wealth": wealth, "education": education, "marital_status": marital,
                "hiv_testing": hiv, "tv_exposure": tv, "radio_exposure": radio,
            }
            out = predict_risk(model, preprocessor, payload)

            # Metric cards
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f"<div class='metric-card'><div class='label'>{T['median']}</div>"
                f"<div class='value'>{out['median']} {T['years']}</div></div>",
                unsafe_allow_html=True,
            )
            m2.markdown(
                f"<div class='metric-card'><div class='label'>{T['window']}</div>"
                f"<div class='value'>{out['window'][0]} – {out['window'][1]} {T['years']}</div></div>",
                unsafe_allow_html=True,
            )
            m3.markdown(
                f"<div class='metric-card'><div class='label'>{T['risk']}</div>"
                f"<div class='value'><span class='risk-pill risk-{out['risk']}'>"
                f"{T[out['risk']]}</span></div></div>",
                unsafe_allow_html=True,
            )

            # Plain-language interpretation
            interp = T["interp_template"].format(
                median=out["median"],
                start=out["window"][0],
                end=out["window"][1],
                risk=T[out["risk"]],
                risk_word=T[f"risk_word_{out['risk']}"],
            )
            st.markdown(f"#### {T['interpretation']}")
            # Convert **bold** markdown into <strong> for the styled box
            import re as _re
            interp_html = _re.sub(
                r"\*\*(.+?)\*\*", r"<strong>\1</strong>", interp
            )
            st.markdown(
                f"<div class='interp-box'>{interp_html}</div>",
                unsafe_allow_html=True,
            )

            # Survival curve
            st.markdown(f"#### {T['curve_title']}")
            st.caption(T["curve_help"])
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ages = list(out["curve"].keys())
            probs = list(out["curve"].values())
            ax.plot(ages, probs, color="#2d5a6e", linewidth=2.4)
            ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
            ax.axvline(out["median"], color="#c97a4a", linestyle="--", linewidth=1)
            ax.set_xlabel("Age" if lang == "en" else "Âge")
            ax.set_ylabel("S(t)")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            fig.patch.set_alpha(0)
            ax.set_facecolor((1, 1, 1, 0.6))
            st.pyplot(fig, clear_figure=True)
        else:
            st.info(T["no_result"])

# -----------------------------------------------------------------------------
# TAB 2 — Variables
# -----------------------------------------------------------------------------
with tab_vars:
    st.markdown(f"### {T['variables_title']}")
    st.caption(T["variables_intro"])
    df_vars = pd.DataFrame(
        VARIABLE_DICT[lang],
        columns=[T["col_code"], T["col_var"], T["col_desc"]],
    )
    st.dataframe(df_vars, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3 — About the project
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown(f"### {T['about_title']}")

    st.markdown(f"#### {T['about_summary']}")
    st.write(T["about_summary_text"])

    st.markdown(f"#### {T['about_aim']}")
    st.write(T["about_aim_text"])

    st.markdown(f"#### {T['about_data']}")
    st.write(T["about_data_text"])

    st.markdown(f"#### {T['about_model']}")
    st.write(T["about_model_text"])

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("DHS Cameroon 2018 • GBSA survival model • Academic project")
