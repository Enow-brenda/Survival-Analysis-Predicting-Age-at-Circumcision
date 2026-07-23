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
    pip install  scikit-survival scikit-learn joblib plotly

Run:
    streamlit run streamlit_app.py
"""

import base64
import re as _re
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Circumcision Age Predictor — DHS Cameroon 2018",
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Background image + dark custom CSS
# =============================================================================
def set_background(image_path: str):
    if not Path(image_path).exists():
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

        /* ── Global dark theme ── */
        .stApp {{
            background-image:
                linear-gradient(160deg, rgba(10,15,22,0.92) 0%, rgba(14,22,32,0.88) 50%, rgba(8,14,22,0.92) 100%),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #c0ced8;
        }}

        /* Restore Material Symbols for Streamlit icons */
        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span {{
            font-family: 'Material Symbols Outlined' !important;
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp label, .stApp li {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        /* ── Headings ── */
        h1, h2, h3, h4, h5, h6 {{ color: #e4eef4 !important; }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] > div {{
            background: rgba(14,20,30,0.97);
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stCaptionContainer {{
            color: #c8d8e4 !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.08) !important;
        }}

        /* ── Streamlit form inputs dark override ── */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stRadio > div {{
            background-color: rgba(22,32,44,0.9) !important;
            color: #d0dce6 !important;
            border-color: rgba(255,255,255,0.1) !important;
        }}
        .stSelectbox > div > div:hover,
        .stNumberInput > div > div > input:hover {{
            border-color: rgba(100,180,220,0.3) !important;
        }}
        .stSelectbox > div > div:focus-within,
        .stNumberInput > div > div > input:focus-within {{
            border-color: rgba(100,180,220,0.5) !important;
            box-shadow: 0 0 0 1px rgba(100,180,220,0.2) !important;
        }}

        /* ── Container padding ── */
        .block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; }}

        /* ── Metric Cards ── */
        .metric-card {{
            background: rgba(22,34,50,0.88);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 24px 20px 20px;
            text-align: center;
            box-shadow:
                0 4px 20px rgba(0,0,0,0.35),
                inset 0 1px 0 rgba(255,255,255,0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: fadeSlideUp 0.5s ease-out both;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow:
                0 8px 30px rgba(0,0,0,0.45),
                inset 0 1px 0 rgba(255,255,255,0.05);
        }}
        .metric-card .label {{
            font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
            color: #6a8898; font-weight: 600;
        }}
        .metric-card .value {{
            font-size: 26px; font-weight: 800; color: #e8f2f8; margin-top: 8px;
            letter-spacing: -0.5px;
        }}
        .metric-card .icon {{
            font-size: 28px; margin-bottom: 6px; opacity: 0.6;
        }}

        /* ── Risk Pills ── */
        .risk-pill {{
            display: inline-block; padding: 7px 18px; border-radius: 999px;
            font-weight: 700; font-size: 14px; letter-spacing: 0.3px;
        }}
        .risk-EARLY  {{ background: linear-gradient(135deg, #e8915a, #d4783f); color: white; box-shadow: 0 2px 10px rgba(212,120,63,0.4); }}
        .risk-MEDIUM {{ background: linear-gradient(135deg, #e8c56a, #d4b050); color: #2a2210; box-shadow: 0 2px 10px rgba(212,176,80,0.35); }}
        .risk-LATE   {{ background: linear-gradient(135deg, #d45a4a, #c03a2a); color: white; box-shadow: 0 2px 10px rgba(192,58,42,0.4); }}

        /* ── Interpretation box ── */
        .interp-box {{
            background: rgba(20,32,48,0.88);
            border-left: 4px solid #4a9ab8;
            border-radius: 12px;
            padding: 20px 22px;
            margin-top: 14px;
            color: #c0d4e0;
            line-height: 1.65;
            font-size: 15px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.25);
        }}

        /* ── Section cards (About tab) ── */
        .info-card {{
            background: rgba(20,32,48,0.80);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 16px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.25);
            transition: transform 0.15s ease;
        }}
        .info-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.35);
        }}
        .info-card .card-icon {{
            font-size: 32px; margin-bottom: 10px;
        }}
        .info-card .card-title {{
            font-size: 16px; font-weight: 700; color: #e0ecf2;
            margin-bottom: 8px;
        }}
        .info-card .card-text {{
            font-size: 14px; color: #8ea8b8; line-height: 1.6;
        }}

        /* ── Form section headers ── */
        .form-section {{
            font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px;
            color: #5a7a8a; font-weight: 600; margin: 8px 0 4px 0;
            padding-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,0.07);
        }}

        /* ── Sidebar info badge ── */
        .sidebar-badge {{
            background: rgba(20,36,52,0.9);
            border-radius: 12px;
            padding: 14px 16px;
            margin: 10px 0;
            border: 1px solid rgba(255,255,255,0.06);
        }}
        .sidebar-badge .badge-title {{
            font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
            color: #5a8a9a; font-weight: 600; margin-bottom: 4px;
        }}
        .sidebar-badge .badge-value {{
            font-size: 15px; font-weight: 700; color: #d8e8f0;
        }}

        /* ── Footer ── */
        .app-footer {{
            text-align: center; padding: 20px 0 10px; margin-top: 24px;
            border-top: 1px solid rgba(255,255,255,0.06);
        }}
        .app-footer span {{
            font-size: 12px; color: #5a7080; letter-spacing: 0.3px;
        }}

        /* ── Animations ── */
        @keyframes fadeSlideUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to   {{ opacity: 1; }}
        }}
        .fade-in {{ animation: fadeIn 0.4s ease-out both; }}

        /* ── Plotly chart bg override ── */
        .stPlotlyChart {{ background: transparent !important; }}
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
            "the probability of remaining uncircumcised at each age and reports the "
            "predicted median age, a \u00b12 year window, and a risk category."
        ),
        "tab_predict": "\U0001f52c Predictor",
        "tab_variables": "\U0001f4cb Variables",
        "tab_about": "\U0001f4cc About the project",
        "form_title": "Individual profile",
        "form_help": "Fill in the respondent's characteristics.",
        "section_demographics": "Demographics",
        "section_socioeconomic": "Socioeconomic",
        "section_exposure": "Media & Health",
        "current_age": "Current age (years)",
        "region": "Region", "religion": "Religion", "residence": "Residence",
        "wealth": "Wealth quintile", "education": "Education", "marital": "Marital status",
        "hiv": "Ever tested for HIV", "tv": "TV exposure", "radio": "Radio exposure",
        "predict": "Predict", "results": "Prediction",
        "median": "Median age", "window": "Window", "risk": "Risk",
        "curve_title": "Estimated survival curve",
        "curve_help": "Probability of still being uncircumcised at each age. Hover for details, drag to zoom.",
        "years": "yrs", "no_result": "Click Predict to see results.",
        "EARLY": "Early", "MEDIUM": "Medium", "LATE": "Late",
        "interpretation": "Plain-language interpretation",
        "interp_template": (
            "Based on this profile, the model estimates that a man with these "
            "characteristics is most likely to be circumcised around age **{median}**, "
            "typically between **{start}** and **{end}** years old. "
            "The risk category is **{risk}**, meaning circumcision tends to happen "
            "{risk_word} compared to the national pattern. "
            "The curve below shows the probability of being still uncircumcised at "
            "each age \u2014 when it drops below 0.5, half of similar individuals have "
            "already been circumcised."
        ),
        "risk_word_EARLY": "earlier (often in childhood)",
        "risk_word_MEDIUM": "around adolescence",
        "risk_word_LATE": "later (in adulthood, or not at all)",
        "variables_title": "Variables used in the model",
        "variables_intro": "All variables come from the DHS / EDS Cameroon 2018 (CMMR71FL) men's individual recode.",
        "about_title": "About this project",
        "about_summary": "Project summary",
        "about_summary_text": (
            "This academic project answers the question: \"How old were you when you were "
            "circumcised?\" using the DHS Cameroon 2018 dataset. We built a survival-analysis "
            "pipeline (Kaplan-Meier, Cox PH, and a final Gradient Boosting Survival Analysis "
            "model) to estimate, for any individual profile, the most likely age at "
            "circumcision and the uncertainty around that estimate."
        ),
        "about_aim": "Aim",
        "about_aim_text": (
            "The aim is to identify the demographic, socio-economic, religious, and "
            "regional factors that influence the timing of male circumcision in Cameroon, "
            "and to provide a simple bilingual tool that public-health stakeholders and "
            "students can use to explore predictions for different population profiles."
        ),
        "about_data": "Data",
        "about_data_text": (
            "Source: Demographic and Health Survey (DHS) \u2014 Cameroon 2018, men's recode "
            "(CMMR71FL), restricted to respondents with a valid circumcision status (MV483) "
            "and age at event (MV483A)."
        ),
        "about_model": "Model",
        "about_model_text": (
            "Final model: Gradient Boosting Survival Analysis (scikit-survival). Inputs "
            "are one-hot encoded categorical variables plus current age. Output is the "
            "estimated survival function S(t), from which we derive the median age, a "
            "\u00b12 year window, and a risk category (Early / Medium / Late)."
        ),
        "col_code": "Code", "col_var": "Variable", "col_desc": "Description",
        "sidebar_model": "Model",
        "sidebar_model_detail": "GBSA (C-Index: 0.718)",
        "sidebar_data": "Dataset",
        "sidebar_data_detail": "DHS Cameroon 2018",
        "sidebar_sample": "Sample",
        "sidebar_sample_detail": "~6,551 respondents",
    },
    "fr": {
        "app_title": "Pr\u00e9dicteur d'\u00e2ge \u00e0 la circoncision",
        "app_subtitle": "Analyse de survie sur l'EDS Cameroun 2018 (CMMR71FL)",
        "methodology": "Methodologie",
        "methodology_text": (
            "Cet outil utilise un mod\u00e8le Gradient Boosting Survival Analysis (GBSA) entra\u00een\u00e9 "
            "sur l'EDS Cameroun 2018. Il estime la courbe de survie S(t) \u2014 la probabilit\u00e9 de "
            "rester non circoncis \u00e0 chaque \u00e2ge \u2014 et fournit l'\u00e2ge m\u00e9dian pr\u00e9dit, une fen\u00eate "
            "\u00b12 ans et une cat\u00e9gorie de risque."
        ),
        "tab_predict": "\U0001f52c Pr\u00e9dicteur",
        "tab_variables": "\U0001f4cb Variables",
        "tab_about": "\U0001f4cc \u00c0 propos du projet",
        "form_title": "Profil individuel",
        "form_help": "Renseignez les caract\u00e9ristiques du r\u00e9pondant.",
        "section_demographics": "D\u00e9mographie",
        "section_socioeconomic": "Socio-\u00e9conomie",
        "section_exposure": "M\u00e9dias & Sant\u00e9",
        "current_age": "\u00c2ge actuel (ann\u00e9es)",
        "region": "R\u00e9gion", "religion": "Religion", "residence": "R\u00e9sidence",
        "wealth": "Quintile de richesse", "education": "\u00c9ducation", "marital": "Statut matrimonial",
        "hiv": "Test VIH d\u00e9j\u00e0 fait", "tv": "Exposition TV", "radio": "Exposition radio",
        "predict": "Pr\u00e9dire", "results": "Pr\u00e9diction",
        "median": "\u00c2ge m\u00e9dian", "window": "Fen\u00eate", "risk": "Risque",
        "curve_title": "Courbe de survie estim\u00e9e",
        "curve_help": "Probabilit\u00e9 d'\u00eatre encore non circoncis \u00e0 chaque \u00e2ge. Survolez pour les d\u00e9tails, glissez pour zoomer.",
        "years": "ans", "no_result": "Cliquez sur Pr\u00e9dire pour voir les r\u00e9sultats.",
        "EARLY": "Pr\u00e9coce", "MEDIUM": "Moyen", "LATE": "Tardif",
        "interpretation": "Interpr\u00e9tation simple",
        "interp_template": (
            "D'apr\u00e8s ce profil, le mod\u00e8le estime qu'un homme avec ces caract\u00e9ristiques "
            "a le plus de chances d'\u00eatre circoncis vers l'\u00e2ge de **{median}** ans, "
            "g\u00e9n\u00e9ralement entre **{start}** et **{end}** ans. "
            "La cat\u00e9gorie de risque est **{risk}**, ce qui signifie que la circoncision "
            "tend \u00e0 survenir {risk_word} par rapport au sch\u00e9ma national. "
            "La courbe ci-dessous montre la probabilit\u00e9 d'\u00eatre encore non circoncis \u00e0 "
            "chaque \u00e2ge \u2014 lorsqu'elle passe sous 0,5, la moiti\u00e9 des individus similaires "
            "sont d\u00e9j\u00e0 circoncis."
        ),
        "risk_word_EARLY": "plus t\u00f4t (souvent durant l'enfance)",
        "risk_word_MEDIUM": "vers l'adolescence",
        "risk_word_LATE": "plus tard (\u00e0 l'\u00e2ge adulte, ou pas du tout)",
        "variables_title": "Variables utilis\u00e9es dans le mod\u00e8le",
        "variables_intro": "Toutes les variables proviennent de l'EDS Cameroun 2018 (CMMR71FL), recode individuel hommes.",
        "about_title": "\u00c0 propos de ce projet",
        "about_summary": "R\u00e9sum\u00e9 du projet",
        "about_summary_text": (
            "Ce projet acad\u00e9mique r\u00e9pond \u00e0 la question : \u00ab Quel \u00e2ge aviez-vous lors de votre "
            "circoncision ? \u00bb \u00e0 partir de l'EDS Cameroun 2018. Nous avons construit un pipeline "
            "d'analyse de survie (Kaplan-Meier, Cox PH, puis Gradient Boosting Survival Analysis) "
            "pour estimer, \u00e0 partir de tout profil individuel, l'\u00e2ge le plus probable de "
            "circoncision et l'incertitude associ\u00e9e."
        ),
        "about_aim": "Objectif",
        "about_aim_text": (
            "L'objectif est d'identifier les facteurs d\u00e9mographiques, socio-\u00e9conomiques, "
            "religieux et r\u00e9gionaux qui influencent le moment de la circoncision masculine "
            "au Cameroun, et de fournir un outil bilingue simple permettant aux acteurs de "
            "sant\u00e9 publique et aux \u00e9tudiants d'explorer les pr\u00e9dictions selon diff\u00e9rents profils."
        ),
        "about_data": "Donn\u00e9es",
        "about_data_text": (
            "Source : Enqu\u00eate D\u00e9mographique et de Sant\u00e9 (EDS) \u2014 Cameroun 2018, recode hommes "
            "(CMMR71FL), restreint aux r\u00e9pondants avec un statut de circoncision valide (MV483) "
            "et un \u00e2ge \u00e0 l'\u00e9v\u00e9nement (MV483A)."
        ),
        "about_model": "Mod\u00e8le",
        "about_model_text": (
            "Mod\u00e8le final : Gradient Boosting Survival Analysis (scikit-survival). Les entr\u00e9es "
            "sont des variables cat\u00e9gorielles encod\u00e9es en one-hot plus l'\u00e2ge actuel. La sortie "
            "est la fonction de survie estim\u00e9e S(t), d'o\u00f9 l'on d\u00e9rive l'\u00e2ge m\u00e9dian, une fen\u00eate "
            "\u00b12 ans et une cat\u00e9gorie de risque (Pr\u00e9coce / Moyen / Tardif)."
        ),
        "col_code": "Code", "col_var": "Variable", "col_desc": "Description",
        "sidebar_model": "Mod\u00e8le",
        "sidebar_model_detail": "GBSA (C-Index: 0.718)",
        "sidebar_data": "Donn\u00e9es",
        "sidebar_data_detail": "EDS Cameroun 2018",
        "sidebar_sample": "\u00c9chantillon",
        "sidebar_sample_detail": "~6 551 r\u00e9pondants",
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
        ("MV190",  "Wealth index",         "Household wealth quintile (poorest \u2192 richest)."),
        ("MV130",  "Religion",             "Religion of the respondent (Catholic, Protestant, Muslim, Animist, \u2026)."),
        ("MV501",  "Marital status",       "Current marital status (never in union, married, \u2026)."),
        ("MV158",  "Radio exposure",       "Frequency of listening to the radio."),
        ("MV159",  "TV exposure",          "Frequency of watching television."),
        ("MV781",  "HIV testing",          "Whether the respondent has ever been tested for HIV."),
        ("MV005",  "Sample weight",        "Individual sample weight (used in survey-weighted analyses)."),
        ("MV021",  "PSU",                  "Primary sampling unit identifier."),
        ("MV022",  "Strata",               "Sample stratum identifier."),
    ],
    "fr": [
        ("MV483",  "Statut de circoncision", "Indique si le r\u00e9pondant est circoncis (indicateur d'\u00e9v\u00e9nement)."),
        ("MV483A", "\u00c2ge \u00e0 la circoncision",  "\u00c2ge d\u00e9clar\u00e9 au moment de la circoncision (temps de l'\u00e9v\u00e9nement)."),
        ("MV012",  "\u00c2ge actuel",             "\u00c2ge actuel du r\u00e9pondant en ann\u00e9es r\u00e9volues."),
        ("MV024",  "R\u00e9gion",                 "R\u00e9gion de r\u00e9sidence (12 r\u00e9gions administratives du Cameroun)."),
        ("MV025",  "R\u00e9sidence",              "Type de lieu de r\u00e9sidence : urbain ou rural."),
        ("MV106",  "Niveau d'\u00e9ducation",     "Plus haut niveau atteint (aucun, primaire, secondaire, sup\u00e9rieur)."),
        ("MV190",  "Indice de richesse",     "Quintile de richesse du m\u00e9nage (tr\u00e8s pauvre \u2192 tr\u00e8s riche)."),
        ("MV130",  "Religion",               "Religion du r\u00e9pondant (catholique, protestant, musulman, animiste, \u2026)."),
        ("MV501",  "Statut matrimonial",     "Statut matrimonial actuel (jamais en union, mari\u00e9, \u2026)."),
        ("MV158",  "Exposition radio",       "Fr\u00e9quence d'\u00e9coute de la radio."),
        ("MV159",  "Exposition TV",          "Fr\u00e9quence de visionnage de la t\u00e9l\u00e9vision."),
        ("MV781",  "Test VIH",               "Si le r\u00e9pondant a d\u00e9j\u00e0 fait un test VIH."),
        ("MV005",  "Poids de l'\u00e9chantillon", "Poids individuel d'\u00e9chantillonnage (analyses pond\u00e9r\u00e9es)."),
        ("MV021",  "UPE",                    "Identifiant de l'unit\u00e9 primaire d'\u00e9chantillonnage."),
        ("MV022",  "Strate",                 "Identifiant de la strate d'\u00e9chantillonnage."),
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
            ("5", "Far North" if L == "en" else "Extr\u00eame-Nord"),
            ("6", "Littoral"),
            ("7", "North" if L == "en" else "Nord"),
            ("8", "North-West" if L == "en" else "Nord-Ouest"),
            ("9", "West" if L == "en" else "Ouest"),
            ("10", "South" if L == "en" else "Sud"),
            ("11", "South-West" if L == "en" else "Sud-Ouest"),
            ("12", "Yaound\u00e9"),
        ],
        "religion": [
            ("1", "Catholic" if L == "en" else "Catholique"),
            ("2", "Protestant"),
            ("3", "Other Christian" if L == "en" else "Autre chr\u00e9tien"),
            ("4", "Muslim" if L == "en" else "Musulman"),
            ("5", "Animist" if L == "en" else "Animiste"),
            ("96", "Other / None" if L == "en" else "Autre / Aucune"),
        ],
        "residence": [("1", "Urban" if L == "en" else "Urbain"), ("2", "Rural")],
        "wealth": [
            ("1", "Poorest" if L == "en" else "Tr\u00e8s pauvre"),
            ("2", "Poorer"  if L == "en" else "Pauvre"),
            ("3", "Middle"  if L == "en" else "Moyen"),
            ("4", "Richer"  if L == "en" else "Riche"),
            ("5", "Richest" if L == "en" else "Tr\u00e8s riche"),
        ],
        "education": [
            ("0", "No education" if L == "en" else "Aucune"),
            ("1", "Primary"      if L == "en" else "Primaire"),
            ("2", "Secondary"    if L == "en" else "Secondaire"),
            ("3", "Higher"       if L == "en" else "Sup\u00e9rieure"),
        ],
        "marital": [
            ("0", "Never in union"   if L == "en" else "Jamais en union"),
            ("1", "Married"          if L == "en" else "Mari\u00e9(e)"),
            ("2", "Living together"  if L == "en" else "En concubinage"),
            ("3", "Widowed"          if L == "en" else "Veuf/Veuve"),
            ("4", "Divorced"         if L == "en" else "Divorc\u00e9(e)"),
            ("5", "Separated"        if L == "en" else "S\u00e9par\u00e9(e)"),
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
# Plotly survival curve builder (dark theme)
# =============================================================================
def build_survival_chart(curve_data: dict, median: int, lang: str):
    ages = list(curve_data.keys())
    probs = list(curve_data.values())

    risk_color = {"EARLY": "#e8915a", "MEDIUM": "#e8c56a", "LATE": "#d45a4a"}
    risk = "EARLY" if median < 14 else ("MEDIUM" if median <= 17 else "LATE")
    main_color = risk_color[risk]

    r, g, b = int(main_color[1:3], 16), int(main_color[3:5], 16), int(main_color[5:7], 16)

    fig = go.Figure()

    # Survival curve with fill
    fig.add_trace(go.Scatter(
        x=ages, y=probs,
        mode="lines",
        name="S(t)",
        line=dict(color=main_color, width=3, shape="hv"),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.12)",
        hovertemplate="Age: %{x}<br>S(t): %{y:.3f}<extra></extra>",
    ))

    # 0.5 threshold line
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="rgba(160,180,200,0.35)", line_width=1.5,
        annotation_text="S(t) = 0.5",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="rgba(160,180,200,0.6)",
    )

    # Median vertical line
    fig.add_vline(
        x=median, line_dash="dot", line_color=main_color, line_width=2,
        annotation_text=f"Median = {median}",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color=main_color,
    )

    # Scatter dot at median crossing
    median_prob = curve_data.get(median, 0.5)
    fig.add_trace(go.Scatter(
        x=[median], y=[median_prob],
        mode="markers",
        marker=dict(color=main_color, size=12, symbol="circle",
                    line=dict(color="#0e1420", width=2)),
        showlegend=False,
        hovertemplate=f"Median: {median} yrs<br>S(t): {median_prob:.3f}<extra></extra>",
    ))

    age_label = "Age" if lang == "en" else "\u00c2ge"
    fig.update_layout(
        xaxis_title=age_label,
        yaxis_title="S(t)",
        yaxis=dict(range=[0, 1.05], dtick=0.2, gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(dtick=5, gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(16,24,36,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#a0b4c0"),
        margin=dict(l=50, r=20, t=30, b=50),
        height=400,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(22,34,50,0.95)",
            bordercolor=main_color,
            font_size=13,
            font_color="#e0ecf2",
            font_family="Inter, sans-serif",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font_size=12, font_color="#a0b4c0",
        ),
    )

    return fig

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### \U0001f310 Language / Langue")
    lang_label = st.radio(
        "lang", ["English", "Fran\u00e7ais"], horizontal=True, label_visibility="collapsed"
    )
    lang = "en" if lang_label == "English" else "fr"
    T = TRANSLATIONS[lang]
    OPTS = options(lang)

    st.markdown("---")

    # Model info badges
    st.markdown(
        f"""<div class="sidebar-badge">
            <div class="badge-title">{T['sidebar_model']}</div>
            <div class="badge-value">{T['sidebar_model_detail']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="sidebar-badge">
            <div class="badge-title">{T['sidebar_data']}</div>
            <div class="badge-value">{T['sidebar_data_detail']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="sidebar-badge">
            <div class="badge-title">{T['sidebar_sample']}</div>
            <div class="badge-value">{T['sidebar_sample_detail']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander(f"{T['methodology']}"):
        st.caption(T["methodology_text"])

T = TRANSLATIONS[lang]
OPTS = options(lang)

# =============================================================================
# Header
# =============================================================================
st.markdown(f"# \U0001fa7a {T['app_title']}")
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
            # -- Demographics section --
            st.markdown(f'<div class="form-section">{T["section_demographics"]}</div>', unsafe_allow_html=True)
            current_age = st.number_input(T["current_age"], 15, 59, 25)

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
            with c2:
                religion = sel("religion", T["religion"])
                marital  = sel("marital", T["marital"])

            # -- Socioeconomic section --
            st.markdown(f'<div class="form-section">{T["section_socioeconomic"]}</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                education = sel("education", T["education"])
            with c4:
                wealth = sel("wealth", T["wealth"])

            # -- Media & Health section --
            st.markdown(f'<div class="form-section">{T["section_exposure"]}</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                tv = st.selectbox(
                    T["tv"], [c for c, _ in OPTS["media"]],
                    format_func=lambda c: dict(OPTS["media"])[c],
                )
            with c6:
                radio = st.selectbox(
                    T["radio"], [c for c, _ in OPTS["media"]],
                    format_func=lambda c: dict(OPTS["media"])[c],
                )
            hiv = sel("hiv", T["hiv"])

            st.markdown("")
            submitted = st.form_submit_button(
                T["predict"], use_container_width=True, type="primary"
            )

    with col_res:
        st.markdown(f"### {T['results']}")

        if submitted:
            with st.spinner("..."):
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

            # -- Metric cards --
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f"""<div class="metric-card" style="animation-delay: 0s">
                    <div class="icon">\U0001f4c5</div>
                    <div class="label">{T['median']}</div>
                    <div class="value">{out['median']} <span style="font-size:14px;font-weight:500;color:#6a8898">{T['years']}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )
            m2.markdown(
                f"""<div class="metric-card" style="animation-delay: 0.1s">
                    <div class="icon">\U0001f4c6</div>
                    <div class="label">{T['window']}</div>
                    <div class="value">{out['window'][0]} \u2013 {out['window'][1]} <span style="font-size:14px;font-weight:500;color:#6a8898">{T['years']}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )
            m3.markdown(
                f"""<div class="metric-card" style="animation-delay: 0.2s">
                    <div class="icon">\U0001f6e1\ufe0f</div>
                    <div class="label">{T['risk']}</div>
                    <div class="value"><span class="risk-pill risk-{out['risk']}">{T[out['risk']]}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("")

            # -- Plain-language interpretation --
            interp = T["interp_template"].format(
                median=out["median"],
                start=out["window"][0],
                end=out["window"][1],
                risk=T[out["risk"]],
                risk_word=T[f"risk_word_{out['risk']}"],
            )
            interp_html = _re.sub(
                r"\*\*(.+?)\*\*", r"<strong>\1</strong>", interp
            )
            st.markdown(
                f"""<div class="interp-box">
                    <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#4a9ab8;margin-bottom:8px">
                        \U0001f4a1 {T['interpretation']}
                    </div>
                    {interp_html}
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("")

            # -- Survival curve (Plotly) --
            st.markdown(f"#### {T['curve_title']}")
            st.caption(T["curve_help"])
            fig = build_survival_chart(out["curve"], out["median"], lang)
            st.plotly_chart(fig, use_container_width=True, config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })

        else:
            st.markdown(
                f"""<div style="text-align:center;padding:60px 20px;color:#5a7a8a">
                    <div style="font-size:48px;margin-bottom:12px;opacity:0.3">\U0001f4ca</div>
                    <div style="font-size:16px;font-weight:500">{T['no_result']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

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
    st.dataframe(
        df_vars,
        use_container_width=True,
        hide_index=True,
        height=520,
        key=f"vars_table_{lang}",
        column_config={
            T["col_code"]: st.column_config.TextColumn(
                T["col_code"], width="small",
            ),
            T["col_var"]: st.column_config.TextColumn(
                T["col_var"], width="medium",
            ),
            T["col_desc"]: st.column_config.TextColumn(
                T["col_desc"], width="large",
            ),
        },
    )

# -----------------------------------------------------------------------------
# TAB 3 — About the project
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown(f"### {T['about_title']}")

    cards = [
        ("\U0001f4ca", T["about_summary"], T["about_summary_text"]),
        ("\U0001f3af", T["about_aim"], T["about_aim_text"]),
        ("\U0001f4c2", T["about_data"], T["about_data_text"]),
        ("\U0001f9e0", T["about_model"], T["about_model_text"]),
    ]

    for icon, title, text in cards:
        st.markdown(
            f"""<div class="info-card fade-in">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-text">{text}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# =============================================================================
# Footer
# =============================================================================
st.markdown(
    f"""<div class="app-footer">
        <span>DHS Cameroon 2018 \u00b7 GBSA survival model \u00b7 Academic project</span>
    </div>""",
    unsafe_allow_html=True,
)
