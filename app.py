import streamlit as st
import numpy as np
import joblib
import os
import time

# Custom CSS avec thème bleu et noir
st.markdown(
    """
    <style>
    /* Couleur de fond globale (noir) */
    body {
        background-color: #000000;
        color: #ffffff;
    }
    /* Conteneur principal stylisé */
    .main {
        background-color: #1a1a1a;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        margin: 2rem;
    }
    /* Titre en bleu */
    h2, h1 {
        color: #007bff;
        text-align: center;
    }
    /* Sous-titre en bleu clair */
    .subtitle {
        color: #66b3ff;
        text-align: center;
        font-size: 1.2rem;
    }
    /* Bouton stylisé en bleu avec texte blanc */
    .stButton button {
        background-color: #007bff;
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.8rem 1.2rem;
    }
    .stButton button:hover {
        background-color: #0056b3;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Encapsulation du contenu dans une div stylisée
st.markdown('<div class="main">', unsafe_allow_html=True)

# Titre et introduction
st.title("Prédiction de Possession de Compte Bancaire")
st.markdown('<p class="subtitle">Entrez les informations du client pour savoir s’il sera susceptible d’avoir un compte bancaire</p>', unsafe_allow_html=True)

# Animation de chargement pendant le chargement du modèle
with st.spinner("Chargement du modèle..."):
    time.sleep(1)

# Fonction pour charger le modèle avec mise en cache
@st.cache_resource
def load_model():
    model_path = "model_final.joblib"
    if not os.path.exists(model_path):
        st.error(f"❌ Le fichier {model_path} est introuvable. Assurez-vous qu'il est bien dans le dossier.")
        return None
    return joblib.load(model_path)

# Charger le modèle Xgboost
model_final = load_model()
if model_final is None:
    st.stop()
st.success("✅ Modèle chargé avec succès !")

# Fonction pour effectuer une prédiction
def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model_final.predict(input_array)
    if prediction[0] == 1:
        return "la personne est susceptible d'avoir un compte bancaire"
    else:
        return "la personne n'est pas susceptible d'avoir un compte bancaire"

# --- MAPPAGES pour encoder les variables catégorielles ---
country_mapping = {
    'Kenya': 0, 
    'Rwanda': 1, 
    'Tanzania': 2, 
    'Uganda': 3
}

location_mapping = {
    'Rural': 0, 
    'Urban': 1
}

cellphone_mapping = {
    'Yes': 1, 
    'No': 0
}

gender_mapping = {
    'Female': 0, 
    'Male': 1
}

relationship_mapping = {
    'Spouse': 0, 
    'Head of Household': 1, 
    'Other relative': 2, 
    'Child': 3, 
    'Parent': 4, 
    'Other non-relatives': 5
}

marital_mapping = {
    'Married/Living together': 0, 
    'Widowed': 1, 
    'Single/Never Married': 2, 
    'Divorced/Seperated': 3, 
    'Dont know': 4
}

education_mapping = {
    'Secondary education': 3, 
    'No formal education': 0, 
    'Vocational/Specialised training': 5,
    'Primary education': 2, 
    'Tertiary education': 4, 
    'Other/Dont know/RTA': 1
}

job_mapping = {
    'Self employed': 0, 
    'Government Dependent': 1, 
    'Formally employed Private': 2,
    'Informally employed': 3, 
    'Formally employed Government': 4, 
    'Farming and Fishing': 5,
    'Remittance Dependent': 6, 
    'Other Income': 7, 
    'Dont Know/Refuse to answer': 8, 
    'No Income': 9
}

# --- Saisie des données utilisateur ---

# Entrées numériques
household_size = st.number_input("Taille du ménage", min_value=1, step=1)
age_of_respondent = st.number_input("Âge du répondant", min_value=0, step=1)

# Variables catégorielles avec mapping
year = st.selectbox("Année", [2018, 2016, 2017])

education_level = st.selectbox("Niveau d'éducation", [
    "Secondary education", 
    "No formal education", 
    "Vocational/Specialised training", 
    "Primary education", 
    "Tertiary education", 
    "Other/Dont know/RTA"
])
education_level_encoded = education_mapping[education_level]

country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
country_encoded = country_mapping[country]

location_type = st.selectbox("Type de localisation", ["Rural", "Urban"])
location_type_encoded = location_mapping[location_type]

cellphone_access = st.selectbox("Accès au téléphone", ["Yes", "No"])
cellphone_access_encoded = cellphone_mapping[cellphone_access]

gender = st.selectbox("Genre", ["Female", "Male"])
gender_of_respondent_encoded = gender_mapping[gender]

relationship = st.selectbox("Relation avec le chef du ménage", [
    "Spouse", 
    "Head of Household", 
    "Other relative", 
    "Child", 
    "Parent", 
    "Other non-relatives"
])
relationship_with_head_encoded = relationship_mapping[relationship]

marital_status = st.selectbox("Statut matrimonial", [
    "Married/Living together", 
    "Widowed", 
    "Single/Never Married", 
    "Divorced/Seperated", 
    "Dont know"
])
marital_status_encoded = marital_mapping[marital_status]

job_type = st.selectbox("Type d'emploi", [
    "Self employed", 
    "Government Dependent", 
    "Formally employed Private", 
    "Informally employed", 
    "Formally employed Government", 
    "Farming and Fishing", 
    "Remittance Dependent", 
    "Other Income", 
    "Dont Know/Refuse to answer", 
    "No Income"
])
job_type_encoded = job_mapping[job_type]

# Bouton de prédiction avec animation
if st.button("Prédire"):
    input_data = [
        year,                              # Année
        household_size,                    # Taille du ménage
        age_of_respondent,                 # Âge du répondant
        education_level_encoded,           # Niveau d'éducation encodé
        country_encoded,                   # Pays encodé
        location_type_encoded,             # Type de localisation encodé
        cellphone_access_encoded,          # Accès au téléphone encodé
        gender_of_respondent_encoded,      # Genre du répondant encodé
        relationship_with_head_encoded,    # Relation avec le chef du ménage encodée
        marital_status_encoded,            # Statut matrimonial encodé
        job_type_encoded                   # Type d'emploi encodé
    ]
    
    #st.write("Données utilisateur :", input_data)
    
    with st.spinner("Réalisation de la prédiction..."):
        time.sleep(1)  # Simulation d'un temps de calcul
    result = predict(input_data)
    
    st.success(f"✅ Prédiction : {result}")
    # Animation si la prédiction est positive
    if "susceptible" in result:
        st.balloons()

# Fin de la div stylisée
st.markdown('</div>', unsafe_allow_html=True)
