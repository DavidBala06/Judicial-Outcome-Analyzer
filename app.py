import streamlit as st
import pandas as pd
import joblib

# --- 1. SETĂRI PAGINĂ ---
st.set_page_config(page_title="LexPredict: Judicial AI", page_icon="⚖️", layout="centered")

# --- 2. ÎNCĂRCAREA MODELULUI ---
# Folosim @st.cache_resource ca să încărcăm modelul o singură dată (pentru viteză)
@st.cache_resource
def load_components():
    model = joblib.load('lexpredict_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, model_columns, le

model, model_columns, label_encoder = load_components()

# --- 3. INTERFAȚA GRAFICĂ (UI) ---
st.title("⚖️ LexPredict: Judicial Outcome Analyzer")
st.markdown("""
Introduceți datele inculpatului și detaliile speței pentru a estima cel mai probabil verdict. 
*Notă: Acesta este un instrument predictiv bazat pe date istorice.*
""")

st.divider()

# Împărțim ecranul în două coloane pentru un design mai curat
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Date Biografice")
    age = st.slider("Vârsta Inculpatului", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gen", ["Male", "Female", "Other"])
    race = st.selectbox("Rasă/Etnie", ["White", "Black", "Hispanic", "Asian", "Other"])

with col2:
    st.subheader("📜 Istoric și Risc")
    prior_convictions = st.number_input("Număr Condamnări Anterioare", min_value=0, max_value=20, value=0)
    risk_score = st.slider("Scor de Risc (Z-Score/Evaluare)", min_value=1, max_value=10, value=5)

st.divider()

st.subheader("🔨 Detaliile Speței Curente")
col3, col4 = st.columns(2)

with col3:
    crime_type = st.selectbox("Tipul Infracțiunii", ["Assault", "Fraud", "Vandalism", "Drug Offense", "Cybercrime"])
with col4:
    arrest_made = st.radio("Arestat la fața locului?", ["Yes", "No"], horizontal=True)
    bail_granted = st.radio("Eliberat pe cauțiune?", ["Yes", "No"], horizontal=True)

st.divider()

# --- 4. LOGICA DE PREDICȚIE ---
if st.button("🔮 Generează Predicția Verdictului", use_container_width=True, type="primary"):
    
    # Creăm un dicționar cu datele introduse de utilizator
    input_data = {
        'Offender_Age': age,
        'Offender_Gender': gender,
        'Offender_Race': race,
        'Crime_Type': crime_type,
        'Prior_Convictions': prior_convictions,
        'Risk_Score': risk_score,
        'Arrest_Made': arrest_made,
        'Bail_Granted': bail_granted
    }
    
    # Transformăm într-un DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Aplicăm One-Hot Encoding la fel cum am făcut la antrenare
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # IMPORTANT: Ne asigurăm că datele noi au FIX aceleași coloane ca datele de antrenament
    # Dacă utilizatorul alege "Male", coloana "Female" va lipsi din input_encoded. O adăugăm cu valoarea 0.
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Reordonăm coloanele ca să se potrivească perfect cu așteptările modelului
    input_encoded = input_encoded[model_columns]
    
    # Facem predicția
    prediction_encoded = model.predict(input_encoded)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    # Extragem probabilitățile (cât de sigur e modelul pe fiecare verdict)
    probabilities = model.predict_proba(input_encoded)[0]
    
    # --- 5. AFIȘAREA REZULTATELOR ---
    st.success(f"### Verdict Estimat: **{prediction_label.upper()}**")
    
    st.markdown("#### Nivelul de încredere al modelului (Probabilități):")
    
    # Creăm un mic grafic de bare pentru probabilități
    prob_df = pd.DataFrame({
        'Verdict': label_encoder.classes_,
        'Probabilitate (%)': [round(p * 100, 1) for p in probabilities]
    }).sort_values(by='Probabilitate (%)', ascending=False)
    
    st.bar_chart(prob_df.set_index('Verdict'))