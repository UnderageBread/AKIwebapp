import os
os.chdir(os.path.split(__file__)[0])

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "胱抑素C": "Cystatin C",
    "手术时长": "Surgery Duration",
    "eGFR": "eGFR",
    "尿酸": "Uric Acid",
    "总胆红素": "Total Bilirubin",
    "胰岛素": "Insulin",
    "血红蛋白量": "Hemoglobin",
    "尿素": "Urea",
    "是否使用胶体": "Colloid Use",
    "中性粒细胞计数": "Neutrophil Count",
    "凝血酶原时间": "Prothrombin Time",
    "ASA12": "ASA12",
}

@st.cache_resource
def load_assets():
    with open('webapp_assets/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('webapp_assets/trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('webapp_assets/selected_features.json', 'r', encoding='utf-8') as f:
        features = json.load(f)
    with open('webapp_assets/feature_meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return scaler, models, features, meta

scaler, models, selected_features, feature_meta = load_assets()

st.title('AKI Prediction System')

model_names = list(models.keys())
default_idx = model_names.index('XGBoost') if 'XGBoost' in model_names else 0
chosen_model = st.selectbox('Select Model', model_names, index=default_idx)

st.markdown('---')
st.subheader('Input Feature Values')

input_values = {}
cols = st.columns(2)

for i, feat in enumerate(selected_features):
    info = feature_meta[feat]
    label = FEATURE_LABELS.get(feat, feat)
    col = cols[i % 2]
    if info['type'] == 'categorical':
        options = info['values']
        input_values[feat] = col.selectbox(label, options, key=feat)
    else:
        input_values[feat] = col.number_input(
            label,
            min_value=float(info['min']),
            max_value=float(info['max']),
            value=float(info['mean']),
            step=float((info['max'] - info['min']) / 100),
            key=feat
        )

st.markdown('---')

if st.button('Predict', type='primary'):
    input_df = pd.DataFrame([input_values], columns=selected_features).astype(float)
    input_scaled = scaler.transform(input_df)
    model = models[chosen_model]
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    if pred == 1:
        st.error('Prediction: **AKI Positive** (At Risk)')
    else:
        st.success('Prediction: **AKI Negative** (Not at Risk)')

    st.metric('AKI Probability', f'{prob[1]*100:.1f}%')
    st.metric('Non-AKI Probability', f'{prob[0]*100:.1f}%')