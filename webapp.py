import os
os.chdir(os.path.split(__file__)[0])

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# FEATURE_LABELS = {
#     "胱抑素C": "Cystatin C",
#     "手术时长": "Surgery Duration",
#     "eGFR": "eGFR",
#     "尿酸": "Uric Acid",
#     "总胆红素": "Total Bilirubin",
#     "胰岛素": "Insulin",
#     "血红蛋白量": "Hemoglobin",
#     "尿素": "Urea",
#     "是否使用胶体": "Colloid Use",
#     "中性粒细胞计数": "Neutrophil Count",
#     "凝血酶原时间": "Prothrombin Time",
#     "ASA12": "ASA12",
# }

FEATURE_LABELS = {
    "胱抑素C": "Cystatin C (mg/L) ",
    "手术时长": "Surgery Duration (min)",
    "eGFR": "estimated Glomerular Filtration Rate (mL/min/1.73 m²)",
    "尿酸": "Uric Acid (μmol/L) ",
    "总胆红素": "Total Bilirubin (μmol/L)",
    "胰岛素": "Insulin Use",
    "血红蛋白量": "Hemoglobin (g/L)",
    "尿素": "Blood Urea Nitrogen (mmol/L)",
    "是否使用胶体": "Colloid Use",
    "中性粒细胞计数": "Neutrophil Count (×10⁹/L)",
    "凝血酶原时间": "Prothrombin Time (s)",
    "ASA12": "ASA Ⅰ/Ⅱ",
}
@st.cache_resource
def load_assets():
    with open('webapp_assets/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('webapp_assets/XGBoost.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('webapp_assets/selected_features.json', 'r', encoding='utf-8') as f:
        features = json.load(f)
    with open('webapp_assets/feature_meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return scaler, model, features, meta

scaler, model, selected_features, feature_meta = load_assets()

st.title('Predictive System for Postoperative Acute Kidney Injury in the Elderly')

st.selectbox('Select Model', ['XGBoost'])

st.markdown('---')
st.subheader('Input Feature Values')

input_values = {}
cols = st.columns(2)
# st.write(feature_meta)
# 胰岛素
# 是否使用胶体
# ASA12
feature_meta['胰岛素']['type'] = 'categorical'
feature_meta['是否使用胶体']['type'] = 'categorical'
feature_meta['ASA12']['type'] = 'categorical'
for i, feat in enumerate(selected_features):
    info = feature_meta[feat]
    label = FEATURE_LABELS.get(feat, feat)
    col = cols[i % 2]
    if info['type'] == 'categorical':
        # options = info['values']
        options = [0,1]
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
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    if pred == 1:
        st.error('Prediction: **AKI Positive** (At Risk)')
    else:
        st.success('Prediction: **AKI Negative** (Not at Risk)')

    st.metric('AKI Probability', f'{prob[1]*100:.1f}%')
    st.metric('Non-AKI Probability', f'{prob[0]*100:.1f}%')
