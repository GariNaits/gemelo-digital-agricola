#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 19:28:39 2025

@author: jordiborras
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import seaborn as sns
import joblib

# Cargar datos
df = pd.read_csv(r"/Users/jordiborras/Documents/Universitat/UNIR/3.Machine Learning/Treballs/Implementacion gemelo Digital/GitHub/gemelo-digital-agricola/Smart_Farming_Crop_Yield_2024.csv")

# Crear etiqueta binaria 'yield_label' usando el percentil 70
threshold_70 = df['yield_kg_per_hectare'].quantile(0.70)
df['yield_label'] = (df['yield_kg_per_hectare'] >= threshold_70).astype(int)

# Cargar modelo
import joblib
model = joblib.load("tree_model.pkl")
X = df[model.feature_names_in_]

# Interfaz de usuario
st.title("Gemelo Digital para predecir y optimizar la cosecha agrícola")
st.header("1. Árbol de Decisión – Clasificación de cosechas Alto Rendimiento")

# Evaluación
y_true = df['yield_label']
y_pred = model.predict(X)
report = classification_report(y_true, y_pred, output_dict=True)
st.write("Métricas del modelo")
st.dataframe(pd.DataFrame(report).transpose())

# Visualización del árbol
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model.named_steps['classifier'],
          feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
          class_names=["No Alta", "Alta"],
          filled=True, ax=ax)
st.pyplot(fig)

# Anomalías
st.header("2.Detección de Anomalías – Isolation Forest")
contamination = st.slider("Nivel de contaminación", 0.01, 0.2, 0.10, step=0.01)

# Entrenar modelo de anomalías
iso = IsolationForest(contamination=contamination, random_state=42)
features = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
            'humidity_%', 'sunlight_hours', 'pesticide_usage_ml', 'NDVI_index']
df['anomaly'] = iso.fit_predict(df[features])
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anómalo'})

st.write(f"Total de anómalos detectados: {(df['anomaly'] == -1).sum()}")

# Scatter plot
st.write("Visualización de anomalías")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='temperature_C', y='soil_moisture_%',
                hue='anomaly_label', palette='Set1', ax=ax2)
st.pyplot(fig2)
