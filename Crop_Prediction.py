#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:50:42 2025

@author: jordiborras
"""
# Árbol de decisión para predicción binaria de alta producción agrícola (percentil 70)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv(r"/Users/jordiborras/Documents/Universitat/UNIR/3.Machine Learning/Treballs/Implementacion gemelo Digital/GitHub/gemelo-digital-agricola/Smart_Farming_Crop_Yield_2024.csv")

# Crear una etiqueta binaria: 1 = Alta producción (>= percentil 70), 0 = No Alta
threshold_70 = df['yield_kg_per_hectare'].quantile(0.70)
df['yield_label'] = (df['yield_kg_per_hectare'] >= threshold_70).astype(int)

# Selección de variables predictoras
numeric_features = ['soil_moisture_%', 'soil_pH', 'temperature_C',
                    'rainfall_mm', 'humidity_%', 'sunlight_hours',
                    'pesticide_usage_ml', 'NDVI_index', 'total_days']
categorical_features = ['crop_type']
features = numeric_features + categorical_features

X = df[features]
y = df['yield_label']

# Preprocesamiento: codificación de variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Definición del pipeline con árbol de decisión
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=6, random_state=0))
])

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
tree_pipeline.fit(X_train, y_train)

# Evaluación del modelo
y_pred = tree_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["No Alta", "Alta"]))

# Visualización del árbol
clf = tree_pipeline.named_steps['classifier']
feature_names_transformed = tree_pipeline.named_steps['preprocessor'].get_feature_names_out()

plt.figure(figsize=(24, 10))
plot_tree(clf, feature_names=feature_names_transformed, class_names=["No Alta", "Alta"], filled=True)
plt.title("Árbol de Decisión - Clasificación Binaria (Percentil 70)")
plt.show()


# Probar modelo Proactivo, deteccion de anomalias usando Isolation Forest
from sklearn.ensemble import IsolationForest
import numpy as np

#Seleccionar variables para detectar anomalias
anomaly_features = ['soil_moisture_%', 'soil_pH', 'temperature_C',
                    'rainfall_mm', 'humidity_%', 'sunlight_hours',
                    'NDVI_index', 'pesticide_usage_ml']

X_anomaly = df[anomaly_features]


# Entrenanar modelo Isolation Forest
iso = IsolationForest(contamination=0.10, random_state=42)
df['anomaly_score'] = iso.fit_predict(X_anomaly)

#Interpretar Resultados
# Ver distribución de etiquetas
print(df['anomaly_score'].value_counts())

# Clasificar como "Normal" o "Anómalo"
df['anomaly_label'] = df['anomaly_score'].map({1: "Normal", -1: "Anómalo"})

# Mostrar algunos ejemplos
print(df[['anomaly_label'] + anomaly_features].head())

#Cambiar nivel de contaminacion para observar efecto en anomalias
for contamination in [0.01, 0.05, 0.10, 0.15, 0.20]:
    iso = IsolationForest(contamination=contamination, random_state=42)
    pred = iso.fit_predict(X_anomaly)
    print(f"Contaminación: {contamination} → Anómalos detectados: {(pred == -1).sum()}")
    
# Guardar el modelo entrenado para usarlo en Streamlit
import joblib
joblib.dump(tree_pipeline, "tree_model.pkl")




