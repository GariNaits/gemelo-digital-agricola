import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns

# Cargar datos
df = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# Crear etiqueta binaria 'yield_label' usando el percentil 70
threshold_70 = df['yield_kg_per_hectare'].quantile(0.70)
df['yield_label'] = (df['yield_kg_per_hectare'] >= threshold_70).astype(int)

# Definir variables
numeric_features = ['soil_moisture_%', 'soil_pH', 'temperature_C',
                    'rainfall_mm', 'humidity_%', 'sunlight_hours',
                    'pesticide_usage_ml', 'NDVI_index', 'total_days']
categorical_features = ['crop_type']
features = numeric_features + categorical_features

X = df[features]
y = df['yield_label']

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_features)],
    remainder='passthrough'
)

# Crear pipeline
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=6, random_state=0))
])

# Entrenar modelo directamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_pipeline.fit(X_train, y_train)
y_pred = tree_pipeline.predict(X_test)

# Interfaz Streamlit
st.title("Gemelo Digital para predecir y optimizar la cosecha agrícola")
st.header("1. Árbol de Decisión – Clasificación de Alto Rendimiento")

# Evaluación
report = classification_report(y_test, y_pred, output_dict=True)
st.write("### Métricas del modelo")
st.dataframe(pd.DataFrame(report).transpose())

# Visualización del árbol
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree_pipeline.named_steps['classifier'],
          feature_names=tree_pipeline.named_steps['preprocessor'].get_feature_names_out(),
          class_names=["No Alta", "Alta"],
          filled=True, ax=ax)
st.pyplot(fig)

# Anomalías
st.header("2. Detección de Anomalías – Isolation Forest")
contamination = st.slider("Nivel de contaminación", 0.01, 0.2, 0.10, step=0.01)

# Entrenar modelo de anomalías
iso = IsolationForest(contamination=contamination, random_state=42)
X_anomaly = df[numeric_features]
df['anomaly'] = iso.fit_predict(X_anomaly)
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anómalo'})

st.write(f"Total de anómalos detectados: {(df['anomaly'] == -1).sum()}")

# Visualización
st.write("### Visualización de anomalías")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='temperature_C', y='soil_moisture_%',
                hue='anomaly_label', palette='Set1', ax=ax2)
st.pyplot(fig2)
