import pandas as pd
import streamlit as st
import numpy as np
st.set_page_config(page_title="Consultia", page_icon=":bar_chart:", layout="wide")


# Show the logo.png centered in the sidebar at the most top posible 
st.sidebar.image("logo.png", use_column_width=True)

st.title("Calculadora de Impacto en Costos de Negocio")
st.write("Compara el rendimiento de diferentes modelos basados en su matriz de confusión, costos asociados, y ajusta parámetros para una comparación de modelos personalizada.")

# Crea explicación para los conceptos de la calculadora
with st.expander("Conceptos Clave"):
    st.markdown("""
- **Tasa de Positivos Verdaderos (TPR):** El porcentaje de casos positivos que el modelo clasifica correctamente.
- **Tasa de Negativos Verdaderos (TNR):** El porcentaje de casos negativos que el modelo clasifica correctamente.
- **Costo de Solución:** El costo asociado con cada caso positivo detectado por el modelo, independientemente de si el modelo lo clasifica correctamente.
- **Costo de Casos Perdidos:** El costo asociado con cada caso positivo no detectado por el modelo.
- **Costo de Revisión:** El costo asociado con revisar cada caso, independientemente de si el modelo lo clasifica correctamente.
- **Random Forest:** Un modelo de clasificación basado en árboles de decisión.
- **XGBoost:** Un modelo de clasificación basado en árboles de decisión optimizado.
- **SMOTE:** Técnica de sobremuestreo de minorías sintéticas.
- **Modelo DQN:** Modelo de clasificación basado en redes neuronales profundas.
""")

# Título para los inputs en la barra lateral

st.sidebar.title("Parámetros de Entrada")

# Barra lateral para parámetros de costos
st.sidebar.header("Parámetros de Costos")
cost_solution = st.sidebar.slider("Costo de solución por caso detectado ($)", 0.0, 3.0, 0.5)
cost_undetected = st.sidebar.number_input("Costo de caso positivo no detectado ($)", value=25)
true_positive_cases_percentage = st.sidebar.number_input("Porcentaje de casos positivos verdaderos entre todos los casos (%)", 0.01, 100.0, 1.4286)
cost_per_case_checked = st.sidebar.number_input("Costo por caso revisado ($)", min_value=0.042, value=0.042, format="%.3f")
st.sidebar.header("Parámetros de Rendimiento de Modelo Personalizado")
custom_tp_rate = st.sidebar.slider("Tasa de Positivos Verdaderos (TPR) (%) para Modelo Personalizado", 0.0, 100.0, 100.0, step=1.0)
custom_tn_rate = st.sidebar.slider("Tasa de Negativos Verdaderos (TNR) (%) para Modelo Personalizado", 0.0, 100.0, 100.0, step=1.0)

# Función para calcular costos de un modelo
def calculate_model_cost(tp_rate, tn_rate):
    total_cases = 1_000_000
    true_positive_cases = total_cases * true_positive_cases_percentage / 100
    other_cases = total_cases - true_positive_cases
    
    tp = true_positive_cases * (tp_rate / 100)
    fp = other_cases * ((100 - tn_rate) / 100)
    fn = true_positive_cases * ((100 - tp_rate) / 100)
    
    cost_tp_fp = (tp + fp) * cost_solution
    cost_fn = fn * cost_undetected
    cost_checking = total_cases * cost_per_case_checked 
    if tp_rate == 0:
        cost_checking = 0
    
    total_cost = cost_tp_fp + cost_fn + cost_checking  # Incluir el costo de revisión en el costo total
    return total_cost, cost_tp_fp, cost_fn, cost_checking

# Métricas de modelos predefinidos
models = {
    "Random Forest": {"TPR": 13.80, "TNR": 98.82},
    "XGBoost": {"TPR": 1.88, "TNR": 99.96},
    "Random Forest con SMOTE": {"TPR": 11.68, "TNR": 99.93},
    "XGBoost con SMOTE": {"TPR": 69.28, "TNR": 41.19},
    "Modelo DQN": {"TPR": 2.46, "TNR": 97.79},
    "Modelo Personalizado": {"TPR": custom_tp_rate, "TNR": custom_tn_rate},
    "Sin modelo": {"TPR": 0, "TNR": 100}
}

# Crear un DataFrame vacío
df_models = pd.DataFrame(columns=["Modelo", "TPR (%)", "TNR (%)", "FPR (%)", "FNR (%)", "Costo Total ($)", "Costo por Solución ($)", "Costo por Casos Perdidos ($)", "Costo por Revisión ($)"])

# Poblar el DataFrame
for model_name, metrics in models.items():
    total_cost, cost_by_solution, cost_by_missed_cases, cost_to_check = calculate_model_cost(metrics["TPR"], metrics["TNR"])
    df_models = pd.concat([df_models, pd.DataFrame({
        "Modelo": [model_name],
        "TPR (%)": [metrics["TPR"]],
        "TNR (%)": [metrics["TNR"]],
        "FPR (%)": [100 - metrics["TNR"]],
        "FNR (%)": [100 - metrics["TPR"]],
        "Costo Total ($)": [f"${total_cost:,.2f}"],
        "Costo por Solución ($)": [f"${cost_by_solution:,.2f}"],
        "Costo por Casos Perdidos ($)": [f"${cost_by_missed_cases:,.2f}"],
        "Costo por Revisión ($)": [f"${cost_to_check:,.2f}"]
    })], ignore_index=True)
# Mostrar comparaciones de modelos usando un DataFrame
st.header("Comparaciones de Modelos")
st.dataframe(df_models, hide_index=True)

# Change the cost components to numeric values
df_models['Costo Total ($)'] = df_models['Costo Total ($)'].str.replace('$', '').str.replace(',', '').astype(float)
df_models['Costo por Solución ($)'] = df_models['Costo por Solución ($)'].str.replace('$', '').str.replace(',', '').astype(float)
df_models['Costo por Casos Perdidos ($)'] = df_models['Costo por Casos Perdidos ($)'].str.replace('$', '').str.replace(',', '').astype(float)
df_models['Costo por Revisión ($)'] = df_models['Costo por Revisión ($)'].str.replace('$', '').str.replace(',', '').astype(float)

# Identifying the best cost model and the best TPR model
best_cost_model = df_models[df_models['Costo Total ($)'] == df_models['Costo Total ($)'].min()]
best_tpr_model = df_models[df_models['TPR (%)'] == df_models['TPR (%)'].max()]

# Extracting the models to compare
models_to_compare = df_models[(df_models['Modelo'] == 'Modelo Personalizado') | 
                              (df_models['Modelo'] == 'Sin modelo') | 
                              (df_models['Modelo'].isin([best_cost_model['Modelo'].iloc[0], 
                                                         best_tpr_model['Modelo'].iloc[0]]))]

# Setting the index to 'Modelo' for easier plotting
models_to_compare.set_index('Modelo', inplace=True)

# Extracting the cost components for the stacked bar chart
cost_components = ['Costo por Solución ($)', 'Costo por Casos Perdidos ($)', 'Costo por Revisión ($)']

# Plotting the stacked bar chart
st.header("Comparación de Modelos")
# Have a check box to ask the user if they want to see the modelo personalizado
modelo_personalizado = st.checkbox("Mostrar Modelo Personalizado")

# Plot the bar chart with the first category red, the second light blue and the third blue
if modelo_personalizado:
    st.bar_chart(models_to_compare[cost_components], color=['#ff0000', '#add8e6', '#0000ff'])
else:
    st.bar_chart(models_to_compare[models_to_compare.index != 'Modelo Personalizado'][cost_components], color=['#ff0000', '#add8e6', '#0000ff'])










# Convert cost and TPR columns to numeric for calculations and comparisons
df_models['Costo Total Numeric'] = df_models['Costo Total ($)'].replace('[\$,]', '', regex=True).astype(float)
df_models['TPR Numeric'] = df_models['TPR (%)'].astype(float)

# Identify the models for comparison
best_cost_model = df_models.iloc[df_models['Costo Total Numeric'].argmin()]
best_tpr_model = df_models.iloc[df_models['TPR Numeric'].argmax()]
custom_model = df_models[df_models['Modelo'] == 'Modelo Personalizado'].iloc[0]
no_model = df_models[df_models['Modelo'] == 'Sin modelo'].iloc[0]

# Calculate savings or improvements
savings_best_cost = no_model['Costo Total Numeric'] - best_cost_model['Costo Total Numeric']
improvement_best_tpr = best_tpr_model['TPR Numeric'] - no_model['TPR Numeric']

# Highlight Key Findings
st.markdown(f"### Ahorros del Mejor Modelo en Términos de Costos")
st.markdown(f"**{best_cost_model['Modelo']}** ahorra **${savings_best_cost:,.2f}** comparado con no usar un modelo.")

st.markdown(f"### Mejora en la Detección de Casos del Mejor Modelo")
st.markdown(f"**{best_tpr_model['Modelo']}** mejora la tasa de detección de casos positivos en **{improvement_best_tpr:.2f}%** comparado con no usar un modelo.")
