import pandas as pd
import streamlit as st
import numpy as np
st.set_page_config(page_title='Consultia', page_icon=':bar_chart:', layout='wide')


# Show the logo.png centered in the sidebar at the most top posible 
st.sidebar.image('logo.png', use_column_width=True)

st.title('Calculadora de Impacto GeoClarity')
st.write('Evaluación de Costos Asociados, Rendimiento Operacional y Servicio a Cliente')

# Crea explicación para los conceptos de la calculadora
with st.expander('Conceptos Clave'):
    st.markdown('''
- **Precisión en la Identificación de Direcciones Incorrectas (TPR):** El porcentaje de casos positivos que el modelo clasifica correctamente.
- **Precisión en la Confirmación de Direcciones Correctas (TNR):** El porcentaje de casos negativos que el modelo clasifica correctamente.
- **Costo de Solución:** El costo asociado con cada caso positivo detectado por el modelo, independientemente de si el modelo lo clasifica correctamente.
- **Costo de Casos Perdidos:** El costo asociado con cada caso positivo no detectado por el modelo.
- **Costo de Revisión:** El costo asociado con revisar cada caso, independientemente de si el modelo lo clasifica correctamente.
- **Random Forest:** Un modelo de clasificación basado en árboles de decisión.
- **XGBoost:** Un modelo de clasificación basado en árboles de decisión optimizado.
- **SMOTE:** Técnica de sobremuestreo de minorías sintéticas.
- **Modelo DQN:** Modelo de clasificación basado en redes neuronales profundas.
''')

# Título para los inputs en la barra lateral

st.sidebar.title('Parámetros de Entrada')

# Barra lateral para parámetros de costos
st.sidebar.header('Parámetros de Costos')
cost_solution = st.sidebar.slider('Inversión por Comunicación Proactiva al Cliente ($)', 0.0, 1.0, 0.1)
cost_undetected = st.sidebar.number_input('Costo por Dirección Incorrecta No Detectada ($)', value=10)
true_positive_cases_percentage = st.sidebar.number_input('Rango de Injerencia del Total de Entregas (%)', 0.01, 100.0, 1.4286)
cost_per_case_checked = st.sidebar.number_input('Costo por Revisión de Dirección Erronea ($)', value=0.042, format='%.3f')
st.sidebar.header('Parámetros de Rendimiento de Modelo Personalizado')
custom_tp_rate = st.sidebar.slider('Precisión en la Identificación de Direcciones Incorrectas (TPR) (%) para Modelo Personalizado', 0.0, 100.0, 1.0, step=1.0)
custom_tn_rate = st.sidebar.slider('Precisión en la Confirmación de Direcciones Correctas (TNR) (%) para Modelo Personalizado', 0.0, 100.0, 1.0, step=1.0)

total_cases = 1_000_000
# Función para calcular costos de un modelo
def calculate_model_cost(tp_rate, tn_rate):
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
    
    total_cost = cost_tp_fp + cost_fn + cost_checking 

    return total_cost, cost_tp_fp, cost_fn, cost_checking

# Métricas de modelos predefinidos
models = {
    'Random Forest': {'TPR': 13.80, 'TNR': 98.82},
    'XGBoost': {'TPR': 1.88, 'TNR': 99.96},
    'Random Forest con SMOTE': {'TPR': 11.68, 'TNR': 99.93},
    'XGBoost con SMOTE': {'TPR': 69.28, 'TNR': 41.19},
    'Modelo DQN': {'TPR': 2.46, 'TNR': 97.79},
    'Modelo Personalizado': {'TPR': custom_tp_rate, 'TNR': custom_tn_rate},
    'Sin modelo': {'TPR': 0, 'TNR': 100}
}

# Crear un DataFrame vacío
df_models = pd.DataFrame(columns=['Modelo', 'Precisión en la Identificación de Direcciones Incorrectas (%)', 'Precisión en la Confirmación de Direcciones Correctas (%)', 'Costos Totales ($)', 'Costos por Detección de Direcciónes Incorrectas ($)', 'Costos Actuales por Direcciones Incorrectas ($)', 'Costos por Revisión ($)', 'Margen de Utilidad'])

# Poblar el DataFrame
for model_name, metrics in models.items():
    total_cost, cost_by_solution, cost_by_missed_cases, cost_to_check = calculate_model_cost(metrics['TPR'], metrics['TNR'])
    df_models = pd.concat([df_models, pd.DataFrame({
        'Modelo': [model_name],
        'Precisión en la Identificación de Direcciones Incorrectas (%)': [metrics['TPR']],
        'Precisión en la Confirmación de Direcciones Correctas (%)': [metrics['TNR']],
        'Costos Totales ($)': [f'${total_cost:,.2f}'],
        'Costos por Detección de Direcciónes Incorrectas ($)': [f'${cost_by_solution:,.2f}'],
        'Costos Actuales por Direcciones Incorrectas ($)': [f'${cost_by_missed_cases:,.2f}'],
        'Costos por Revisión ($)': [f'${cost_to_check:,.2f}']

    })], ignore_index=True)
# Mostrar comparaciones de modelos usando un DataFrame
    
# Update the 'Margen de Utilidad' column in the dataframe by showing the difference between the total cost of each model and the cost of not using a model
df_models['Margen de Utilidad'] = df_models['Costos Totales ($)'].apply(
    lambda x: f'${float(str(x).replace("$", "").replace(",", "")) - float(str(df_models[df_models["Modelo"] == "Sin modelo"]["Costos Totales ($)"].iloc[0]).replace("$", "").replace(",", "")):,.2f}'
)

st.header('Resumen Comparativo de Efectividad de Modelos')
st.dataframe(df_models, hide_index=True)

# Change the cost components to numeric values
df_models['Costos Totales ($)'] = df_models['Costos Totales ($)'].str.replace('$', '').str.replace(',', '').astype(float)
df_models['Costos por Detección de Direcciónes Incorrectas ($)'] = df_models['Costos por Detección de Direcciónes Incorrectas ($)'].str.replace('$', '').str.replace(',', '').astype(float)
df_models['Costos Actuales por Direcciones Incorrectas ($)'] = df_models['Costos Actuales por Direcciones Incorrectas ($)'].str.replace('$', '').str.replace(',', '').astype(float)
df_models['Costos por Revisión ($)'] = df_models['Costos por Revisión ($)'].str.replace('$', '').str.replace(',', '').astype(float)

# Identifying the best cost model and the best TPR model
best_cost_model = df_models[df_models['Costos Totales ($)'] == df_models['Costos Totales ($)'].min()]
best_tpr_model = df_models[df_models['Precisión en la Identificación de Direcciones Incorrectas (%)'] == df_models['Precisión en la Identificación de Direcciones Incorrectas (%)'].max()]

# Extracting the models to compare
models_to_compare = df_models[(df_models['Modelo'] == 'Modelo Personalizado') | 
                              (df_models['Modelo'] == 'Sin modelo') | 
                              (df_models['Modelo'].isin([best_cost_model['Modelo'].iloc[0], 
                                                         best_tpr_model['Modelo'].iloc[0]]))]

# Setting the index to 'Modelo' for easier plotting
models_to_compare.set_index('Modelo', inplace=True)

# Extracting the cost components for the stacked bar chart
cost_components = ['Costos por Detección de Direcciónes Incorrectas ($)', 'Costos Actuales por Direcciones Incorrectas ($)', 'Costos por Revisión ($)']

# Plotting the stacked bar chart

# Have a check box to ask the user if they want to see the modelo personalizado
modelo_personalizado = st.checkbox('Mostrar Modelo Personalizado')


# Here we plot the number of customers affected by the no model as red, the customers affected by solution as blue as a stacked bar chart, the number of customers.
import matplotlib.pyplot as plt

# Calculate the number of clients affected by no model
clients_affected_no_model = total_cases * (true_positive_cases_percentage / 100)

# Calculate the number of clients missed and reached out for XGBoost+SMOTE and Random Forest
# Using the rates from the models considering that error rates causes models to reach out to unnecessary clients
import math

cases_missed_xgboost_smote = math.ceil(clients_affected_no_model * ((100 - models['XGBoost con SMOTE']['TPR']) / 100))
cases_correctly_reached_xgboost_smote = math.ceil(clients_affected_no_model * models['XGBoost con SMOTE']['TPR'] / 100)
cases_reached_out_xgboost_smote = math.ceil(cases_correctly_reached_xgboost_smote + (total_cases * (100 - models['XGBoost con SMOTE']['TNR']) / 100))

cases_missed_random_forest = math.ceil(clients_affected_no_model * ((100 - models['Random Forest']['TPR']) / 100))
cases_correctly_reached_random_forest = math.ceil(clients_affected_no_model * models['Random Forest']['TPR'] / 100)
cases_reached_out_random_forest = math.ceil(cases_correctly_reached_random_forest + (total_cases * (100 - models['Random Forest']['TNR']) / 100))

# Create a dataframe to hold the data
data = {
    'Modelo': ['Sin modelo', 'XGBoost con SMOTE', 'Random Forest'],
    'Clientes Afectados por No Detectar Direcciones Incorrectas': [clients_affected_no_model, cases_missed_xgboost_smote, cases_missed_random_forest],
    'Clientes Proactivamente Alcanzados': [0, cases_reached_out_xgboost_smote, cases_reached_out_random_forest],
}

df_clients_affected = pd.DataFrame(data)

# Set the index to 'Modelo' for easier plotting
df_clients_affected.set_index('Modelo', inplace=True)

import altair as alt

# Melt the DataFrame to long format
df_long = df_clients_affected.reset_index().melt('Modelo')

# Create the Altair chart
chart = alt.Chart(df_long).mark_bar().encode(
    x='Modelo:N',
    y=alt.Y('value:Q', stack='zero', title='Value'),
    color=alt.Color('variable:N', 
                    scale=alt.Scale(domain=df_long.variable.unique(), range=['#ff0000', '#0000ff']),
                    legend=alt.Legend(title=None, orient='bottom')),
    order=alt.Order(
        'variable:N',
        sort='ascending'
    ),
    tooltip=['value', 'variable']
).interactive()


# If the user wants to see the modelo personalizado, we create chart_with_personalizado
cases_missed_personalizado = math.ceil(clients_affected_no_model * ((100 - models['Modelo Personalizado']['TPR']) / 100))
cases_correctly_reached_personalizado = math.ceil(clients_affected_no_model * models['Modelo Personalizado']['TPR'] / 100)
cases_reached_out_personalizado = math.ceil(cases_correctly_reached_personalizado + (total_cases * (100 - models['Modelo Personalizado']['TNR']) / 100))

# Use and update df_clients_affected to include the personalizado model
df_clients_affected.loc['Modelo Personalizado'] = [cases_missed_personalizado, cases_reached_out_personalizado]

# Melt the DataFrame to long format
df_long_personalizado = df_clients_affected.reset_index().melt('Modelo')

# Create the Altair chart
chart_with_personalizado = alt.Chart(df_long_personalizado).mark_bar().encode(
    x='Modelo:N',
    y=alt.Y('value:Q', stack='zero', title='Value'),
    color=alt.Color('variable:N', 
                    scale=alt.Scale(domain=df_long_personalizado.variable.unique(), range=['#ff0000', '#0000ff']),
                    legend=alt.Legend(title=None, orient='bottom')),
    order=alt.Order(
        'variable:N',
        sort='ascending'
    ),
    tooltip=['value', 'variable']
).interactive()


# Plot the bar chart with the first category red, the second light blue and the third blue
if modelo_personalizado:
    st.header('Visualización de Impacto del Modelo Óptimo')
    st.bar_chart(models_to_compare[cost_components], color=['#ff0000', '#add8e6', '#0000ff'])
    st.header('Visualización de Volumen de Clientes Afectados')
    st.altair_chart(chart_with_personalizado,use_container_width=True)
else:
    st.header('Visualización de Impacto del Modelo Óptimo')
    st.bar_chart(models_to_compare[models_to_compare.index != 'Modelo Personalizado'][cost_components], color=['#ff0000', '#add8e6', '#0000ff'])
    st.header('Visualización de Volumen de Clientes Afectados')
    st.altair_chart(chart,use_container_width=True)










# Convert cost and TPR columns to numeric for calculations and comparisons
df_models['Costo Total Numeric'] = df_models['Costos Totales ($)'].replace('[\$,]', '', regex=True).astype(float)
df_models['TPR Numeric'] = df_models['Precisión en la Identificación de Direcciones Incorrectas (%)'].astype(float)

# Identify the models for comparison
best_cost_model = df_models.iloc[df_models['Costo Total Numeric'].argmin()]
best_tpr_model = df_models.iloc[df_models['TPR Numeric'].argmax()]
custom_model = df_models[df_models['Modelo'] == 'Modelo Personalizado'].iloc[0]
no_model = df_models[df_models['Modelo'] == 'Sin modelo'].iloc[0]

# Calculate savings or improvements
savings_best_cost = no_model['Costo Total Numeric'] - best_cost_model['Costo Total Numeric']
improvement_best_tpr = best_tpr_model['TPR Numeric'] - no_model['TPR Numeric']

# Highlight Key Findings
st.markdown(f'### Ahorros del Mejor Modelo en Términos de Costos')
st.markdown(f'**{best_cost_model["Modelo"]}** ahorra **${savings_best_cost:,.2f}** comparado con no usar un modelo.')

st.markdown(f'### Mejora en la Detección de Casos del Mejor Modelo')
st.markdown(f'**{best_tpr_model["Modelo"]}** mejora la detección de casos en **{improvement_best_tpr:.2f}%** comparado con no usar un modelo.')

# Show the number of customers reached out by xbgoost+smote
st.markdown(f'### Número de Clientes Alcanzados por el Mejor Modelo')
st.markdown("El modelo **XGBoost con SMOTE** alcanza **{:.0f}** clientes proactivamente.".format(cases_reached_out_xgboost_smote))