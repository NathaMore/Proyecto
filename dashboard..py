import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# --- 1. Carga y Preparación de Datos ---

# Cargar el dataset procesado
try:
    data = pd.read_csv("inc_final.csv")
    data['opened_at'] = pd.to_datetime(data['opened_at'])
except FileNotFoundError:
    print("Error: 'inc_final.csv' no encontrado. Asegúrate de que el archivo esté en la misma carpeta.")
    exit()

# Cargar el dataset original para obtener columnas categóricas adicionales
try:
    # Cargar solo las columnas necesarias para optimizar memoria
    data_orig = pd.read_csv("incident_event_log.csv", usecols=['number', 'priority', 'assignment_group', 'location'])
    # Eliminar duplicados para tener una fila por incidente, manteniendo el último estado
    data_orig = data_orig.drop_duplicates(subset=['number'], keep='last')
except FileNotFoundError:
    print("Error: 'incident_event_log.csv' no encontrado.")
    exit()


# Unir los datasets
data = pd.merge(data, data_orig, on='number', how='left')


# --- 2. Entrenamiento del Modelo de Machine Learning ---

# Definir las variables para el modelo (igual que en tu script de regresión)
features = [
    "reassignment_count", "reopen_count", "sys_mod_count",
    "impact_ord", "urgency_ord", "know_ord"
]
target = "time_min"

# Eliminar filas donde las variables del modelo puedan ser nulas
model_data = data.dropna(subset=features + [target])

X = model_data[features]
# Usamos log1p para estabilizar la varianza, como en tu script original
y = np.log1p(model_data[target])

# Entrenar el modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# --- 3. Inicialización de la App Dash ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- 4. Layout del Dashboard ---
app.layout = html.Div(style={'backgroundColor': '#f2f2f2', 'fontFamily': 'Arial, sans-serif'}, children=[
    
    html.Div(style={'backgroundColor': '#2c3e50', 'padding': '15px', 'color': 'white', 'textAlign': 'center'}, children=[
        html.H1("Dashboard de rendimiento y prediccion de la resolución de incidentes"),
    ]),
    
    html.Div(style={'padding': '20px'}, children=[
        html.Label("Rango de fechas:", style={'fontWeight': 'bold'}),
        dcc.DatePickerRange(
            id='date-range-picker',
            min_date_allowed=data['opened_at'].min().date(),
            max_date_allowed=data['opened_at'].max().date(),
            start_date=data['opened_at'].min().date(),
            end_date=data['opened_at'].max().date(),
            display_format='DD-MM-YY'
        )
    ]),

    dcc.Tabs(id="tabs-main", value='tab-1', children=[
        
        dcc.Tab(label='Análisis Descriptivo', value='tab-1', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.Div(id='kpi-cards', style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
                html.Div(style={'display': 'flex'}, children=[
                    dcc.Graph(id='histogram-resolucion', style={'width': '50%'}),
                    dcc.Graph(id='line-mttr-evolucion', style={'width': '50%'})
                ])
            ])
        ]),
        
        dcc.Tab(label='Análisis de Factores', value='tab-2', children=[
            html.Div(style={'padding': '20px', 'display': 'flex'}, children=[
                dcc.Graph(id='heatmap-correlacion', style={'width': '50%'}),
                html.Div(style={'width': '50%', 'paddingLeft': '20px'}, children=[
                    html.Label("Analizar rendimiento por:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='dropdown-categoria-barra',
                        options=[
                            {'label': 'Grupo de Asignación', 'value': 'assignment_group'},
                            {'label': 'Prioridad', 'value': 'priority'},
                            {'label': 'Localización', 'value': 'location'}
                        ],
                        value='assignment_group'
                    ),
                    dcc.Graph(id='bar-mttr-categoria')
                ])
            ])
        ]),
        
        dcc.Tab(label='Simulador Predictivo ¿Qué pasaría si...?', value='tab-3', children=[
            html.Div(style={'padding': '20px', 'display': 'flex'}, children=[
                html.Div(style={'width': '40%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '5px'}, children=[
                    html.H4("Calculadora de Tiempo Estimado"),
                    html.Label("Conteos de Reasignación:"),
                    dcc.Input(id='input-reassignment', type='number', value=0, min=0, step=1, style={'width': '100%'}),
                    html.Br(), html.Br(),
                    html.Label("Conteos de Reapertura:"),
                    dcc.Input(id='input-reopen', type='number', value=0, min=0, step=1, style={'width': '100%'}),
                    html.Br(), html.Br(),
                    html.Label("Conteos de Modificación del Sistema:"),
                    dcc.Input(id='input-sys-mod', type='number', value=1, min=0, step=1, style={'width': '100%'}),
                    html.Br(), html.Br(),
                    html.Label("Impacto:"),
                    dcc.Dropdown(id='input-impact', options=[{'label': '2 - Medium', 'value': 2}, {'label': '3 - Low', 'value': 3}, {'label': '1 - High', 'value': 1}], value=2),
                    html.Br(),
                    html.Label("Urgencia:"),
                    dcc.Dropdown(id='input-urgency', options=[{'label': '2 - Medium', 'value': 2}, {'label': '3 - Low', 'value': 3}, {'label': '1 - High', 'value': 1}], value=2),
                    html.Br(),
                    html.Label("Conocimiento Usado:"),
                    dcc.Dropdown(id='input-knowledge', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=1),
                    html.Br(),
                    html.Button('Predecir Tiempo', id='boton-predecir', n_clicks=0, style={'marginTop': '20px', 'width': '100%', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px'})
                ]),
                
                html.Div(style={'width': '60%', 'paddingLeft': '40px'}, children=[
                    html.Div(id='output-prediccion', style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '5px'}),
                    dcc.Graph(id='graph-feature-importance')
                ])
            ])
        ]),
    ])
])

# --- 5. Callbacks de Interacción ---

@app.callback(
    [Output('kpi-cards', 'children'),
     Output('histogram-resolucion', 'figure'),
     Output('line-mttr-evolucion', 'figure')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_tab1(start_date, end_date):
    # --- MODIFICACIÓN AQUÍ ---
    # Se añade .copy() para asegurar que trabajamos con una copia y evitar el warning.
    dff = data[(data['opened_at'] >= start_date) & (data['opened_at'] <= end_date)].copy()
    
    if dff.empty:
        return [], go.Figure().update_layout(title="No hay datos para el rango seleccionado."), go.Figure().update_layout(title="No hay datos para el rango seleccionado.")

    mttr = dff['time_min'].mean()
    total_incidents = len(dff)
    reopen_rate = dff[dff['reopen_count'] > 0].shape[0] / total_incidents if total_incidents > 0 else 0

    kpi_cards = [
        html.Div(style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'width': '30%'}, children=[
            html.H3(f"{mttr/1440:.2f} Días"), html.P("Tiempo Medio de Resolución (MTTR)")
        ]),
        html.Div(style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'width': '30%'}, children=[
            html.H3(f"{total_incidents:,}"), html.P("Total de Incidentes")
        ]),
        html.Div(style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'width': '30%'}, children=[
            html.H3(f"{reopen_rate:.2%}"), html.P("Tasa de Reapertura")
        ]),
    ]
    
    # Crear nueva columna temporal en días
    dff['time_days'] = dff['time_min'] / 1440
    
    # Usar la nueva columna para el histograma y actualizar etiquetas
    fig_hist = px.histogram(dff, x='time_days', title='Distribución del Tiempo de Resolución (días)', labels={'time_days': 'Tiempo (días)'})
    
    # Establecer el límite izquierdo del eje X al valor mínimo del rango de datos actual
    if not dff.empty:
        fig_hist.update_xaxes(range=[dff['time_days'].min(), None])

    # Asegurarse de que las líneas de media/mediana usen la nueva columna
    fig_hist.add_vline(x=dff['time_days'].mean(), line_dash="dash", line_color="red", annotation_text="Media")
    fig_hist.add_vline(x=dff['time_days'].median(), line_dash="dot", line_color="green", annotation_text="Mediana")

    # --- MODIFICACIÓN AQUÍ ---
    # 1. Agrupar por día ('D') y calcular la media de 'time_days'
    dff_time = dff.set_index('opened_at').resample('D')['time_days'].mean().reset_index()
    # 2. Actualizar títulos, etiquetas y el eje Y para usar días
    fig_line = px.line(dff_time, x='opened_at', y='time_days', 
                       title='Evolución Diaria del Tiempo Medio de Resolución (días)', 
                       labels={'opened_at': 'Día', 'time_days': 'Tiempo Medio de Resolución (días)'})
    
    return kpi_cards, fig_hist, fig_line


@app.callback(
    [Output('heatmap-correlacion', 'figure'),
     Output('bar-mttr-categoria', 'figure')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('dropdown-categoria-barra', 'value')]
)
def update_tab2(start_date, end_date, dropdown_value):
    # --- MODIFICACIÓN AQUÍ ---
    # Se añade .copy() de forma preventiva para mantener buenas prácticas.
    dff = data[(data['opened_at'] >= start_date) & (data['opened_at'] <= end_date)].copy()

    if dff.empty:
        return go.Figure().update_layout(title="No hay datos para el rango seleccionado."), go.Figure().update_layout(title="No hay datos para el rango seleccionado.")

    corr_matrix = dff[features + ['time_min']].corr()
    fig_heat = px.imshow(corr_matrix, text_auto=True, title='Mapa de Calor de Correlación de Variables', color_continuous_scale='RdYlGn', aspect="auto")

    if dropdown_value and dff[dropdown_value].notna().any():
      bar_data = dff.groupby(dropdown_value)['time_min'].mean().sort_values(ascending=False).head(20).reset_index()
      fig_bar = px.bar(bar_data, 
                       x='time_min', 
                       y=dropdown_value, 
                       orientation='h',
                       title=f'Top 20 - Tiempo Medio de Resolución por {dropdown_value.replace("_", " ").title()}',
                       labels={'time_min': 'Tiempo Medio (minutos)', dropdown_value: dropdown_value.replace("_", " ").title()})
      fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
      fig_bar = go.Figure().update_layout(title="No hay datos para la selección.")

    return fig_heat, fig_bar


@app.callback(
    Output('output-prediccion', 'children'),
    [Input('boton-predecir', 'n_clicks')],
    [State('input-reassignment', 'value'),
     State('input-reopen', 'value'),
     State('input-sys-mod', 'value'),
     State('input-impact', 'value'),
     State('input-urgency', 'value'),
     State('input-knowledge', 'value')],
    prevent_initial_call=True
)
def update_prediction_text(n_clicks, reassignment, reopen, sys_mod, impact, urgency, knowledge):
    input_features = np.array([[reassignment, reopen, sys_mod, impact, urgency, knowledge]])
    pred_log = rf_model.predict(input_features)[0]
    pred_min = np.expm1(pred_log)
    
    pred_horas = int(pred_min // 60)
    pred_minutos_restantes = int(pred_min % 60)
    
    return f"Tiempo Estimado: {pred_horas} horas y {pred_minutos_restantes} minutos"


@app.callback(
    Output('graph-feature-importance', 'figure'),
    [Input('tabs-main', 'value'),
     Input('boton-predecir', 'n_clicks')]
)
def update_importance_graph(tab_value, n_clicks):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    if tab_value == 'tab-3':
        importances = pd.DataFrame({
            'Variable': features,
            'Importancia': rf_model.feature_importances_
        }).sort_values(by='Importancia', ascending=True)
        
        fig_importance = px.bar(importances, 
                                x='Importancia', 
                                y='Variable', 
                                orientation='h', 
                                title='Importancia Global de los Factores en la Predicción')
        return fig_importance
    
    raise dash.exceptions.PreventUpdate

# --- 6. Ejecución de la App ---
if __name__ == '__main__':
    app.run(debug=True)

