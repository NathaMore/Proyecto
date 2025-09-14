import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

#region Carga de datos y unificacion

data = pd.read_csv("inc_final.csv")
data['opened_at'] = pd.to_datetime(data['opened_at'])
data_orig = pd.read_csv("incident_event_log.csv", usecols=['number', 'priority', 'assignment_group', 'location'])
data_orig = data_orig.drop_duplicates(subset=['number'], keep='last')

data = pd.merge(data, data_orig, on='number', how='left')

#endregion

#region Entrenamiento del Modelo de Machine Learning
# Usamos las mismas variables del modelo de regresión más eficiente (RandomForest)
features = [
    "reassignment_count", "reopen_count", "sys_mod_count",
    "impact_ord", "urgency_ord", "know_ord"
]
target = "time_min"
model_data = data.dropna(subset=features + [target])
X = model_data[features]
# Usamos log1p para estabilizar la varianza, como en el script
y = np.log1p(model_data[target])

# Entrenamiento
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)
#endregion

#region Diseño boceto app

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

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
                    dcc.Graph(id='bar-mttr-categoria', style={'height': '500px'})
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
                    dcc.Graph(id='graph-feature-importance', style={'height': '500px'})
                ])
            ])
        ]),
    ])
])

#endregion