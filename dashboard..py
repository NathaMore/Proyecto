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

# Asegurar que la columna 'knowledge' sea numérica (0/1) para el modelo
data['knowledge'] = data['knowledge'].astype(int)

#endregion

#region Entrenamiento del Modelo de Machine Learning
# Usamos las mismas variables del modelo de regresión más eficiente (RandomForest)
features = [
    'reassignment_count', 'reopen_count', 'sys_mod_count',
    'impact_ord', 'knowledge'
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
        
        dcc.Tab(label='Análisis descriptivo', value='tab-1', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.Div(id='kpi-cards', style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
                html.Div(style={'display': 'flex'}, children=[
                    dcc.Graph(id='histogram-resolucion', style={'width': '50%'}),
                    dcc.Graph(id='line-mttr-evolucion', style={'width': '50%'})
                ])
            ])
        ]),
        
        dcc.Tab(label='Análisis de factores', value='tab-2', children=[
            html.Div(style={'padding': '20px', 'display': 'flex'}, children=[
                dcc.Graph(id='heatmap-correlacion', style={'width': '50%'}),
                html.Div(style={'width': '50%', 'paddingLeft': '20px'}, children=[
                    html.Label("Analizar tiempo de resolucion por:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='dropdown-categoria-barra',
                        options=[
                            {'label': 'Grupo de asignación', 'value': 'assignment_group'},
                            {'label': 'Prioridad', 'value': 'priority'},
                            {'label': 'Localización', 'value': 'location'}
                        ],
                        value='assignment_group'
                    ),
                    dcc.Graph(id='bar-mttr-categoria', style={'height': '500px'})
                ])
            ])
        ]),
        
        dcc.Tab(label='Simulador predictivo ¿Qué pasaría si...?', value='tab-3', children=[
            html.Div(style={'padding': '20px', 'display': 'flex'}, children=[
                html.Div(style={'width': '40%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '5px'}, children=[
                    html.H4("Calculadora de tiempo estimado"),
                    html.Label("Conteos de reasignación:"),
                    dcc.Input(id='input-reassignment', type='number', value=0, min=0, step=1, style={'width': '100%'}),
                    html.Br(), html.Br(),
                    html.Label("Conteos de reapertura:"),
                    dcc.Input(id='input-reopen', type='number', value=0, min=0, step=1, style={'width': '100%'}),
                    html.Br(), html.Br(),
                    html.Label("Conteos de modificación del sistema:"),
                    dcc.Input(id='input-sys-mod', type='number', value=1, min=0, step=1, style={'width': '100%'}),
                    html.Br(), html.Br(),
                    html.Label("Impacto:"),
                    dcc.Dropdown(id='input-impact', options=[{'label': '2 - Medium', 'value': 2}, {'label': '3 - Low', 'value': 3}, {'label': '1 - High', 'value': 1}], value=2),
                    html.Br(),
                    html.Label("Conocimiento usado:"),
                    dcc.Dropdown(id='input-knowledge', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=1),
                    html.Br(),
                    html.Button('Predecir tiempo', id='boton-predecir', n_clicks=0, style={'marginTop': '20px', 'width': '100%', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px'})
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
#region Callbacks

@app.callback(
    [Output('kpi-cards', 'children'),
     Output('histogram-resolucion', 'figure'),
     Output('line-mttr-evolucion', 'figure')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)

#region Fechas
def update_tab1(start_date, end_date):
    
    dff = data[(data['opened_at'] >= start_date) & (data['opened_at'] <= end_date)].copy()
    
    if dff.empty:
        return [], go.Figure().update_layout(title="No hay datos para el rango seleccionado."), go.Figure().update_layout(title="No hay datos para el rango seleccionado.")

    mttr = dff['time_min'].mean()
    total_incidents = len(dff)
    reopen_rate = dff[dff['reopen_count'] > 0].shape[0] / total_incidents if total_incidents > 0 else 0

    kpi_cards = [
        html.Div(style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'width': '30%'}, children=[
            html.H3(f"{mttr/1440:.2f} Días"), html.P("Tiempo medio de resolución (MTTR)")
        ]),
        html.Div(style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'width': '30%'}, children=[
            html.H3(f"{total_incidents:,}"), html.P("Total de incidentes")
        ]),
        html.Div(style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'width': '30%'}, children=[
            html.H3(f"{reopen_rate:.2%}"), html.P("Tasa de reapertura")
        ]),
    ]
    
    #endregion
    #region Histograma

    dff['time_days'] = dff['time_min'] / 1440
    
    fig_hist = px.histogram(dff, x='time_days', title='Distribución del tiempo de resolución (días)', labels={'time_days': 'Tiempo (días)'})
    
    if not dff.empty:
        fig_hist.update_xaxes(range=[dff['time_days'].min(), None])

    fig_hist.add_vline(x=dff['time_days'].mean(), line_dash="dash", line_color="red", annotation_text="Media")
    fig_hist.add_vline(x=dff['time_days'].median(), line_dash="dot", line_color="green", annotation_text="Mediana")

    dff_time = dff.set_index('opened_at').resample('D')['time_days'].mean().reset_index()

    fig_line = px.line(dff_time, x='opened_at', y='time_days', 
                       title='Evolución diaria del tiempo medio de resolución (días)', 
                       labels={'opened_at': 'Día', 'time_days': 'Tiempo medio de resolución (días)'})
    
    return kpi_cards, fig_hist, fig_line

#endregion

@app.callback(
    [Output('heatmap-correlacion', 'figure'),
     Output('bar-mttr-categoria', 'figure')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('dropdown-categoria-barra', 'value')]
)

#region Mapa de calor
def update_tab2(start_date, end_date, dropdown_value):

    dff = data[(data['opened_at'] >= start_date) & (data['opened_at'] <= end_date)].copy()

    if dff.empty:
        return go.Figure().update_layout(title="No hay datos para el rango seleccionado."), go.Figure().update_layout(title="No hay datos para el rango seleccionado.")

    corr_matrix = dff[features + ['time_min']].corr()

    corr_with_target = corr_matrix[['time_min']].sort_values(by='time_min', ascending=False)

    corr_with_target = corr_with_target.drop('time_min', axis=0)

    fig_heat = px.imshow(
        corr_with_target,
        text_auto='.2f',
        title='Correlación con el tiempo de resolución',
        labels=dict(color="Correlación"),
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig_heat.update_traces(textfont_size=16)
    fig_heat.update_xaxes(showticklabels=False)

    if dropdown_value and dff[dropdown_value].notna().any():
      
      bar_data = dff.groupby(dropdown_value)['time_min'].mean().reset_index()
      bar_data['time_days'] = bar_data['time_min'] / 1440
      bar_data = bar_data.sort_values(by='time_days', ascending=False).head(20)
      
      fig_bar = px.bar(bar_data, 
                       x='time_days', 
                       y=dropdown_value, 
                       orientation='h',
                       title=f'Top 20 - Tiempo medio de resolución por {dropdown_value.replace("_", " ").title()}',
                       labels={'time_days': 'Tiempo medio (días)', dropdown_value: dropdown_value.replace("_", " ").title()})
      fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
      fig_bar = go.Figure().update_layout(title="No hay datos para la selección.")

    return fig_heat, fig_bar

#endregion

@app.callback(
    Output('output-prediccion', 'children'),
    [Input('boton-predecir', 'n_clicks')],
    [State('input-reassignment', 'value'),
     State('input-reopen', 'value'),
     State('input-sys-mod', 'value'),
     State('input-impact', 'value'),
     State('input-knowledge', 'value')],
    prevent_initial_call=True
)

#region Prediccion dias
def update_prediction_text(n_clicks, reassignment, reopen, sys_mod, impact, knowledge):
    
    input_df = pd.DataFrame(
        [[reassignment, reopen, sys_mod, impact, knowledge]],
        columns=features
    )
    
    pred_log = rf_model.predict(input_df)[0]
    pred_min = np.expm1(pred_log)
    
    pred_dias = pred_min / 1440
    
    return f"Tiempo estimado: {pred_dias:.2f} días"

#endregion

@app.callback(
    Output('graph-feature-importance', 'figure'),
    [Input('tabs-main', 'value'),
     Input('boton-predecir', 'n_clicks')]
)

#region Barras de importancia
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
                                title='Importancia de los Factores en la predicción')
        return fig_importance
    
    raise dash.exceptions.PreventUpdate
#endregion

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
