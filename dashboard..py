import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

data = pd.read_csv("inc_final.csv")
data['opened_at'] = pd.to_datetime(data['opened_at'])
data_orig = pd.read_csv("incident_event_log.csv", usecols=['number', 'priority', 'assignment_group', 'location'])
data_orig = data_orig.drop_duplicates(subset=['number'], keep='last')

data = pd.merge(data, data_orig, on='number', how='left')
