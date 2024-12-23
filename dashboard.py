from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from db import get_regions, get_region_count, get_logs, get_active_density_event

# Initialize app with a modern theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-card {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
                margin: 10px;
                flex: 1;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #2C3E50;
            }
            .metric-title {
                color: #7F8C8D;
                font-size: 1.2rem;
                margin-bottom: 10px;
            }
            .dashboard-container {
                background-color: #F8F9FA;
                min-height: 100vh;
                padding: 20px;
            }
            .graph-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Region Monitoring Dashboard", 
                   className="text-primary mb-4 mt-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[],
                        value=None,
                        placeholder="Select a region",
                        className="mb-3"
                    )
                ])
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Current Count", className="metric-title"),
                    html.H4(id='current-count', children='0', 
                           className="metric-value")
                ])
            ], className="text-center")
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Density Status", className="metric-title"),
                    html.H4(id='density-status', children='Normal',
                           className="metric-value")
                ])
            ], className="text-center")
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='count-history')
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='dwell-times')
                ])
            ])
        ], width=12)
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=5*1000,
        n_intervals=0
    )
], fluid=True, className="dashboard-container")

# Update the callback to include styling based on status
@app.callback(
    [Output('current-count', 'children'),
     Output('current-count', 'style'),
     Output('density-status', 'children'),
     Output('density-status', 'style')],
    [Input('interval-component', 'n_intervals'),
     Input('region-dropdown', 'value')]
)
def update_metrics(_, selected_region):
    if not selected_region:
        return '0', {'color': '#2C3E50'}, 'No region selected', {'color': '#7F8C8D'}
    
    # Get your actual data here
    count = get_region_count(selected_region)
    density_event = get_active_density_event(selected_region)
    
    # Define status colors
    status_colors = {
        'Normal': '#27AE60',
        'Warning': '#F39C12',
        'Critical': '#E74C3C',
        'No region selected': '#7F8C8D'
    }
    
    status = 'Normal'  # Replace with actual logic
    if density_event:
        status = density_event['status']
    
    return (
        str(count),
        {'color': '#2C3E50'},
        status,
        {'color': status_colors.get(status, '#7F8C8D')}
    )

# Callback for count history graph - update styling
@app.callback(
    Output('count-history', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('region-dropdown', 'value')]
)
def update_count_history(_, selected_region):
    if not selected_region:
        return go.Figure()
    
    # Get your data here
    logs = get_logs(selected_region)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[log['timestamp'] for log in logs],
        y=[log['count'] for log in logs],
        mode='lines+markers',
        name='Count',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Historical Count',
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title='Time',
        yaxis_title='Count',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)