import dash
from dash import html, dcc, Input, Output
import requests

app = dash.Dash(__name__)

currencies = ['USD', 'GBP', 'JPY', 'CHF', 'SEK']

# Endpoints requiring currency param
currency_endpoints = [
    'bandwidth_volatility',
    'cip_deviation_vs_band',
]

# Global endpoints that work (only cip_deviations for now)
global_endpoints = [
    'cip_deviations',
]

app.layout = html.Div([
    html.H1("Financial Analysis Dashboard"),

    # Currency dropdown
    html.Label("Select Currency:"),
    dcc.Dropdown(
        id='currency-dropdown',
        options=[{'label': c, 'value': c} for c in currencies],
        value=currencies[0]  # default selection
    ),

    html.H2("Currency-Specific Charts"),
    html.Div(id='currency-charts'),

    html.H2("Global Charts"),
    html.Div(id='global-charts')
])

@app.callback(
    Output('currency-charts', 'children'),
    Input('currency-dropdown', 'value')
)
def update_currency_charts(selected_currency):
    charts = []
    for endpoint in currency_endpoints:
        url = f"http://localhost:8000/api/charts/{endpoint}?currency={selected_currency}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Extract image from nested 'data' key
            img_data = data.get('data', {}).get('image', '')
            if img_data:
                img_src = f"data:image/png;base64,{img_data}"
                charts.append(html.Div([
                    html.H4(endpoint.replace('_', ' ').title()),
                    html.Img(src=img_src, style={'width': '600px', 'margin-bottom': '30px'})
                ]))
            else:
                charts.append(html.Div(f"No image data for {endpoint}"))
        else:
            charts.append(html.Div(f"Failed to load {endpoint}"))

    return charts

@app.callback(
    Output('global-charts', 'children'),
    Input('currency-dropdown', 'value')  # dummy input to trigger once on load
)
def update_global_charts(_):
    charts = []
    for endpoint in global_endpoints:
        url = f"http://localhost:8000/api/charts/{endpoint}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Extract image from nested 'data' key
            img_data = data.get('data', {}).get('image', '')
            if img_data:
                img_src = f"data:image/png;base64,{img_data}"
                charts.append(html.Div([
                    html.H4(endpoint.replace('_', ' ').title()),
                    html.Img(src=img_src, style={'width': '600px', 'margin-bottom': '30px'})
                ]))
            else:
                charts.append(html.Div(f"No image data for {endpoint}"))
        else:
            charts.append(html.Div(f"Failed to load {endpoint}"))

    return charts

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8056))
    app.run_server(debug=True, port=port, host='0.0.0.0')