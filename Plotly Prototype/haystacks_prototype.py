# Initialize dash application by importing necessary packages
import dash # v 1.16.2
from dash import dcc # v 1.12.1
import dash_bootstrap_components as dbc # v 0.10.3
from dash import html # v 1.1.1
import pandas as pd
import plotly.express as px # plotly v 4.7.1
import plotly.graph_objects as go
import numpy as np

# Define external stylesheets and apply them
external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Prototype Real Estate Dashboard (Haystacks)', external_stylesheets=[external_stylesheets])

# Read in external data into dash application
df = pd.read_csv('Data/clean_Ames_HousePrice.csv')

# Define features to be utilized for generating scatter plots
features = ['GrLivArea', 'LotFrontage', 'LotArea', 'GarageArea']

# Define sales prices based on models to be utilized for generating scatter plots
models = ['SalePrice']

# Calculate averages of selected features
df_average = df[features].mean()

# Calculate maximum value of entire dataframe to set limit of point plot
max_val = df.max()

#Define layout of dash application
app.layout = html.Div([
    # Title and caption
    html.Div(
        children=[
            html.H1(
                children="Real Estate Analytics"
                ),
            html.P(
                children="Analyze the final sale price of homes"
                " based on quantitative and categorical features.",
            ),
        ]
    ),
    html.Div([

        html.Div([
            # Model selection caption
            html.Div([
                html.Label('Model Selection'),], style={'font-size': '18px'}),
            # Model selection dropdown menu based on those defined at the beginning
            dcc.Dropdown(
                id='crossfilter-model',
                options=[
                    {'label': 'Final Sale Price', 'value': 'SalePrice'},
                ],
                value=models[0],
                clearable=False

            )], style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div([
            # Feature selection caption
            html.Div([
                html.Label('Feature Selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),
            # Radio button for color schema of scatter plots
            html.Div([
                dcc.RadioItems(
                    id='gradient-scheme',
                    options=[
                        {'label': 'Orange to Red', 'value': 'OrRd'},
                        {'label': 'Viridis', 'value': 'Viridis'},
                        {'label': 'Plasma', 'value': 'Plasma'}
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            # Feature selection dropdown menu based on those defined at the beginning
            dcc.Dropdown(
                id='crossfilter-feature',
                options=[{'label': i, 'value': i} for i in features],
                value=features[0],
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
        
        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),
    html.Div([
        html.Div([
            # Scatter plot of correlation between selected feature and sale price
            dcc.Graph(
                id='scatter-plot',
                hoverData={'points': [{'customdata': 0}]}
            )
        ], style={'width': '74%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([
            # Place SHAP values here
            html.H1(
                children="SHAP Values"
                ),
        ], style={'width': '24%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),
    ]),
    # Point plot of average values of selected features
    html.Div([
        dcc.Graph(id='point-plot'),
    ], style={'display': 'inline-block', 'width': '100%'}),

    ], style={'backgroundColor': 'rgb(17, 17, 17)'},
)

# First callback to coordinate scatterplot with feature, model, and color gradient schema
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-feature', 'value'),
        dash.dependencies.Input('crossfilter-model', 'value'),
        dash.dependencies.Input('gradient-scheme', 'value')
    ]
)

# Function to update graph in response to feature, model, and color gradient schema being updated
def update_graph(feature, model, gradient):
    # Define points and respective colors of scatter plot
    cols = df[feature].values #df[feature].values
    hover_names = []
    for ix, val in zip(df["PID"].values, df[feature].values):
        hover_names.append(f'Customer {ix:03d}<br>{feature} value of {val}')
    # Generate scatter plot
    fig = px.scatter(
        df,
        x=df[feature],
        y=df[model],
        color=cols,
        opacity=0.8,
        hover_name=hover_names,
        hover_data=features,
        template='plotly_dark',
        color_continuous_scale=gradient,
    )
    # Update feature information when user hovers over data point
    # customdata_set = list(df[features].to_numpy())
    fig.update_traces(customdata=df.index)
    
    # Layout of scatter plot
    fig.update_layout(
        coloraxis_colorbar={'title': f'{feature}'},
        coloraxis_showscale=True,
        legend_title_text='Spend',
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        template='plotly_dark'
    )
    # Axis settings for scatter plot
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    return fig

# Function to create point plot of averages of selected features
def create_point_plot(df, title):

    fig = go.Figure(
        data=[
            go.Bar(name='Average', x=features, y=df_average.values, marker_color='#c178f6'),
            go.Bar(name=title, x=features, y=df.values, marker_color='#89efbd')
        ]
    )
    # Bar layout for point plot
    fig.update_layout(
        barmode='group',
        height=225,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
        template='plotly_dark'
    )
    # Axis settings for point plot
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(type="log", range=[0,5])

    return fig

# Callback to update point plot based on user hovering over points in scatter plot
@app.callback(
    dash.dependencies.Output('point-plot', 'figure'),
    [
        dash.dependencies.Input('scatter-plot', 'hoverData')
    ]
)

# Function to trigger last function in response to user hover over point in scatter plot
def update_point_plot(hoverData):
    index = hoverData['points'][0]['customdata']
    title = f'Customer {index}'
    return create_point_plot(df[features].iloc[index], title)

# The final two critical lines of code to RUN the application
if __name__ == '__main__':
    app.run_server(debug=True)