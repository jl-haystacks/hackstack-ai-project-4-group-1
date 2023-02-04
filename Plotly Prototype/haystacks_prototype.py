# Initialize Dash application by importing necessary packages
import dash # v 1.16.2
from dash import dcc # v 1.12.1
import dash_bootstrap_components as dbc # v 0.10.3
from dash import html # v 1.1.1
from jupyter_dash import JupyterDash
import pandas as pd
import plotly.express as px # plotly v 4.7.1
import plotly.graph_objects as go
import numpy as np
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# Define external stylesheets and apply them
external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Prototype Real Estate Dashboard (Haystacks)', external_stylesheets=[external_stylesheets])

# Read in external data into dash application
df = pd.read_csv('Data/Raw/haystacks_ga_clean_new_format.csv')
df1 = df.copy()
################################# Need to integrate below mask into pre-processing and take it out of here
mask = {'F': 0,
       'D-': 1,
       'D': 2,
       'D+': 3,
       'C-': 4,
       'C': 5,
       'C+': 6,
       'B-': 7,
       'B': 8,
       'B+': 9,
       'A-': 10,
       'A': 11}
df['overall_crime_grade'] = df['overall_crime_grade'].apply(lambda row: mask[row])
df['property_crime_grade'] = df['property_crime_grade'].apply(lambda row: mask[row])
#################################

# Define features to be utilized for generating scatter plots
# features = ['HS_rating', 'MS_rating', 'ES_rating', 'overall_crime_grade',
# 'property_crime_grade', 'rent', 'beds', 'baths_full',
# 'baths_half', 'square_footage', 'lot_size', 'year_built']

# Reordering below... will delete above when it feels safe.

features = ['listing_status', 'square_footage', 'overall_crime_grade',
       'ES_rating', 'lot_size', 'baths_half', 'MS_rating', 'HS_rating',
       'listing_special_features', 'beds', 'baths_full', 'year_built',
       'property_crime_grade', 'transaction_type']

# Define sales prices based on models to be utilized for generating scatter plots
models = ['price']

# Calculate averages of selected features
df_average = df[features].mean()

# Calculate maximum value of entire dataframe to set limit of point plot
max_val = df.max()

# Load models indexed by zipcode
# Includes a model, a shap_value object, and the corresponding explainer

# Series of zipcode data frames. May want to save elsewhere.
zip_dfs = pd.Series([], dtype='O')
for zipcode in set(df.zipcode.values):
    zip_dfs[zipcode] = df.loc[df.zipcode == zipcode]

# Read pickle file that can be obtained by running the first half or so of MLR.ipynb
MLR_MS_df = pd.read_pickle('Data/Pickle/MLR_modNshap.P')

## Predictions... should probably pre-load for each model.
df['MLR_price'] = df.apply(lambda row: MLR_MS_df.loc[row.zipcode,'model'].predict(row[features].to_numpy().reshape(1,-1)).item(), axis=1)
df['MLR_caprate'] = 100*12*(df.rent/df.MLR_price)

#############################################################################################

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
                html.Label('Model'),], style={'font-size': '18px'}),
            # Model selection dropdown menu based on those defined at the beginning
            dcc.Dropdown(
                id='crossfilter-model',
                options=[
                    #{'label': 'Final Sale Price', 'value': 'price'},
                    {'label': 'Multiple Linear Regression: Price', 'value': 'MLR_price'},
                    #{'label': 'Multiple Linear Regression: Caprate', 'value': 'MLR_caprate'},
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
            # Zip code selection
            html.Div([
                html.Label('Resolution'),], style={'font-size': '18px'}),
            dcc.Dropdown(
                id='crossfilter-resolution',
                options=[
                    {'label': 'State', 'value': 'state'},
                    #{'label': 'County', 'value': 'county'},
                    {'label': 'Zip code', 'value': 'zipcode'},
                ],
                value='state',
                clearable=False, style={'width': '49%', 'display': 'inline-block'}

            ),
            html.Div(id = 'reso_list', 
                     children = [dcc.Dropdown(id='filter-dropdown',
                        value= 'Georgia')
            ], style={'width': '49%', 'display': 'inline-block'}
        ), 
    
        ], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),
    html.Div([
        html.Div([
            # Scatter plot of correlation between selected feature and sale price
            dcc.Graph(
                id='scatter-plot',
                # Update graph on click
                clickData={'points': [{'customdata': 0}]}
                # Update graph on hover
                # hoverData={'points': [{'customdata': 0}]}
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

################################################################################################

@app.callback(
    dash.dependencies.Output('reso_list', 'children'),
    dash.dependencies.Input('crossfilter-resolution', 'value')
)
## Creates dynamic dropdown. Effects change in scatter-plot as well.
def update_scale(resolution_dropdown):
    new_dropdown = dcc.Dropdown(
    list(set(df[resolution_dropdown])),
    id='filter-dropdown',
        value=df[resolution_dropdown].values[0]
    )
    return new_dropdown

## can't use shap figures without adjustments

# @app.callback(
#     dash.dependencies.Output('shap-bee', 'figure'),
#     [
#         dash.dependencies.Input('filter-dropdown','value'),
#         #dash.dependencies.State('crossfilter-resolution', 'value') ## Useful if we add county as well
#         #dash.dependencies.Input('crossfilter-model', 'State')    ## need to figure out a good way to do this. Will probably use a Series of those model dataframes
#         dash.dependencies.State('shap-bee', 'figure')
#     ]
# )
# def update_shap(focus, current):
#     if focus == 'Georgia':
#         return current
#     fig = shap.plots.beeswarm(MLR_MS_df.loc[focus,'shap_values'], max_display = None) 
#     return fig


# First callback to coordinate scatterplot with feature, model, and color gradient schema
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-feature', 'value'),
        dash.dependencies.Input('crossfilter-model', 'value'),
        dash.dependencies.Input('gradient-scheme', 'value'),
        dash.dependencies.State('crossfilter-resolution', 'value'),
        dash.dependencies.Input('filter-dropdown','value')
    ]
)

# Function to update graph in response to feature, model, and color gradient schema being updated
def update_graph(feature, model, gradient, resolution, focus):
    # Filter dataframe 
    df1 = df.loc[df[resolution] == focus]
    # Define points and respective colors of scatter plot
    cols = df1[feature].values #df[feature].values
    hover_names = []
    for ix, val in zip(df1.index.values, df1[feature].values):
        hover_names.append(f'Customer {ix:03d}<br>{feature} value of {val}')
    # Generate scatter plot
    fig = px.scatter(
        df1,
        x=df1[feature],
        y=df1[model],
        color=cols,
        opacity=0.8,
        hover_name=hover_names,
        hover_data=features,
        template='plotly_dark',
        color_continuous_scale=gradient,
    )
    # Update feature information when user hovers over data point
    # customdata_set = list(df[features].to_numpy())
    fig.update_traces(customdata=df1.index)
    
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
    for ix, val in zip(df.index.values, df[feature].values):
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
        # Update graph on click
        dash.dependencies.Input('scatter-plot', 'clickData')
        # Update graph on hover
        # dash.dependencies.Input('scatter-plot', 'hoverData')
    ]
)

# Function to trigger last function in response to user click point in scatter plot
def update_point_plot(clickData):
    index = clickData['points'][0]['customdata']
    title = f'Customer {index}'
    return create_point_plot(df[features].iloc[index], title)

# Function to trigger last function in response to user hover over point in scatter plot
# def update_point_plot(hoverData):
#     index = hoverData['points'][0]['customdata']
#     title = f'Customer {index}'
#     return create_point_plot(df[features].iloc[index], title)

# The final two critical lines of code to RUN the application
if __name__ == '__main__':
    app.run_server(debug=True)